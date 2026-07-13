"""Multi-turn Exa search-agent rollout for the async RL recipe.

One ``rollout_fn`` call = one trajectory: the model calls the ``search`` /
``get_contents`` tools, reads back ``role="tool"`` observations, and ends the
episode by replying without a tool call; an LLM judge grades that final
answer.  The loss mask is ``1`` on assistant tokens, ``0`` on the prompt and
tool observations.

Terminal rewards: ``judge(answer)`` on a final answer,
``context_overflow_penalty`` when the trajectory would exceed
``max_trajectory_tokens``, ``no_answer_penalty`` when ``max_turns`` pass
without an answer.  Recoverable failures return ``None`` (drop the
trajectory) -- raising here would abort the whole training run.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Sequence

from training.examples.rl.exa_search.exa_search import (
    EXA_SEARCH_TOOLS,
    TOOL_NAME_GET_CONTENTS,
    TOOL_NAME_SEARCH,
    ExaSearchConfig,
    ExaSearchTool,
    extract_answer,
    parse_tool_calls,
)
from training.examples.rl.exa_search.reward import AnswerJudge, JudgeConfig, JudgeError
from training.examples.rl.vanilla_sampler import build_deployment_sampler
from training.utils.rl.rollout import (
    MessageTrajectoryAssembler,
    RolloutRun,
    RolloutSample,
    TITOTokenizer,
)

if TYPE_CHECKING:
    from training.recipes.async_rl_loop import RolloutFn, RolloutSetup

logger = logging.getLogger(__name__)

_PARSE_FEEDBACK = (
    "Your tool call was malformed. Call one tool per turn as "
    '<tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call> '
    'or <tool_call>{"name": "get_contents", "arguments": {"url": "..."}}'
    "</tool_call>, or reply without any tool call to give your final answer."
)

_FINAL_TURN_NUDGE = (
    "You are out of tool-call turns. Reply now without calling any tool and "
    'give your final answer after the prefix "Answer:".'
)


def _completion_logprobs(completion, *, attr: str) -> list[float] | None:
    """Per-output-token logprobs, or ``None`` if missing or misaligned.

    Handles echoed-prompt logprobs and their off-by-one alignment.
    """
    values = getattr(completion, attr, None)
    if values is None:
        return None
    values = list(values)
    prompt_len = int(completion.prompt_len)
    output_len = len(completion.full_tokens) - prompt_len
    if getattr(completion, "logprobs_echoed", False):
        full_len = len(completion.full_tokens)
        if len(values) == full_len:
            values = values[prompt_len:]
        elif len(values) == max(0, full_len - 1):
            values = values[max(0, prompt_len - 1):]
    if len(values) != output_len or any(v is None for v in values):
        return None
    return [float(v) for v in values]


def _gold_answers(sample_prompt: dict) -> list[str]:
    """Normalize the row's gold answer(s) into a list of strings."""
    gold = sample_prompt.get("ground_truth", sample_prompt.get("answer"))
    if gold is None:
        return []
    if isinstance(gold, str):
        return [gold]
    if isinstance(gold, Sequence):
        return [str(g) for g in gold if str(g).strip()]
    return [str(gold)]


def _question_text(sample_prompt: dict, messages: list[dict]) -> str:
    """The natural-language question, for the judge prompt."""
    q = sample_prompt.get("question")
    if q:
        return str(q)
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def _append_jsonl(path: str, record: dict) -> None:
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.warning("Could not append rollout stats to %s", path, exc_info=True)


def make_rollout_fn(setup: "RolloutSetup") -> "RolloutFn":
    sampler = build_deployment_sampler(setup)
    sample_kwargs = dict(setup.sample_kwargs)
    tokenizer = setup.tokenizer
    extras = setup.extras

    stats_path = extras.get("stats_path")
    if stats_path:
        os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)

    max_turns = int(extras.get("max_turns", 6))
    if max_turns < 1:
        raise ValueError(f"max_turns must be >= 1, got {max_turns}")
    max_trajectory_tokens = int(extras.get("max_trajectory_tokens", 30720))
    max_completion_tokens = int(sample_kwargs.get("max_tokens", 0) or 0)
    overflow_penalty = float(extras.get("context_overflow_penalty", -0.25))
    no_answer_penalty = float(extras.get("no_answer_penalty", -0.1))

    exa_tool = ExaSearchTool(
        ExaSearchConfig(
            search_type=str(extras.get("search_type", "auto")),
            num_results=int(extras.get("num_results", 5)),
            max_requests_per_second=float(extras.get("exa_qps", 10.0)),
        )
    )
    judge = AnswerJudge(
        JudgeConfig(
            **({"model": extras["judge_model"]} if extras.get("judge_model") else {}),
        )
    )

    async def rollout_fn(sample_prompt: dict) -> RolloutRun | None:
        messages = list(sample_prompt.get("messages") or [])
        if not messages:
            return None
        gold_answers = _gold_answers(sample_prompt)
        question = _question_text(sample_prompt, messages)

        assembler = MessageTrajectoryAssembler(TITOTokenizer(tokenizer))
        current_messages = messages
        reward = no_answer_penalty
        done = False
        outcome = "no_answer"
        final_answer = ""
        turns_used = 0
        n_search = 0
        n_get_contents = 0
        n_parse_errors = 0

        for turn in range(max_turns):
            prompt_tokens = assembler.prepare_next_input(
                current_messages, tools=EXA_SEARCH_TOOLS,
            )

            if len(prompt_tokens) + max_completion_tokens > max_trajectory_tokens:
                if turn == 0:
                    # Nothing sampled yet, so there is no trajectory to train on.
                    logger.warning(
                        "Prompt (%d tokens) exceeds max_trajectory_tokens=%d "
                        "before the first turn; dropping row.",
                        len(prompt_tokens), max_trajectory_tokens,
                    )
                    return None
                reward = overflow_penalty
                done = True
                outcome = "overflow"
                break

            try:
                completions = await sampler.sample_with_prompt_tokens(
                    prompt_tokens, n=1, **sample_kwargs,
                )
            except Exception:  # noqa: BLE001
                logger.warning("Sampler call failed; dropping trajectory", exc_info=True)
                return None
            if not completions:
                return None
            completion = completions[0]

            prompt_len = int(completion.prompt_len)
            output_tokens = list(completion.full_tokens[prompt_len:])
            output_logprobs = _completion_logprobs(completion, attr="sampling_logprobs")
            if not output_tokens or output_logprobs is None:
                return None

            assistant_text = getattr(completion, "text", "") or tokenizer.decode(output_tokens)
            assistant_message = {"role": "assistant", "content": assistant_text}
            assembler.add_assistant_response(
                request_messages=current_messages,
                assistant_message=assistant_message,
                prompt_token_ids=prompt_tokens,
                completion_token_ids=output_tokens,
                completion_logprobs=output_logprobs,
                finish_reason=getattr(completion, "finish_reason", "stop"),
            )

            turns_used = turn + 1
            tool_calls, had_invalid = parse_tool_calls(assistant_text)
            if had_invalid:
                n_parse_errors += 1

            if not tool_calls and not had_invalid:
                # No tool call: this message is the final answer.
                answer = extract_answer(assistant_text)
                try:
                    reward = await judge(question, answer, gold_answers)
                except JudgeError:
                    logger.warning("Judge could not grade trajectory; dropping", exc_info=True)
                    return None
                done = True
                outcome = "answered"
                final_answer = answer
                break

            if turn + 1 >= max_turns:
                # Out of turns: don't pay for calls whose results are never seen.
                break

            next_messages: list[dict] = []
            for call in tool_calls:
                if call.name == TOOL_NAME_SEARCH:
                    n_search += 1
                    observation = await exa_tool.search(str(call.arguments.get("query", "")))
                elif call.name == TOOL_NAME_GET_CONTENTS:
                    n_get_contents += 1
                    observation = await exa_tool.get_contents(str(call.arguments.get("url", "")))
                else:
                    observation = (
                        f"Unknown tool {call.name!r}. Available tools: "
                        f"{TOOL_NAME_SEARCH!r}, {TOOL_NAME_GET_CONTENTS!r}."
                    )
                next_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.tool_call_id,
                        "name": call.name,
                        "content": observation,
                    }
                )
            if not next_messages:
                next_messages.append({"role": "user", "content": _PARSE_FEEDBACK})

            if turn + 2 >= max_turns:
                next_messages.append({"role": "user", "content": _FINAL_TURN_NUDGE})
            current_messages = current_messages + [assistant_message] + next_messages

        if not done:
            reward = no_answer_penalty

        tokens, logprobs, loss_mask = assembler.trajectory.to_flat()
        if stats_path:
            _append_jsonl(
                stats_path,
                {
                    "ts": round(time.time(), 3),
                    "id": sample_prompt.get("id"),
                    "source": sample_prompt.get("source"),
                    "reward": reward,
                    "outcome": outcome,
                    "turns": turns_used,
                    "search_calls": n_search,
                    "get_contents_calls": n_get_contents,
                    "parse_errors": n_parse_errors,
                    "trajectory_tokens": len(tokens),
                    "answer": final_answer[:300],
                },
            )
        sample = RolloutSample(
            tokens=tokens,
            logprobs=logprobs,
            loss_mask=loss_mask,
            reward=reward,
        )
        return RolloutRun(segments=[sample])

    return rollout_fn
