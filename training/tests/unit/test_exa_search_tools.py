"""Unit tests for the exa_search RL example's pure helpers.

Covers the tool-call parser, answer extraction, tool specs, and the LLM judge
(verdict parsing, drop-on-failure, and fail-fast).  No network, no Exa/Fireworks
credentials -- the judge's OpenAI client is faked.
"""

from __future__ import annotations

import asyncio

import pytest

from training.examples.rl.exa_search.exa_search import (
    EXA_SEARCH_TOOLS,
    TOOL_NAME_GET_CONTENTS,
    TOOL_NAME_SEARCH,
    _AsyncRateLimiter,
    extract_answer,
    parse_tool_calls,
    strip_think,
)
from training.examples.rl.exa_search.reward import (
    AnswerJudge,
    JudgeConfig,
    JudgeError,
)


# ---------------------------------------------------------------------------
# Fake OpenAI-compatible client (no network)
# ---------------------------------------------------------------------------

class _FakeCompletions:
    def __init__(self, content=None, exc=None):
        self._content = content
        self._exc = exc

    async def create(self, **kwargs):
        if self._exc is not None:
            raise self._exc
        msg = type("Msg", (), {"content": self._content})()
        choice = type("Choice", (), {"message": msg})()
        return type("Resp", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, content=None, exc=None):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content, exc)})()


def _judge_with_client(content=None, exc=None, **cfg):
    judge = AnswerJudge(JudgeConfig(api_key="test-key", **cfg))
    judge._client = _FakeClient(content=content, exc=exc)  # bypass _get_client
    return judge


# ---------------------------------------------------------------------------
# Tool specs
# ---------------------------------------------------------------------------

def test_tool_specs_are_openai_function_schemas():
    assert [t["function"]["name"] for t in EXA_SEARCH_TOOLS] == [
        TOOL_NAME_SEARCH,
        TOOL_NAME_GET_CONTENTS,
    ]
    for tool in EXA_SEARCH_TOOLS:
        assert tool["type"] == "function"
        parameters = tool["function"]["parameters"]
        assert parameters["type"] == "object"
        assert parameters["required"]
        assert tool["function"]["description"]


# ---------------------------------------------------------------------------
# parse_tool_calls
# ---------------------------------------------------------------------------

def test_parse_single_tagged_call():
    calls, had_invalid = parse_tool_calls(
        '<tool_call>\n{"name": "search", "arguments": {"query": "q"}}\n</tool_call>'
    )
    assert not had_invalid
    assert [(c.name, c.arguments) for c in calls] == [("search", {"query": "q"})]


def test_parse_strips_think_blocks():
    calls, had_invalid = parse_tool_calls(
        "<think>let me search</think>"
        '<tool_call>{"name": "get_contents", "arguments": {"url": "https://x.com"}}</tool_call>'
    )
    assert not had_invalid
    assert calls[0].name == "get_contents"
    assert calls[0].arguments == {"url": "https://x.com"}


def test_parse_preserves_order_of_parallel_calls():
    calls, _ = parse_tool_calls(
        '<tool_call>{"name": "search", "arguments": {"query": "a"}}</tool_call>\n'
        '<tool_call>{"name": "search", "arguments": {"query": "b"}}</tool_call>'
    )
    assert [c.arguments["query"] for c in calls] == ["a", "b"]
    assert calls[0].tool_call_id != calls[1].tool_call_id


def test_parse_unterminated_tag_still_parses():
    calls, had_invalid = parse_tool_calls(
        '<tool_call>{"name": "search", "arguments": {"query": "x"}}'
    )
    assert not had_invalid
    assert calls[0].arguments == {"query": "x"}


def test_parse_stringified_arguments():
    calls, _ = parse_tool_calls(
        '<tool_call>{"name": "search", "arguments": "{\\"query\\": \\"x\\"}"}</tool_call>'
    )
    assert calls[0].arguments == {"query": "x"}


def test_parse_malformed_block_flags_invalid():
    calls, had_invalid = parse_tool_calls(
        '<tool_call>{"name": "search", "arguments": broken</tool_call>'
    )
    assert calls == []
    assert had_invalid


def test_parse_bare_json_fallback_only_for_known_tools():
    calls, had_invalid = parse_tool_calls(
        '{"name": "search", "arguments": {"query": "bare"}}'
    )
    assert not had_invalid
    assert calls[0].name == "search"

    calls, had_invalid = parse_tool_calls('{"result": "42"}')
    assert calls == []
    assert not had_invalid


def test_parse_plain_text_is_final_answer():
    calls, had_invalid = parse_tool_calls("The capital of France is Paris.\n\nAnswer: Paris")
    assert calls == []
    assert not had_invalid


# ---------------------------------------------------------------------------
# extract_answer / strip_think
# ---------------------------------------------------------------------------

def test_extract_answer_takes_text_after_last_answer_prefix():
    assert extract_answer("Based on the results...\nAnswer: 406 AD") == "406 AD"
    assert extract_answer("Answer: A. No wait. Answer: B") == "B"


def test_extract_answer_without_prefix_returns_whole_message():
    assert extract_answer("It was in 406.") == "It was in 406."


def test_extract_answer_strips_think():
    assert extract_answer("<think>reasoning</think>Answer: 406") == "406"


def test_strip_think_handles_closed_and_truncated_blocks():
    assert strip_think("<think>a</think>x") == "x"
    assert strip_think("x<think>truncated tail") == "x"


def test_rate_limiter_paces_concurrent_acquires():
    async def run() -> float:
        limiter = _AsyncRateLimiter(rate=100.0)
        loop = asyncio.get_running_loop()
        start = loop.time()
        await asyncio.gather(*(limiter.acquire() for _ in range(10)))
        return loop.time() - start

    # 10 acquires at 100/s occupy at least 9 x 10ms slots.
    assert asyncio.run(run()) >= 0.09


# ---------------------------------------------------------------------------
# Reward: LLM judge (faked client) -- verdicts, drop-on-failure, fail-fast
# ---------------------------------------------------------------------------

def test_judge_correct_verdict_scores_one():
    judge = _judge_with_client(content="CORRECT")
    assert asyncio.run(judge("Q", "Paris", ["Paris"])) == 1.0


def test_judge_incorrect_verdict_scores_zero():
    judge = _judge_with_client(content="INCORRECT")
    assert asyncio.run(judge("Q", "London", ["Paris"])) == 0.0


def test_judge_incorrect_wins_over_substring_correct():
    # "INCORRECT" contains "CORRECT"; the verdict must resolve to 0.0.
    judge = _judge_with_client(content="INCORRECT")
    assert asyncio.run(judge("Q", "London", ["Paris"])) == 0.0


def test_empty_answer_is_incorrect_without_calling_judge():
    judge = AnswerJudge(JudgeConfig(api_key=None))  # no client needed
    assert asyncio.run(judge("Q", "", ["Paris"])) == 0.0


def test_missing_api_key_raises_instead_of_silently_grading():
    judge = AnswerJudge(JudgeConfig(api_key=None))
    with pytest.raises(RuntimeError, match="judge API key"):
        asyncio.run(judge("Q", "Paris", ["Paris"]))


def test_unparseable_verdict_raises_judge_error():
    judge = _judge_with_client(content="maybe?")
    with pytest.raises(JudgeError, match="unparseable"):
        asyncio.run(judge("Q", "Paris", ["Paris"]))


def test_api_failure_raises_judge_error():
    judge = _judge_with_client(exc=RuntimeError("boom"))
    with pytest.raises(JudgeError, match="judge call failed"):
        asyncio.run(judge("Q", "Paris", ["Paris"]))


def test_judge_fail_fast_after_consecutive_failures():
    judge = AnswerJudge(JudgeConfig(max_consecutive_failures=3))
    judge._register_failure()  # 1
    judge._register_failure()  # 2
    with pytest.raises(RuntimeError, match="failed 3 times in a row"):
        judge._register_failure()  # 3 -> fatal


def test_judge_env_overrides(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "accounts/test/models/my-judge")
    monkeypatch.setenv("JUDGE_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "test-key")
    cfg = JudgeConfig()
    assert cfg.model == "accounts/test/models/my-judge"
    assert cfg.base_url == "http://localhost:1234/v1"
    assert cfg.api_key == "test-key"
