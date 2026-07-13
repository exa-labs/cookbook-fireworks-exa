"""Reward for the Exa search agent: LLM-as-judge answer grading.

An LLM judge grades the final answer against the gold answer(s), 1.0 or 0.0.
Multi-hop QA gold answers are short free-text strings that correct models
routinely rephrase, so exact matching under-counts (and the agent learns to
game it); the judge is any OpenAI-compatible endpoint, configured via
``JudgeConfig`` or environment variables:

    JUDGE_MODEL             (default: accounts/fireworks/models/qwen3p7-plus)
    JUDGE_BASE_URL          (default: https://api.fireworks.ai/inference/v1)
    JUDGE_API_KEY           (default: $FIREWORKS_API_KEY)
    JUDGE_REASONING_EFFORT  (default: "none"; set empty to omit the param)

The judge must answer without a reasoning trace, or the small ``max_tokens``
truncates the verdict: use a non-thinking instruct model, or a hybrid model
with reasoning disabled (``reasoning_effort="none"``).

A judge call that fails or returns an unparseable verdict raises ``JudgeError``;
the caller drops that trajectory. ``max_consecutive_failures`` failures in a row
raise a fatal error instead, so a misconfigured judge fails loudly rather than
silently draining the dataset.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Sequence

logger = logging.getLogger(__name__)


class JudgeError(RuntimeError):
    """The judge could not produce a verdict for one candidate answer."""


_JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader for a question-answering system. You are given a "
    "question, the reference (gold) answer(s), and a candidate answer produced "
    "by a model. Decide whether the candidate answer is correct.\n\n"
    "A candidate is CORRECT if it conveys the same factual answer as any gold "
    "answer, allowing for paraphrase, extra detail, or different formatting. It "
    "is INCORRECT if it states a different fact, is missing, hedges without "
    "committing, or contradicts the gold answer.\n\n"
    'Respond with a single word: "CORRECT" or "INCORRECT".'
)

_JUDGE_USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Gold answer(s):\n{gold}\n\n"
    "Candidate answer:\n{candidate}\n\n"
    "Is the candidate answer correct? Reply with CORRECT or INCORRECT."
)

_CORRECT_RE = re.compile(r"\bCORRECT\b", re.IGNORECASE)
_INCORRECT_RE = re.compile(r"\bINCORRECT\b", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


@dataclass
class JudgeConfig:
    """Configuration for the LLM judge."""

    model: str = field(
        default_factory=lambda: os.environ.get(
            "JUDGE_MODEL",
            "accounts/fireworks/models/qwen3p7-plus",
        )
    )
    reasoning_effort: str | None = field(
        default_factory=lambda: os.environ.get("JUDGE_REASONING_EFFORT", "none") or None
    )
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "JUDGE_BASE_URL", "https://api.fireworks.ai/inference/v1"
        )
    )
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("JUDGE_API_KEY")
        or os.environ.get("FIREWORKS_API_KEY")
    )
    max_tokens: int = 16
    temperature: float = 0.0
    timeout: float = 30.0
    # Consecutive failures before raising a fatal error instead of dropping the
    # trajectory -- guards against a misconfigured judge silently draining data.
    max_consecutive_failures: int = 10


class AnswerJudge:
    """Grades a candidate answer against gold answer(s) with an LLM judge.

    Returns 1.0 (correct) or 0.0 (incorrect). A failed or unparseable judge
    call raises ``JudgeError`` so the caller can drop that one trajectory;
    ``max_consecutive_failures`` in a row raise a fatal error instead.
    """

    def __init__(self, config: JudgeConfig | None = None):
        self.config = config or JudgeConfig()
        self._client = None
        self._consecutive_failures = 0

    def _get_client(self):
        if self._client is None:
            if not self.config.api_key:
                raise RuntimeError(
                    "No judge API key. Set FIREWORKS_API_KEY (or JUDGE_API_KEY)."
                )
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
        return self._client

    def _register_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.config.max_consecutive_failures:
            raise RuntimeError(
                f"LLM judge failed {self._consecutive_failures} times in a row "
                f"(model={self.config.model!r}, base_url={self.config.base_url!r}). "
                "Check the judge model id and API key."
            )

    async def __call__(
        self, question: str, answer: str, gold_answers: Sequence[str]
    ) -> float:
        """Return 1.0 if ``answer`` is judged correct for ``question``, else 0.0.

        Raises ``JudgeError`` if the judge call fails or is unparseable.
        """
        answer = (answer or "").strip()
        gold_answers = [g for g in gold_answers if g] or [""]
        if not answer:
            return 0.0

        client = self._get_client()  # missing key -> RuntimeError (fatal config error)
        gold_str = "\n".join(f"- {g}" for g in gold_answers)
        try:
            resp = await client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _JUDGE_USER_TEMPLATE.format(
                            question=question, gold=gold_str, candidate=answer
                        ),
                    },
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                extra_body=(
                    {"reasoning_effort": self.config.reasoning_effort}
                    if self.config.reasoning_effort
                    else {}
                ),
            )
            verdict = _THINK_RE.sub("", resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM judge call failed", exc_info=True)
            self._register_failure()
            raise JudgeError(f"judge call failed: {exc}") from exc

        # Check INCORRECT first: it contains the substring "CORRECT".
        if _INCORRECT_RE.search(verdict):
            self._consecutive_failures = 0
            return 0.0
        if _CORRECT_RE.search(verdict):
            self._consecutive_failures = 0
            return 1.0

        logger.warning("Judge returned unparseable verdict %r", verdict)
        self._register_failure()
        raise JudgeError(f"unparseable verdict: {verdict!r}")
