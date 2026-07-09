"""Reward for the Exa search agent: LLM-as-judge answer grading.

An LLM judge grades the final answer against the gold answer(s), 1.0 or 0.0.
Multi-hop QA gold answers are short free-text strings that correct models
routinely rephrase, so exact matching under-counts; the judge is any
OpenAI-compatible endpoint, configured via ``JudgeConfig`` or environment
variables:

    JUDGE_MODEL       (default: accounts/fireworks/models/qwen3-30b-a3b-instruct-2507)
    JUDGE_BASE_URL    (default: https://api.fireworks.ai/inference/v1)
    JUDGE_API_KEY     (default: $FIREWORKS_API_KEY)

The judge must be a non-thinking instruct model: a reasoning judge opens with
``<think>`` and the small ``max_tokens`` would truncate the verdict.
``token_f1`` is the offline fallback (``JudgeConfig(enabled=False)``).
"""

from __future__ import annotations

import logging
import os
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-F1 fallback (SQuAD-style; no network, no API key)
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    """SQuAD-style token-level F1 between a prediction and one gold answer."""
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def best_token_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    """Max token-F1 over a set of acceptable gold answers."""
    return max((token_f1(prediction, g) for g in gold_answers), default=0.0)


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

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

    enabled: bool = True
    model: str = field(
        default_factory=lambda: os.environ.get(
            "JUDGE_MODEL",
            "accounts/fireworks/models/qwen3-30b-a3b-instruct-2507",
        )
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
    # Failures in a row before raising instead of falling back to F1.
    max_consecutive_failures: int = 10


class AnswerJudge:
    """Grades a candidate answer against gold answer(s) with an LLM judge.

    Isolated judge failures fall back to token-F1; ``max_consecutive_failures``
    failures in a row raise instead of silently mis-grading the run.
    """

    def __init__(self, config: JudgeConfig | None = None, *, f1_threshold: float = 0.6):
        self.config = config or JudgeConfig()
        self.f1_threshold = f1_threshold
        self._client = None
        self._consecutive_failures = 0

    def _get_client(self):
        if self._client is None:
            # Lazy import: `token_f1` works with no optional deps.
            from openai import AsyncOpenAI

            if not self.config.api_key:
                raise RuntimeError(
                    "No judge API key. Set FIREWORKS_API_KEY (or JUDGE_API_KEY), "
                    "or disable the judge with JudgeConfig(enabled=False)."
                )
            self._client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
        return self._client

    def _f1_reward(self, answer: str, gold_answers: Sequence[str]) -> float:
        return 1.0 if best_token_f1(answer, gold_answers) >= self.f1_threshold else 0.0

    def _note_failure(self, answer: str, gold_answers: Sequence[str]) -> float:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.config.max_consecutive_failures:
            raise RuntimeError(
                f"LLM judge failed {self._consecutive_failures} times in a row "
                f"(model={self.config.model!r}, base_url={self.config.base_url!r}). "
                "Refusing to silently grade the run with the F1 fallback -- "
                "check the judge model id and API key, or pass --no-judge."
            )
        return self._f1_reward(answer, gold_answers)

    async def __call__(
        self, question: str, answer: str, gold_answers: Sequence[str]
    ) -> float:
        """Return 1.0 if ``answer`` is judged correct for ``question``, else 0.0."""
        answer = (answer or "").strip()
        gold_answers = [g for g in gold_answers if g] or [""]
        if not answer:
            return 0.0

        if not self.config.enabled:
            return self._f1_reward(answer, gold_answers)

        try:
            client = self._get_client()
            gold_str = "\n".join(f"- {g}" for g in gold_answers)
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
            )
            verdict = _THINK_RE.sub("", resp.choices[0].message.content or "").strip()
            # Check INCORRECT first: it contains the substring "CORRECT".
            if _INCORRECT_RE.search(verdict):
                self._consecutive_failures = 0
                return 0.0
            if _CORRECT_RE.search(verdict):
                self._consecutive_failures = 0
                return 1.0
            logger.warning("Judge returned unparseable verdict %r; using F1 fallback", verdict)
            return self._note_failure(answer, gold_answers)
        except RuntimeError:
            raise
        except Exception:
            logger.warning("LLM judge call failed; using token-F1 fallback", exc_info=True)
            return self._note_failure(answer, gold_answers)
