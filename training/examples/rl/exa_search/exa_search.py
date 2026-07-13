"""Exa web-search tools for the RL search agent.

Two tools, declared as OpenAI-compatible function specs and rendered through
the model's chat template:

- ``search(query)``     -- Exa web search; returns ranked result snippets.
- ``get_contents(url)`` -- fetch the contents of one result page.

There is no ``submit_answer`` tool: an episode ends when the model replies
without calling a tool.  Rollouts sample at the token level, so the model's
Hermes-style ``<tool_call>{...}</tool_call>`` blocks are parsed client-side
by ``parse_tool_calls``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

TOOL_NAME_SEARCH = "search"
TOOL_NAME_GET_CONTENTS = "get_contents"

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions using web search.\n"
    "1. If you lack the knowledge to answer, call the `search` tool to find "
    "relevant web pages.\n"
    "2. If a search snippet looks promising but is not enough, call "
    "`get_contents` with that result's URL to read the page.\n"
    "3. Break multi-hop questions into sub-questions and search for each hop.\n"
    "4. Once you have all the information you need, reply WITHOUT calling any "
    'tool, and include your final answer after the prefix "Answer:". The '
    "answer should be concise -- a name, date, number, or short phrase."
)

# Injected before the last turn in both training (rollout.py) and eval
# (eval.py); the two loops must present the identical environment.
FINAL_TURN_NUDGE = (
    "You are out of tool-call turns. Reply now without calling any tool and "
    'give your final answer after the prefix "Answer:".'
)

EXA_SEARCH_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME_SEARCH,
            "description": (
                "Search the web. Returns the most relevant pages with a short "
                "snippet from each."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME_GET_CONTENTS,
            "description": (
                "Fetch the contents of a web page from a previous search "
                "result. Use this when a snippet looks relevant but you need "
                "the full page to answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of a page returned by search.",
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Exa search tools
# ---------------------------------------------------------------------------

@dataclass
class ExaSearchConfig:
    """Knobs for the Exa search backend.

    ``search_type`` maps to Exa's ``type`` parameter; see
    https://docs.exa.ai/reference/search.
    """

    api_key: str | None = None
    search_type: str = "auto"
    num_results: int = 5
    max_snippet_chars: int = 2000
    max_contents_chars: int = 10000
    max_observation_chars: int = 15000
    max_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    cache_size: int = 4096
    # Exa accounts are rate-limited (default 10 requests/second); pace all
    # API calls across concurrent rollouts to stay under it.  0 disables.
    max_requests_per_second: float = 10.0


class _AsyncRateLimiter:
    """Evenly spaces acquisitions at ``rate`` per second across coroutines."""

    def __init__(self, rate: float):
        self._interval = 1.0 / rate
        self._lock = asyncio.Lock()
        self._next_free = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next_free)
            self._next_free = start + self._interval
            wait = start - now
        if wait > 0:
            await asyncio.sleep(wait)


class ExaSearchTool:
    """Async wrapper over the Exa SDK that returns compact text observations.

    ``search``/``get_contents`` retry transient failures and never raise:
    after exhausted retries they return a short error string so the rollout
    can continue.
    """

    def __init__(self, config: ExaSearchConfig | None = None):
        self.config = config or ExaSearchConfig()
        api_key = self.config.api_key or os.environ.get("EXA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "EXA_API_KEY is not set. Get one at https://dashboard.exa.ai/api-keys "
                "and put it in your .env (EXA_API_KEY=...)."
            )
        # Lazy import: the parser and prompts are usable without the SDK.
        from exa_py import AsyncExa

        self._exa = AsyncExa(api_key=api_key)
        self._cache: dict[tuple, str] = {}
        self._limiter = (
            _AsyncRateLimiter(self.config.max_requests_per_second)
            if self.config.max_requests_per_second > 0
            else None
        )

    async def search(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "No query provided. Call search with a non-empty query string."
        cache_key = ("search", query, self.config.search_type, self.config.num_results)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            result = await self._with_retries(
                lambda: self._exa.search(
                    query,
                    type=self.config.search_type,
                    num_results=self.config.num_results,
                    contents={"highlights": True},
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Exa search failed for %r: %s", query, exc)
            return f"Search error: {exc}. Try a different query."
        observation = self._format_search_results(getattr(result, "results", []) or [])
        self._cache_put(cache_key, observation)
        return observation

    async def get_contents(self, url: str) -> str:
        url = (url or "").strip()
        if not url:
            return "No URL provided. Call get_contents with a URL from a search result."
        cache_key = ("contents", url)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            result = await self._with_retries(
                lambda: self._exa.get_contents(
                    [url],
                    text={"max_characters": self.config.max_contents_chars},
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Exa get_contents failed for %r: %s", url, exc)
            return f"Fetch error: {exc}. Try a different URL or another search."
        results = getattr(result, "results", []) or []
        if not results:
            return "No contents found for that URL. Try a different URL or another search."
        page = results[0]
        observation = json.dumps(
            {
                "url": getattr(page, "url", url) or url,
                "title": (getattr(page, "title", None) or "Untitled").strip(),
                "contents": ((getattr(page, "text", None) or "").strip())[
                    : self.config.max_contents_chars
                ],
            },
            ensure_ascii=False,
        )[: self.config.max_observation_chars]
        self._cache_put(cache_key, observation)
        return observation

    async def _with_retries(self, thunk):
        last_exc: Exception | None = None
        for attempt in range(self.config.max_attempts):
            try:
                if self._limiter is not None:
                    await self._limiter.acquire()
                return await thunk()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt + 1 < self.config.max_attempts:
                    await asyncio.sleep(self.config.retry_backoff_seconds * (2 ** attempt))
        assert last_exc is not None
        raise last_exc

    def _cache_put(self, key: tuple, value: str) -> None:
        if len(self._cache) >= self.config.cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = value

    def _format_search_results(self, results: list[Any]) -> str:
        if not results:
            return "No results found. Try a different query."
        hits = []
        for i, r in enumerate(results, 1):
            highlights = getattr(r, "highlights", None) or []
            snippet = " ... ".join(h.strip() for h in highlights if h)
            if not snippet:
                snippet = (getattr(r, "text", None) or "").strip()
            hits.append(
                {
                    "id": i,
                    "title": (getattr(r, "title", None) or "Untitled").strip(),
                    "url": getattr(r, "url", "") or "",
                    "snippet": snippet[: self.config.max_snippet_chars],
                }
            )
        return json.dumps(hits, indent=1, ensure_ascii=False)[
            : self.config.max_observation_chars
        ]


# ---------------------------------------------------------------------------
# Tool-call parsing (Hermes-style <tool_call>{...}</tool_call> blocks)
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_TAIL_RE = re.compile(r"<think>.*", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks (e.g. Qwen3)."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_TAIL_RE.sub("", text)
    return text.strip()


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]
    tool_call_id: str = field(
        default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}"
    )


def _decode_call_object(obj: Any) -> ParsedToolCall | None:
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("name", "")).strip()
    if not name:
        return None
    arguments = obj.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return ParsedToolCall(name=name, arguments=arguments)


def parse_tool_calls(output_text: str) -> tuple[list[ParsedToolCall], bool]:
    """Parse Hermes-style tool calls from raw model output.

    Returns ``(calls, had_invalid)`` in emission order.  No calls and no
    invalid blocks means the message is a final answer.  A bare top-level
    ``{"name": ..., "arguments": ...}`` object naming a known tool is also
    accepted (a common early-training slip).
    """
    text = strip_think(output_text)
    calls: list[ParsedToolCall] = []
    had_invalid = False

    blocks = _TOOL_CALL_RE.findall(text)
    for block in blocks:
        brace = block.find("{")
        if brace < 0:
            had_invalid = True
            continue
        try:
            obj, _ = json.JSONDecoder().raw_decode(block[brace:])
        except json.JSONDecodeError:
            had_invalid = True
            continue
        call = _decode_call_object(obj)
        if call is None:
            had_invalid = True
        else:
            calls.append(call)

    if not blocks:
        brace = text.find("{")
        if brace >= 0:
            try:
                obj, _ = json.JSONDecoder().raw_decode(text[brace:])
            except json.JSONDecodeError:
                obj = None
            call = _decode_call_object(obj)
            if call is not None and call.name in (TOOL_NAME_SEARCH, TOOL_NAME_GET_CONTENTS):
                calls.append(call)

    return calls, had_invalid


_ANSWER_PREFIX_RE = re.compile(r"answer\s*:", re.IGNORECASE)


def extract_answer(output_text: str) -> str:
    """Text after the last ``Answer:`` prefix, else the whole message."""
    text = strip_think(output_text)
    matches = list(_ANSWER_PREFIX_RE.finditer(text))
    if matches:
        return text[matches[-1].end():].strip()
    return text.strip()
