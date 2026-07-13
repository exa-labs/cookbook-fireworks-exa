#!/usr/bin/env python3
"""Evaluate the Exa search agent on held-out multi-hop QA, base vs. tuned.

Runs the same search/answer loop as training, but through the standard
Fireworks chat-completions API with ``tools=``, and grades answers with the
same LLM judge. Reports accuracy, searches per episode, and turns per episode
for each model, and dumps full transcripts for qualitative comparison.

By default this provisions one dedicated deployment of the base model (with
addons enabled), attaches each ``--lora`` adapter to it, evaluates every model
through it, and deletes it on exit::

    python eval.py \\
        --lora accounts/<acct>/models/exa-search-agent \\
        --dataset mixed --limit 100

Reuse an existing deployment with ``--deployment <id>`` (skips create/delete);
keep the created one with ``--keep-deployment``. Requires FIREWORKS_API_KEY
and EXA_API_KEY (training/.env is loaded automatically).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time

import httpx
from dotenv import load_dotenv

from training.examples.rl.exa_search.exa_search import (
    EXA_SEARCH_TOOLS,
    SYSTEM_PROMPT,
    TOOL_NAME_GET_CONTENTS,
    TOOL_NAME_SEARCH,
    ExaSearchConfig,
    ExaSearchTool,
    extract_answer,
    parse_tool_calls,
)
from training.examples.rl.exa_search.reward import AnswerJudge, JudgeConfig, JudgeError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_TRAINING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
load_dotenv(os.path.join(_TRAINING_ROOT, ".env"))
load_dotenv(os.path.join(_TRAINING_ROOT, "..", ".env"))

API_BASE = "https://api.fireworks.ai"
INFERENCE_BASE = "https://api.fireworks.ai/inference/v1"

_FINAL_TURN_NUDGE = (
    "You are out of tool-call turns. Reply now without calling any tool and "
    'give your final answer after the prefix "Answer:".'
)


# ---------------------------------------------------------------------------
# Control plane: deployment + LoRA addon lifecycle
# ---------------------------------------------------------------------------

class ControlPlane:
    def __init__(self, api_key: str):
        self._http = httpx.Client(
            base_url=API_BASE,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        self._account_id: str | None = None

    @staticmethod
    def _raise_with_body(resp: httpx.Response) -> None:
        if resp.status_code >= 400:
            raise RuntimeError(f"{resp.request.method} {resp.request.url} -> {resp.status_code}: {resp.text[:500]}")

    @property
    def account_id(self) -> str:
        if self._account_id is None:
            resp = self._http.get("/v1/accounts", params={"pageSize": 2})
            resp.raise_for_status()
            accounts = resp.json().get("accounts") or []
            if not accounts:
                raise RuntimeError("API key is not associated with any Fireworks account.")
            self._account_id = accounts[0]["name"].removeprefix("accounts/")
        return self._account_id

    def create_deployment(self, base_model: str, accelerator: str) -> str:
        deployment_id = f"exa-search-eval-{int(time.time()) % 10_000_000}"
        resp = self._http.post(
            f"/v1/accounts/{self.account_id}/deployments",
            params={"deploymentId": deployment_id},
            json={
                "baseModel": base_model,
                "minReplicaCount": 1,
                "maxReplicaCount": 1,
                "acceleratorType": accelerator,
                "acceleratorCount": 1,
                "enableAddons": True,
                "displayName": "exa-search eval",
            },
        )
        self._raise_with_body(resp)
        name = resp.json().get("name") or f"accounts/{self.account_id}/deployments/{deployment_id}"
        logger.info("Created deployment %s (%s)", name, accelerator)
        return name

    def wait_deployment_ready(self, name: str, timeout_s: float = 1200.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            resp = self._http.get(f"/v1/{name}")
            resp.raise_for_status()
            dep = resp.json()
            state = dep.get("state")
            ready = (dep.get("replicaStats") or {}).get("readyReplicaCount", 0)
            if state == "READY" or ready >= 1:
                logger.info("Deployment %s is ready", name)
                return
            if state in ("FAILED", "DELETING", "DELETED"):
                raise RuntimeError(f"Deployment {name} entered state {state}")
            status = (dep.get("status") or {}).get("message", "")
            logger.info("Deployment %s: %s %s...", name, state, f"({status}) " if status else "")
            time.sleep(15)
        raise TimeoutError(f"Deployment {name} not ready after {timeout_s:.0f}s")

    def load_lora(self, model: str, deployment: str, timeout_s: float = 900.0) -> str:
        deadline = time.monotonic() + timeout_s
        while True:
            resp = self._http.post(
                f"/v1/accounts/{self.account_id}/deployedModels",
                json={"model": model, "deployment": deployment},
            )
            # Addon loads are refused until the deployment leaves CREATING,
            # which can lag behind its replicas actually serving.
            if resp.status_code == 400 and "state CREATING" in resp.text:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Deployment {deployment} stuck in CREATING")
                logger.info("Deployment still CREATING; retrying addon load in 15s...")
                time.sleep(15)
                continue
            self._raise_with_body(resp)
            name = resp.json()["name"]
            logger.info("Loading LoRA %s onto %s (%s)", model, deployment, name)
            return name

    def wait_lora_ready(self, deployed_model_name: str, timeout_s: float = 600.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            resp = self._http.get(f"/v1/{deployed_model_name}")
            resp.raise_for_status()
            state = resp.json().get("state")
            if state == "DEPLOYED":
                logger.info("LoRA addon %s is deployed", deployed_model_name)
                return
            if state in ("UNDEPLOYING", "FAILED"):
                raise RuntimeError(f"Addon {deployed_model_name} entered state {state}")
            logger.info("Addon %s: %s ...", deployed_model_name, state)
            time.sleep(10)
        raise TimeoutError(f"Addon {deployed_model_name} not deployed after {timeout_s:.0f}s")

    def delete_deployment(self, name: str) -> None:
        resp = self._http.delete(f"/v1/{name}", params={"ignoreChecks": True})
        if resp.status_code >= 400:
            logger.warning("Could not delete deployment %s: %s", name, resp.text[:200])
        else:
            logger.info("Deleted deployment %s", name)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_eval_rows(dataset: str, limit: int, difficulty: str, seed: int) -> list[dict]:
    from training.examples.rl.exa_search.prepare_data import load_hotpotqa, load_musique

    rows: list[dict] = []
    if dataset == "mixed":
        half = limit // 2
        rows += load_hotpotqa("validation", difficulty, limit - half)
        rows += load_musique("validation", half)
    elif dataset == "hotpotqa":
        rows = load_hotpotqa("validation", difficulty, limit)
    elif dataset == "musique":
        rows = load_musique("validation", limit)
    random.Random(seed).shuffle(rows)
    return rows[:limit]


# ---------------------------------------------------------------------------
# Agent loop over chat completions
# ---------------------------------------------------------------------------

def _fallback_tool_calls(content: str) -> list[dict]:
    calls, _ = parse_tool_calls(content or "")
    return [
        {
            "id": c.tool_call_id,
            "type": "function",
            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
        }
        for c in calls
    ]


async def run_episode(
    client,
    model_route: str,
    row: dict,
    exa_tool: ExaSearchTool,
    judge: AnswerJudge,
    *,
    max_turns: int,
    temperature: float,
    max_tokens: int,
) -> dict:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["question"]},
    ]
    outcome = "no_answer"
    answer = ""
    correct: float | None = None
    n_search = n_get_contents = turns_used = 0

    for turn in range(max_turns):
        resp = await client.chat.completions.create(
            model=model_route,
            messages=messages,
            tools=EXA_SEARCH_TOOLS,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        msg = resp.choices[0].message
        turns_used = turn + 1

        tool_calls = [tc.model_dump() for tc in (msg.tool_calls or [])]
        if not tool_calls:
            tool_calls = _fallback_tool_calls(msg.content or "")

        if not tool_calls:
            answer = extract_answer(msg.content or "")
            messages.append({"role": "assistant", "content": msg.content})
            try:
                correct = await judge(row["question"], answer, row["ground_truth"])
            except JudgeError:
                correct = None
            outcome = "answered"
            break

        messages.append(
            {"role": "assistant", "content": msg.content, "tool_calls": tool_calls}
        )
        if turn + 1 >= max_turns:
            break
        for call in tool_calls:
            fn = call["function"]
            try:
                arguments = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if fn["name"] == TOOL_NAME_SEARCH:
                n_search += 1
                observation = await exa_tool.search(str(arguments.get("query", "")))
            elif fn["name"] == TOOL_NAME_GET_CONTENTS:
                n_get_contents += 1
                observation = await exa_tool.get_contents(str(arguments.get("url", "")))
            else:
                observation = (
                    f"Unknown tool {fn['name']!r}. Available tools: "
                    f"{TOOL_NAME_SEARCH!r}, {TOOL_NAME_GET_CONTENTS!r}."
                )
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": observation}
            )
        if turn + 2 >= max_turns:
            messages.append({"role": "user", "content": _FINAL_TURN_NUDGE})

    return {
        "id": row["id"],
        "source": row["source"],
        "question": row["question"],
        "ground_truth": row["ground_truth"],
        "answer": answer,
        "correct": correct,
        "outcome": outcome,
        "turns": turns_used,
        "search_calls": n_search,
        "get_contents_calls": n_get_contents,
        "messages": messages,
    }


async def eval_model(
    label: str,
    model_route: str,
    rows: list[dict],
    out_dir: str,
    args: argparse.Namespace,
) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=INFERENCE_BASE,
        api_key=os.environ["FIREWORKS_API_KEY"],
        timeout=180.0,
        max_retries=3,
    )
    exa_tool = ExaSearchTool(
        ExaSearchConfig(search_type=args.search_type, num_results=args.num_results)
    )
    judge = AnswerJudge(JudgeConfig())
    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []

    async def one(row: dict) -> None:
        async with semaphore:
            try:
                result = await run_episode(
                    client, model_route, row, exa_tool, judge,
                    max_turns=args.max_turns,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except Exception:  # noqa: BLE001
                logger.warning("[%s] episode failed for %s", label, row["id"], exc_info=True)
                return
            results.append(result)
            done = len(results)
            if done % 10 == 0 or done == len(rows):
                acc = [r for r in results if r["correct"] is not None]
                logger.info(
                    "[%s] %d/%d | acc so far %.1f%%",
                    label, done, len(rows),
                    100 * sum(r["correct"] for r in acc) / max(1, len(acc)),
                )

    await asyncio.gather(*(one(r) for r in rows))

    transcript_path = os.path.join(out_dir, f"{label}.transcripts.jsonl")
    with open(transcript_path, "w") as f:
        for r in sorted(results, key=lambda r: r["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    judged = [r for r in results if r["correct"] is not None]
    summary = {
        "label": label,
        "model": model_route,
        "episodes": len(results),
        "judged": len(judged),
        "accuracy": sum(r["correct"] for r in judged) / max(1, len(judged)),
        "avg_search_calls": sum(r["search_calls"] for r in results) / max(1, len(results)),
        "avg_turns": sum(r["turns"] for r in results) / max(1, len(results)),
        "no_answer_rate": sum(r["outcome"] != "answered" for r in results) / max(1, len(results)),
        "transcripts": transcript_path,
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the Exa search agent (base vs tuned)")
    p.add_argument("--base-model", default="accounts/fireworks/models/qwen3-4b-instruct-2507")
    p.add_argument("--lora", action="append", default=[],
                   help="Promoted adapter model id to evaluate (repeatable).")
    p.add_argument("--skip-base", action="store_true",
                   help="Evaluate only the --lora model(s), not the base.")
    p.add_argument("--deployment", default=None,
                   help="Existing deployment id/name to route through (skips create+delete).")
    p.add_argument("--keep-deployment", action="store_true",
                   help="Do not delete the deployment this script created.")
    p.add_argument("--accelerator", default="NVIDIA_B200_180GB")
    p.add_argument("--dataset", default="mixed", choices=["mixed", "hotpotqa", "musique"])
    p.add_argument("--difficulty", default="hard", choices=["hard", "medium", "easy", "all"])
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--search-type", default="auto",
                   choices=["auto", "fast", "instant", "deep-lite", "deep", "deep-reasoning"])
    p.add_argument("--num-results", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out-dir", default=None,
                   help="Results directory (default: ./eval_results/<timestamp>).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for var in ("FIREWORKS_API_KEY", "EXA_API_KEY"):
        if not os.environ.get(var):
            raise RuntimeError(f"{var} is not set (put it in training/.env).")

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "eval_results", time.strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=True)

    rows = load_eval_rows(args.dataset, args.limit, args.difficulty, args.seed)
    logger.info("Evaluating on %d held-out rows (%s)", len(rows), args.dataset)

    cp = ControlPlane(os.environ["FIREWORKS_API_KEY"])
    deployment = args.deployment
    created_deployment = False
    if deployment and not deployment.startswith("accounts/"):
        deployment = f"accounts/{cp.account_id}/deployments/{deployment}"
    if deployment is None:
        deployment = cp.create_deployment(args.base_model, args.accelerator)
        created_deployment = True

    models: list[tuple[str, str]] = []
    if not args.skip_base:
        models.append(("base", f"{args.base_model}#{deployment}"))
    for i, lora in enumerate(args.lora):
        label = "tuned" if len(args.lora) == 1 else f"tuned-{i}"
        models.append((label, f"{lora}#{deployment}"))

    if not models:
        raise RuntimeError("Nothing to evaluate: pass --lora and/or drop --skip-base.")

    summaries: list[dict] = []
    try:
        cp.wait_deployment_ready(deployment)
        for lora in args.lora:
            addon = cp.load_lora(lora, deployment)
            cp.wait_lora_ready(addon)

        for label, route in models:
            logger.info("=== Evaluating %s (%s) ===", label, route)
            summaries.append(asyncio.run(eval_model(label, route, rows, out_dir, args)))
    finally:
        if created_deployment and not args.keep_deployment:
            cp.delete_deployment(deployment)
        elif created_deployment:
            logger.info("Keeping deployment %s (billing until deleted)", deployment)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"dataset": args.dataset, "limit": args.limit, "models": summaries}, f, indent=2)

    print(f"\n{'model':<10} {'episodes':>8} {'accuracy':>9} {'searches/ep':>12} {'turns/ep':>9} {'no-answer':>10}")
    for s in summaries:
        print(
            f"{s['label']:<10} {s['episodes']:>8} {s['accuracy']:>8.1%} "
            f"{s['avg_search_calls']:>12.2f} {s['avg_turns']:>9.2f} {s['no_answer_rate']:>9.1%}"
        )
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()
