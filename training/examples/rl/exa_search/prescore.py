#!/usr/bin/env python3
"""Offline difficulty scoring to filter the training set before RL.

GRPO only learns from prompt groups whose rewards disagree; all-correct and
all-wrong groups have zero advantage and waste rollout budget. This pass runs
the *base* model over every training question ``--samples`` times, counts how
many come back correct, and keeps only the mixed-difficulty band
``0 < correct < samples`` -- the questions RL can actually learn from.

It reuses ``eval.py``'s deployment control plane and agent loop, then writes
the surviving rows (plus ``base_correct`` / ``base_samples``) to
``dataset.prescored.jsonl``. Point ``train.py --dataset-path`` at that file and
drop the runtime filter (``--no-filter-constant-reward``): the offline pass has
already done the filtering, so nearly every group yields a gradient.

    python prescore.py --samples 8 --temperature 1.0

Requires FIREWORKS_API_KEY and EXA_API_KEY (training/.env is loaded).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections import Counter

from dotenv import load_dotenv

from training.examples.rl.exa_search.eval import ControlPlane, run_episode
from training.examples.rl.exa_search.exa_search import ExaSearchConfig, ExaSearchTool
from training.examples.rl.exa_search.reward import AnswerJudge, JudgeConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_TRAINING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
load_dotenv(os.path.join(_TRAINING_ROOT, ".env"))
load_dotenv(os.path.join(_TRAINING_ROOT, "..", ".env"))

INFERENCE_BASE = "https://api.fireworks.ai/inference/v1"
DEFAULT_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.jsonl")


def _load_rows(path: str, max_rows: int | None) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


async def score_rows(rows: list[dict], model_route: str, args: argparse.Namespace) -> list[dict]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=INFERENCE_BASE,
        api_key=os.environ["FIREWORKS_API_KEY"],
        timeout=180.0,
        max_retries=3,
    )
    exa_tool = ExaSearchTool(
        ExaSearchConfig(
            search_type=args.search_type,
            num_results=args.num_results,
            max_requests_per_second=args.exa_qps,
        )
    )
    judge = AnswerJudge(JudgeConfig())
    semaphore = asyncio.Semaphore(args.concurrency)
    scored: list[dict] = []

    async def one_sample(row: dict) -> float | None:
        async with semaphore:
            try:
                result = await run_episode(
                    client, model_route, row, exa_tool, judge,
                    max_turns=args.max_turns,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except Exception:  # noqa: BLE001
                logger.warning("sample failed for %s", row["id"], exc_info=True)
                return None
            return result["correct"]

    async def score_one(row: dict) -> None:
        verdicts = await asyncio.gather(*(one_sample(row) for _ in range(args.samples)))
        judged = [v for v in verdicts if v is not None]
        correct = int(sum(v for v in judged))
        scored.append({**row, "base_correct": correct, "base_samples": len(judged)})
        done = len(scored)
        if done % 25 == 0 or done == len(rows):
            kept = sum(1 for r in scored if 0 < r["base_correct"] < r["base_samples"])
            logger.info("scored %d/%d | mixed-difficulty so far: %d", done, len(rows), kept)

    await asyncio.gather(*(score_one(r) for r in rows))
    return scored


def main() -> None:
    args = parse_args()
    for var in ("FIREWORKS_API_KEY", "EXA_API_KEY"):
        if not os.environ.get(var):
            raise RuntimeError(f"{var} is not set (put it in training/.env).")

    rows = _load_rows(args.dataset_path, args.max_rows)
    logger.info("Scoring %d rows x %d samples against %s", len(rows), args.samples, args.base_model)

    cp = ControlPlane(os.environ["FIREWORKS_API_KEY"])
    deployment = args.deployment
    created = False
    if deployment and not deployment.startswith("accounts/"):
        deployment = f"accounts/{cp.account_id}/deployments/{deployment}"
    if deployment is None:
        deployment = cp.create_deployment(args.base_model, args.accelerator)
        created = True

    try:
        cp.wait_deployment_ready(deployment)
        route = f"{args.base_model}#{deployment}"
        scored = asyncio.run(score_rows(rows, route, args))
    finally:
        if created and not args.keep_deployment:
            cp.delete_deployment(deployment)
        elif created:
            logger.info("Keeping deployment %s (billing until deleted)", deployment)

    dist = Counter(
        "all_wrong" if r["base_samples"] and r["base_correct"] == 0
        else "all_right" if r["base_correct"] == r["base_samples"]
        else "unjudged" if r["base_samples"] == 0
        else "mixed"
        for r in scored
    )
    kept = [r for r in scored if 0 < r["base_correct"] < r["base_samples"]]
    with open(args.out_path, "w") as f:
        for r in sorted(kept, key=lambda r: r["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Distribution: %s", dict(dist))
    logger.info("Kept %d/%d mixed-difficulty rows -> %s", len(kept), len(scored), args.out_path)
    print(
        f"\nprescored {len(scored)} rows | mixed-difficulty (trainable): {len(kept)} "
        f"({100 * len(kept) / max(1, len(scored)):.0f}%)\n"
        f"all-right (too easy): {dist.get('all_right', 0)} | "
        f"all-wrong (too hard): {dist.get('all_wrong', 0)} | "
        f"unjudged: {dist.get('unjudged', 0)}\n"
        f"-> {args.out_path}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline base-model difficulty scoring for RL data")
    p.add_argument("--base-model", default="accounts/fireworks/models/qwen3-4b-instruct-2507",
                   help="Model to score difficulty against (must match the RL base model).")
    p.add_argument("--dataset-path", default=DEFAULT_DATASET)
    p.add_argument("--out-path", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset.prescored.jsonl"))
    p.add_argument("--max-rows", type=int, default=None,
                   help="Score only the first N rows (default: all).")
    p.add_argument("--samples", type=int, default=8,
                   help="Base-model samples per question (match --completions-per-prompt).")
    p.add_argument("--deployment", default=None,
                   help="Existing deployment id/name to route through (skips create+delete).")
    p.add_argument("--keep-deployment", action="store_true")
    p.add_argument("--accelerator", default="NVIDIA_B200_180GB")
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (match training's 1.0 so difficulty is on-distribution).")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--search-type", default="auto",
                   choices=["auto", "fast", "instant", "deep-lite", "deep", "deep-reasoning"])
    p.add_argument("--num-results", type=int, default=5)
    p.add_argument("--exa-qps", type=float, default=10.0)
    p.add_argument("--concurrency", type=int, default=24)
    return p.parse_args()


if __name__ == "__main__":
    main()
