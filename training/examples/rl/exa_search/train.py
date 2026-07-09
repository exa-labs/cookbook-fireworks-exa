#!/usr/bin/env python3
"""Async RL (GRPO) training of an Exa web-search agent on multi-hop QA.

Run ``prepare_data.py`` first to materialize ``dataset.jsonl``, then::

    python train.py \\
        --base-model accounts/fireworks/models/qwen3-4b \\
        --tokenizer-model Qwen/Qwen3-4B \\
        --max-turns 6 \\
        --output-model-id accounts/<acct>/models/exa-search-agent

Required environment (put these in ``training/.env``; loaded via python-dotenv):
    FIREWORKS_API_KEY   -- Fireworks training + inference (and the default judge)
    EXA_API_KEY         -- Exa search backend (https://dashboard.exa.ai/api-keys)
    WANDB_API_KEY       -- optional, for metric logging
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Iterator

from dotenv import load_dotenv

from training.examples.rl.exa_search.rollout import make_rollout_fn
from training.recipes.async_rl_loop import Config, main
from training.utils import DeployConfig, TrainerConfig, WandBConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_TRAINING_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
load_dotenv(os.path.join(_TRAINING_ROOT, ".env"))
load_dotenv(os.path.join(_TRAINING_ROOT, "..", ".env"))

DEFAULT_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.jsonl")


def _iter_rows(path: str, max_rows: int | None) -> Iterator[dict]:
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if max_rows is not None and n >= max_rows:
                return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async RL for an Exa web-search agent")
    # Model / data
    p.add_argument("--base-model", default="accounts/fireworks/models/qwen3-4b-instruct-2507",
                   help="Fireworks model resource name to fine-tune (blog: Qwen3-4B-Instruct-2507).")
    p.add_argument("--tokenizer-model", default="Qwen/Qwen3-4B-Instruct-2507",
                   help="HF tokenizer id (must match the base model's tokenizer).")
    p.add_argument("--dataset-path", default=DEFAULT_DATASET)
    p.add_argument("--output-model-id", default=None,
                   help="Promote the final checkpoint to this model id.")
    p.add_argument("--max-rows", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    # GRPO / optimization
    p.add_argument("--completions-per-prompt", type=int, default=8,
                   help="GRPO group size per question.")
    p.add_argument("--prompt-groups-per-step", type=int, default=8,
                   help="Questions per optimizer step (batch = this x completions-per-prompt).")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--kl-beta", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-completion-tokens", type=int, default=2048,
                   help="Per-turn generation budget (thinking + one tool call).")
    p.add_argument("--lora-rank", type=int, default=32,
                   help="LoRA rank (blog uses LoRA; 0 = full-parameter training).")
    p.add_argument("--max-head-offpolicy-versions", type=int, default=0,
                   help="Off-policy staleness budget (0 = strict on-policy).")
    p.add_argument("--max-concurrency-rollout-sample", type=int, default=None,
                   help="Cap on in-flight LLM calls against the deployment (>= completions-per-prompt).")
    p.add_argument("--no-filter-constant-reward", action="store_true",
                   help="Keep prompt groups whose rewards are all identical "
                        "(by default they are dropped -- GRPO advantage is 0 there).")
    # Agent / search / judge
    p.add_argument("--max-turns", type=int, default=6,
                   help="Max search/answer turns per question.")
    p.add_argument("--search-type", default="auto",
                   choices=["auto", "fast", "instant", "deep-lite", "deep", "deep-reasoning"],
                   help="Exa search type. 'fast' is cheaper/lower-latency for high-throughput RL.")
    p.add_argument("--num-results", type=int, default=5,
                   help="Exa results returned per search (blog: 5).")
    p.add_argument("--max-trajectory-tokens", type=int, default=30720,
                   help="Hard cap on prompt+completion tokens per trajectory; exceeding it "
                        "ends the episode with the context-overflow penalty (blog: 30720).")
    p.add_argument("--context-overflow-penalty", type=float, default=-0.25,
                   help="Reward when the trajectory exceeds --max-trajectory-tokens (blog: -0.25).")
    p.add_argument("--no-answer-penalty", type=float, default=-0.1,
                   help="Reward when the agent burns all turns without giving a final answer.")
    p.add_argument("--judge-model", default=None,
                   help="Override the LLM judge model (default: "
                        "accounts/fireworks/models/qwen3-30b-a3b-instruct-2507; use a "
                        "non-thinking instruct model).")
    p.add_argument("--no-judge", action="store_true",
                   help="Disable the LLM judge; grade with token-F1 instead (offline / no judge key).")
    # Infra / logging
    p.add_argument("--training-shape-id", default=os.environ.get("TRAINING_SHAPE") or None,
                   help="Training shape resource name; auto-selected if unset.")
    p.add_argument("--replica-count", type=int, default=None,
                   help="Fixed inference-deployment replica count (fans out rollout sampling).")
    p.add_argument("--log-path", default="./exa_search_logs")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "exa-search-rl"))
    p.add_argument("--wandb-run-name", default=None)
    return p.parse_args()


def run() -> None:
    args = parse_args()
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. Run `python prepare_data.py` first."
        )
    if not os.environ.get("EXA_API_KEY"):
        raise RuntimeError(
            "EXA_API_KEY is not set. Add it to training/.env "
            "(get a key at https://dashboard.exa.ai/api-keys)."
        )

    rows = list(_iter_rows(args.dataset_path, args.max_rows))
    logger.info("Loaded %d rows from %s", len(rows), args.dataset_path)

    cfg = Config(
        log_path=args.log_path,
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        kl_beta=args.kl_beta,
        completions_per_prompt=args.completions_per_prompt,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
        epochs=args.epochs,
        max_rows=args.max_rows,
        lora_rank=args.lora_rank,
        prompt_groups_per_step=args.prompt_groups_per_step,
        max_head_offpolicy_versions=args.max_head_offpolicy_versions,
        max_concurrency_rollout_sample=args.max_concurrency_rollout_sample,
        output_model_id=args.output_model_id,
        trainer=TrainerConfig(training_shape_id=args.training_shape_id),
        deployment=DeployConfig(
            tokenizer_model=args.tokenizer_model,
            replica_count=args.replica_count,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=args.wandb_run_name or f"exa-search-{int(time.time()) % 100000}",
        ),
    )

    # Constant-reward groups have zero GRPO advantage; drop them by default.
    dynamic_filter_fn = (
        None if args.no_filter_constant_reward
        else (lambda pg: len(set(pg.rewards)) > 1)
    )

    rollout_extras = {
        "max_turns": args.max_turns,
        "search_type": args.search_type,
        "num_results": args.num_results,
        "max_trajectory_tokens": args.max_trajectory_tokens,
        "context_overflow_penalty": args.context_overflow_penalty,
        "no_answer_penalty": args.no_answer_penalty,
        "judge_enabled": not args.no_judge,
        "judge_model": args.judge_model,
    }

    logger.info(
        "base=%s | rows=%d | epochs=%d | cpp=%d | groups/step=%d | max_turns=%d | search=%s",
        args.base_model, len(rows), args.epochs, args.completions_per_prompt,
        args.prompt_groups_per_step, args.max_turns, args.search_type,
    )

    main(
        cfg,
        rollout_fn_factory=make_rollout_fn,
        rows=rows,
        rollout_extras=rollout_extras,
        dynamic_filter_fn=dynamic_filter_fn,
    )


if __name__ == "__main__":
    run()
