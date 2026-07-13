#!/usr/bin/env python3
"""Async RL (GRPO) training of an Exa web-search agent on multi-hop QA.

Run ``prepare_data.py`` first to materialize ``dataset.jsonl``, then::

    python train.py \\
        --base-model accounts/fireworks/models/qwen3-4b-instruct-2507 \\
        --tokenizer-model Qwen/Qwen3-4B-Instruct-2507 \\
        --max-turns 6 \\
        --output-model-id exa-search-agent

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
                   help="Fireworks model resource name to fine-tune.")
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
                   help="LoRA rank (0 = full-parameter training).")
    p.add_argument("--max-head-offpolicy-versions", type=int, default=0,
                   help="Off-policy staleness budget (0 = strict on-policy).")
    p.add_argument("--max-concurrency-rollout-sample", type=int, default=None,
                   help="Cap on in-flight LLM calls against the deployment (>= completions-per-prompt).")
    p.add_argument("--no-filter-constant-reward", action="store_true",
                   help="Keep constant-reward prompt groups (dropped by default; "
                        "GRPO advantage is 0 there).")
    # Agent / search / judge
    p.add_argument("--max-turns", type=int, default=6,
                   help="Max search/answer turns per question.")
    p.add_argument("--search-type", default="auto",
                   choices=["auto", "fast", "instant", "deep-lite", "deep", "deep-reasoning"],
                   help="Exa search type. 'fast' is cheaper/lower-latency for high-throughput RL.")
    p.add_argument("--num-results", type=int, default=5,
                   help="Exa results returned per search.")
    p.add_argument("--exa-qps", type=float, default=10.0,
                   help="Client-side cap on Exa API requests/second across all "
                        "concurrent rollouts (Exa's default account limit is 10; "
                        "0 disables).")
    p.add_argument("--max-trajectory-tokens", type=int, default=30720,
                   help="Hard cap on prompt+completion tokens per trajectory; exceeding it "
                        "ends the episode with the context-overflow penalty.")
    # Infra / logging
    p.add_argument("--dcp-save-interval", type=int, default=10,
                   help="Save a resumable checkpoint every N optimizer steps (0 = final only).")
    p.add_argument("--init-from-checkpoint", default=None,
                   help="Resume from a prior checkpoint: \"<job-id>:step-N\" cross-job, "
                        "bare \"step-N\" within the same job. See README: Resuming.")
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
    if args.output_model_id:
        from fireworks.training.sdk import validate_output_model_id

        errors = validate_output_model_id(args.output_model_id)
        if errors:
            raise ValueError("Invalid --output-model-id: " + "; ".join(errors))

    rows = list(_iter_rows(args.dataset_path, args.max_rows))
    logger.info("Loaded %d rows from %s", len(rows), args.dataset_path)

    # One identifier per run, shared by the stats file and the wandb run so
    # artifacts correlate by the run's identity, not by a fragile timestamp.
    run_tag = args.wandb_run_name or args.output_model_id or f"run-{int(time.time())}"

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
        dcp_save_interval=args.dcp_save_interval,
        init_from_checkpoint=args.init_from_checkpoint,
        output_model_id=args.output_model_id,
        trainer=TrainerConfig(training_shape_id=args.training_shape_id),
        deployment=DeployConfig(
            tokenizer_model=args.tokenizer_model,
            replica_count=args.replica_count,
        ),
        wandb=WandBConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            run_name=run_tag,
        ),
    )

    # Constant-reward groups have zero GRPO advantage; drop them by default.
    dynamic_filter_fn = (
        None if args.no_filter_constant_reward
        else (lambda pg: len(set(pg.rewards)) > 1)
    )

    stats_path = os.path.join(args.log_path, f"rollout_stats-{run_tag}.jsonl")
    rollout_extras = {
        "stats_path": stats_path,
        "max_turns": args.max_turns,
        "search_type": args.search_type,
        "num_results": args.num_results,
        "exa_qps": args.exa_qps,
        "max_trajectory_tokens": args.max_trajectory_tokens,
    }

    logger.info(
        "base=%s | rows=%d | epochs=%d | cpp=%d | groups/step=%d | max_turns=%d | search=%s",
        args.base_model, len(rows), args.epochs, args.completions_per_prompt,
        args.prompt_groups_per_step, args.max_turns, args.search_type,
    )
    logger.info("Per-rollout stats -> %s", stats_path)

    main(
        cfg,
        rollout_fn_factory=make_rollout_fn,
        rows=rows,
        rollout_extras=rollout_extras,
        dynamic_filter_fn=dynamic_filter_fn,
    )


if __name__ == "__main__":
    run()
