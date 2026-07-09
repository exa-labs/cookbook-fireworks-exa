#!/usr/bin/env python3
"""Download multi-hop QA data and write JSONL rows for the Exa search agent.

The agent searches the open web, so rows carry only the question and gold
answer(s).  ``--dataset mixed`` (default) is the blog's 50/50 HotpotQA +
MuSiQue train mix (``PeterJinGo/nq_hotpotqa_train`` filtered to HotpotQA
rows, ``dgslibisey/MuSiQue``); ``hotpotqa`` / ``musique`` / ``2wiki`` load a
single source.  Train splits keep the standard validation sets clean for
evaluating the result.

Row format (extra keys reach ``rollout_fn`` as ``sample_prompt``)::

    {"id": "hotpotqa-0",
     "messages": [{"role": "system", ...}, {"role": "user", "content": "<question>"}],
     "question": "<question>",
     "ground_truth": ["<gold answer>"],
     "source": "hotpotqa"}

Usage::

    python prepare_data.py                                  # blog mix, 2000 rows
    python prepare_data.py --dataset musique --max-rows 1000
    python prepare_data.py --max-rows 16 --output smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random

from datasets import load_dataset

from training.examples.rl.exa_search.exa_search import SYSTEM_PROMPT

DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.jsonl")


def _row(idx: int, source: str, question: str, gold: list[str]) -> dict:
    return {
        "id": f"{source}-{idx}",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "question": question,
        "ground_truth": gold,
        "source": source,
    }


def load_peterjin_hotpotqa(split: str, max_rows: int) -> list[dict]:
    """HotpotQA rows from the blog's source set, skipping its single-hop NQ
    rows.  Streamed: the full split is ~170k rows."""
    hf_split = "train" if split == "train" else "test"
    ds = load_dataset("PeterJinGo/nq_hotpotqa_train", split=hf_split, streaming=True)
    rows: list[dict] = []
    for row in ds:
        if row.get("data_source") != "hotpotqa":
            continue
        gold = [str(g) for g in (row.get("golden_answers") or []) if str(g).strip()]
        if not gold:
            continue
        rows.append(_row(len(rows), "hotpotqa", row["question"], gold))
        if len(rows) >= max_rows:
            break
    print(f"Loaded {len(rows)} HotpotQA rows from PeterJinGo/nq_hotpotqa_train ({hf_split})")
    return rows


def load_hotpotqa(split: str, difficulty: str, max_rows: int) -> list[dict]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    print(f"Loaded {len(ds)} rows from HotpotQA ({split})")
    rows: list[dict] = []
    for i, row in enumerate(ds):
        if difficulty != "all" and row.get("level", "") != difficulty:
            continue
        rows.append(_row(len(rows), "hotpotqa", row["question"], [row["answer"]]))
        if len(rows) >= max_rows:
            break
    print(f"  Kept {len(rows)} rows (difficulty={difficulty})")
    return rows


def load_musique(split: str, max_rows: int) -> list[dict]:
    ds = load_dataset("dgslibisey/MuSiQue", split=split)
    print(f"Loaded {len(ds)} rows from MuSiQue ({split})")
    rows: list[dict] = []
    for row in ds:
        if not row.get("answerable", True):
            continue
        gold = [row.get("answer", "")]
        gold += list(row.get("answer_aliases", []) or [])
        rows.append(_row(len(rows), "musique", row["question"], [g for g in gold if g]))
        if len(rows) >= max_rows:
            break
    print(f"  Kept {len(rows)} rows")
    return rows


def load_2wiki(split: str, max_rows: int) -> list[dict]:
    ds = load_dataset("ohjoonhee/2WikiMultihopQA", split=split)
    print(f"Loaded {len(ds)} rows from 2WikiMultiHopQA ({split})")
    rows: list[dict] = []
    for row in ds:
        rows.append(_row(len(rows), "2wikimultihopqa", row["question"], [row.get("answer", "")]))
        if len(rows) >= max_rows:
            break
    print(f"  Kept {len(rows)} rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multi-hop QA data for the Exa search agent")
    parser.add_argument("--dataset", default="mixed",
                        choices=["mixed", "hotpotqa", "musique", "2wiki", "all"],
                        help="'mixed' (default) = the blog's 50/50 HotpotQA+MuSiQue mix")
    parser.add_argument("--split", default="train",
                        help="HF split; 'train' (default) or 'validation' -- keep "
                             "validation for eval, not training")
    parser.add_argument("--difficulty", default="hard",
                        choices=["hard", "medium", "easy", "all"],
                        help="HotpotQA difficulty filter (ignored for other datasets)")
    parser.add_argument("--max-rows", type=int, default=2000,
                        help="Max rows to write (total across datasets)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    val_split = "train" if args.split == "train" else "validation"
    all_rows: list[dict] = []

    if args.dataset == "mixed":
        half = args.max_rows // 2
        all_rows.extend(load_peterjin_hotpotqa(args.split, args.max_rows - half))
        all_rows.extend(load_musique(val_split, half))
        random.shuffle(all_rows)
    if args.dataset in ("hotpotqa", "all"):
        budget = args.max_rows if args.dataset == "hotpotqa" else args.max_rows // 3
        all_rows.extend(load_hotpotqa(args.split, args.difficulty, budget))
    if args.dataset in ("musique", "all"):
        budget = args.max_rows if args.dataset == "musique" else args.max_rows // 3
        try:
            all_rows.extend(load_musique(val_split, budget))
        except Exception as exc:  # noqa: BLE001
            print(f"  Skipping MuSiQue: {exc}")
    if args.dataset in ("2wiki", "all"):
        budget = args.max_rows if args.dataset == "2wiki" else args.max_rows // 3
        try:
            all_rows.extend(load_2wiki(val_split, budget))
        except Exception as exc:  # noqa: BLE001
            print(f"  Skipping 2WikiMultiHopQA: {exc}")

    if args.dataset == "all":
        random.shuffle(all_rows)
        all_rows = all_rows[: args.max_rows]

    with open(args.output, "w") as f:
        for entry in all_rows:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for r in all_rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    print(f"\nWrote {len(all_rows)} rows to {args.output}")
    for src, c in sorted(counts.items()):
        print(f"  {src}: {c}")


if __name__ == "__main__":
    main()
