#!/usr/bin/env bash
# Canned end-to-end run for the Exa search agent.
#
#   FIREWORKS_API_KEY=... EXA_API_KEY=... bash run.sh          # full canned run
#   SMOKE=1 FIREWORKS_API_KEY=... EXA_API_KEY=... bash run.sh  # tiny/cheap smoke run
#
# Keys are read from the environment or from training/.env (train.py loads it).
# Set OUTPUT_MODEL_ID=<bare-id, e.g. exa-search-agent> to promote the final
# checkpoint to accounts/<your-acct>/models/<id>; left unset, it can be
# promoted later.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

BASE_MODEL="${BASE_MODEL:-accounts/fireworks/models/qwen3-4b-instruct-2507}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
export TRAINING_SHAPE="${TRAINING_SHAPE:-accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora}"

if [[ "${SMOKE:-0}" == "1" ]]; then
    # Tiny/cheap: a few questions, one small step -- validates the whole loop.
    MAX_ROWS=16; CPP=4; PROMPT_GROUPS=4; EPOCHS=1; MAX_TURNS=4; SEARCH_TYPE=fast
    EXTRA_ARGS="--no-filter-constant-reward"
else
    MAX_ROWS="${MAX_ROWS:-512}"; CPP="${CPP:-8}"; PROMPT_GROUPS="${PROMPT_GROUPS:-8}"
    EPOCHS="${EPOCHS:-1}"; MAX_TURNS="${MAX_TURNS:-6}"; SEARCH_TYPE="${SEARCH_TYPE:-auto}"
    EXTRA_ARGS=""
fi

if [[ ! -f "$HERE/dataset.jsonl" ]] || [[ "$(wc -l < "$HERE/dataset.jsonl")" -lt "$MAX_ROWS" ]]; then
    echo "dataset.jsonl missing or smaller than MAX_ROWS=$MAX_ROWS; preparing the HotpotQA+MuSiQue blog mix..."
    python "$HERE/prepare_data.py" --max-rows "$MAX_ROWS"
fi

python "$HERE/train.py" \
    --base-model "$BASE_MODEL" \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --dataset-path "$HERE/dataset.jsonl" \
    --max-rows "$MAX_ROWS" \
    --epochs "$EPOCHS" \
    --completions-per-prompt "$CPP" \
    --prompt-groups-per-step "$PROMPT_GROUPS" \
    --max-turns "$MAX_TURNS" \
    --search-type "$SEARCH_TYPE" \
    --learning-rate 1e-5 \
    --kl-beta 0.0 \
    ${OUTPUT_MODEL_ID:+--output-model-id "$OUTPUT_MODEL_ID"} \
    $EXTRA_ARGS
