# Exa Web-Search Agent (async RL)

Train a multi-turn **web-search agent** with reinforcement learning on Fireworks,
using [Exa](https://exa.ai) as the live search backend. The model learns to
search the open web, read results, and answer multi-hop questions. RL
rewards it for getting the answer right.

> **Reference**: Exa, *"How Search Quality Shapes RL Outcomes"*
> ([exa.ai/blog/rl-search-outcomes](https://exa.ai/blog/rl-search-outcomes), 2026).
> That study holds everything fixed except the search backend and finds that
> agents trained with **Exa** reach higher pass@k at **lower training and
> inference cost** than agents trained with a SERP (Google-proxy) backend: a
> stronger retriever surfaces the supporting evidence in fewer turns, so more
> rollouts reach a correct answer and the reward signal is denser.

## Why this example

The cookbook already ships [`training/examples/multihop_qa`](../../multihop_qa),
whose agent searches a **local TF-IDF index over a small paragraph pool bundled
with each question**: a closed-book toy retriever where the answer is
guaranteed to be in the pool. This example swaps that retriever for **real Exa
web search over the open web**. That single swap is the axis the blog studies:
*the search tool is part of the environment the policy learns in, so its quality
shapes what RL can learn.*

The recipe underneath is the same one every async RL example uses
(`recipes.async_rl_loop`); **the only thing customized here is the rollout
function**. See
[`/skills/dev/references/rl/async-rl.md`](/skills/dev/references/rl/async-rl.md)
for the recipe API and gate sizing.

## How it works

One `rollout_fn` call = one trajectory:

```
question ─▶ model ─▶ <tool_call> search(query) ──────▶ Exa web search
              ▲      <tool_call> get_contents(url) ──▶ Exa page fetch
              │                                              │
              └──────────── role="tool" observations ◀───────┘
              │
              └▶ reply with NO tool call ("Answer: ...") ─▶ LLM judge ─▶ reward ∈ {0,1}
```

- **Tools** (`exa_search.py`): `search(query)` runs an Exa web search and
  returns ranked result snippets; `get_contents(url)` fetches a result page.
  Both are declared as OpenAI-compatible function specs and rendered through
  the model's chat template, the same
  [tool-calling contract Fireworks serves at inference time](https://docs.fireworks.ai/guides/function-calling),
  so the trained adapter drops straight into a standard `tools=` agent loop.
  Results come back as `role="tool"` messages. Exa calls are async
  (`AsyncExa`), retried with backoff on transient errors, and cached across
  the GRPO group's duplicate queries.
- **Episode end**: idiomatically, there is no "submit" tool: the episode ends
  when the model replies **without** calling a tool; that message is its final
  answer (extracted after the `Answer:` prefix the system prompt requests).
  This matches both the Fireworks agent loop and the blog's reference
  implementation.
- **Rollout** (`rollout.py`): runs the tool loop for up to `--max-turns`,
  packing the whole conversation into one `RolloutRun`.
  `MessageTrajectoryAssembler` keeps the per-token loss mask aligned: `1` on
  assistant-generated tokens, `0` on the prompt and the injected tool
  observations. A trajectory that would exceed `--max-trajectory-tokens` ends
  early with the **−0.25 context-overflow penalty** (the exact condition the
  blog penalizes); burning all turns without answering costs a smaller
  `--no-answer-penalty`.
- **Reward** (`reward.py`): an **LLM judge** grades the final answer against the
  gold answer(s) (1.0 / 0.0), following the SimpleQA-style grader the blog uses
  (exact-match was dropped because the agent learned to reward-hack answer
  formatting). A judge call that fails or returns an unparseable verdict drops
  that one trajectory; a persistently broken judge (many failures in a row)
  aborts the run rather than silently mis-grading it.
- The recipe owns everything else: GRPO advantage, reference forwards, weight
  sync, KL/TIS, checkpointing.

## Files

| File | Description |
|---|---|
| `exa_search.py` | Exa `search`/`get_contents` tools (async, retried, cached), OpenAI-style tool specs, tool-call parser, and the agent system prompt. |
| `reward.py` | LLM-as-judge answer grading (Fireworks-hosted by default): binary correct/incorrect. |
| `rollout.py` | `make_rollout_fn(setup)`: the multi-turn search/answer loop. |
| `prepare_data.py` | Builds `dataset.jsonl`; defaults to the blog's 50/50 HotpotQA+MuSiQue train mix. |
| `train.py` | Wires the dataset + rollout factory into `recipes.async_rl_loop.main`. |
| `eval.py` | Held-out eval (base vs. tuned adapter): accuracy, searches/episode, transcripts. |
| `run.sh` | Canned end-to-end run (`SMOKE=1` for a tiny/cheap version). |
| `requirements.txt` | Extra dep: `exa-py`. |

## Setup

1. Install the cookbook `training` package (see [`../../../README.md`](../../../README.md)),
   then this example's extra dep:

   ```bash
   pip install -r requirements.txt   # or: pip install -e ".[exa-search]"
   ```

2. Put your keys in `training/.env` (loaded automatically by `train.py`):

   ```
   FIREWORKS_API_KEY=...   # Fireworks training + inference, and the default LLM judge
   EXA_API_KEY=...         # Exa search backend: https://dashboard.exa.ai/api-keys
   WANDB_API_KEY=...        # optional, for metric logging
   ```

   The judge defaults to a Fireworks-hosted model; point it elsewhere with
   `JUDGE_MODEL` / `JUDGE_BASE_URL` / `JUDGE_API_KEY`.

## Quick start

```bash
# 1. Prepare data: the blog's 50/50 HotpotQA + MuSiQue mix (train splits).
python prepare_data.py --max-rows 2000

# 2. Train. The training shape selects the trainer + deployment GPU pair;
#    its profile must reference a deployment shape (see skills/dev shapes.md).
TRAINING_SHAPE=accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora \
python train.py \
    --base-model accounts/fireworks/models/qwen3-4b-instruct-2507 \
    --tokenizer-model Qwen/Qwen3-4B-Instruct-2507 \
    --max-rows 512 \
    --completions-per-prompt 8 \
    --max-turns 6 \
    --search-type auto \
    --output-model-id exa-search-agent
```

`--output-model-id` is the bare id (lowercase a-z, 0-9, hyphens; max 63 chars)
-- the promoted model lands at `accounts/<your-acct>/models/<id>`. It is
validated at startup so a bad id fails before training, not after.

Or `bash run.sh` for the canned config, and **`SMOKE=1 bash run.sh`** for a
tiny/cheap run (16 questions, a few steps, `search-type=fast`) that validates the
whole loop before you commit to a full run.

Every trajectory is appended to `<log-path>/rollout_stats-<run>.jsonl` (reward,
outcome, turns, search calls, tokens) — this is the raw data for
searches-per-episode and reward-over-training curves. `<run>` is the
`--output-model-id` (or `--wandb-run-name`, else a timestamp), so each run gets
its own file and the wandb run shares the same name. Resumable checkpoints
are saved every `--dcp-save-interval` optimizer steps (default 10).

## Resuming / extending a run

Those resumable checkpoints let you continue a finished (or interrupted) run
with `--init-from-checkpoint`, which restores the optimizer state, step
counter, and dataset cursor:

```bash
python train.py \
    --base-model accounts/fireworks/models/qwen3-4b-instruct-2507 \
    --tokenizer-model Qwen/Qwen3-4B-Instruct-2507 \
    --init-from-checkpoint "<job-id>:step-60" \
    --epochs 3 \
    --output-model-id exa-search-agent-v2
```

- **`<job-id>`** is the trainer job id printed at startup (checkpoints persist
  ~30 days under it). Use `"<job-id>:step-N"` to resume in a fresh job, or a
  bare `"step-N"` within the same job.
- **Raise the budget.** Because resume restores how many rows were already
  consumed, a run that finished its `--epochs`/`--max-rows` has no data left —
  bump `--epochs` (or `--max-rows`) so there is fresh data to train on,
  otherwise it resumes at the last step and immediately exits.
- Use a **new `--output-model-id`** (e.g. `-v2`) so the extended run promotes
  alongside the original instead of overwriting it, which keeps both available
  to compare in `eval.py`.

## Evaluating the result

`eval.py` runs the same agent loop through the standard chat-completions API
(`tools=`) on held-out validation splits, grades with the same judge, and
compares the base model to the tuned adapter:

```bash
python eval.py \
    --lora accounts/<acct>/models/exa-search-agent \
    --dataset mixed --limit 100
```

It provisions one dedicated deployment of the base model, attaches the
adapter(s) as LoRA addons, evaluates every model through that single GPU, and
deletes the deployment on exit (`--keep-deployment` / `--deployment <id>` to
reuse one). Per-question transcripts land in `eval_results/<timestamp>/`; the
summary table reports accuracy, searches/episode, turns/episode, and
no-answer rate per model.

## Key hyperparameters

| Flag | Default | Blog | Notes |
|---|---|---|---|
| `--base-model` | `qwen3-4b-instruct-2507` | Qwen3-4B-Instruct-2507 | The model being fine-tuned. |
| `--lora-rank` | `32` | LoRA | Blog trains LoRA adapters; `0` = full-parameter. |
| `--max-turns` | `6` | 10 | Max tool-call turns per question. |
| `--num-results` | `5` | 5 | Exa results per search. |
| `--search-type` | `auto` | `fast` | `fast` is cheaper/lower-latency for high-throughput RL. |
| `--completions-per-prompt` | `8` | 8 | GRPO group size per question. |
| `--prompt-groups-per-step` | `8` | n/a | Questions per optimizer step. |
| `--max-trajectory-tokens` | `30720` | 30720 | Token budget per trajectory; exceeding it ends the episode. |
| `--context-overflow-penalty` | `-0.25` | −0.25 | Reward when the trajectory exceeds the token budget. |
| `--no-answer-penalty` | `-0.1` | ~−0.1 | Reward when all turns pass without a final answer (blog: format penalty). |
| `--kl-beta` | `0.0` | 0.0 | KL penalty vs. the reference policy. |

## Data

`prepare_data.py` defaults to the blog's exact sources and mix: 50/50
**HotpotQA** (via `PeterJinGo/nq_hotpotqa_train`, the Search-R1 processed set,
filtered to HotpotQA rows with gold-answer alias lists) + **MuSiQue** (via
`dgslibisey/MuSiQue`), **train** splits only; the standard validation sets
stay clean for evaluating the trained agent. Single-source variants
(`--dataset hotpotqa|musique|2wiki`) are available for ablations.

One difference from the blog remains, by design: the blog also **pre-scores**
every question offline (base model, 8 samples each) and trains only on
mixed-difficulty questions (`0 < correct < 8`), so no rollout budget is spent
on questions the whole GRPO group gets uniformly right or wrong. This example
instead drops those constant-reward groups **at runtime** (the default
`dynamic_filter_fn` in `train.py`): zero up-front cost, at the price of some
wasted rollout throughput. If you scale this up, reproducing the offline
scoring pass is the first optimization worth making.

The blog trains with **Dr. GRPO**; the recipe's default `policy_loss="grpo"`
(REINFORCE + KL) is the closest available variant; see the async-RL reference
for the full list.

## Reproducing the backend comparison

The blog's headline is comparative (Exa vs. a Google/SERP proxy). This example
ships the **Exa** side and makes the backend a one-line swap: `ExaSearchTool` is
the only search-specific piece the rollout depends on. To reproduce the
head-to-head, implement a `SerpSearchTool` with the same
`search(query) -> observation` interface, select it via a flag, and train a
second run with everything else fixed, then compare reward curves and per-rollout
token/turn/search-call counts (Figures 2–4 in the blog). We keep this example
Exa-only; see the blog for the published comparison.

## Cost

A real run bills mainly on **GPU-hours** (trainer + rollout deployment) for the
run's wall-clock, plus Exa searches (~pennies each; cheaper with
`--search-type fast`) and the LLM-judge calls (a small completion per rollout).
Start with `SMOKE=1 bash run.sh` (a few dollars) to validate end-to-end, and
scale `--max-rows` / `--completions-per-prompt` up from there.

## Model compatibility

The default is Qwen3-4B-Instruct-2507, whose chat template renders `tools=`
specs and emits Hermes-style `<tool_call>{...}</tool_call>` blocks: the format
`parse_tool_calls` expects. Any instruct model with a Hermes-style tool
template (most Qwen-family models) works out of the box; for a different
tool-call format, adjust the parser in `exa_search.py`. The parser strips
`<think>...</think>` reasoning blocks before parsing, so thinking variants are
handled too. The judge must answer **without a reasoning trace** so its
one-word verdict is not swallowed by truncation: the default is
`qwen3p7-plus` with `reasoning_effort="none"` (see `JUDGE_MODEL` /
`JUDGE_REASONING_EFFORT` in `reward.py`).

One serving note: this recipe samples at `--temperature 1.0` because on-policy
RL requires drawing from the full policy distribution. When you *deploy* the
trained adapter behind the standard `tools=` API, follow the
[Fireworks tool-calling guidance](https://docs.fireworks.ai/guides/function-calling)
and use a low temperature (0–0.3) for reliable tool selection.

## References

- Exa, *How Search Quality Shapes RL Outcomes*: [exa.ai/blog/rl-search-outcomes](https://exa.ai/blog/rl-search-outcomes)
- [Exa Search API](https://docs.exa.ai/reference/search)
- [Fireworks tool calling](https://docs.fireworks.ai/guides/function-calling): the tool-spec contract the agent is trained against.
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1): the system-prompt lineage.
- `../multi_turn_message_in/`: the message-in async rollout this example builds on.
- `../../multihop_qa/`: the local-TF-IDF (closed-book) counterpart.
