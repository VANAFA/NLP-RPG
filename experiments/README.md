# Experiments

Research harness for evaluating narrator model configurations, separate
from the game itself (`src/`, `backend/`). Implements both experiments from
the plan: **Experiment 1** (model quality / thematic control) and
**Experiment 2** (CAVEMAN log compression / memory retention).

No real results are checked in — `results/` and `figures/` are gitignored
and start empty. Run the experiments to populate them; nothing in this repo
should be presented as data until you've actually run it.

## Experiment 1 — what it does

Compares four model configurations on a battery of English prompts (30
normal fantasy actions, 64 "adversarial" prompts across four difficulty
tiers — `obvious`/`subtle`/`meta`/`false_lore`, see
`data/prompts_adversarial.json` — that try to break the fantasy setting):

| key | config |
|---|---|
| `base` | `Qwen/Qwen2.5-0.5B-Instruct`, unmodified |
| `base_steered` | base model + activation steering |
| `finetuned` | base model + `fireball-nlp/fireball-qwen2.5-0.5b-lora` |
| `finetuned_steered` | finetuned model + activation steering |

For each (model, prompt) pair it records the response, latency, and a
deterministic truncation/repetition signal, then sends every response to
the judge (config.JUDGE_BACKEND — see below) for: **thematic adherence**
(does the model reject/recontextualize the anachronism, or embrace it? —
scored on a deliberately strict rubric, see `judge.py`'s `JUDGE_SYSTEM`),
**four separate story-quality dimensions** (fluency, creativity, coherence,
immersion, each 1-5 — a single "fluency" number can't tell a competent-but-
generic response from a vivid one), **cut off** (does the response look
truncated?), and **repetition**. Results land in
`results/experiment1_summary.csv`, plus a difficulty-tier breakdown in
`results/experiment1_adherence_by_difficulty.csv` (the overall adherence
rate saturates for the stronger configs; the breakdown is what actually
discriminates).

Activation steering is Contrastive Activation Addition (CAA): a steering
direction is derived per-model (not shared between base/finetuned — each
model's residual stream gets its own vector) from `data/contrastive_pairs.json`,
then added to one decoder layer's output at every generated token. See the
docstring in `steering.py` for the method and references.

Run: `run_experiment1.py` (or the stages standalone: `run_generate.py` ->
`judge.py` -> `aggregate.py`).

## Experiment 2 — what it does

Plays a 23-turn scripted campaign through **one** model config (the winner
of Experiment 1 — set `EXPERIMENT2_MODEL_KEY` in `config.py` once you know
it; defaults to `finetuned_steered` per the plan's stated expectation) to
build a real history log, via `data/campaign_script.json`. Five of those
turns are written to prompt the GM toward stating a specific fact (where a
key is hidden, an NPC's name, the current objective, an item found, a
detail about an enemy).

Important design note: the reference answer for each fact is **not**
hardcoded — the scripted turns only *prompt* the GM, they don't dictate
what it says, and a 0.5B model won't reliably produce the exact wording you
might guess. So right after the log is built, each fact's ground truth is
extracted from what the model *actually* said on that turn (via a Claude
call), and grading later compares against that extracted fact. If a turn
doesn't clearly establish its fact, that question is dropped rather than
scored against a guess.

At checkpoints of increasing log length (`EXPERIMENT2_CHECKPOINTS`, default
`[6, 12, 18, 23]` turns), it:

1. Takes the log prefix up to that checkpoint, and its `caveman.py`-compressed
   version.
2. Computes **TCR** (Token Compression Ratio) from the two token counts.
3. Asks the model each applicable closed question — once with the original
   log as context, once with the compressed log — and grades each answer
   (via Claude) against the extracted reference fact, giving **IRA**
   (Information Retention Accuracy) for original vs. compressed at that
   checkpoint.

Results land in `results/experiment2_summary.csv`; the full generated log
plus extracted references land in `results/experiment2_campaign_log.json`
(useful for manually sanity-checking that the facts were actually
established before trusting the numbers).

**On `caveman.py`**: this is our own operationalization of the CAVEMAN
description given for this project (strip function words and redundancy,
keep entities/relations/hard facts) — heuristic stopword removal + a
capitalized-token/number "hard data" filter + near-duplicate sentence
dedup, no extra ML dependencies. It is **not** a verified reproduction of a
specific published algorithm — no paper/spec was available when this was
written. If you have the actual source, say so and this should be swapped
for a faithful port before the report cites it as "CAVEMAN".

Run: `run_experiment2.py`. Must be run after Experiment 1 gives you a
winner to set as `EXPERIMENT2_MODEL_KEY` (it does *not* run automatically
based on Experiment 1's output — that's a one-line manual edit to
`config.py` since it's your call to make from the results, not something to
automate silently).

## Figures

`make_figures.py` reads whichever of `results/experiment1_summary.csv` /
`results/experiment2_summary.csv` exist and writes PNGs to `figures/`:

- `figures/experiment1_metrics.png` — 2x2 bar charts (one per rubric axis)
  comparing the four configs.
- `figures/experiment2_memory.png` — line charts: IRA (original vs.
  compressed) and TCR, both vs. campaign length in turns.

It errors clearly (not silently) per-figure if the corresponding CSV
doesn't exist yet — it never invents data to plot.

## Setup

Reuses `backend/.venv` (already has `torch`/`transformers`/`peft`/`accelerate`)
instead of downloading a second CUDA environment — run `backend/setup.ps1`
first if that venv doesn't exist yet.

```powershell
.\experiments\setup.ps1
copy experiments\.env.example experiments\.env   # then fill in ANTHROPIC_API_KEY, and ideally HF_TOKEN
```

`HF_TOKEN` is optional but strongly recommended: anonymous Hugging Face
downloads of large model files get bandwidth-throttled; an authenticated
request (free account, "Read" token at huggingface.co/settings/tokens)
generally isn't. Without it, downloading the base model the first time can
take a couple of hours instead of a couple of minutes on an otherwise-normal
connection.

## Running everything

```powershell
cd experiments
..\backend\.venv\Scripts\python.exe run_experiment1.py
# read results\experiment1_summary.csv, set EXPERIMENT2_MODEL_KEY in config.py to the winner
..\backend\.venv\Scripts\python.exe run_experiment2.py
..\backend\.venv\Scripts\python.exe make_figures.py
```

Both models download from Hugging Face on first run (cached after that).
To re-run only one stage of Experiment 1 (e.g. after editing the judge
rubric in `judge.py`), run `run_generate.py`, `judge.py`, or `aggregate.py`
directly — each stage reads/writes the same `results/*.jsonl` files.

## Tuning knobs (`config.py`)

- `STEERING_LAYER_FRACTION` / `STEERING_ALPHA` — which decoder layer to
  steer and how strongly. The defaults (mid-depth layer, alpha=8 on a
  *unit* vector) are a starting point, not a validated setting — for a
  report you should sweep `STEERING_ALPHA` and sanity-check that the
  steered model still produces coherent (not garbled) text before trusting
  the thematic-adherence numbers. `run_generate.py`/`run_experiment2.py`
  print the picked layer index on startup.
- `MAX_NEW_TOKENS` / `TEMPERATURE` / `SEED` — Experiment 1 generation
  params, shared across all four configs for a fair comparison.
- `EXPERIMENT2_MODEL_KEY` / `EXPERIMENT2_CHECKPOINTS` / `EXPERIMENT2_TEMPERATURE`
  — which config, at which log lengths, and how deterministic the campaign
  generation is (lower than Experiment 1's, since fact-consistency matters
  more here than narrative creativity).
- `data/prompts_normal.json` / `data/prompts_adversarial.json` — Experiment
  1's prompt battery; extend these to widen coverage before drawing strong
  conclusions from n=12 per category.
- `data/campaign_script.json` — Experiment 2's scripted turns and questions;
  extend the `turns` list and add more `questions` (with new `source_turn`s)
  for a richer memory-retention test.
