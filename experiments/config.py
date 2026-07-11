"""Shared configuration for the Experiment 1 and Experiment 2 harnesses.
Single source of truth for model ids, generation params, steering
hyperparameters, and file paths, so every script agrees on them.
"""

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

# Loaded once here (every script imports config) so ANTHROPIC_API_KEY and
# HF_TOKEN in experiments/.env reach every script, including huggingface_hub
# itself — HF_TOKEN is picked up automatically by transformers/huggingface_hub
# and lifts the bandwidth throttling anonymous downloads get on large files.
# Pointed at ROOT/.env explicitly (rather than default cwd-search) so this
# works regardless of which directory a script is launched from.
load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_ADAPTER_ID = "fireball-nlp/fireball-qwen2.5-0.5b-lora"

# The four configurations compared in Experiment 1. `use_lora`/`use_steering`
# drive models.py's loader; `key` is used as the row id everywhere
# downstream (generations.jsonl, judged.jsonl, the summary table).
MODEL_CONFIGS = [
    {"key": "base", "label": "Base (Qwen2.5-0.5B-Instruct)", "use_lora": False, "use_steering": False},
    {"key": "base_steered", "label": "Base + Activation Steering", "use_lora": False, "use_steering": True},
    {"key": "finetuned", "label": "Finetuned (FIREBALL LoRA)", "use_lora": True, "use_steering": False},
    {"key": "finetuned_steered", "label": "Finetuned + Activation Steering", "use_lora": True, "use_steering": True},
]

# --- Generation ------------------------------------------------------------

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9
SEED = 42

GM_SYSTEM_PROMPT = (
    "You are the Game Master of a dark fantasy RPG. You narrate the world and react to the "
    "player's actions in 2-4 sentences of vivid, in-world prose. Stay strictly within a "
    "medieval fantasy setting: swords, magic, castles, forests, dungeons, mythical creatures. "
    "Never introduce modern technology or science-fiction elements — if the player mentions "
    "something anachronistic, reinterpret or reject it in a way that keeps the scene coherent "
    "with a fantasy world."
)

# --- Activation steering (Contrastive Activation Addition) -----------------
#
# Steering vector = mean(activations on fantasy-continuation text) -
# mean(activations on anachronism-continuation text), captured at the last
# token of each contrastive example, at one decoder layer. Added to every
# generated token's hidden state at that layer during generation. See
# steering.py for the extraction/hook implementation.

STEERING_LAYER_FRACTION = 0.5  # fraction of model depth; layer = round(num_layers * fraction)
STEERING_ALPHA = 8.0  # multiplier on the *unit* steering vector; tune per model, see README
CONTRASTIVE_PAIRS_PATH = DATA_DIR / "contrastive_pairs.json"

# --- Prompt battery ----------------------------------------------------------

NORMAL_PROMPTS_PATH = DATA_DIR / "prompts_normal.json"
ADVERSARIAL_PROMPTS_PATH = DATA_DIR / "prompts_adversarial.json"

# --- Judge (LLM-as-a-Judge) ---------------------------------------------------
#
# "anthropic": Claude via judge_client.py — best judge quality, needs
# ANTHROPIC_API_KEY. "local": a locally-run model with schema-constrained
# decoding (local_judge.py) — free, no key, but a much weaker judge (see
# that module's docstring). Set to "local" here since no Anthropic key is
# available for this run.
JUDGE_BACKEND = "local"

JUDGE_MODEL = "claude-sonnet-5"
JUDGE_MAX_RETRIES = 3

# Local judge model: deliberately NOT one of the 0.5B configs being
# evaluated (self-judging would be circular) — reuses the 4B model the
# actual game backend already runs (backend/server.py), so it's a
# known-good choice for this environment.
LOCAL_JUDGE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LOCAL_JUDGE_MAX_NEW_TOKENS = 350
LOCAL_JUDGE_MAX_RETRIES = 3

# --- Outputs (Experiment 1) --------------------------------------------------

GENERATIONS_PATH = RESULTS_DIR / "generations.jsonl"
JUDGED_PATH = RESULTS_DIR / "judged.jsonl"
SUMMARY_CSV_PATH = RESULTS_DIR / "experiment1_summary.csv"
ADHERENCE_BY_DIFFICULTY_CSV_PATH = RESULTS_DIR / "experiment1_adherence_by_difficulty.csv"

# --- Experiment 2 (CAVEMAN compression / memory retention) -------------------
#
# Which Experiment 1 config to use as "the winner". Set from real
# Experiment 1 results (results/experiment1_summary.csv), not the plan's
# original assumption: that run showed STEERING_ALPHA=8.0 pushing
# hit_max_tokens_rate to 0.54-0.75 (the model rambles instead of
# converging) with the worst latency of the four configs, while plain
# "finetuned" had equal-best thematic adherence (1.0), the lowest
# truncation rate (0.042), and the lowest latency.
EXPERIMENT2_MODEL_KEY = "finetuned"

# Lower than Experiment 1's TEMPERATURE: this experiment depends on the GM
# actually stating specific planted facts, so less sampling variance is
# more important here than narrative creativity.
EXPERIMENT2_TEMPERATURE = 0.3
EXPERIMENT2_MAX_NEW_TOKENS = 160

# Log lengths (in turns) at which TCR/IRA are measured — all >= the last
# question's source_turn (16), so every checkpoint can evaluate every
# question. 16 planted facts (up from 10) gives more headroom against
# extraction dropout: the first pass (5 facts) lost 2/5 to dropout (n=3
# questions); the second pass (10 facts) lost 4/10 (n=6). At a similar
# ~50-60% yield, 16 facts should land around n=8-10 usable questions.
#
# Capped total campaign length at 40 turns (not higher): build_campaign_log
# reprocesses the ENTIRE growing conversation from scratch every turn (no
# KV-cache reuse across turns), so latency/VRAM use grows roughly
# quadratically with turn count. A 58-turn attempt crashed with a raw
# segfault (exit 139, no Python traceback) right as the log finished
# building — almost certainly a CUDA OOM under that quadratic growth,
# compounded by then needing to load the 4B judge model on top. 40 turns
# stays inside the range that was still fast (turns 1-47 took ~1-2s each in
# that run; the slowdown only kicked in around turn 48+).
CAMPAIGN_SCRIPT_PATH = DATA_DIR / "campaign_script.json"
EXPERIMENT2_CHECKPOINTS = [18, 26, 34, 40]

EXPERIMENT2_LOG_PATH = RESULTS_DIR / "experiment2_campaign_log.json"
EXPERIMENT2_SUMMARY_CSV_PATH = RESULTS_DIR / "experiment2_summary.csv"
