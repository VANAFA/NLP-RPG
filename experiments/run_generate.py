"""Experiment 1 - Step 1: generate responses from all four model configs
over the full prompt battery (normal + adversarial), recording latency and
a deterministic truncation/repetition signal alongside each response.

Judging (judge.py) and aggregation (aggregate.py) are separate steps so a
judge-only or aggregate-only re-run doesn't require re-generating (useful
if you tweak the rubric or just want to recompute the summary table).
"""

import json

import torch
from tqdm import tqdm

import config as cfg
import models


def distinct_ngram_ratio(text: str, n: int = 2) -> float:
    """Cheap, deterministic repetition proxy: fraction of n-grams that are
    unique. Complements (doesn't replace) the judge's `repetition_detected`
    call, which can catch semantic/paraphrased loops this can't."""
    words = text.split()
    if len(words) < n:
        return 1.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def load_prompts():
    prompts = []
    with open(cfg.NORMAL_PROMPTS_PATH, "r", encoding="utf-8") as f:
        for p in json.load(f):
            prompts.append({**p, "is_adversarial": False})
    with open(cfg.ADVERSARIAL_PROMPTS_PATH, "r", encoding="utf-8") as f:
        for p in json.load(f):
            prompts.append({**p, "is_adversarial": True})
    return prompts


def main():
    torch.manual_seed(cfg.SEED)
    prompts = load_prompts()

    print(f"Loading base model ({cfg.BASE_MODEL_ID}) + computing its steering vector...")
    base = models.load_base()
    print(f"Loading finetuned model ({cfg.LORA_ADAPTER_ID}) + computing its steering vector...")
    finetuned = models.load_finetuned()

    loaded_by_key = {
        "base": base,
        "base_steered": base,
        "finetuned": finetuned,
        "finetuned_steered": finetuned,
    }

    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    with open(cfg.GENERATIONS_PATH, "w", encoding="utf-8") as out:
        for model_cfg in cfg.MODEL_CONFIGS:
            loaded = loaded_by_key[model_cfg["key"]]
            for prompt in tqdm(prompts, desc=model_cfg["key"]):
                text, latency_ms, hit_max_tokens = models.generate(
                    loaded, cfg.GM_SYSTEM_PROMPT, prompt["text"], use_steering=model_cfg["use_steering"]
                )
                row = {
                    "model_key": model_cfg["key"],
                    "model_label": model_cfg["label"],
                    "prompt_id": prompt["id"],
                    "prompt_text": prompt["text"],
                    "is_adversarial": prompt["is_adversarial"],
                    "difficulty": prompt.get("difficulty", "n/a"),
                    "response": text,
                    "latency_ms": round(latency_ms, 1),
                    "hit_max_tokens": hit_max_tokens,
                    "distinct_bigram_ratio": round(distinct_ngram_ratio(text), 3),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()

    print(f"Wrote {cfg.GENERATIONS_PATH}")


if __name__ == "__main__":
    main()
