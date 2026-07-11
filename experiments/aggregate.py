"""Experiment 1 - Step 3: join generations.jsonl + judged.jsonl and compute
per-model summary metrics: thematic adherence, conversational quality
(truncation, repetition), four separate story-quality dimensions (fluency,
creativity, coherence, immersion — see judge.py's rubric), and latency.
Also breaks thematic adherence down by prompt difficulty tier
(data/prompts_adversarial.json's "difficulty" field) — the overall rate
saturates near 1.0 for most configs on the easier prompts alone, so the
breakdown is what actually discriminates between them.
"""

import pandas as pd

import config as cfg

ADHERENT_LABELS = ["rejected", "recontextualized"]


def main():
    gens = pd.read_json(cfg.GENERATIONS_PATH, lines=True)
    judged = pd.read_json(cfg.JUDGED_PATH, lines=True)
    df = gens.merge(judged, on=["model_key", "prompt_id"], how="left", validate="one_to_one")

    order = [c["key"] for c in cfg.MODEL_CONFIGS]

    # Rows where the judge failed even after retries (judge.py's safety net)
    # carry None for every judged field — excluded from every metric below
    # rather than silently counted as e.g. "not adherent", which would bias
    # the rates. They still count toward n_prompts so failures stay visible.
    n_failed_total = int(df["judge_failed"].sum())
    judged_ok = df[~df["judge_failed"]]

    rows = []
    for model_key, group in df.groupby("model_key", sort=False):
        ok_group = judged_ok[judged_ok["model_key"] == model_key]
        adversarial = ok_group[ok_group["is_adversarial"]]
        adherent = adversarial["thematic_adherence"].isin(ADHERENT_LABELS)
        n_failed = int(group["judge_failed"].sum())

        rows.append({
            "model_key": model_key,
            "model_label": group["model_label"].iloc[0],
            "n_prompts": len(group),
            "n_judge_failed": n_failed,
            "n_adversarial": len(adversarial),
            "thematic_adherence_rate": round(adherent.mean(), 3) if len(adversarial) else float("nan"),
            "judge_cut_off_rate": round(ok_group["cut_off"].mean(), 3),
            "hit_max_tokens_rate": round(group["hit_max_tokens"].mean(), 3),
            "avg_fluency_score": round(ok_group["fluency_score"].mean(), 2),
            "avg_creativity_score": round(ok_group["creativity_score"].mean(), 2),
            "avg_coherence_score": round(ok_group["coherence_score"].mean(), 2),
            "avg_immersion_score": round(ok_group["immersion_score"].mean(), 2),
            "judge_repetition_rate": round(ok_group["repetition_detected"].mean(), 3),
            "avg_distinct_bigram_ratio": round(group["distinct_bigram_ratio"].mean(), 3),
            "avg_latency_ms": round(group["latency_ms"].mean(), 1),
            "p95_latency_ms": round(group["latency_ms"].quantile(0.95), 1),
        })

    if n_failed_total:
        print(f"[warn] {n_failed_total} row(s) had a failed judge call and were excluded from judge-derived metrics.\n")

    summary = pd.DataFrame(rows).set_index("model_key").loc[order]
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(cfg.SUMMARY_CSV_PATH)
    print(summary.to_string())
    print(f"\nWrote {cfg.SUMMARY_CSV_PATH}")

    adversarial_all = judged_ok[judged_ok["is_adversarial"]].copy()
    adversarial_all["adherent"] = adversarial_all["thematic_adherence"].isin(ADHERENT_LABELS)
    by_difficulty = (
        adversarial_all.groupby(["model_key", "difficulty"])["adherent"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "adherence_rate", "count": "n"})
        .reset_index()
    )
    by_difficulty["adherence_rate"] = by_difficulty["adherence_rate"].round(3)
    pivot = by_difficulty.pivot(index="model_key", columns="difficulty", values="adherence_rate").loc[order]
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    pivot.to_csv(cfg.ADHERENCE_BY_DIFFICULTY_CSV_PATH)
    print("\nThematic adherence rate by difficulty tier:")
    print(pivot.to_string())
    print(f"\nWrote {cfg.ADHERENCE_BY_DIFFICULTY_CSV_PATH}")


if __name__ == "__main__":
    main()
