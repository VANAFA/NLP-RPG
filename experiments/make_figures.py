"""Generates report-ready figures from the CSV summaries produced by
aggregate.py (Experiment 1) and run_experiment2.py (Experiment 2), saved to
experiments/figures/. Run after the corresponding experiment has produced
real results — this does NOT fabricate data; it errors clearly (per
figure) if the expected CSV doesn't exist yet.
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as cfg

MODEL_ORDER = [c["key"] for c in cfg.MODEL_CONFIGS]
MODEL_LABELS = {c["key"]: c["label"] for c in cfg.MODEL_CONFIGS}
COLORS = ["#8c8c8c", "#4c72b0", "#dd8452", "#55a868"]  # base, base_steered, finetuned, finetuned_steered


def _require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} doesn't exist yet - run the experiment that produces it before making figures."
        )


QUALITY_DIMENSIONS = [
    ("avg_fluency_score", "Fluency"),
    ("avg_creativity_score", "Creativity"),
    ("avg_coherence_score", "Coherence"),
    ("avg_immersion_score", "Immersion"),
]


def plot_experiment1():
    _require(cfg.SUMMARY_CSV_PATH)
    df = pd.read_csv(cfg.SUMMARY_CSV_PATH, index_col="model_key").loc[MODEL_ORDER]
    labels = [MODEL_LABELS[k] for k in df.index]
    x = range(len(labels))
    width = 0.35

    # Single column, stacked vertically (portrait paper layout) rather than
    # a 2x2 grid — see plot_experiment2 for the same treatment.
    fig, axes = plt.subplots(4, 1, figsize=(7, 20))
    fig.suptitle("Experiment 1 - Narrative Quality and Thematic Control", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.bar(labels, df["thematic_adherence_rate"], color=COLORS)
    ax.set_title("Thematic adherence rate (higher = better)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1]
    rel_width = 0.8 / 3
    ax.bar([i - rel_width for i in x], df["judge_cut_off_rate"], rel_width, label="judge: cut off", color="#c44e52")
    ax.bar(list(x), df["hit_max_tokens_rate"], rel_width, label="hit max_tokens", color="#937860")
    ax.bar([i + rel_width for i in x], df["judge_repetition_rate"], rel_width, label="repetition", color="#8c8c34")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title("Reliability issues (lower = better)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    # Story quality: one group of bars per dimension (fluency/creativity/
    # coherence/immersion), one color per model — same layout as the
    # difficulty-breakdown figure, so the two are visually consistent.
    ax = axes[2]
    dim_keys = [k for k, _ in QUALITY_DIMENSIONS]
    dim_labels = [label for _, label in QUALITY_DIMENSIONS]
    n_models = len(df.index)
    qwidth = 0.8 / n_models
    xd = range(len(dim_keys))
    for i, (model_key, color, label) in enumerate(zip(df.index, COLORS, labels)):
        offsets = [xi + (i - (n_models - 1) / 2) * qwidth for xi in xd]
        values = [df.loc[model_key, k] for k in dim_keys]
        ax.bar(offsets, values, qwidth, label=label, color=color)
    ax.set_xticks(list(xd))
    ax.set_xticklabels(dim_labels)
    ax.set_title("Story quality (1-5, higher = better)")
    ax.set_ylim(0, 5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[3]
    ax.bar([i - width / 2 for i in x], df["avg_latency_ms"], width, label="avg", color="#4c72b0")
    ax.bar([i + width / 2 for i in x], df["p95_latency_ms"], width, label="p95", color="#8172b2")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title("Latency, ms (lower = better)")
    ax.legend(fontsize=8)

    # Explicit top margin: with 4 tall stacked panels, tight_layout's
    # automatic spacing doesn't reliably clear room for suptitle, and it was
    # overlapping the first panel's own title.
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    cfg.FIGURES_DIR.mkdir(exist_ok=True)
    out_path = cfg.FIGURES_DIR / "experiment1_metrics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


DIFFICULTY_ORDER = ["obvious", "subtle", "meta", "false_lore"]


def plot_experiment1_difficulty():
    _require(cfg.ADHERENCE_BY_DIFFICULTY_CSV_PATH)
    df = pd.read_csv(cfg.ADHERENCE_BY_DIFFICULTY_CSV_PATH, index_col="model_key").loc[MODEL_ORDER]
    tiers = [t for t in DIFFICULTY_ORDER if t in df.columns]
    labels = [MODEL_LABELS[k] for k in df.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Experiment 1 - Thematic Adherence by Prompt Difficulty", fontsize=14, fontweight="bold")

    n_models = len(df.index)
    width = 0.8 / n_models
    x = range(len(tiers))
    for i, (model_key, color, label) in enumerate(zip(df.index, COLORS, labels)):
        offsets = [xi + (i - (n_models - 1) / 2) * width for xi in x]
        values = [df.loc[model_key, t] for t in tiers]
        ax.bar(offsets, values, width, label=label, color=color)

    ax.set_xticks(list(x))
    ax.set_xticklabels([t.replace("_", " ") for t in tiers])
    ax.set_ylabel("Thematic adherence rate (higher = better)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    cfg.FIGURES_DIR.mkdir(exist_ok=True)
    out_path = cfg.FIGURES_DIR / "experiment1_adherence_by_difficulty.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


LEVEL_ORDER = ["none", "light", "medium", "aggressive"]
LEVEL_COLORS = {"none": "#8c8c8c", "light": "#4c72b0", "medium": "#dd8452", "aggressive": "#c44e52"}
LEVEL_LABELS = {"none": "None (raw log)", "light": "Light", "medium": "Medium", "aggressive": "Aggressive"}


def plot_experiment2():
    _require(cfg.EXPERIMENT2_SUMMARY_CSV_PATH)
    df = pd.read_csv(cfg.EXPERIMENT2_SUMMARY_CSV_PATH)
    levels = [l for l in LEVEL_ORDER if l in df["level"].unique()]

    # Single column, stacked vertically (portrait paper layout).
    fig, axes = plt.subplots(3, 1, figsize=(7, 15))
    ax1, ax2, ax3 = axes
    fig.suptitle("Experiment 2 - CAVEMAN Compression & Memory Retention", fontsize=14, fontweight="bold")

    for level in levels:
        sub = df[df["level"] == level].sort_values("checkpoint_turns")
        color = LEVEL_COLORS[level]
        label = LEVEL_LABELS[level]
        ax1.plot(sub["checkpoint_turns"], sub["ira"], marker="o", label=label, color=color)
        ax2.plot(sub["checkpoint_turns"], sub["tcr"], marker="o", label=label, color=color)

    ax1.set_xlabel("Campaign length (turns)")
    ax1.set_ylabel("Information Retention Accuracy (IRA)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("IRA vs. campaign length, by compression level")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Campaign length (turns)")
    ax2.set_ylabel("Token Compression Ratio (TCR)")
    ax2.set_title("TCR vs. campaign length, by compression level")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # The actual trade-off curve: how much IRA costs as compression gets
    # stronger, averaged across checkpoints per level (n is small per
    # checkpoint, so this trades some granularity for a less noisy line).
    avg = df.groupby("level")[["tcr", "ira"]].mean().reindex(levels)
    ax3.plot(avg["tcr"], avg["ira"], marker="o", color="#55a868", linewidth=2)
    for level in levels:
        ax3.annotate(LEVEL_LABELS[level], (avg.loc[level, "tcr"], avg.loc[level, "ira"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax3.set_xlabel("Token Compression Ratio (TCR), avg across checkpoints")
    ax3.set_ylabel("Information Retention Accuracy (IRA), avg across checkpoints")
    ax3.set_ylim(0, 1.05)
    ax3.set_xlim(left=-0.02)
    ax3.set_title("Compression vs. retention trade-off")
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    cfg.FIGURES_DIR.mkdir(exist_ok=True)
    out_path = cfg.FIGURES_DIR / "experiment2_memory.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    figures = [
        ("Experiment 1", plot_experiment1),
        ("Experiment 1 (by difficulty)", plot_experiment1_difficulty),
        ("Experiment 2", plot_experiment2),
    ]
    skipped = []
    for name, fn in figures:
        try:
            fn()
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
            skipped.append(name)
    if skipped:
        print(f"\n{len(skipped)}/{len(figures)} figure(s) skipped - run the corresponding experiment first.")
        sys.exit(1 if len(skipped) == len(figures) else 0)


if __name__ == "__main__":
    main()
