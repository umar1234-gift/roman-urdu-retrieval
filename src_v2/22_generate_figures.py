"""
Generate publication-quality figures for the paper.
====================================================
Reads from existing result JSON files (no re-computation needed).

Outputs to: paper/figures/*.png

Run: python src_v2/22_generate_figures.py
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LEVELS = ["L0", "L1", "L2", "L3", "L4"]
LEVEL_LABELS = ["L0\n(English)", "L1\n(Mixed)", "L2\n(Balanced)",
                "L3\n(Urdu-dom.)", "L4\n(Roman Urdu)"]


def load_json(p):
    if not Path(p).exists():
        print(f"  [WARN] Missing: {p}")
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def p1(records, level=None):
    """Compute @1 accuracy from records."""
    if level:
        records = [r for r in records if r["level"] == level]
    if not records:
        return 0.0
    g1 = sum(1 for r in records if r["gold_rank"] == 1)
    return g1 / len(records)


def p3(records, level=None):
    if level:
        records = [r for r in records if r["level"] == level]
    if not records:
        return 0.0
    g3 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 3)
    return g3 / len(records)


def mrr(records, level=None):
    if level:
        records = [r for r in records if r["level"] == level]
    if not records:
        return 0.0
    return sum(1/r["gold_rank"] for r in records if r["gold_rank"]) / len(records)


# ============================================================
# LOAD ALL RESULTS
# ============================================================

def load_results():
    r = {}

    # Original queries (full 450)
    r["BM25"] = load_json(PROJECT_ROOT / "data_v2" / "results" / "bm25_results.json")
    r["E5-base"] = load_json(PROJECT_ROOT / "data_v2" / "results" / "dense_results.json")

    e5_all = load_json(PROJECT_ROOT / "data_v2" / "results" / "dense_model_comparison.json")
    r["E5-large"] = e5_all["e5-large"] if e5_all and "e5-large" in e5_all else None

    r["Hybrid-base"] = load_json(PROJECT_ROOT / "data_v2" / "results" / "hybrid_results.json")
    r["Hybrid-large"] = load_json(PROJECT_ROOT / "data_v2" / "results" / "hybrid_e5large_results.json")

    adapt = load_json(PROJECT_ROOT / "data_v2" / "results" / "adaptive_rrf_results.json")
    r["Adaptive-RRF"] = adapt["per_query"] if adapt and "per_query" in adapt else None

    # Rewritten queries
    r["Rewritten"] = load_json(PROJECT_ROOT / "data_v2" / "results" / "rewritten_retrieval.json")

    # Test set results
    r["test_orig"] = load_json(PROJECT_ROOT / "data_v2" / "results" / "train_val_test_results.json")
    r["test_llm"] = load_json(PROJECT_ROOT / "data_v2" / "results" / "llm_rewriting_test_set_results.json")

    return r


# ============================================================
# FIGURE 1: Per-level @1 bar chart (6 systems, full 450)
# ============================================================

def fig1(results):
    systems = ["BM25", "E5-base", "E5-large", "Hybrid-base", "Hybrid-large", "Adaptive-RRF"]
    available = {k: results[k] for k in systems if results[k]}

    if not available:
        print("  [SKIP] fig1 — no data")
        return

    x = np.arange(len(LEVELS))
    n = len(available)
    width = 0.13

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    for i, (name, recs) in enumerate(available.items()):
        vals = [p1(recs, lvl) for lvl in LEVELS]
        offset = (i - n/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=colors[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xlabel("Linguistic Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("@1 Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Per-Level Top-1 Accuracy — 6 Retrieval Systems (n=450)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="lower left", fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = FIG_DIR / "fig1_per_level_p1.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ============================================================
# FIGURE 2: Degradation curve (L0 → L4)
# ============================================================

def fig2(results):
    systems = ["BM25", "E5-base", "E5-large", "Hybrid-base", "Hybrid-large", "Adaptive-RRF"]
    available = {k: results[k] for k in systems if results[k]}

    if not available:
        print("  [SKIP] fig2 — no data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v", "p"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(available)))

    for i, (name, recs) in enumerate(available.items()):
        vals = [p1(recs, lvl) for lvl in LEVELS]
        ax.plot(LEVELS, vals, marker=markers[i], linewidth=2.2,
                markersize=10, label=name, color=colors[i])

    # Highlight hard region (L3–L4)
    ax.axvspan(2.5, 4.5, alpha=0.08, color="red")
    ax.text(3.5, 0.68, "Hard\nregion", ha="center", fontsize=11,
            color="darkred", fontweight="bold")

    ax.set_xlabel("Linguistic Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("@1 Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Degradation Curve: English (L0) → Roman Urdu (L4)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.65, 1.02)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = FIG_DIR / "fig2_degradation_curve.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ============================================================
# FIGURE 3: LLM Rewriting Effect (before/after)
# ============================================================

def fig3(results):
    orig = results["Adaptive-RRF"]
    rw = results["Rewritten"]

    if not orig or not rw:
        print("  [SKIP] fig3 — no data")
        return

    x = np.arange(len(LEVELS))
    width = 0.35

    orig_vals = [p1(orig, lvl) for lvl in LEVELS]
    rw_vals = [p1(rw, lvl) for lvl in LEVELS]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars1 = ax.bar(x - width/2, orig_vals, width,
                    label="Original (Adaptive-RRF)",
                    color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width/2, rw_vals, width,
                    label="Rewritten (LLM + Adaptive-RRF)",
                    color="#27ae60", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

    # Annotate deltas for L3, L4
    for i, lvl in enumerate(LEVELS):
        if lvl in ["L3", "L4"]:
            delta = rw_vals[i] - orig_vals[i]
            ax.annotate(f"Δ={delta:+.3f}",
                        xy=(x[i] + width/2, rw_vals[i]),
                        xytext=(x[i] + width/2 + 0.15, rw_vals[i] + 0.06),
                        ha="center", fontsize=10, fontweight="bold",
                        color="darkgreen",
                        arrowprops=dict(arrowstyle="->", color="darkgreen",
                                        lw=1.5))

    ax.set_xlabel("Linguistic Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("@1 Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Effect of LLM Query Rewriting on Retrieval Accuracy (n=450)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = FIG_DIR / "fig3_llm_rewriting_effect.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ============================================================
# FIGURE 4: Weight Sensitivity Heatmap
# ============================================================

def fig4():
    weights = [(1.0, 1.0), (0.7, 0.3), (0.5, 0.5), (0.4, 0.6),
               (0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.0, 1.0)]

    # From script 16 grid search output
    data = {
        "L0": [0.9889, 0.9889, 0.9889, 0.9889, 0.9889, 0.9889, 0.9889, 0.9889],
        "L1": [0.9889, 0.9889, 0.9889, 0.9889, 0.9889, 0.9889, 0.9889, 0.9889],
        "L2": [0.9889, 0.9889, 0.9889, 0.9889, 0.9889, 1.0000, 1.0000, 1.0000],
        "L3": [0.8000, 0.8000, 0.8000, 0.8111, 0.8444, 0.8667, 0.8444, 0.8444],
        "L4": [0.7667, 0.7667, 0.7667, 0.7667, 0.7778, 0.7778, 0.7889, 0.7889],
    }

    matrix = np.array([data[lvl] for lvl in LEVELS])

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.74, vmax=1.0)

    ax.set_xticks(np.arange(len(weights)))
    ax.set_xticklabels([f"{w[0]:.1f}/{w[1]:.1f}" for w in weights],
                        rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(LEVELS)))
    ax.set_yticklabels(LEVELS, fontsize=11)

    ax.set_xlabel("Weight Configuration  (w_BM25 / w_E5)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Linguistic Level", fontsize=12, fontweight="bold")
    ax.set_title("Weight Sensitivity: @1 Accuracy across RRF Weight Configurations",
                 fontsize=13, fontweight="bold")

    for i in range(len(LEVELS)):
        for j in range(len(weights)):
            v = matrix[i, j]
            color = "white" if v < 0.82 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="@1 Accuracy")
    plt.tight_layout()
    out = FIG_DIR / "fig4_weight_sensitivity.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ============================================================
# FIGURE 5: Statistical Significance
# ============================================================

def fig5():
    comparisons = [
        "BM25\nvs\nE5-base",
        "BM25\nvs\nE5-large",
        "E5-base\nvs\nE5-large",
        "Hybrid-large\nvs\nAdaptive-RRF",
        "BM25\nvs\nAdaptive-RRF",
        "Original\nvs\nRewritten\n(L4)",
    ]
    pvals = [1.000, 0.0015, 0.0052, 0.0352, 0.0005, 0.000015]

    # -log10(p), cap at 6
    neg_log = [-np.log10(p) if p > 0 else 6 for p in pvals]
    neg_log = [min(x, 6) for x in neg_log]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    colors = ["#e74c3c" if p >= 0.05 else "#27ae60" for p in pvals]
    bars = ax.bar(comparisons, neg_log, color=colors, alpha=0.85, edgecolor="black")

    # p=0.05 threshold
    threshold = -np.log10(0.05)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=2,
                label=f"p = 0.05 threshold")

    for bar, p in zip(bars, pvals):
        v = bar.get_height()
        label = f"p={p:.4f}" if p >= 0.0001 else "p<0.0001"
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.15,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("$-\\log_{10}(p)$", fontsize=12, fontweight="bold")
    ax.set_title("Statistical Significance of System Comparisons\n"
                 "(Above threshold = statistically significant)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 7)

    plt.xticks(rotation=0, fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "fig5_significance.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ============================================================
# FIGURE 6: Test Set Honest Comparison (before/after LLM)
# ============================================================

def fig6(results):
    test_llm = results["test_llm"]
    if not test_llm:
        print("  [SKIP] fig6 — no test_llm data")
        return

    # Systems on test set
    labels = [
        "ORIG:\nBM25",
        "ORIG:\nE5-large",
        "ORIG:\nStd RRF",
        "ORIG:\nAdaptive RRF",
        "RW:\nBM25+LLM",
        "RW:\nE5-large+LLM",
        "RW:\nStd RRF+LLM",
        "RW:\nAdap RRF+LLM",
    ]

    orig = test_llm["test_original"]
    rw = test_llm["test_rewritten"]

    vals = [
        orig["bm25"]["@1"],
        orig["e5_large"]["@1"],
        orig["standard_rrf"]["@1"],
        orig["adaptive_rrf"]["@1"],
        rw["bm25_llm"]["@1"],
        rw["e5_large_llm"]["@1"],
        rw["standard_rrf_llm"]["@1"],
        rw["adaptive_rrf_llm"]["@1"],
    ]

    colors = ["#e74c3c"]*4 + ["#27ae60"]*4

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="black")

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Highlight best
    best_idx = int(np.argmax(vals))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)

    ax.set_ylabel("@1 Accuracy (Test Set, n=90)", fontsize=12, fontweight="bold")
    ax.set_title("Held-Out Test Set: Original vs LLM-Rewritten Queries\n"
                 "(Red = original, Green = rewritten)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.axhline(0.95, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    out = FIG_DIR / "fig6_test_set_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ============================================================
# FIGURE 7: RRF Failure Story (original vs test set)
# ============================================================

def fig7():
    """Show how Adaptive RRF looked great on full data but failed on test set."""

    systems = ["BM25", "E5-large", "Std RRF", "Adaptive RRF"]

    # Full-dataset (overfitted)
    full_data = [0.8844, 0.9222, 0.9067, 0.9267]

    # Test set (honest)
    test_data = [0.8667, 0.9222, 0.8778, 0.9000]

    x = np.arange(len(systems))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    bars1 = ax.bar(x - width/2, full_data, width,
                    label="Full dataset (450, overfitted)",
                    color="#95a5a6", alpha=0.85)
    bars2 = ax.bar(x + width/2, test_data, width,
                    label="Held-out test set (90, honest)",
                    color="#2c3e50", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Highlight the RRF failure
    idx_rrf = 3
    ax.annotate("RRF appeared best\non full data...",
                xy=(x[idx_rrf] - width/2, full_data[idx_rrf]),
                xytext=(x[idx_rrf] - 0.7, 0.95),
                ha="center", fontsize=9, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5))
    ax.annotate("...but FAILS on\ntest set (negative finding)",
                xy=(x[idx_rrf] + width/2, test_data[idx_rrf]),
                xytext=(x[idx_rrf] + 0.7, 0.82),
                ha="center", fontsize=9, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5))

    ax.set_ylabel("@1 Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("The Overfitting Story: Adaptive RRF on Full Data vs Test Set",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=11)
    ax.set_ylim(0.80, 1.02)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = FIG_DIR / "fig7_overfitting_story.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 100)
    print(f"Output dir: {FIG_DIR}\n")

    results = load_results()

    print("Generating figures...")
    fig1(results)
    fig2(results)
    fig3(results)
    fig4()
    fig5()
    fig6(results)
    fig7()

    print()
    print("=" * 100)
    print(f"All figures saved to: {FIG_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()