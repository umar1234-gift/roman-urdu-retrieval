"""
Final Figure Generation — Frozen Numbers
==========================================
Regenerates all 8 figures to match the frozen final paper.

Run: python src_v2/49_final_figures.py
"""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LEVELS = ["L0", "L1", "L2", "L3", "L4"]
LEVEL_LABELS = ["L0\n(English)", "L1\n(Mixed)", "L2\n(Balanced)",
                "L3\n(Urdu-dom.)", "L4\n(Roman Urdu)"]


# ============================================================
# FROZEN NUMBERS (from clean pipeline rerun, Oct 1, 2026)
# ============================================================

# System: {level: @1}
ORIGINAL = {
    "BM25":                  [96.67, 97.78, 97.78, 75.56, 74.44],
    "E5-large":              [98.89, 98.89, 100.0, 84.44, 78.89],
    "Hybrid-large (RRF)":    [98.89, 98.89, 98.89, 84.44, 80.00],
    "Weighted-RRF":          [98.89, 98.89, 98.89, 86.67, 80.00],
    "Convex (in-fold)":      [98.89, 98.89, 98.89, 86.67, 82.22],
}

# A_single retry per-level (frozen)
A_SINGLE = [98.89, 98.89, 100.0, 98.89, 96.67]

# Overall metrics
OVERALL = {
    "BM25": 88.44,
    "E5-large": 92.22,
    "Hybrid-large (RRF)": 92.22,
    "Weighted-RRF": 92.67,
    "Convex (in-fold)": 93.11,
}

# Original vs A_single (per-level)
ORIGINAL_E5 = [98.89, 98.89, 100.0, 84.44, 78.89]

# Zero-score analysis
ZERO_BY_LEVEL = {"L0": 0, "L1": 0, "L2": 0, "L3": 9, "L4": 7}

# RRF variants
RRF_VARIANTS = {
    "A: Arbitrary tie": 90.67,
    "B: Skip BM25 (tie-aware)": 92.22,
    "C: Random tie": 91.11,
    "E5-large alone": 92.22,
}

# Significance data (frozen)
SIGNIFICANCE = [
    ("L0 vs L4\n(BM25)", 22.22, 0.0002),
    ("L0 vs L4\n(E5-base)", 27.78, 0.0002),
    ("L0 vs L4\n(E5-large)", 20.00, 0.0002),
    ("A_single vs Orig\n(L2-L4)", 10.74, 0.0004),
    ("Convex vs E5\n(L3-L4)", 2.78, 0.028),
    ("E5-large vs\nE5-base", 3.56, 0.0504),
]


def fig1_per_level():
    """Per-level @1 bar chart — 5 systems (no Adaptive-RRF)."""
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(LEVELS))
    n = len(ORIGINAL)
    width = 0.15

    colors = plt.cm.tab10(np.linspace(0, 1, n))

    for i, (name, vals) in enumerate(ORIGINAL.items()):
        offset = (i - n/2 + 0.5) * width
        bars = ax.bar(x + offset, [v/100 for v in vals], width,
                       label=name, color=colors[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v/100 + 0.005,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xlabel("Linguistic Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("@1 Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Per-Level Top-1 Accuracy — 5 Retrieval Systems (n=450)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = FIG_DIR / "fig1_per_level_p1.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def fig2_degradation():
    """Degradation curve — 5 systems."""
    fig, ax = plt.subplots(figsize=(11, 6))

    markers = ["o", "s", "^", "D", "v"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(ORIGINAL)))

    for i, (name, vals) in enumerate(ORIGINAL.items()):
        ax.plot(LEVELS, [v/100 for v in vals],
                marker=markers[i], linewidth=2.2, markersize=10,
                label=name, color=colors[i])

    ax.axvspan(2.5, 4.5, alpha=0.08, color="red")
    ax.text(3.5, 0.72, "Hard region\n(Roman Urdu)",
            ha="center", fontsize=11, color="darkred", fontweight="bold")

    ax.set_xlabel("Linguistic Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("@1 Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Degradation Curve: English (L0) to Roman Urdu (L4)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.70, 1.02)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = FIG_DIR / "fig2_degradation_curve.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def fig3_llm_effect():
    """LLM rewriting effect — Original E5 vs A_single retry."""
    x = np.arange(len(LEVELS))
    width = 0.35

    orig = [v/100 for v in ORIGINAL_E5]
    rewr = [v/100 for v in A_SINGLE]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, orig, width,
                    label="Original (E5-large)",
                    color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width/2, rewr, width,
                    label="A_single retry (LLM + E5-large)",
                    color="#27ae60", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

    for i, lvl in enumerate(LEVELS):
        if lvl in ["L3", "L4"]:
            delta = rewr[i] - orig[i]
            ax.annotate(f"delta={delta:+.3f}",
                        xy=(x[i] + width/2, rewr[i]),
                        xytext=(x[i] + width/2 + 0.15, rewr[i] + 0.06),
                        ha="center", fontsize=10, fontweight="bold",
                        color="darkgreen",
                        arrowprops=dict(arrowstyle="->",
                                        color="darkgreen", lw=1.5))

    ax.set_xlabel("Linguistic Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("@1 Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Effect of LLM Query Rewriting (A_single retry)",
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


def fig4_zero_score():
    """Zero-score BM25 queries by level (NEW - replaces weight sensitivity)."""
    fig, ax = plt.subplots(figsize=(9, 5))

    levels = list(ZERO_BY_LEVEL.keys())
    counts = list(ZERO_BY_LEVEL.values())
    colors = ["#27ae60", "#27ae60", "#f39c12", "#e74c3c", "#e74c3c"]

    bars = ax.bar(levels, counts, color=colors, alpha=0.85, edgecolor="black")

    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.15,
                str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Linguistic Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("Zero-Score Queries (of 90)", fontsize=12, fontweight="bold")
    ax.set_title("BM25 Zero-Score Queries by Level\n(no term match to any document)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = FIG_DIR / "fig4_zero_score_queries.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def fig5_rrf_variants():
    """RRF variants — tie-handling fix (NEW)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    names = list(RRF_VARIANTS.keys())
    vals = list(RRF_VARIANTS.values())
    colors = ["#e74c3c", "#27ae60", "#f39c12", "#3498db"]

    bars = ax.bar(names, vals, color=colors, alpha=0.85, edgecolor="black")

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.15,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_ylabel("@1 Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("RRF Variants: Zero-Score Tie-Breaking Artifact\n"
                 "(Proper handling recovers E5-large parity)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(88, 94)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.xticks(fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "fig5_rrf_tie_artifact.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def fig6_convex_per_level():
    """Convex vs E5-large per-level (NEW)."""
    fig, ax = plt.subplots(figsize=(11, 6))

    e5 = ORIGINAL["E5-large"]
    cvx = ORIGINAL["Convex (in-fold)"]
    x = np.arange(len(LEVELS))
    width = 0.35

    bars1 = ax.bar(x - width/2, [v/100 for v in e5], width,
                    label="E5-large", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width/2, [v/100 for v in cvx], width,
                    label="Convex (in-fold)", color="#27ae60", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Highlight L3-L4 improvement
    for i in [3, 4]:
        delta = (cvx[i] - e5[i]) / 100
        ax.annotate(f"+{delta*100:.2f}pp",
                    xy=(x[i] + width/2, cvx[i]/100),
                    xytext=(x[i] + width/2, cvx[i]/100 + 0.06),
                    ha="center", fontsize=10, fontweight="bold",
                    color="darkgreen",
                    arrowprops=dict(arrowstyle="->",
                                    color="darkgreen", lw=1.5))

    ax.set_xlabel("Linguistic Level", fontsize=12, fontweight="bold")
    ax.set_ylabel("@1 Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Convex Fusion vs E5-large: Improvement on Hard Subgroups",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_ylim(0.70, 1.10)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = FIG_DIR / "fig6_convex_per_level.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def fig7_significance():
    """Statistical significance bar chart (frozen p-values)."""
    names = [s[0] for s in SIGNIFICANCE]
    deltas = [s[1] for s in SIGNIFICANCE]
    pvals = [s[2] for s in SIGNIFICANCE]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    colors = ["#27ae60" if p < 0.05 else "#e74c3c" for p in pvals]
    bars = ax.bar(names, deltas, color=colors, alpha=0.85,
                   edgecolor="black")

    for bar, p, d in zip(bars, pvals, deltas):
        v = bar.get_height()
        label = f"p={p:.4f}" if p >= 0.0002 else "p<0.0002"
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                label, ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_ylabel("Delta @1 (percentage points)", fontsize=12, fontweight="bold")
    ax.set_title("Statistical Significance of Key Comparisons\n"
                 "Green = significant at p<0.05, Red = not significant",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.axhline(0, color="black", linewidth=1)

    plt.xticks(fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "fig7_significance.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def fig8_overall_comparison():
    """Overall @1 comparison (5 systems) - NEW."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(OVERALL.keys())
    vals = list(OVERALL.values())
    colors = ["#95a5a6", "#3498db", "#2980b9", "#9b59b6", "#27ae60"]

    bars = ax.barh(names, vals, color=colors, alpha=0.85, edgecolor="black")

    for bar, v in zip(bars, vals):
        ax.text(v + 0.15, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}%", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("@1 Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Overall Performance on Original Queries (n=450)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(86, 95)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = FIG_DIR / "fig8_overall_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def main():
    print("=" * 100)
    print("FINAL FIGURE GENERATION — FROZEN NUMBERS")
    print("=" * 100)
    print(f"Output: {FIG_DIR}\n")

    print("Generating 8 figures...")
    fig1_per_level()
    fig2_degradation()
    fig3_llm_effect()
    fig4_zero_score()
    fig5_rrf_variants()
    fig6_convex_per_level()
    fig7_significance()
    fig8_overall_comparison()

    print()
    print("=" * 100)
    print("ALL FIGURES REGENERATED")
    print("=" * 100)
    print()
    print("Old figures (delete these):")
    print("  fig1_per_level_p1.png   — was OK, now updated")
    print("  fig2_degradation_curve.png — was OK, now updated")
    print("  fig3_llm_rewriting_effect.png — was Adaptive-RRF, now A_single")
    print("  fig4_weight_sensitivity.png — REPLACED with fig4_zero_score_queries.png")
    print("  fig5_significance.png — was OK, now updated")
    print("  fig6_test_set_comparison.png — REPLACED with fig6_convex_per_level.png")
    print("  fig7_overfitting_story.png — REPLACED with fig7_significance.png")
    print("  fig8_weight_sweep.png — REPLACED with fig8_overall_comparison.png")


if __name__ == "__main__":
    main()