from pathlib import Path
import json
import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BM25_FILE = PROJECT_ROOT / "data_v2" / "results" / "bm25_results.json"
E5BASE_FILE = PROJECT_ROOT / "data_v2" / "results" / "dense_results.json"
E5LARGE_FILE = PROJECT_ROOT / "data_v2" / "results" / "dense_model_comparison.json"
HYBRID_BASE_FILE = PROJECT_ROOT / "data_v2" / "results" / "hybrid_results.json"
HYBRID_LARGE_FILE = PROJECT_ROOT / "data_v2" / "results" / "hybrid_e5large_results.json"
ADAPTIVE_FILE = PROJECT_ROOT / "data_v2" / "results" / "adaptive_rrf_results.json"

OUTPUT_FILE = PROJECT_ROOT / "data_v2" / "results" / "statistical_tests_v3.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "statistical_tests_v3.txt"


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def rr_vec(records, ids):
    by_id = {r["query_id"]: r for r in records}
    return np.array([
        1.0 / by_id[q]["gold_rank"] if by_id[q]["gold_rank"] else 0.0
        for q in ids
    ])


def top1_vec(records, ids):
    by_id = {r["query_id"]: r for r in records}
    return np.array([
        1 if by_id[q]["gold_rank"] == 1 else 0
        for q in ids
    ])


def mcnemar(a, b):
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0}
    n = n01 + n10
    k = min(n01, n10)
    p = 2 * stats.binom.cdf(k, n, 0.5)
    p = min(p, 1.0)
    return {"n01": n01, "n10": n10, "p_value": float(p)}


def wilcoxon(a, b):
    diffs = a - b
    if np.all(diffs == 0):
        return {"statistic": 0.0, "p_value": 1.0}
    stat, p = stats.wilcoxon(a, b)
    return {"statistic": float(stat), "p_value": float(p)}


def bootstrap_ci(values, n=5000, alpha=0.05):
    rng = np.random.default_rng(42)
    boots = [np.mean(rng.choice(values, size=len(values), replace=True))
             for _ in range(n)]
    return float(np.quantile(boots, alpha/2)), float(np.quantile(boots, 1-alpha/2))


def main():
    # Load all systems
    bm25 = load_json(BM25_FILE)
    e5base = load_json(E5BASE_FILE)
    e5large = load_json(E5LARGE_FILE)["e5-large"]
    hyb_base = load_json(HYBRID_BASE_FILE)
    hyb_large = load_json(HYBRID_LARGE_FILE)
    adaptive = load_json(ADAPTIVE_FILE)["per_query"]

    ids = sorted({r["query_id"] for r in bm25})

    systems = {
        "BM25": bm25,
        "E5-base": e5base,
        "E5-large": e5large,
        "Hybrid-base": hyb_base,
        "Hybrid-large": hyb_large,
        "Adaptive-RRF": adaptive,
    }

    # Precompute RR and top1 vectors
    rr = {name: rr_vec(recs, ids) for name, recs in systems.items()}
    t1 = {name: top1_vec(recs, ids) for name, recs in systems.items()}

    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out("=" * 100)
    out("STATISTICAL SIGNIFICANCE TESTS v3")
    out("=" * 100)
    out(f"Queries: {len(ids)}")

    # ============================================================
    # PAIRWISE COMPARISONS
    # ============================================================
    pairwise = [
        ("BM25 vs E5-base", "BM25", "E5-base"),
        ("BM25 vs E5-large", "BM25", "E5-large"),
        ("E5-base vs E5-large", "E5-base", "E5-large"),
        ("Hybrid-base vs Hybrid-large", "Hybrid-base", "Hybrid-large"),
        ("E5-large vs Hybrid-large", "E5-large", "Hybrid-large"),
        ("E5-large vs Adaptive-RRF", "E5-large", "Adaptive-RRF"),
        ("Hybrid-large vs Adaptive-RRF", "Hybrid-large", "Adaptive-RRF"),
        ("BM25 vs Adaptive-RRF", "BM25", "Adaptive-RRF"),
    ]

    all_results = {}

    for label, a_name, b_name in pairwise:
        out()
        out("=" * 100)
        out(label)
        out("=" * 100)

        mc = mcnemar(t1[a_name], t1[b_name])
        out(f"McNemar (top-1) : n01={mc['n01']}, n10={mc['n10']}, p={mc['p_value']:.6f} "
            f"{'*** SIGNIFICANT' if mc['p_value'] < 0.05 else '(not sig.)'}")

        wl = wilcoxon(rr[a_name], rr[b_name])
        out(f"Wilcoxon (RR)   : stat={wl['statistic']:.2f}, p={wl['p_value']:.6f} "
            f"{'*** SIGNIFICANT' if wl['p_value'] < 0.05 else '(not sig.)'}")

        a_ci = bootstrap_ci(rr[a_name])
        b_ci = bootstrap_ci(rr[b_name])
        out(f"{a_name} MRR 95% CI : [{a_ci[0]:.4f}, {a_ci[1]:.4f}]")
        out(f"{b_name} MRR 95% CI : [{b_ci[0]:.4f}, {b_ci[1]:.4f}]")

        all_results[label] = {
            "mcnemar": mc,
            "wilcoxon": wl,
            "a_mrr_ci": a_ci,
            "b_mrr_ci": b_ci,
        }

    # ============================================================
    # DEGRADATION L0 vs L4
    # ============================================================
    out()
    out("=" * 100)
    out("DEGRADATION L0 vs L4 (Wilcoxon on Reciprocal Rank)")
    out("=" * 100)

    for name, recs in systems.items():
        l0 = [r for r in recs if r["level"] == "L0"]
        l4 = [r for r in recs if r["level"] == "L4"]
        l0_rr = np.array([1/r["gold_rank"] if r["gold_rank"] else 0 for r in l0])
        l4_rr = np.array([1/r["gold_rank"] if r["gold_rank"] else 0 for r in l4])
        wl = wilcoxon(l0_rr, l4_rr)
        sig = "*** SIGNIFICANT" if wl["p_value"] < 0.05 else "(not sig.)"
        out(f"{name:<15} L0 vs L4 : stat={wl['statistic']:>7.2f}, p={wl['p_value']:.6f}  {sig}")

    # ============================================================
    # MRR BOOTSTRAP CIs (all systems)
    # ============================================================
    out()
    out("=" * 100)
    out("MRR 95% BOOTSTRAP CONFIDENCE INTERVALS")
    out("=" * 100)
    out(f"{'System':<15} {'MRR':>8} {'95% CI':>24}")
    out("-" * 100)

    for name, recs in systems.items():
        mrr = rr[name].mean()
        lo, hi = bootstrap_ci(rr[name])
        out(f"{name:<15} {mrr:>8.4f} [{lo:.4f}, {hi:.4f}]")

    # ============================================================
    # SAVE
    # ============================================================
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    out()
    out(f"Saved: {OUTPUT_FILE}")
    out(f"Saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()