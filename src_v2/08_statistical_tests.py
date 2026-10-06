from pathlib import Path
import json
import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BM25_FILE = PROJECT_ROOT / "data_v2" / "results" / "bm25_results.json"
DENSE_FILE = PROJECT_ROOT / "data_v2" / "results" / "dense_results.json"
HYBRID_FILE = PROJECT_ROOT / "data_v2" / "results" / "hybrid_results.json"

OUTPUT_FILE = PROJECT_ROOT / "data_v2" / "results" / "statistical_tests.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "statistical_tests.txt"


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
    bm25 = load_json(BM25_FILE)
    dense = load_json(DENSE_FILE)
    hybrid = load_json(HYBRID_FILE)

    ids = sorted({r["query_id"] for r in bm25})

    bm25_rr = rr_vec(bm25, ids)
    e5_rr = rr_vec(dense, ids)
    hyb_rr = rr_vec(hybrid, ids)

    bm25_t1 = top1_vec(bm25, ids)
    e5_t1 = top1_vec(dense, ids)
    hyb_t1 = top1_vec(hybrid, ids)

    report = []
    def out(s):
        print(s)
        report.append(s)

    out("=" * 100)
    out("STATISTICAL SIGNIFICANCE TESTS")
    out("=" * 100)

    pairs = [
        ("BM25 vs E5", bm25_rr, e5_rr, bm25_t1, e5_t1),
        ("Hybrid vs BM25", hyb_rr, bm25_rr, hyb_t1, bm25_t1),
        ("Hybrid vs E5", hyb_rr, e5_rr, hyb_t1, e5_t1),
    ]

    results = {}

    for name, a_rr, b_rr, a_t1, b_t1 in pairs:
        out("")
        out("=" * 100)
        out(name)
        out("=" * 100)

        mc = mcnemar(a_t1, b_t1)
        out(f"McNemar (top-1) : n01={mc['n01']}, n10={mc['n10']}, p={mc['p_value']:.6f}")

        wl = wilcoxon(a_rr, b_rr)
        out(f"Wilcoxon (RR)   : stat={wl['statistic']:.2f}, p={wl['p_value']:.6f}")

        a_ci = bootstrap_ci(a_rr)
        b_ci = bootstrap_ci(b_rr)
        out(f"A MRR 95% CI    : [{a_ci[0]:.4f}, {a_ci[1]:.4f}]")
        out(f"B MRR 95% CI    : [{b_ci[0]:.4f}, {b_ci[1]:.4f}]")

        results[name] = {
            "mcnemar": mc,
            "wilcoxon": wl,
            "A_mrr_ci": a_ci,
            "B_mrr_ci": b_ci
        }

    # Level-wise
    out("")
    out("=" * 100)
    out("LEVEL-WISE DEGRADATION (L0 vs L4, RR)")
    out("=" * 100)

    for lvl_sys, records in [("BM25", bm25), ("E5", dense), ("Hybrid", hybrid)]:
        l0 = [r for r in records if r["level"] == "L0"]
        l4 = [r for r in records if r["level"] == "L4"]
        l0_rr = np.array([1/r["gold_rank"] if r["gold_rank"] else 0 for r in l0])
        l4_rr = np.array([1/r["gold_rank"] if r["gold_rank"] else 0 for r in l4])
        wl = wilcoxon(l0_rr, l4_rr)
        out(f"{lvl_sys:8s} L0 vs L4 Wilcoxon: stat={wl['statistic']:.2f}, p={wl['p_value']:.6f}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    out("")
    out(f"Saved: {OUTPUT_FILE}")
    out(f"Saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()