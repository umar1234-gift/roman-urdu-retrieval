from pathlib import Path
import json
import numpy as np
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ADAPTIVE_FILE = PROJECT_ROOT / "data_v2" / "results" / "adaptive_rrf_results.json"
REWRITTEN_FILE = PROJECT_ROOT / "data_v2" / "results" / "rewritten_retrieval.json"

OUTPUT = PROJECT_ROOT / "data_v2" / "results" / "rewriting_significance.json"


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def rr(records, ids):
    by_id = {r["query_id"]: r for r in records}
    return np.array([1.0 / by_id[q]["gold_rank"] if by_id[q]["gold_rank"] else 0.0 for q in ids])


def top1(records, ids):
    by_id = {r["query_id"]: r for r in records}
    return np.array([1 if by_id[q]["gold_rank"] == 1 else 0 for q in ids])


def mcnemar(a, b):
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    n = n01 + n10
    k = min(n01, n10)
    p = 2 * stats.binom.cdf(k, n, 0.5)
    return {"n01": n01, "n10": n10, "p": float(min(p, 1.0))}


def wilcoxon(a, b):
    if np.all(a - b == 0):
        return {"stat": 0.0, "p": 1.0}
    s, p = stats.wilcoxon(a, b)
    return {"stat": float(s), "p": float(p)}


def main():
    original = load_json(ADAPTIVE_FILE)["per_query"]
    rewritten = load_json(REWRITTEN_FILE)

    ids = sorted({r["query_id"] for r in original})

    o_rr = rr(original, ids)
    r_rr = rr(rewritten, ids)
    o_t1 = top1(original, ids)
    r_t1 = top1(rewritten, ids)

    print("=" * 100)
    print("SIGNIFICANCE: ORIGINAL vs REWRITTEN (Adaptive RRF)")
    print("=" * 100)

    mc = mcnemar(o_t1, r_t1)
    print(f"\nMcNemar (top-1): n01={mc['n01']}, n10={mc['n10']}, p={mc['p']:.6f}")

    wl = wilcoxon(o_rr, r_rr)
    print(f"Wilcoxon (RR)  : stat={wl['stat']:.2f}, p={wl['p']:.6f}")

    # Per level
    for lvl in ["L0", "L1", "L2", "L3", "L4"]:
        lvl_ids = [r["query_id"] for r in original if r["level"] == lvl]
        o = top1(original, lvl_ids)
        r = top1(rewritten, lvl_ids)
        m = mcnemar(o, r)
        print(f"  {lvl}: n01={m['n01']}, n10={m['n10']}, p={m['p']:.6f}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump({"mcnemar_overall": mc, "wilcoxon_overall": wl}, f, indent=2)


if __name__ == "__main__":
    main()