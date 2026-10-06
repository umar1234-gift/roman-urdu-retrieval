"""
Missing Cluster Tests (FIXED)
==============================
Uses:
- dense_model_comparison.json for E5-large/small on ORIGINAL queries
- retrieval_all_rewritten.json for E5-large on REWRITTEN queries
- bm25_results.json, dense_results.json, hybrid_results.json for others

Tests:
1. L0 vs L4 degradation (cluster-aware) — for each system
2. Before vs After rewriting (cluster-aware) — E5-large
3. E5-base vs E5-large (cluster-aware) — model size
4. Hybrid-base vs E5-base (cluster-aware) — fusion when balanced

Run: python src_v2/35_missing_cluster_tests.py
"""

from pathlib import Path
import json
import numpy as np
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files
BM25_FILE = PROJECT_ROOT / "data_v2" / "results" / "bm25_results.json"
E5BASE_FILE = PROJECT_ROOT / "data_v2" / "results" / "dense_results.json"
DENSE_COMPARE_FILE = PROJECT_ROOT / "data_v2" / "results" / "dense_model_comparison.json"
HYBRID_BASE_FILE = PROJECT_ROOT / "data_v2" / "results" / "hybrid_results.json"
RERANK_FILE = PROJECT_ROOT / "data_v2" / "results" / "retrieval_all_rewritten.json"
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "missing_cluster_tests.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "missing_cluster_tests.txt"

N_BOOTSTRAP = 5000
SEED = 42


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_gold_ranks(records):
    """Convert list of records to {query_id: gold_rank}"""
    out = {}
    for r in records:
        if isinstance(r, dict) and "query_id" in r:
            out[r["query_id"]] = r.get("gold_rank")
    return out


def paired_cluster_bootstrap(records_a, records_b, doc_ids,
                              n_boot=5000, seed=42):
    """
    Paired cluster bootstrap at document level.
    records_a, records_b: {query_id: gold_rank}
    Returns: dict with delta, CI, p-value
    """
    rng = np.random.default_rng(seed)

    # Only common query_ids
    common_qids = sorted(set(records_a.keys()) & set(records_b.keys()))
    if not common_qids:
        return None

    # Per-query top-1 delta (A - B)
    deltas = np.array([
        (1 if records_a[q] == 1 else 0) - (1 if records_b[q] == 1 else 0)
        for q in common_qids
    ])

    # Group by doc
    by_doc = defaultdict(list)
    idx_map = {q: i for i, q in enumerate(common_qids)}
    for q in common_qids:
        d = doc_ids.get(q)
        if d is not None:
            by_doc[d].append(idx_map[q])

    docs = list(by_doc.keys())
    n_docs = len(docs)

    if n_docs < 2:
        return None

    obs_delta = float(np.mean(deltas))

    # Bootstrap
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled_docs = rng.choice(docs, size=n_docs, replace=True)
        sampled_idx = []
        for d in sampled_docs:
            sampled_idx.extend(by_doc[d])
        boot[i] = np.mean(deltas[sampled_idx])

    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))

    # Two-sided p-value
    p = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    p = float(min(p, 1.0))

    return {
        "delta_at1": obs_delta,
        "ci_95": [lo, hi],
        "p_value": p,
        "n_queries": len(common_qids),
        "n_docs": n_docs,
    }


def degradation_test(ranks, levels, doc_ids, system_name,
                      n_boot=5000, seed=42):
    """
    L0 vs L4 degradation test at document level.
    For each doc, compute mean@1 on L0 queries and mean@1 on L4 queries.
    Paired delta = L0_mean - L4_mean.
    """
    rng = np.random.default_rng(seed)

    # Group by doc
    doc_l0 = defaultdict(list)
    doc_l4 = defaultdict(list)

    for q, r in ranks.items():
        if r is None:
            continue
        lvl = levels.get(q)
        d = doc_ids.get(q)
        if d is None:
            continue
        top1 = 1 if r == 1 else 0
        if lvl == "L0":
            doc_l0[d].append(top1)
        elif lvl == "L4":
            doc_l4[d].append(top1)

    common_docs = sorted(set(doc_l0.keys()) & set(doc_l4.keys()))
    if len(common_docs) < 2:
        return None

    # Per-doc delta
    deltas = np.array([
        np.mean(doc_l0[d]) - np.mean(doc_l4[d])
        for d in common_docs
    ])

    obs = float(np.mean(deltas))

    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(common_docs, size=len(common_docs), replace=True)
        boot[i] = np.mean([
            np.mean(doc_l0[d]) - np.mean(doc_l4[d])
            for d in sampled
        ])

    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))
    p = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    p = float(min(p, 1.0))

    mean_l0 = float(np.mean([np.mean(v) for v in doc_l0.values()]))
    mean_l4 = float(np.mean([np.mean(v) for v in doc_l4.values()]))

    return {
        "system": system_name,
        "L0_mean_at1": mean_l0,
        "L4_mean_at1": mean_l4,
        "delta": obs,
        "ci_95": [lo, hi],
        "p_value": p,
        "n_docs": len(common_docs),
    }


def main():
    print("=" * 100)
    print("MISSING CLUSTER TESTS (FIXED)")
    print("=" * 100)

    # Load queries for level & doc mapping
    queries = load_json(QUERIES_FILE)
    doc_ids = {q["query_id"]: q["doc_id"] for q in queries}
    levels = {q["query_id"]: q["level"] for q in queries}

    # Load all results
    bm25 = load_json(BM25_FILE)
    e5base = load_json(E5BASE_FILE)
    hybrid_base = load_json(HYBRID_BASE_FILE)
    dense_compare = load_json(DENSE_COMPARE_FILE)
    rerank = load_json(RERANK_FILE)

    # Extract gold ranks
    bm25_ranks = extract_gold_ranks(bm25)
    e5base_ranks = extract_gold_ranks(e5base)
    hybrid_base_ranks = extract_gold_ranks(hybrid_base)

    # E5-large on ORIGINAL queries
    e5large_orig_ranks = extract_gold_ranks(dense_compare.get("e5-large", []))

    # E5-large on REWRITTEN queries
    e5large_rewritten_ranks = {}
    if "per_query" in rerank and "E5-large" in rerank["per_query"]:
        e5large_rewritten_ranks = extract_gold_ranks(rerank["per_query"]["E5-large"])

    report = []
    def out(s=""):
        print(s)
        report.append(s)

    results = {}

    # =========================================================
    # TEST 1: L0 vs L4 Degradation
    # =========================================================
    out()
    out("=" * 100)
    out("TEST 1: L0 vs L4 DEGRADATION (cluster-aware, per document)")
    out("=" * 100)

    systems_for_test1 = {
        "BM25": bm25_ranks,
        "E5-base": e5base_ranks,
        "E5-large": e5large_orig_ranks,
        "Hybrid-base": hybrid_base_ranks,
    }

    for sys_name, ranks in systems_for_test1.items():
        if not ranks:
            continue
        r = degradation_test(ranks, levels, doc_ids, sys_name, N_BOOTSTRAP, SEED)
        if r:
            out()
            out(f"  {sys_name}:")
            out(f"    L0 mean @1 : {r['L0_mean_at1']:.4f}")
            out(f"    L4 mean @1 : {r['L4_mean_at1']:.4f}")
            out(f"    Δ (L0-L4)  : {r['delta']:+.4f}")
            out(f"    95% CI     : [{r['ci_95'][0]:+.4f}, {r['ci_95'][1]:+.4f}]")
            out(f"    p-value    : {r['p_value']:.4f} "
                f"{'* SIGNIFICANT' if r['p_value'] < 0.05 else '(not sig.)'}")
            results[f"L0_vs_L4_{sys_name}"] = r

    # =========================================================
    # TEST 2: Before vs After Rewriting (E5-large)
    # =========================================================
    out()
    out("=" * 100)
    out("TEST 2: BEFORE vs AFTER REWRITING (E5-large, cluster-aware)")
    out("=" * 100)
    out()
    out("Compare E5-large on original vs rewritten queries.")
    out("Restrict to L2-L4 (L0/L1 unchanged by rewriting).")

    if e5large_orig_ranks and e5large_rewritten_ranks:
        # L2-L4 only
        l2l4 = [q for q in e5large_rewritten_ranks if levels.get(q) in ["L2", "L3", "L4"]]
        orig_sub = {q: e5large_orig_ranks[q] for q in l2l4 if q in e5large_orig_ranks}
        rew_sub = {q: e5large_rewritten_ranks[q] for q in l2l4 if q in e5large_rewritten_ranks}

        r = paired_cluster_bootstrap(rew_sub, orig_sub, doc_ids, N_BOOTSTRAP, SEED)
        if r:
            out()
            out(f"  Rewritten vs Original (L2-L4 only):")
            out(f"    Δ @1    : {r['delta_at1']:+.4f}")
            out(f"    95% CI  : [{r['ci_95'][0]:+.4f}, {r['ci_95'][1]:+.4f}]")
            out(f"    p-value : {r['p_value']:.4f} "
                f"{'* SIGNIFICANT' if r['p_value'] < 0.05 else '(not sig.)'}")
            results["rewritten_vs_original_L2L4"] = r
    else:
        out("  Missing data — cannot run this test.")

    # =========================================================
    # TEST 3: E5-base vs E5-large (Model Size)
    # =========================================================
    out()
    out("=" * 100)
    out("TEST 3: E5-base vs E5-large on ORIGINAL queries (cluster-aware)")
    out("=" * 100)

    if e5large_orig_ranks and e5base_ranks:
        r = paired_cluster_bootstrap(e5large_orig_ranks, e5base_ranks,
                                      doc_ids, N_BOOTSTRAP, SEED)
        if r:
            out()
            out(f"  E5-large vs E5-base:")
            out(f"    Δ @1    : {r['delta_at1']:+.4f}")
            out(f"    95% CI  : [{r['ci_95'][0]:+.4f}, {r['ci_95'][1]:+.4f}]")
            out(f"    p-value : {r['p_value']:.4f} "
                f"{'* SIGNIFICANT' if r['p_value'] < 0.05 else '(not sig.)'}")
            results["E5-large_vs_E5-base"] = r

    # =========================================================
    # TEST 4: Hybrid-base vs E5-base (Fusion when balanced)
    # =========================================================
    out()
    out("=" * 100)
    out("TEST 4: Hybrid-base vs E5-base on ORIGINAL queries")
    out("=" * 100)

    r = paired_cluster_bootstrap(hybrid_base_ranks, e5base_ranks,
                                  doc_ids, N_BOOTSTRAP, SEED)
    if r:
        out()
        out(f"  Hybrid-base vs E5-base:")
        out(f"    Δ @1    : {r['delta_at1']:+.4f}")
        out(f"    95% CI  : [{r['ci_95'][0]:+.4f}, {r['ci_95'][1]:+.4f}]")
        out(f"    p-value : {r['p_value']:.4f} "
            f"{'* SIGNIFICANT' if r['p_value'] < 0.05 else '(not sig.)'}")
        results["Hybrid-base_vs_E5-base"] = r

        if r['p_value'] >= 0.05:
            out()
            out("  -> NOT significant. Cannot claim 'fusion helps when balanced'.")

    # =========================================================
    # SUMMARY
    # =========================================================
    out()
    out("=" * 100)
    out("SUMMARY")
    out("=" * 100)
    out()
    out(f"{'Test':<45} {'Δ':>10} {'p-value':>12}")
    out("-" * 100)

    for k, v in results.items():
        if "delta" in v:
            d = v["delta"]
        elif "delta_at1" in v:
            d = v["delta_at1"]
        else:
            continue
        p = v.get("p_value", 0)
        sig = "*" if p < 0.05 else ""
        out(f"{k:<45} {d:>+10.4f} {p:>12.4f} {sig}")

    save_json(RESULTS_FILE, results)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    out()
    out(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()