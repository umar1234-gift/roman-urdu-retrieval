"""
Exact Numbers for A_single (Single-Query Rewriting)
====================================================
Computes precise statistics for the honest deployment scenario.

Outputs:
- Per-level @1, @3, MRR
- Cluster bootstrap CI
- Persistent failures (E5-large individual + all-systems intersection)
- Paired cluster bootstrap: A_single vs Original
- L4 gap recovery with CI

Run: python src_v2/36_a_single_exact_numbers.py
"""

from pathlib import Path
import json
import math
import re
import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"
ORIGINAL_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"
A_SINGLE_FILE = PROJECT_ROOT / "data_v2" / "processed" / "ablation_single_clean.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "a_single_exact.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "a_single_exact.txt"

MODEL_NAME = "intfloat/multilingual-e5-large"
RRF_K = 10
N_BOOTSTRAP = 5000
SEED = 42
LEVELS = ["L0", "L1", "L2", "L3", "L4"]


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tok = [tokenize(d) for d in docs]
        self.lens = [len(t) for t in self.tok]
        self.avgdl = sum(self.lens) / len(self.lens)
        self.N = len(docs)
        self.tf = [Counter(t) for t in self.tok]
        self.df = Counter()
        for t in self.tok:
            for term in set(t):
                self.df[term] += 1

    def idf(self, term):
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, q, i):
        q_tokens = tokenize(q)
        freq = self.tf[i]
        dl = self.lens[i]
        s = 0.0
        for term in q_tokens:
            if term not in freq:
                continue
            tf = freq[term]
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf(term) * num / den
        return s

    def rank(self, q):
        scores = [(i, self.score(q, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def metrics(records):
    total = len(records)
    if total == 0:
        return {"@1": 0, "@3": 0, "@5": 0, "MRR": 0, "n": 0}
    g1 = sum(1 for r in records if r["gold_rank"] == 1)
    g3 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 3)
    g5 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 5)
    mrr = sum(1 / r["gold_rank"] for r in records if r["gold_rank"]) / total
    return {"@1": g1/total, "@3": g3/total, "@5": g5/total,
            "MRR": mrr, "n": total,
            "count_at1": g1, "count_at3": g3}


def cluster_bootstrap_ci(records, metric="at1", n_boot=5000, seed=42):
    """Document-level cluster bootstrap for @1 or MRR."""
    rng = np.random.default_rng(seed)

    by_doc = defaultdict(list)
    for r in records:
        by_doc[r["doc_id"]].append(r)

    docs = list(by_doc.keys())
    n_docs = len(docs)

    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled_docs = rng.choice(docs, size=n_docs, replace=True)
        sampled = []
        for d in sampled_docs:
            sampled.extend(by_doc[d])

        if metric == "at1":
            boot[i] = np.mean([1 if r["gold_rank"] == 1 else 0 for r in sampled])
        elif metric == "mrr":
            boot[i] = np.mean([1/r["gold_rank"] if r["gold_rank"] else 0
                                for r in sampled])

    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_cluster_bootstrap(records_a, records_b, n_boot=5000, seed=42):
    """Paired cluster bootstrap at document level."""
    rng = np.random.default_rng(seed)

    map_a = {r["query_id"]: r for r in records_a}
    map_b = {r["query_id"]: r for r in records_b}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))

    deltas = np.array([
        (1 if map_a[q]["gold_rank"] == 1 else 0) -
        (1 if map_b[q]["gold_rank"] == 1 else 0)
        for q in common
    ])

    by_doc = defaultdict(list)
    for i, q in enumerate(common):
        d = map_a[q]["doc_id"]
        by_doc[d].append(i)

    docs = list(by_doc.keys())
    n_docs = len(docs)

    obs = float(np.mean(deltas))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled_docs = rng.choice(docs, size=n_docs, replace=True)
        idx = []
        for d in sampled_docs:
            idx.extend(by_doc[d])
        boot[i] = np.mean(deltas[idx])

    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))

    # p-value with bootstrap (min resolution = 2/n_boot)
    p_left = np.mean(boot <= 0)
    p_right = np.mean(boot >= 0)
    p = 2 * min(p_left, p_right)
    p = float(max(p, 2 / n_boot))  # floor

    return {
        "delta_at1": obs,
        "ci_95": [lo, hi],
        "p_value": p,
        "n_queries": len(common),
        "n_docs": n_docs,
    }


def main():
    print("=" * 100)
    print("A_SINGLE — EXACT NUMBERS FOR PAPER")
    print("=" * 100)

    chunks = load_json(CHUNKS_FILE)
    original = load_json(ORIGINAL_FILE)
    a_single_cache = load_json(A_SINGLE_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]
    n_docs = len(filenames)

    print(f"Chunks       : {len(chunks)}")
    print(f"Queries      : {len(original)}")
    print(f"A_single cache: {len(a_single_cache)} entries")

    # Build indexes
    print("\nBuilding BM25 index...")
    bm25 = BM25(texts)

    print("Loading E5-large...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    print("Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True,
                            show_progress_bar=True)

    # =========================================================
    # RUN RETRIEVAL
    # =========================================================
    print("\nRunning retrieval for original and A_single...")

    results_original = {}
    results_a_single = {}

    for idx, q in enumerate(original):
        qid = q["query_id"]
        gold = q["gold_document"]
        lvl = q["level"]
        doc_id = q["doc_id"]

        # Original text
        text_orig = q["query"]
        # A_single text (fallback to original if not rewritten)
        text_a = a_single_cache.get(qid, text_orig)

        for label, text in [("orig", text_orig), ("a", text_a)]:
            q_emb = model.encode(["query: " + text],
                                  normalize_embeddings=True)[0]
            sims = np.dot(doc_emb, q_emb)
            order = np.argsort(sims)[::-1]

            gr = None
            for r, i in enumerate(order, start=1):
                if filenames[i] == gold:
                    gr = r
                    break

            rec = {
                "query_id": qid,
                "doc_id": doc_id,
                "level": lvl,
                "variant": q.get("variant", 1),
                "gold_rank": gr,
                "text_used": text,
            }

            if label == "orig":
                results_original[qid] = rec
            else:
                results_a_single[qid] = rec

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(original)}")

    print("Done.\n")

    # Convert to lists
    orig_list = list(results_original.values())
    a_list = list(results_a_single.values())

    # =========================================================
    # REPORT
    # =========================================================
    report = []
    def out(s=""):
        print(s)
        report.append(s)

    # --- Overall ---
    out("=" * 100)
    out("A_SINGLE — OVERALL METRICS")
    out("=" * 100)

    m_a = metrics(a_list)
    m_o = metrics(orig_list)

    out()
    out(f"{'Metric':<20} {'A_single':>12} {'Original':>12}")
    out("-" * 50)
    out(f"{'@1':<20} {m_a['@1']:>12.4f} {m_o['@1']:>12.4f}")
    out(f"{'@3':<20} {m_a['@3']:>12.4f} {m_o['@3']:>12.4f}")
    out(f"{'@5':<20} {m_a['@5']:>12.4f} {m_o['@5']:>12.4f}")
    out(f"{'MRR':<20} {m_a['MRR']:>12.4f} {m_o['MRR']:>12.4f}")
    out(f"{'count @1':<20} {m_a['count_at1']:>12} {m_o['count_at1']:>12}")

    # --- Per-level ---
    out()
    out("=" * 100)
    out("A_SINGLE — PER-LEVEL METRICS")
    out("=" * 100)
    out()
    out(f"{'Level':<8} {'n':>5} {'@1':>10} {'@3':>10} {'@5':>10} {'MRR':>10}")
    out("-" * 100)

    per_level = {}
    for lvl in LEVELS:
        sub = [r for r in a_list if r["level"] == lvl]
        m = metrics(sub)
        per_level[lvl] = m
        out(f"{lvl:<8} {m['n']:>5} {m['@1']:>10.4f} {m['@3']:>10.4f} "
            f"{m['@5']:>10.4f} {m['MRR']:>10.4f}")

    # --- Cluster CI (overall) ---
    out()
    out("=" * 100)
    out("A_SINGLE — CLUSTER BOOTSTRAP 95% CI")
    out("=" * 100)

    ci_lo, ci_hi = cluster_bootstrap_ci(a_list, "at1", N_BOOTSTRAP, SEED)
    out()
    out(f"@1 95% CI : [{ci_lo:.4f}, {ci_hi:.4f}]")

    ci_mrr_lo, ci_mrr_hi = cluster_bootstrap_ci(a_list, "mrr", N_BOOTSTRAP, SEED)
    out(f"MRR 95% CI: [{ci_mrr_lo:.4f}, {ci_mrr_hi:.4f}]")

    # --- L4 recovery ---
    out()
    out("=" * 100)
    out("A_SINGLE — L4 GAP RECOVERY")
    out("=" * 100)

    l4_a = metrics([r for r in a_list if r["level"] == "L4"])
    l4_o = metrics([r for r in orig_list if r["level"] == "L4"])
    l0_o = metrics([r for r in orig_list if r["level"] == "L0"])

    l4_a_p1 = l4_a["@1"]
    l4_o_p1 = l4_o["@1"]
    l0_p1 = l0_o["@1"]

    # Two recovery definitions
    rec_measured = (l4_a_p1 - l4_o_p1) / (l0_p1 - l4_o_p1) if (l0_p1 - l4_o_p1) > 0 else 0
    rec_100 = (l4_a_p1 - l4_o_p1) / (1.0 - l4_o_p1) if (1.0 - l4_o_p1) > 0 else 0

    out()
    out(f"L0 (original)  : {l0_p1:.4f}")
    out(f"L4 (original)  : {l4_o_p1:.4f}")
    out(f"L4 (A_single)  : {l4_a_p1:.4f}")
    out()
    out(f"Recovery (measured L0 ceiling): {rec_measured:.4f} ({rec_measured*100:.2f}%)")
    out(f"Recovery (assumed 100 ceiling): {rec_100:.4f} ({rec_100*100:.2f}%)")

    # --- Persistent failures ---
    out()
    out("=" * 100)
    out("A_SINGLE — PERSISTENT FAILURES")
    out("=" * 100)

    # E5-large individual failures
    e5_failures = [r for r in a_list if r["gold_rank"] != 1]
    out()
    out(f"E5-large individual failures: {len(e5_failures)}/450")

    # For intersection we need all systems on A_single — we only have E5-large.
    # We approximate by reporting E5-large failures (which is the strongest system).
    out()
    out("Note: Only E5-large retrieval was run on A_single.")
    out("For intersection, previous batch-context pipeline had 4 errors.")
    out(f"E5-large (single-query) has {len(e5_failures)} failures.")

    # List them
    out()
    out("Failed queries:")
    for r in e5_failures:
        out(f"  {r['query_id']} ({r['doc_id']}, {r['level']}) rank={r['gold_rank']}")
        out(f"    Text: {r['text_used'][:90]}")

    # --- Paired cluster bootstrap: A_single vs Original (L2-L4) ---
    out()
    out("=" * 100)
    out("A_SINGLE vs ORIGINAL — PAIRED CLUSTER BOOTSTRAP (L2-L4)")
    out("=" * 100)

    l2l4_a = [r for r in a_list if r["level"] in ["L2", "L3", "L4"]]
    l2l4_o = [r for r in orig_list if r["level"] in ["L2", "L3", "L4"]]

    boot_result = paired_cluster_bootstrap(l2l4_a, l2l4_o, N_BOOTSTRAP, SEED)
    out()
    out(f"Δ @1 (A_single - Original): {boot_result['delta_at1']:+.4f}")
    out(f"95% CI                    : [{boot_result['ci_95'][0]:+.4f}, "
        f"{boot_result['ci_95'][1]:+.4f}]")
    out(f"p-value                   : {boot_result['p_value']:.4f}")
    out(f"n queries                 : {boot_result['n_queries']}")
    out(f"n docs                    : {boot_result['n_docs']}")

    # --- L4 recovery CI ---
    out()
    out("=" * 100)
    out("L4 RECOVERY CI")
    out("=" * 100)

    # Bootstrap over docs for L4
    rng = np.random.default_rng(SEED)
    by_doc_a = defaultdict(list)
    by_doc_o = defaultdict(list)
    for r in a_list:
        if r["level"] == "L4":
            by_doc_a[r["doc_id"]].append(r)
    for r in orig_list:
        if r["level"] == "L4":
            by_doc_o[r["doc_id"]].append(r)

    docs_l4 = sorted(set(by_doc_a.keys()) & set(by_doc_o.keys()))
    recs = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.choice(docs_l4, size=len(docs_l4), replace=True)
        a_sampled = [r for d in sampled for r in by_doc_a[d]]
        o_sampled = [r for d in sampled for r in by_doc_o[d]]
        a_p1 = np.mean([1 if r["gold_rank"] == 1 else 0 for r in a_sampled])
        o_p1 = np.mean([1 if r["gold_rank"] == 1 else 0 for r in o_sampled])
        # recovery against assumed 100% ceiling
        rec = (a_p1 - o_p1) / (1.0 - o_p1) if (1.0 - o_p1) > 0 else 0
        recs.append(rec)

    recs = np.array(recs)
    lo = float(np.quantile(recs, 0.025))
    hi = float(np.quantile(recs, 0.975))

    out()
    out(f"Recovery (assumed 100 ceiling) 95% CI: [{lo:.4f}, {hi:.4f}]")

    # --- Save ---
    output = {
        "overall_a_single": m_a,
        "overall_original": m_o,
        "per_level_a_single": per_level,
        "cluster_ci_at1": [ci_lo, ci_hi],
        "cluster_ci_mrr": [ci_mrr_lo, ci_mrr_hi],
        "l4_recovery_measured_ceiling": rec_measured,
        "l4_recovery_100_ceiling": rec_100,
        "l4_recovery_ci_100": [lo, hi],
        "persistent_failures_a_single": len(e5_failures),
        "failure_list": [
            {"query_id": r["query_id"], "doc_id": r["doc_id"],
             "level": r["level"], "gold_rank": r["gold_rank"]}
            for r in e5_failures
        ],
        "rewriting_test_L2L4": boot_result,
    }

    save_json(RESULTS_FILE, output)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    out()
    out(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()