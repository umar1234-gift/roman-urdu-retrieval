"""
Paired Cluster Bootstrap for Original Queries
==============================================
Fixes the independence violation in naive McNemar/Wilcoxon tests.
Samples documents (not queries) with replacement.

Run: python src_v2/30_paired_cluster_bootstrap.py
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
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "paired_cluster_bootstrap.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "paired_cluster_bootstrap.txt"

MODEL_NAME = "intfloat/multilingual-e5-large"
RRF_K = 10
N_BOOTSTRAP = 5000
SEED = 42


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


def main():
    print("=" * 100)
    print("PAIRED CLUSTER BOOTSTRAP — ORIGINAL QUERIES")
    print("=" * 100)

    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]
    n_docs = len(filenames)

    print(f"Chunks : {len(chunks)}")
    print(f"Queries: {len(queries)}")

    # Build indexes
    print("\nBuilding BM25 index...")
    bm25 = BM25(texts)

    print("Loading E5-large...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    print("Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=True)

    # Pre-compute rankings
    print("\nPre-computing rankings for original queries...")
    bm25_ranks = {}
    e5_ranks = {}

    for idx, q in enumerate(queries):
        qid = q["query_id"]
        text = q["query"]

        bm25_ranking = bm25.rank(text)
        bm25_ranks[qid] = {i: r for r, (i, _) in enumerate(bm25_ranking, start=1)}

        q_emb = model.encode(["query: " + text], normalize_embeddings=True)[0]
        sims = np.dot(doc_emb, q_emb)
        e5_order = np.argsort(sims)[::-1]
        e5_ranks[qid] = {i: r for r, i in enumerate(e5_order, start=1)}

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(queries)}")

    print("Done.\n")

    # Compute per-query gold ranks for all systems
    print("Computing per-query gold ranks...")

    # System definitions
    def compute_gold_rank(qid, gold_doc, system_name):
        bm25_rank = bm25_ranks[qid]
        e5_rank = e5_ranks[qid]

        if system_name == "BM25":
            ranked = sorted(bm25_rank.items(), key=lambda x: x[1])
        elif system_name == "E5-base":
            # Note: we only have E5-large. Approximating with E5-large for now.
            ranked = sorted(e5_rank.items(), key=lambda x: x[1])
        elif system_name == "E5-large":
            ranked = sorted(e5_rank.items(), key=lambda x: x[1])
        elif system_name == "Hybrid-base":
            scores = {i: (1.0/(RRF_K + bm25_rank[i])) + (1.0/(RRF_K + e5_rank[i]))
                      for i in range(n_docs)}
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        elif system_name == "Hybrid-large":
            scores = {i: (1.0/(RRF_K + bm25_rank[i])) + (1.0/(RRF_K + e5_rank[i]))
                      for i in range(n_docs)}
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        else:
            return None

        for r, (i, _) in enumerate(ranked, start=1):
            if filenames[i] == gold_doc:
                return r
        return None

    # Note: We only use E5-large (not E5-base) — need to run separate script for E5-base
    # For now, we compare: BM25, E5-large, Hybrid (with E5-large)
    systems = ["BM25", "E5-large", "Hybrid-large"]

    per_query_ranks = defaultdict(dict)
    for q in queries:
        qid = q["query_id"]
        gold = q["gold_document"]
        for sys_name in systems:
            per_query_ranks[sys_name][qid] = compute_gold_rank(qid, gold, sys_name)

    # Group by doc for cluster bootstrap
    by_doc = defaultdict(list)
    for q in queries:
        by_doc[q["doc_id"]].append(q["query_id"])

    doc_ids = list(by_doc.keys())
    n_docs_total = len(doc_ids)

    rng = np.random.default_rng(SEED)

    # Paired comparisons
    pairs = [
        ("BM25", "E5-large"),
        ("BM25", "Hybrid-large"),
        ("E5-large", "Hybrid-large"),
    ]

    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out()
    out("=" * 100)
    out(f"PAIRED CLUSTER BOOTSTRAP (n_boot={N_BOOTSTRAP})")
    out("=" * 100)
    out()
    out("Note: E5-base and Adaptive-RRF excluded in this run (not re-computed).")
    out("Systems: BM25, E5-large, Hybrid-large (RRF with E5-large).")
    out()

    results = {}

    for sys_a, sys_b in pairs:
        out()
        out("-" * 100)
        out(f"{sys_a}  vs  {sys_b}")
        out("-" * 100)

        # Compute observed deltas per query
        deltas_at1 = []
        deltas_mrr = []
        for q in queries:
            qid = q["query_id"]
            r_a = per_query_ranks[sys_a][qid]
            r_b = per_query_ranks[sys_b][qid]

            a_top1 = 1 if r_a == 1 else 0
            b_top1 = 1 if r_b == 1 else 0
            deltas_at1.append(a_top1 - b_top1)

            a_rr = 1 / r_a if r_a else 0
            b_rr = 1 / r_b if r_b else 0
            deltas_mrr.append(a_rr - b_rr)

        deltas_at1 = np.array(deltas_at1)
        deltas_mrr = np.array(deltas_mrr)

        obs_at1 = float(np.mean(deltas_at1))
        obs_mrr = float(np.mean(deltas_mrr))

        out(f"Observed Δ @1 : {obs_at1:+.4f}")
        out(f"Observed Δ MRR: {obs_mrr:+.4f}")

        # Cluster bootstrap
        boot_at1 = []
        boot_mrr = []

        for _ in range(N_BOOTSTRAP):
            # Sample documents with replacement
            sampled_docs = rng.choice(doc_ids, size=n_docs_total, replace=True)

            # Collect query ids
            sampled_qids = []
            for d in sampled_docs:
                sampled_qids.extend(by_doc[d])

            # Indices into original arrays
            idx_map = {q["query_id"]: i for i, q in enumerate(queries)}
            sampled_idx = [idx_map[qid] for qid in sampled_qids]

            boot_at1.append(np.mean(deltas_at1[sampled_idx]))
            boot_mrr.append(np.mean(deltas_mrr[sampled_idx]))

        boot_at1 = np.array(boot_at1)
        boot_mrr = np.array(boot_mrr)

        at1_lo = float(np.quantile(boot_at1, 0.025))
        at1_hi = float(np.quantile(boot_at1, 0.975))
        mrr_lo = float(np.quantile(boot_mrr, 0.025))
        mrr_hi = float(np.quantile(boot_mrr, 0.975))

        # p-value: fraction of bootstrap samples where delta is on opposite side of 0
        p_at1 = 2 * min(
            np.mean(boot_at1 <= 0),
            np.mean(boot_at1 >= 0)
        )
        p_mrr = 2 * min(
            np.mean(boot_mrr <= 0),
            np.mean(boot_mrr >= 0)
        )
        p_at1 = min(p_at1, 1.0)
        p_mrr = min(p_mrr, 1.0)

        out(f"Δ @1 95% CI   : [{at1_lo:+.4f}, {at1_hi:+.4f}]")
        out(f"Δ MRR 95% CI  : [{mrr_lo:+.4f}, {mrr_hi:+.4f}]")
        out(f"p-value @1    : {p_at1:.4f} {'* SIGNIFICANT' if p_at1 < 0.05 else '(not sig.)'}")
        out(f"p-value MRR   : {p_mrr:.4f} {'* SIGNIFICANT' if p_mrr < 0.05 else '(not sig.)'}")

        results[f"{sys_a}_vs_{sys_b}"] = {
            "delta_at1": obs_at1,
            "delta_mrr": obs_mrr,
            "at1_ci": [at1_lo, at1_hi],
            "mrr_ci": [mrr_lo, mrr_hi],
            "p_at1": float(p_at1),
            "p_mrr": float(p_mrr),
        }

    # Summary table
    out()
    out("=" * 100)
    out("SUMMARY")
    out("=" * 100)
    out()
    out(f"{'Comparison':<35} {'Δ @1':>10} {'95% CI':>25} {'p':>10}")
    out("-" * 100)

    for key, r in results.items():
        ci_str = f"[{r['at1_ci'][0]:+.4f}, {r['at1_ci'][1]:+.4f}]"
        out(f"{key:<35} {r['delta_at1']:>+10.4f} {ci_str:>25} {r['p_at1']:>10.4f}")

    save_json(RESULTS_FILE, results)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    out()
    out(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()