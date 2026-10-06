"""
RRF with Proper Zero-Score Handling
=====================================
When BM25 scores are all zero, RRF injects arbitrary ranks.
Fix: skip BM25 contribution for those queries (or use E5-only rank).

Run: python src_v2/44_rrf_tie_fixed.py
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

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "rrf_tie_fixed.json"

MODEL_NAME = "intfloat/multilingual-e5-large"
RRF_K = 10
LEVELS = ["L0", "L1", "L2", "L3", "L4"]


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


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

    def scores_all(self, q):
        q_tokens = tokenize(q)
        out = []
        for i in range(self.N):
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
            out.append(s)
        return np.array(out)

    def ranks_with_zero_handling(self, q):
        """Return ranks. If all scores zero, return None (no ranking)."""
        scores = self.scores_all(q)
        if scores.max() < 1e-9:
            return None  # cannot rank

        # Use stable argsort (ties broken by doc index ascending)
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(self.N, dtype=int)
        for r, i in enumerate(order, start=1):
            ranks[i] = r
        return ranks


def metrics(records):
    total = len(records)
    if total == 0:
        return {"@1": 0, "@3": 0, "MRR": 0, "n": 0}
    g1 = sum(1 for r in records if r["gold_rank"] == 1)
    g3 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 3)
    mrr = sum(1 / r["gold_rank"] for r in records if r["gold_rank"]) / total
    return {"@1": g1/total, "@3": g3/total, "MRR": mrr, "n": total}


def main():
    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)

    filenames = [c["filename"] for c in chunks]
    texts = [c["text"] for c in chunks]
    n_docs = len(filenames)

    print("=" * 100)
    print("RRF WITH PROPER ZERO-SCORE HANDLING")
    print("=" * 100)

    print("Building BM25...")
    bm25 = BM25(texts)

    print("Loading E5-large...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    doc_emb = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=True)

    print("\nRunning 3 RRF variants:")
    print("  A: RRF with buggy rank (BM25 arbitrary tie)")
    print("  B: RRF skipping BM25 when all-zero (E5-only for those queries)")
    print("  C: RRF with random tie-breaking (fixed seed)")
    print()

    results = {"A": [], "B": [], "C": []}
    zero_count = 0

    rng = np.random.default_rng(42)

    for idx, q in enumerate(queries):
        qid = q["query_id"]
        lvl = q["level"]
        gold = q["gold_document"]
        text = q["query"]

        bm25_scores = bm25.scores_all(text)
        e5_scores = model.encode(["query: " + text], normalize_embeddings=True)[0]
        e5_scores = np.dot(doc_emb, e5_scores)

        all_zero = bm25_scores.max() < 1e-9
        if all_zero:
            zero_count += 1

        # E5 ranks (always well-defined)
        e5_order = np.argsort(-e5_scores, kind="stable")
        e5_ranks = np.empty(n_docs, dtype=int)
        for r, i in enumerate(e5_order, start=1):
            e5_ranks[i] = r

        # Variant A: buggy BM25 rank (index order)
        bm25_order_a = np.argsort(-bm25_scores, kind="stable")
        bm25_ranks_a = np.empty(n_docs, dtype=int)
        for r, i in enumerate(bm25_order_a, start=1):
            bm25_ranks_a[i] = r

        # Variant B: skip BM25 if all zero
        if all_zero:
            rrf_scores_b = 1.0 / (RRF_K + e5_ranks)
        else:
            bm25_order_b = np.argsort(-bm25_scores, kind="stable")
            bm25_ranks_b = np.empty(n_docs, dtype=int)
            for r, i in enumerate(bm25_order_b, start=1):
                bm25_ranks_b[i] = r
            rrf_scores_b = 1.0 / (RRF_K + bm25_ranks_b) + 1.0 / (RRF_K + e5_ranks)

        # Variant C: random tie-breaking
        if all_zero:
            shuffled = rng.permutation(n_docs)
            bm25_ranks_c = np.empty(n_docs, dtype=int)
            for r, i in enumerate(shuffled, start=1):
                bm25_ranks_c[i] = r
        else:
            bm25_order_c = np.argsort(-bm25_scores, kind="stable")
            bm25_ranks_c = np.empty(n_docs, dtype=int)
            for r, i in enumerate(bm25_order_c, start=1):
                bm25_ranks_c[i] = r
        rrf_scores_c = 1.0 / (RRF_K + bm25_ranks_c) + 1.0 / (RRF_K + e5_ranks)

        # Compute gold ranks
        for label, scores in [("A", 1/(RRF_K + bm25_ranks_a) + 1/(RRF_K + e5_ranks)),
                                ("B", rrf_scores_b),
                                ("C", rrf_scores_c)]:
            order = np.argsort(-scores, kind="stable")
            gr = None
            for r, i in enumerate(order, start=1):
                if filenames[i] == gold:
                    gr = r
                    break
            results[label].append({"query_id": qid, "level": lvl, "gold_rank": gr})

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(queries)}")

    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(f"\nAll-zero BM25 queries: {zero_count}/450")
    print()
    print(f"{'Variant':<40} {'@1':>10} {'@3':>10} {'MRR':>10}")
    print("-" * 80)

    for label, name in [("A", "A: Buggy rank (script 40)"),
                         ("B", "B: Skip BM25 when zero"),
                         ("C", "C: Random tie-breaking")]:
        m = metrics(results[label])
        print(f"{name:<40} {m['@1']:>10.4f} {m['@3']:>10.4f} {m['MRR']:>10.4f}")

    print()
    print("Per-level @1:")
    print(f"{'Variant':<40} " + " ".join(f"{lvl:>8}" for lvl in LEVELS))
    print("-" * 100)
    for label, name in [("A", "A: Buggy rank (script 40)"),
                         ("B", "B: Skip BM25 when zero"),
                         ("C", "C: Random tie-breaking")]:
        row = []
        for lvl in LEVELS:
            subset = [r for r in results[label] if r["level"] == lvl]
            row.append(metrics(subset)["@1"])
        print(f"{name:<40} " + " ".join(f"{v:>8.4f}" for v in row))

    # Save
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "zero_count": zero_count,
            "variant_A_buggy": metrics(results["A"]),
            "variant_B_skip": metrics(results["B"]),
            "variant_C_random": metrics(results["C"]),
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()