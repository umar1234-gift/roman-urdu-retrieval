"""
Convex Combination of BM25 and E5 Scores
==========================================
Tests whether score-based fusion outperforms rank-based RRF.

Formula: score(d) = alpha * norm(bm25_score) + (1-alpha) * norm(e5_score)

Run: python src_v2/40_convex_combination.py
"""

from pathlib import Path
import json
import math
import re
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "convex_combination.json"

MODEL_NAME = "intfloat/multilingual-e5-large"
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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

    def scores_all(self, q):
        return np.array([self.score(q, i) for i in range(self.N)])


def main():
    print("=" * 100)
    print("CONVEX COMBINATION OF BM25 AND E5 SCORES")
    print("=" * 100)

    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]
    n_docs = len(filenames)

    print(f"Chunks : {len(chunks)}")
    print(f"Queries: {len(queries)}\n")

    # BM25
    print("Building BM25...")
    bm25 = BM25(texts)

    # E5-large
    print("Loading E5-large...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    print("Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=True)

    # Precompute raw scores for all queries
    print("\nComputing scores for all queries...")
    bm25_scores_all = {}
    e5_scores_all = {}

    for idx, q in enumerate(queries):
        qid = q["query_id"]
        text = q["query"]

        bm25_s = bm25.scores_all(text)
        q_emb = model.encode(["query: " + text], normalize_embeddings=True)[0]
        e5_s = np.dot(doc_emb, q_emb)

        bm25_scores_all[qid] = bm25_s
        e5_scores_all[qid] = e5_s

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(queries)}")

    print("Done.\n")

    # Evaluate each alpha
    print("Sweeping alpha...")
    results = {}

    for alpha in ALPHAS:
        correct = 0
        for q in queries:
            qid = q["query_id"]
            gold = q["gold_document"]

            bm25_s = bm25_scores_all[qid]
            e5_s = e5_scores_all[qid]

            # Min-max normalize each score list
            def norm(x):
                lo, hi = x.min(), x.max()
                if hi - lo < 1e-9:
                    return np.zeros_like(x)
                return (x - lo) / (hi - lo)

            combined = alpha * norm(bm25_s) + (1 - alpha) * norm(e5_s)

            # Find gold rank
            ranked = np.argsort(combined)[::-1]
            for r, i in enumerate(ranked, start=1):
                if filenames[i] == gold:
                    if r == 1:
                        correct += 1
                    break

        results[alpha] = correct / len(queries)
        print(f"  alpha={alpha:.1f} @1={results[alpha]:.4f}")

    # Summary
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\n{'alpha':<8} {'@1':>10}")
    print("-" * 30)
    for alpha in ALPHAS:
        print(f"{alpha:<8.1f} {results[alpha]:>10.4f}")

    best_alpha = max(results, key=results.get)
    print(f"\nBest: alpha={best_alpha} @1={results[best_alpha]:.4f}")

    # Compare with RRF
    print()
    print("Comparison with RRF on original queries:")
    print(f"  E5-large alone     : 92.22%")
    print(f"  Best convex (alpha={best_alpha}) : {results[best_alpha]:.4f}")
    print(f"  Hybrid-base (RRF)  : 90.44%")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()