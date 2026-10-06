"""
Zero-Score Query Analysis
==========================
Counts queries where BM25 scores are all zero (no term match).
These are the tie-breaking-sensitive queries.

Run: python src_v2/43_zero_score_analysis.py
"""

from pathlib import Path
import json
import math
import re
import numpy as np
from collections import Counter, defaultdict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "zero_score_analysis.json"

LEVELS = ["L0", "L1", "L2", "L3", "L4"]


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


def main():
    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)
    with open(QUERIES_FILE) as f:
        queries = json.load(f)

    bm25 = BM25([c["text"] for c in chunks])

    by_level = defaultdict(lambda: {"total": 0, "all_zero": 0, "partial_zero": 0})
    zero_ids = []
    partial_ids = []

    for q in queries:
        qid = q["query_id"]
        lvl = q["level"]
        scores = bm25.scores_all(q["query"])

        by_level[lvl]["total"] += 1

        if scores.max() < 1e-9:
            by_level[lvl]["all_zero"] += 1
            zero_ids.append({"query_id": qid, "level": lvl, "doc_id": q["doc_id"]})
        elif np.sum(scores < 1e-9) > 0:
            by_level[lvl]["partial_zero"] += 1
            partial_ids.append({"query_id": qid, "level": lvl})

    print("=" * 100)
    print("ZERO-SCORE BM25 QUERY ANALYSIS")
    print("=" * 100)
    print()
    print(f"{'Level':<8} {'Total':>8} {'All-Zero':>10} {'Partial-Zero':>14}")
    print("-" * 50)
    for lvl in LEVELS:
        d = by_level[lvl]
        print(f"{lvl:<8} {d['total']:>8} {d['all_zero']:>10} {d['partial_zero']:>14}")

    total = sum(d["total"] for d in by_level.values())
    total_zero = sum(d["all_zero"] for d in by_level.values())
    total_partial = sum(d["partial_zero"] for d in by_level.values())

    print()
    print(f"Total queries        : {total}")
    print(f"All-zero BM25 scores : {total_zero} ({total_zero/total*100:.1f}%)")
    print(f"Partial-zero         : {total_partial}")
    print()

    print("All-zero query IDs (first 30):")
    for z in zero_ids[:30]:
        print(f"  {z['query_id']} ({z['level']}, {z['doc_id']})")

    # Save
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "by_level": dict(by_level),
            "total": total,
            "all_zero": total_zero,
            "partial_zero": total_partial,
            "zero_ids": zero_ids,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()