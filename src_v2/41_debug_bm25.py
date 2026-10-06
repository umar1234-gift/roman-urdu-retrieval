"""
Debug BM25 discrepancy between script 05 (88.44%) and script 40 (88.89%)
=========================================================================
Both should give identical rankings — but they don't. Find why.

Run: python src_v2/41_debug_bm25.py
"""

from pathlib import Path
import json
import math
import re
import numpy as np
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"
BM25_SCRIPT05_FILE = PROJECT_ROOT / "data_v2" / "results" / "bm25_results.json"


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
    with open(BM25_SCRIPT05_FILE) as f:
        script05 = json.load(f)

    filenames = [c["filename"] for c in chunks]
    bm25 = BM25([c["text"] for c in chunks])

    s05_by_id = {r["query_id"]: r["gold_rank"] for r in script05}

    mismatches = []
    total_05_correct = 0
    total_method1_correct = 0
    total_method2_correct = 0

    for q in queries:
        qid = q["query_id"]
        gold = q["gold_document"]

        # Method 1: raw scores + argsort
        raw = bm25.scores_all(q["query"])
        ranked1 = np.argsort(raw)[::-1]
        rank1 = None
        for r, i in enumerate(ranked1, start=1):
            if filenames[i] == gold:
                rank1 = r
                break

        # Method 2: normalized + argsort
        denom = raw.max() - raw.min()
        if denom < 1e-12:
            norm = np.zeros_like(raw)
        else:
            norm = (raw - raw.min()) / denom
        ranked2 = np.argsort(norm)[::-1]
        rank2 = None
        for r, i in enumerate(ranked2, start=1):
            if filenames[i] == gold:
                rank2 = r
                break

        r05 = s05_by_id.get(qid)

        if r05 == 1:
            total_05_correct += 1
        if rank1 == 1:
            total_method1_correct += 1
        if rank2 == 1:
            total_method2_correct += 1

        if rank1 != r05 or rank2 != r05:
            mismatches.append({
                "qid": qid,
                "level": q["level"],
                "doc_id": q["doc_id"],
                "script05": r05,
                "raw_argsort": rank1,
                "norm_argsort": rank2,
            })

    print("=" * 100)
    print("BM25 DISCREPANCY DEBUG")
    print("=" * 100)
    print()
    print(f"Total queries              : {len(queries)}")
    print(f"Script 05 correct @1       : {total_05_correct} ({total_05_correct/len(queries)*100:.2f}%)")
    print(f"Raw BM25 argsort correct   : {total_method1_correct} ({total_method1_correct/len(queries)*100:.2f}%)")
    print(f"Normalized argsort correct : {total_method2_correct} ({total_method2_correct/len(queries)*100:.2f}%)")
    print()
    print(f"Total mismatches: {len(mismatches)}")

    if mismatches:
        print()
        print("First 20 mismatches:")
        print("-" * 100)
        for m in mismatches[:20]:
            print(f"  {m['qid']} {m['doc_id']} {m['level']}: "
                  f"script05={m['script05']}, raw={m['raw_argsort']}, norm={m['norm_argsort']}")

    # Save
    out_file = PROJECT_ROOT / "data_v2" / "results" / "bm25_debug.json"
    with open(out_file, "w") as f:
        json.dump({
            "total_queries": len(queries),
            "script05_correct": total_05_correct,
            "raw_argsort_correct": total_method1_correct,
            "norm_argsort_correct": total_method2_correct,
            "mismatches": mismatches,
        }, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()