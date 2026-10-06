"""
In-Fold Weight Tuning on A_single
===================================
Claude requested: evaluate fusion on the honest A_single queries,
not just batch-context.

Run: python src_v2/38_in_fold_a_single.py
"""

from pathlib import Path
import json
import math
import re
import random
import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"
A_SINGLE_FILE = PROJECT_ROOT / "data_v2" / "processed" / "ablation_single.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "in_fold_a_single.json"

MODEL_NAME = "intfloat/multilingual-e5-large"
RRF_K = 10
N_FOLDS = 5
SEED = 42

WEIGHT_GRID = [
    (0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
    (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
    (0.8, 0.2), (0.9, 0.1), (1.0, 0.0), (1.0, 1.0),
]


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


def group_folds(queries, n_folds=5, seed=42):
    random.seed(seed)
    docs = sorted({q["doc_id"] for q in queries})
    random.shuffle(docs)
    folds = [[] for _ in range(n_folds)]
    for i, d in enumerate(docs):
        folds[i % n_folds].append(d)
    return folds


def evaluate(items, weight, bm25_ranks, e5_ranks, filenames, n_docs):
    w_b, w_e = weight
    correct = 0
    for it in items:
        qid = it["query_id"]
        gold = it["gold_document"]
        bm25_rank = bm25_ranks[qid]
        e5_rank = e5_ranks[qid]

        scores = {k: (w_b/(RRF_K + bm25_rank[k])) + (w_e/(RRF_K + e5_rank[k]))
                  for k in range(n_docs)}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for r, (k, _) in enumerate(ranked, start=1):
            if filenames[k] == gold:
                if r == 1:
                    correct += 1
                break
    return correct / len(items) if items else 0


def main():
    print("=" * 100)
    print("IN-FOLD WEIGHT TUNING ON A_SINGLE QUERIES")
    print("=" * 100)

    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)
    a_single = load_json(A_SINGLE_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]
    n_docs = len(filenames)

    print(f"Chunks    : {len(chunks)}")
    print(f"Queries   : {len(queries)}")
    print(f"A_single  : {len(a_single)} entries")

    # Build item list with A_single text
    items = []
    for q in queries:
        qid = q["query_id"]
        items.append({
            "query_id": qid,
            "doc_id": q["doc_id"],
            "level": q["level"],
            "gold_document": q["gold_document"],
            "text": a_single.get(qid, q["query"]),
        })

    # Build BM25 and E5 ranks using A_single text
    print("\nBuilding BM25...")
    bm25 = BM25(texts)

    print("Loading E5-large...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    print("Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=True)

    print("\nPrecomputing rankings on A_single text...")
    bm25_ranks = {}
    e5_ranks = {}

    for idx, it in enumerate(items):
        qid = it["query_id"]
        text = it["text"]

        bm25_r = bm25.rank(text)
        bm25_ranks[qid] = {i: r for r, (i, _) in enumerate(bm25_r, start=1)}

        q_emb = model.encode(["query: " + text], normalize_embeddings=True)[0]
        sims = np.dot(doc_emb, q_emb)
        e5_order = np.argsort(sims)[::-1]
        e5_ranks[qid] = {i: r for r, i in enumerate(e5_order, start=1)}

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(items)}")

    print("Done.\n")

    # Folds
    folds = group_folds(items, N_FOLDS, SEED)

    print("=" * 100)
    print("IN-FOLD TUNING (A_single)")
    print("=" * 100)

    fold_results = []

    for fold_idx in range(N_FOLDS):
        test_docs = folds[fold_idx]
        train_docs = [d for i, f in enumerate(folds) if i != fold_idx for d in f]

        train_items = [it for it in items if it["doc_id"] in train_docs]
        test_items = [it for it in items if it["doc_id"] in test_docs]

        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")

        best_w = None
        best_p1 = -1
        for w in WEIGHT_GRID:
            p1 = evaluate(train_items, w, bm25_ranks, e5_ranks, filenames, n_docs)
            if p1 > best_p1:
                best_p1 = p1
                best_w = w

        test_p1 = evaluate(test_items, best_w, bm25_ranks, e5_ranks, filenames, n_docs)
        oracle_p1 = max(
            evaluate(test_items, w, bm25_ranks, e5_ranks, filenames, n_docs)
            for w in WEIGHT_GRID
        )

        print(f"  Best weight: {best_w} (train @1={best_p1:.4f})")
        print(f"  Test @1: {test_p1:.4f}  (oracle: {oracle_p1:.4f})")

        fold_results.append({
            "fold": fold_idx + 1,
            "best_weight": list(best_w),
            "train_p1": best_p1,
            "test_p1": test_p1,
            "oracle_test_p1": oracle_p1,
        })

    # Summary
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    test_p1s = [f["test_p1"] for f in fold_results]
    train_p1s = [f["train_p1"] for f in fold_results]
    oracle_p1s = [f["oracle_test_p1"] for f in fold_results]

    print(f"\nTrain @1 : {np.mean(train_p1s):.4f} ± {np.std(train_p1s):.4f}")
    print(f"Test  @1 : {np.mean(test_p1s):.4f} ± {np.std(test_p1s):.4f}")
    print(f"Oracle   : {np.mean(oracle_p1s):.4f} ± {np.std(oracle_p1s):.4f}")
    print(f"\nGap (train − test)  : {np.mean(train_p1s) - np.mean(test_p1s):+.4f}")
    print(f"Gap (oracle − test) : {np.mean(oracle_p1s) - np.mean(test_p1s):+.4f}")

    # Best weights distribution
    weight_counts = Counter(tuple(f["best_weight"]) for f in fold_results)
    print(f"\nBest weights across folds:")
    for w, count in weight_counts.most_common():
        print(f"  {w}: {count} folds")

    save_json(RESULTS_FILE, {
        "folds": fold_results,
        "summary": {
            "train_mean": float(np.mean(train_p1s)),
            "train_std": float(np.std(train_p1s)),
            "test_mean": float(np.mean(test_p1s)),
            "test_std": float(np.std(test_p1s)),
            "oracle_mean": float(np.mean(oracle_p1s)),
            "oracle_std": float(np.std(oracle_p1s)),
        }
    })

    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()