"""
In-Fold Alpha Tuning for Convex Combination + Paired Cluster Test
==================================================================
Fixes the tuning leakage Claude identified.

1. Grouped 5-fold CV for α selection (train-fold only)
2. Test @1 on held-out folds (mean ± std)
3. Paired cluster bootstrap: convex vs E5-large

Run: python src_v2/42_in_fold_convex.py
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

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "in_fold_convex.json"

MODEL_NAME = "intfloat/multilingual-e5-large"
N_FOLDS = 5
SEED = 42
N_BOOTSTRAP = 5000
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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


def minmax(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def rank_of(scores, filenames, gold):
    ranked = np.argsort(scores)[::-1]
    for r, i in enumerate(ranked, start=1):
        if filenames[i] == gold:
            return r
    return None


def eval_alpha(items, alpha, bm25_scores, e5_scores, filenames):
    """Evaluate α on a list of items. Returns @1."""
    correct = 0
    for it in items:
        qid = it["query_id"]
        gold = it["gold_document"]
        b = minmax(bm25_scores[qid])
        e = minmax(e5_scores[qid])
        combined = alpha * b + (1 - alpha) * e
        r = rank_of(combined, filenames, gold)
        if r == 1:
            correct += 1
    return correct / len(items) if items else 0.0


def group_folds(queries, n_folds=5, seed=42):
    random.seed(seed)
    docs = sorted({q["doc_id"] for q in queries})
    random.shuffle(docs)
    folds = [[] for _ in range(n_folds)]
    for i, d in enumerate(docs):
        folds[i % n_folds].append(d)
    return folds


def paired_cluster_bootstrap(ranks_a, ranks_b, doc_ids, n_boot=5000, seed=42):
    """Paired cluster bootstrap on top-1 accuracy."""
    rng = np.random.default_rng(seed)
    common = sorted(set(ranks_a.keys()) & set(ranks_b.keys()))
    deltas = np.array([
        (1 if ranks_a[q] == 1 else 0) - (1 if ranks_b[q] == 1 else 0)
        for q in common
    ])
    by_doc = defaultdict(list)
    idx_map = {q: i for i, q in enumerate(common)}
    for q in common:
        d = doc_ids[q]
        by_doc[d].append(idx_map[q])

    docs = list(by_doc.keys())
    n_docs = len(docs)
    obs = float(np.mean(deltas))

    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(docs, size=n_docs, replace=True)
        idx = []
        for d in sampled:
            idx.extend(by_doc[d])
        boot[i] = np.mean(deltas[idx])

    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))
    p = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    p = float(max(p, 2 / n_boot))

    return {
        "delta": obs,
        "ci_95": [lo, hi],
        "p_value": p,
        "n_queries": len(common),
    }


def main():
    print("=" * 100)
    print("IN-FOLD ALPHA TUNING FOR CONVEX COMBINATION")
    print("=" * 100)

    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]
    doc_ids = {q["query_id"]: q["doc_id"] for q in queries}

    print(f"Chunks : {len(chunks)}")
    print(f"Queries: {len(queries)}\n")

    print("Building BM25...")
    bm25 = BM25(texts)

    print("Loading E5-large...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    print("Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=True)

    print("\nPrecomputing scores...")
    bm25_scores = {}
    e5_scores = {}
    for idx, q in enumerate(queries):
        qid = q["query_id"]
        bm25_scores[qid] = bm25.scores_all(q["query"])
        q_emb = model.encode(["query: " + q["query"]], normalize_embeddings=True)[0]
        e5_scores[qid] = np.dot(doc_emb, q_emb)
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(queries)}")
    print("Done.\n")

    # Folds
    folds = group_folds(queries, N_FOLDS, SEED)

    print("=" * 100)
    print("IN-FOLD α TUNING")
    print("=" * 100)

    fold_rows = []

    for fold_idx in range(N_FOLDS):
        test_docs = folds[fold_idx]
        train_docs = [d for i, f in enumerate(folds) if i != fold_idx for d in f]

        train_items = [q for q in queries if q["doc_id"] in train_docs]
        test_items = [q for q in queries if q["doc_id"] in test_docs]

        best_alpha = None
        best_train = -1.0
        for a in ALPHAS:
            p1 = eval_alpha(train_items, a, bm25_scores, e5_scores, filenames)
            if p1 > best_train:
                best_train = p1
                best_alpha = a

        test_p1 = eval_alpha(test_items, best_alpha, bm25_scores, e5_scores, filenames)
        oracle_p1 = max(
            eval_alpha(test_items, a, bm25_scores, e5_scores, filenames)
            for a in ALPHAS
        )

        print(f"\n--- Fold {fold_idx+1}/{N_FOLDS} ---")
        print(f"  Best α (train): {best_alpha} (train @1={best_train:.4f})")
        print(f"  Test @1: {test_p1:.4f}  (oracle: {oracle_p1:.4f})")

        fold_rows.append({
            "fold": fold_idx + 1,
            "best_alpha": float(best_alpha),
            "train_p1": best_train,
            "test_p1": test_p1,
            "oracle_p1": oracle_p1,
        })

    # Summary
    test_p1s = [f["test_p1"] for f in fold_rows]
    train_p1s = [f["train_p1"] for f in fold_rows]
    oracle_p1s = [f["oracle_p1"] for f in fold_rows]

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\nTrain @1 (mean ± std): {np.mean(train_p1s):.4f} ± {np.std(train_p1s, ddof=1):.4f}")
    print(f"Test  @1 (mean ± std): {np.mean(test_p1s):.4f} ± {np.std(test_p1s, ddof=1):.4f}")
    print(f"Oracle   (mean ± std): {np.mean(oracle_p1s):.4f} ± {np.std(oracle_p1s, ddof=1):.4f}")
    print(f"\nTrain-Test gap: {np.mean(train_p1s) - np.mean(test_p1s):+.4f}")
    print(f"Oracle-Test gap: {np.mean(oracle_p1s) - np.mean(test_p1s):+.4f}")

    # Best α distribution
    alpha_counts = Counter(f["best_alpha"] for f in fold_rows)
    print(f"\nBest α distribution across folds:")
    for a, c in alpha_counts.most_common():
        print(f"  α={a}: {c} folds")

    # Paired cluster bootstrap on OUT-OF-FOLD predictions
    # We use the per-fold selected α to make out-of-fold predictions
    print()
    print("=" * 100)
    print("PAIRED CLUSTER BOOTSTRAP (convex vs E5-large, out-of-fold)")
    print("=" * 100)

    convex_ranks = {}
    e5_ranks = {}

    for fold_idx in range(N_FOLDS):
        test_docs = folds[fold_idx]
        best_alpha = fold_rows[fold_idx]["best_alpha"]
        test_items = [q for q in queries if q["doc_id"] in test_docs]

        for it in test_items:
            qid = it["query_id"]
            gold = it["gold_document"]
            b = minmax(bm25_scores[qid])
            e = minmax(e5_scores[qid])

            # Convex prediction
            combined = best_alpha * b + (1 - best_alpha) * e
            r_c = rank_of(combined, filenames, gold)
            convex_ranks[qid] = r_c

            # E5-large alone
            r_e = rank_of(e5_scores[qid], filenames, gold)
            e5_ranks[qid] = r_e

    result = paired_cluster_bootstrap(convex_ranks, e5_ranks, doc_ids, N_BOOTSTRAP, SEED)

    print(f"\nΔ @1 (convex - E5-large) : {result['delta']:+.4f}")
    print(f"95% CI                   : [{result['ci_95'][0]:+.4f}, {result['ci_95'][1]:+.4f}]")
    print(f"p-value                  : {result['p_value']:.4f}")

    if result["p_value"] < 0.05:
        print("→ SIGNIFICANT")
    else:
        print("→ NOT significant (CI may include zero)")

    # Save
    save_json(RESULTS_FILE, {
        "folds": fold_rows,
        "summary": {
            "train_mean": float(np.mean(train_p1s)),
            "train_std": float(np.std(train_p1s, ddof=1)),
            "test_mean": float(np.mean(test_p1s)),
            "test_std": float(np.std(test_p1s, ddof=1)),
            "oracle_mean": float(np.mean(oracle_p1s)),
            "oracle_std": float(np.std(oracle_p1s, ddof=1)),
        },
        "paired_bootstrap": result,
    })

    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()