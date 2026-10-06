"""
Clean Pipeline Re-run with Tie-Aware Handling
==============================================
Re-runs all BM25-dependent systems with proper tie-handling.

Systems:
- BM25 (tie-aware, report zero-score separately)
- E5-large (dense baseline)
- Hybrid-large (RRF with tie-aware BM25)
- Weighted RRF (in-fold, tie-aware)
- Convex combination (in-fold, tie-aware)

Outputs clean numbers for the paper.

Run: python src_v2/45_clean_pipeline_rerun.py
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

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "clean_pipeline_rerun.json"

MODEL_NAME = "intfloat/multilingual-e5-large"
RRF_K = 10
N_FOLDS = 5
SEED = 42
N_BOOTSTRAP = 5000
LEVELS = ["L0", "L1", "L2", "L3", "L4"]

WEIGHT_GRID = [
    (0.0, 1.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7),
    (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3),
    (0.8, 0.2), (0.9, 0.1), (1.0, 0.0),
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

    def ranks_stable(self, q):
        """Return ranks with stable tie-breaking (doc index ascending).
           Returns None if all scores are zero."""
        scores = self.scores_all(q)
        if scores.max() < 1e-9:
            return None
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
    return {"@1": g1/total, "@3": g3/total, "MRR": mrr, "n": total,
            "count_at1": g1}


def group_folds(queries, n_folds=5, seed=42):
    random.seed(seed)
    docs = sorted({q["doc_id"] for q in queries})
    random.shuffle(docs)
    folds = [[] for _ in range(n_folds)]
    for i, d in enumerate(docs):
        folds[i % n_folds].append(d)
    return folds


def main():
    print("=" * 100)
    print("CLEAN PIPELINE RE-RUN WITH TIE-AWARE HANDLING")
    print("=" * 100)

    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]
    n_docs = len(filenames)

    print(f"Chunks : {len(chunks)}")
    print(f"Queries: {len(queries)}\n")

    # Build BM25
    print("Building BM25...")
    bm25 = BM25(texts)

    # Build E5-large
    print("Loading E5-large...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    print("Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=True)

    # Precompute
    print("\nPrecomputing scores...")
    bm25_scores = {}
    e5_scores = {}
    bm25_ranks = {}  # None if all-zero
    zero_ids = []

    for idx, q in enumerate(queries):
        qid = q["query_id"]
        text = q["query"]

        bm25_s = bm25.scores_all(text)
        bm25_scores[qid] = bm25_s

        # E5
        q_emb = model.encode(["query: " + text], normalize_embeddings=True)[0]
        e5_s = np.dot(doc_emb, q_emb)
        e5_scores[qid] = e5_s

        # BM25 ranks (stable)
        if bm25_s.max() < 1e-9:
            bm25_ranks[qid] = None
            zero_ids.append(qid)
        else:
            order = np.argsort(-bm25_s, kind="stable")
            ranks = np.empty(n_docs, dtype=int)
            for r, i in enumerate(order, start=1):
                ranks[i] = r
            bm25_ranks[qid] = ranks

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(queries)}")

    print(f"\nAll-zero BM25 queries: {len(zero_ids)}/450 ({len(zero_ids)/450*100:.1f}%)")

    # ============================================================
    # SYSTEMS
    # ============================================================

    systems = {
        "BM25": [],
        "E5-large": [],
        "Hybrid-large (RRF tie-aware)": [],
        "Weighted-RRF (in-fold, tie-aware)": [],
        "Convex (in-fold)": [],
    }

    # ---- BM25 alone ----
    for q in queries:
        qid = q["query_id"]
        gold = q["gold_document"]
        ranks = bm25_ranks[qid]

        if ranks is None:
            # No ranking possible — treat as not found
            gr = None
        else:
            gr = None
            order = np.argsort(ranks)
            for r, i in enumerate(order, start=1):
                if filenames[i] == gold:
                    gr = ranks[i]
                    break

        systems["BM25"].append({
            "query_id": qid, "level": q["level"],
            "doc_id": q["doc_id"], "gold_rank": gr,
        })

    # ---- E5-large alone ----
    for q in queries:
        qid = q["query_id"]
        gold = q["gold_document"]
        order = np.argsort(-e5_scores[qid], kind="stable")
        gr = None
        for r, i in enumerate(order, start=1):
            if filenames[i] == gold:
                gr = r
                break
        systems["E5-large"].append({
            "query_id": qid, "level": q["level"],
            "doc_id": q["doc_id"], "gold_rank": gr,
        })

    # ---- Hybrid-large (RRF tie-aware) ----
    for q in queries:
        qid = q["query_id"]
        gold = q["gold_document"]
        bm25_r = bm25_ranks[qid]
        e5_order = np.argsort(-e5_scores[qid], kind="stable")
        e5_r = np.empty(n_docs, dtype=int)
        for r, i in enumerate(e5_order, start=1):
            e5_r[i] = r

        if bm25_r is None:
            scores = 1.0 / (RRF_K + e5_r)
        else:
            scores = 1.0 / (RRF_K + bm25_r) + 1.0 / (RRF_K + e5_r)

        order = np.argsort(-scores, kind="stable")
        gr = None
        for r, i in enumerate(order, start=1):
            if filenames[i] == gold:
                gr = r
                break
        systems["Hybrid-large (RRF tie-aware)"].append({
            "query_id": qid, "level": q["level"],
            "doc_id": q["doc_id"], "gold_rank": gr,
        })

    # ---- Weighted RRF (in-fold, tie-aware) ----
    folds = group_folds(queries, N_FOLDS, SEED)

    def eval_weights(items, weight):
        w_b, w_e = weight
        correct = 0
        for it in items:
            qid = it["query_id"]
            gold = it["gold_document"]
            bm25_r = bm25_ranks[qid]
            e5_order = np.argsort(-e5_scores[qid], kind="stable")
            e5_r = np.empty(n_docs, dtype=int)
            for r, i in enumerate(e5_order, start=1):
                e5_r[i] = r

            if bm25_r is None:
                scores = w_e / (RRF_K + e5_r)
            else:
                scores = w_b / (RRF_K + bm25_r) + w_e / (RRF_K + e5_r)

            order = np.argsort(-scores, kind="stable")
            for r, i in enumerate(order, start=1):
                if filenames[i] == gold:
                    if r == 1:
                        correct += 1
                    break
        return correct / len(items) if items else 0

    for fold_idx in range(N_FOLDS):
        test_docs = folds[fold_idx]
        train_docs = [d for i, f in enumerate(folds) if i != fold_idx for d in f]
        train_items = [q for q in queries if q["doc_id"] in train_docs]
        test_items = [q for q in queries if q["doc_id"] in test_docs]

        best_w = None
        best_p1 = -1
        for w in WEIGHT_GRID:
            p1 = eval_weights(train_items, w)
            if p1 > best_p1:
                best_p1 = p1
                best_w = w

        w_b, w_e = best_w
        for it in test_items:
            qid = it["query_id"]
            gold = it["gold_document"]
            bm25_r = bm25_ranks[qid]
            e5_order = np.argsort(-e5_scores[qid], kind="stable")
            e5_r = np.empty(n_docs, dtype=int)
            for r, i in enumerate(e5_order, start=1):
                e5_r[i] = r

            if bm25_r is None:
                scores = w_e / (RRF_K + e5_r)
            else:
                scores = w_b / (RRF_K + bm25_r) + w_e / (RRF_K + e5_r)

            order = np.argsort(-scores, kind="stable")
            gr = None
            for r, i in enumerate(order, start=1):
                if filenames[i] == gold:
                    gr = r
                    break
            systems["Weighted-RRF (in-fold, tie-aware)"].append({
                "query_id": qid, "level": it["level"],
                "doc_id": it["doc_id"], "gold_rank": gr,
            })

    # ---- Convex (in-fold) ----
    def minmax(x):
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    def eval_convex(items, alpha):
        correct = 0
        for it in items:
            qid = it["query_id"]
            gold = it["gold_document"]
            b = minmax(bm25_scores[qid])
            e = minmax(e5_scores[qid])
            combined = alpha * b + (1 - alpha) * e
            order = np.argsort(-combined, kind="stable")
            for r, i in enumerate(order, start=1):
                if filenames[i] == gold:
                    if r == 1:
                        correct += 1
                    break
        return correct / len(items) if items else 0

    alpha_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for fold_idx in range(N_FOLDS):
        test_docs = folds[fold_idx]
        train_docs = [d for i, f in enumerate(folds) if i != fold_idx for d in f]
        train_items = [q for q in queries if q["doc_id"] in train_docs]
        test_items = [q for q in queries if q["doc_id"] in test_docs]

        best_a = None
        best_p1 = -1
        for a in alpha_grid:
            p1 = eval_convex(train_items, a)
            if p1 > best_p1:
                best_p1 = p1
                best_a = a

        for it in test_items:
            qid = it["query_id"]
            gold = it["gold_document"]
            b = minmax(bm25_scores[qid])
            e = minmax(e5_scores[qid])
            combined = best_a * b + (1 - best_a) * e
            order = np.argsort(-combined, kind="stable")
            gr = None
            for r, i in enumerate(order, start=1):
                if filenames[i] == gold:
                    gr = r
                    break
            systems["Convex (in-fold)"].append({
                "query_id": qid, "level": it["level"],
                "doc_id": it["doc_id"], "gold_rank": gr,
            })

    # ============================================================
    # REPORT
    # ============================================================
    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out()
    out("=" * 100)
    out("CLEAN PIPELINE — OVERALL METRICS")
    out("=" * 100)
    out(f"\n{'System':<40} {'@1':>8} {'@3':>8} {'MRR':>8}")
    out("-" * 100)

    for name, recs in systems.items():
        m = metrics(recs)
        out(f"{name:<40} {m['@1']:>8.4f} {m['@3']:>8.4f} {m['MRR']:>8.4f}")

    # Per-level
    out()
    out("=" * 100)
    out("PER-LEVEL @1")
    out("=" * 100)
    out(f"\n{'System':<40} " + " ".join(f"{lvl:>8}" for lvl in LEVELS))
    out("-" * 100)

    for name, recs in systems.items():
        row = []
        for lvl in LEVELS:
            subset = [r for r in recs if r["level"] == lvl]
            row.append(metrics(subset)["@1"])
        out(f"{name:<40} " + " ".join(f"{v:>8.4f}" for v in row))

    # Degradation
    out()
    out("=" * 100)
    out("DEGRADATION L0 → L4")
    out("=" * 100)
    for name, recs in systems.items():
        l0 = metrics([r for r in recs if r["level"] == "L0"])["@1"]
        l4 = metrics([r for r in recs if r["level"] == "L4"])["@1"]
        out(f"{name:<40} L0={l0:.4f}  L4={l4:.4f}  Drop={l0-l4:+.4f}")

    # Save
    save_json(RESULTS_FILE, {
        "systems": {name: metrics(recs) for name, recs in systems.items()},
        "per_level": {
            name: {lvl: metrics([r for r in recs if r["level"] == lvl])["@1"]
                   for lvl in LEVELS}
            for name, recs in systems.items()
        },
        "zero_count": len(zero_ids),
        "zero_ids": zero_ids,
    })

    with open(PROJECT_ROOT / "data_v2" / "results" / "clean_pipeline_rerun.txt", "w") as f:
        f.write("\n".join(report))

    out()
    out(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()