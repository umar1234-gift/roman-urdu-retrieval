"""
Convex vs E5-large — Paired Cluster Bootstrap (Clean Tie-Aware)
================================================================
Tests significance on overall and per-level, especially L4.

Run: python src_v2/48_convex_significance.py
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

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "convex_significance.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "convex_significance.txt"

MODEL_NAME = "intfloat/multilingual-e5-large"
N_FOLDS = 5
SEED = 42
N_BOOTSTRAP = 5000
LEVELS = ["L0", "L1", "L2", "L3", "L4"]
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


def group_folds(queries, n_folds=5, seed=42):
    random.seed(seed)
    docs = sorted({q["doc_id"] for q in queries})
    random.shuffle(docs)
    folds = [[] for _ in range(n_folds)]
    for i, d in enumerate(docs):
        folds[i % n_folds].append(d)
    return folds


def paired_cluster_bootstrap(recs_a, recs_b, subset_levels=None,
                              n_boot=5000, seed=42):
    """Paired cluster bootstrap on top-1 accuracy."""
    if subset_levels:
        recs_a = [r for r in recs_a if r["level"] in subset_levels]
        recs_b = [r for r in recs_b if r["level"] in subset_levels]

    rng = np.random.default_rng(seed)
    map_a = {r["query_id"]: r for r in recs_a}
    map_b = {r["query_id"]: r for r in recs_b}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))

    if not common:
        return None

    deltas = np.array([
        (1 if map_a[q]["gold_rank"] == 1 else 0) -
        (1 if map_b[q]["gold_rank"] == 1 else 0)
        for q in common
    ])

    by_doc = defaultdict(list)
    for i, q in enumerate(common):
        by_doc[map_a[q]["doc_id"]].append(i)

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
        "n_docs": n_docs,
    }


def main():
    print("=" * 100)
    print("CONVEX vs E5-LARGE — PAIRED CLUSTER BOOTSTRAP (CLEAN)")
    print("=" * 100)

    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]
    n_docs = len(filenames)

    print("Building BM25...")
    bm25 = BM25(texts)

    print("Loading E5-large...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    print("Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True,
                            show_progress_bar=True)

    print("\nPrecomputing scores...")
    bm25_scores = {}
    e5_scores = {}

    for idx, q in enumerate(queries):
        qid = q["query_id"]
        bm25_scores[qid] = bm25.scores_all(q["query"])
        q_emb = model.encode(["query: " + q["query"]],
                              normalize_embeddings=True)[0]
        e5_scores[qid] = np.dot(doc_emb, q_emb)

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(queries)}")

    print("Done.\n")

    # E5-large records
    e5_records = []
    for q in queries:
        qid = q["query_id"]
        order = np.argsort(-e5_scores[qid], kind="stable")
        gr = None
        for r, i in enumerate(order, start=1):
            if filenames[i] == q["gold_document"]:
                gr = r
                break
        e5_records.append({
            "query_id": qid, "doc_id": q["doc_id"],
            "level": q["level"], "gold_rank": gr,
        })

    # In-fold convex records
    folds = group_folds(queries, N_FOLDS, SEED)
    convex_records = []

    for fold_idx in range(N_FOLDS):
        test_docs = folds[fold_idx]
        train_docs = [d for i, f in enumerate(folds) if i != fold_idx for d in f]
        train_items = [q for q in queries if q["doc_id"] in train_docs]
        test_items = [q for q in queries if q["doc_id"] in test_docs]

        # Tune alpha on train
        best_a = None
        best_p1 = -1
        for a in ALPHAS:
            correct = 0
            for it in train_items:
                qid = it["query_id"]
                b = minmax(bm25_scores[qid])
                e = minmax(e5_scores[qid])
                combined = a * b + (1 - a) * e
                order = np.argsort(-combined, kind="stable")
                for r, i in enumerate(order, start=1):
                    if filenames[i] == it["gold_document"]:
                        if r == 1:
                            correct += 1
                        break
            p1 = correct / len(train_items)
            if p1 > best_p1:
                best_p1 = p1
                best_a = a

        # Apply best_a to test
        for it in test_items:
            qid = it["query_id"]
            b = minmax(bm25_scores[qid])
            e = minmax(e5_scores[qid])
            combined = best_a * b + (1 - best_a) * e
            order = np.argsort(-combined, kind="stable")
            gr = None
            for r, i in enumerate(order, start=1):
                if filenames[i] == it["gold_document"]:
                    gr = r
                    break
            convex_records.append({
                "query_id": qid, "doc_id": it["doc_id"],
                "level": it["level"], "gold_rank": gr,
            })

    # Overall test
    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out()
    out("=" * 100)
    out("OVERALL: Convex vs E5-large")
    out("=" * 100)

    r_all = paired_cluster_bootstrap(convex_records, e5_records,
                                       None, N_BOOTSTRAP, SEED)
    out(f"\nΔ @1        : {r_all['delta']:+.4f}")
    out(f"95% CI      : [{r_all['ci_95'][0]:+.4f}, {r_all['ci_95'][1]:+.4f}]")
    out(f"p-value     : {r_all['p_value']:.4f} "
        f"{'* SIGNIFICANT' if r_all['p_value'] < 0.05 else '(not sig.)'}")
    out(f"n queries   : {r_all['n_queries']}")

    # Per-level
    out()
    out("=" * 100)
    out("PER-LEVEL: Convex vs E5-large")
    out("=" * 100)
    out()

    per_level_results = {}
    for lvl in LEVELS:
        r = paired_cluster_bootstrap(convex_records, e5_records,
                                       [lvl], N_BOOTSTRAP, SEED)
        if r:
            sig = "* SIGNIFICANT" if r["p_value"] < 0.05 else "(not sig.)"
            out(f"{lvl}: Δ={r['delta']:+.4f}  "
                f"CI=[{r['ci_95'][0]:+.4f}, {r['ci_95'][1]:+.4f}]  "
                f"p={r['p_value']:.4f}  {sig}")
            per_level_results[lvl] = r

    # L2-L4 subgroup
    out()
    out("=" * 100)
    out("L2-L4 SUBGROUP: Convex vs E5-large")
    out("=" * 100)
    r_l2l4 = paired_cluster_bootstrap(convex_records, e5_records,
                                        ["L2", "L3", "L4"], N_BOOTSTRAP, SEED)
    out(f"\nΔ @1        : {r_l2l4['delta']:+.4f}")
    out(f"95% CI      : [{r_l2l4['ci_95'][0]:+.4f}, {r_l2l4['ci_95'][1]:+.4f}]")
    out(f"p-value     : {r_l2l4['p_value']:.4f} "
        f"{'* SIGNIFICANT' if r_l2l4['p_value'] < 0.05 else '(not sig.)'}")

    # L3-L4 subgroup (Roman Urdu)
    out()
    out("=" * 100)
    out("L3-L4 SUBGROUP (Roman Urdu): Convex vs E5-large")
    out("=" * 100)
    r_l3l4 = paired_cluster_bootstrap(convex_records, e5_records,
                                        ["L3", "L4"], N_BOOTSTRAP, SEED)
    out(f"\nΔ @1        : {r_l3l4['delta']:+.4f}")
    out(f"95% CI      : [{r_l3l4['ci_95'][0]:+.4f}, {r_l3l4['ci_95'][1]:+.4f}]")
    out(f"p-value     : {r_l3l4['p_value']:.4f} "
        f"{'* SIGNIFICANT' if r_l3l4['p_value'] < 0.05 else '(not sig.)'}")

    # Save
    save_json(RESULTS_FILE, {
        "overall": r_all,
        "per_level": per_level_results,
        "L2_L4": r_l2l4,
        "L3_L4": r_l3l4,
    })

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    out()
    out(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()