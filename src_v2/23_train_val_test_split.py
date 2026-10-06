"""
Train/Val/Test Split with Proper Weight Tuning
"""

from pathlib import Path
import json
import math
import re
import numpy as np
from collections import Counter, defaultdict
import random
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "train_val_test_results.json"

MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K = 10
RRF_K = 10
LEVELS = ["L0", "L1", "L2", "L3", "L4"]

SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

WEIGHT_GRID = [
    (1.0, 1.0), (0.7, 0.3), (0.5, 0.5), (0.4, 0.6),
    (0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.0, 1.0),
]


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
    mrr = sum(1/r["gold_rank"] for r in records if r["gold_rank"]) / total
    return {"@1": g1/total, "@3": g3/total, "@5": g5/total, "MRR": mrr, "n": total}


def stratified_split(queries, seed=42):
    random.seed(seed)
    by_level = defaultdict(list)
    for q in queries:
        by_level[q["level"]].append(q)

    train, val, test = [], [], []

    for lvl in LEVELS:
        items = by_level[lvl][:]
        random.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    return train, val, test


def eval_weights(items, weight, bm25_ranks, e5_ranks, filenames):
    """Evaluate a weight config on a set of items."""
    w_b, w_e = weight
    results = []

    for item in items:
        bm25_rank = bm25_ranks[item["query_id"]]
        e5_rank = e5_ranks[item["query_id"]]
        gold_doc = item["gold_document"]

        scores = {}
        for i in range(len(filenames)):
            r_b = bm25_rank[i]
            r_e = e5_rank[i]
            scores[i] = (w_b / (RRF_K + r_b)) + (w_e / (RRF_K + r_e))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        gold_rank = None
        for r, (i, _) in enumerate(ranked, start=1):
            if filenames[i] == gold_doc:
                gold_rank = r
                break
        results.append({"gold_rank": gold_rank})

    return metrics(results)


def retrieve_with_weights(items, weight, bm25_ranks, e5_ranks, filenames):
    """Return per-query records for a set of items with given weight."""
    w_b, w_e = weight
    out = []
    for item in items:
        bm25_rank = bm25_ranks[item["query_id"]]
        e5_rank = e5_ranks[item["query_id"]]
        gold_doc = item["gold_document"]

        scores = {}
        for i in range(len(filenames)):
            r_b = bm25_rank[i]
            r_e = e5_rank[i]
            scores[i] = (w_b / (RRF_K + r_b)) + (w_e / (RRF_K + r_e))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        gold_rank = None
        for r, (i, _) in enumerate(ranked, start=1):
            if filenames[i] == gold_doc:
                gold_rank = r
                break

        out.append({
            "query_id": item["query_id"],
            "level": item["level"],
            "gold_rank": gold_rank,
            "weight": weight
        })
    return out


def main():
    print("=" * 100)
    print("TRAIN / VAL / TEST SPLIT — PROPER WEIGHT TUNING")
    print("=" * 100)
    print(f"Seed: {SEED}")
    print(f"Split: {int(TRAIN_RATIO*100)}% / {int(VAL_RATIO*100)}% / {int(TEST_RATIO*100)}%\n")

    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]

    # ============================================================
    # STEP 1: Split
    # ============================================================
    train, val, test = stratified_split(queries, seed=SEED)

    print("=" * 100)
    print("SPLIT SUMMARY")
    print("=" * 100)
    print(f"Train      : {len(train)} queries")
    print(f"Validation : {len(val)} queries")
    print(f"Test       : {len(test)} queries")

    print()
    print(f"{'Level':<6} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-" * 40)
    for lvl in LEVELS:
        n_tr = sum(1 for q in train if q["level"] == lvl)
        n_v = sum(1 for q in val if q["level"] == lvl)
        n_te = sum(1 for q in test if q["level"] == lvl)
        print(f"{lvl:<6} {n_tr:>8} {n_v:>8} {n_te:>8}")

    # ============================================================
    # STEP 2: Build indexes
    # ============================================================
    print()
    print("=" * 100)
    print("BUILDING INDEXES")
    print("=" * 100)

    print("Building BM25...")
    bm25 = BM25(texts)

    print(f"Loading E5-large...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    print("Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True,
                            show_progress_bar=True)

    # ============================================================
    # STEP 3: Pre-compute rankings
    # ============================================================
    print()
    print("Pre-computing rankings for all queries...")

    bm25_ranks = {}
    e5_ranks = {}

    for idx, q in enumerate(queries):
        bm25_ranking = bm25.rank(q["query"])
        bm25_ranks[q["query_id"]] = {i: r for r, (i, _) in
                                       enumerate(bm25_ranking, start=1)}

        q_emb = model.encode(["query: " + q["query"]],
                             normalize_embeddings=True)[0]
        sims = np.dot(doc_emb, q_emb)
        e5_order = np.argsort(sims)[::-1]
        e5_ranks[q["query_id"]] = {i: r for r, i in enumerate(e5_order, start=1)}

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(queries)}")

    print("Done.\n")

    # ============================================================
    # STEP 4: Tune weights on TRAIN
    # ============================================================
    print("=" * 100)
    print("STEP 1: TUNE WEIGHTS ON TRAINING SET")
    print("=" * 100)

    best_train_weights = {}

    for lvl in LEVELS:
        lvl_train = [q for q in train if q["level"] == lvl]
        if not lvl_train:
            continue

        print(f"\n{lvl} ({len(lvl_train)} train queries):")

        results_by_w = {}
        for w in WEIGHT_GRID:
            m = eval_weights(lvl_train, w, bm25_ranks, e5_ranks, filenames)
            results_by_w[w] = m

        best_w = max(results_by_w.keys(), key=lambda w: results_by_w[w]["@1"])
        best_train_weights[lvl] = best_w

        for w, m in results_by_w.items():
            marker = " ← BEST" if w == best_w else ""
            print(f"  ({w[0]:.1f}/{w[1]:.1f}) @1={m['@1']:.4f} "
                  f"@3={m['@3']:.4f} MRR={m['MRR']:.4f}{marker}")

    print()
    print("Selected weights (tuned on TRAIN):")
    for lvl in LEVELS:
        print(f"  {lvl}: {best_train_weights[lvl]}")

    # ============================================================
    # STEP 5: Validate on VAL
    # ============================================================
    print()
    print("=" * 100)
    print("STEP 2: VALIDATE ON VALIDATION SET")
    print("=" * 100)

    print(f"\n{'Level':<6} {'@1':>8} {'@3':>8} {'MRR':>8} {'n':>6}")
    print("-" * 40)

    for lvl in LEVELS:
        lvl_val = [q for q in val if q["level"] == lvl]
        w = best_train_weights[lvl]
        m = eval_weights(lvl_val, w, bm25_ranks, e5_ranks, filenames)
        print(f"{lvl:<6} {m['@1']:>8.4f} {m['@3']:>8.4f} {m['MRR']:>8.4f} {m['n']:>6}")

    val_all = []
    for lvl in LEVELS:
        lvl_val = [q for q in val if q["level"] == lvl]
        w = best_train_weights[lvl]
        val_all.extend(retrieve_with_weights(lvl_val, w, bm25_ranks, e5_ranks, filenames))
    val_overall = metrics(val_all)
    print(f"\n{'Overall':<6} {val_overall['@1']:>8.4f} {val_overall['@3']:>8.4f} "
          f"{val_overall['MRR']:>8.4f} {val_overall['n']:>6}")

    # ============================================================
    # STEP 6: FINAL TEST
    # ============================================================
    print()
    print("=" * 100)
    print("STEP 3: FINAL TEST ON HELD-OUT TEST SET")
    print("=" * 100)

    print(f"\n{'Level':<6} {'@1':>8} {'@3':>8} {'MRR':>8} {'n':>6}")
    print("-" * 40)

    for lvl in LEVELS:
        lvl_test = [q for q in test if q["level"] == lvl]
        w = best_train_weights[lvl]
        m = eval_weights(lvl_test, w, bm25_ranks, e5_ranks, filenames)
        print(f"{lvl:<6} {m['@1']:>8.4f} {m['@3']:>8.4f} {m['MRR']:>8.4f} {m['n']:>6}")

    test_all = []
    for lvl in LEVELS:
        lvl_test = [q for q in test if q["level"] == lvl]
        w = best_train_weights[lvl]
        test_all.extend(retrieve_with_weights(lvl_test, w, bm25_ranks, e5_ranks, filenames))
    test_overall = metrics(test_all)
    print(f"\n{'Overall':<6} {test_overall['@1']:>8.4f} {test_overall['@3']:>8.4f} "
          f"{test_overall['MRR']:>8.4f} {test_overall['n']:>6}")

    # ============================================================
    # STEP 7: Baselines on TEST
    # ============================================================
    print()
    print("=" * 100)
    print("STEP 4: BASELINES ON TEST SET")
    print("=" * 100)

    # E5-large alone
    e5_test = []
    for item in test:
        e5_rank = e5_ranks[item["query_id"]]
        ranked = sorted(e5_rank.items(), key=lambda x: x[1])
        gr = None
        for r, (i, _) in enumerate(ranked, start=1):
            if filenames[i] == item["gold_document"]:
                gr = r
                break
        e5_test.append({"gold_rank": gr})
    e5_test_m = metrics(e5_test)

    # BM25 alone
    bm25_test = []
    for item in test:
        bm25_rank = bm25_ranks[item["query_id"]]
        ranked = sorted(bm25_rank.items(), key=lambda x: x[1])
        gr = None
        for r, (i, _) in enumerate(ranked, start=1):
            if filenames[i] == item["gold_document"]:
                gr = r
                break
        bm25_test.append({"gold_rank": gr})
    bm25_test_m = metrics(bm25_test)

    # Standard RRF (no tuning)
    std_rrf_test = []
    for item in test:
        bm25_rank = bm25_ranks[item["query_id"]]
        e5_rank = e5_ranks[item["query_id"]]
        scores = {}
        for i in range(len(filenames)):
            r_b = bm25_rank[i]
            r_e = e5_rank[i]
            scores[i] = (1.0/(RRF_K + r_b)) + (1.0/(RRF_K + r_e))
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        gr = None
        for r, (i, _) in enumerate(ranked, start=1):
            if filenames[i] == item["gold_document"]:
                gr = r
                break
        std_rrf_test.append({"gold_rank": gr})
    std_rrf_m = metrics(std_rrf_test)

    print(f"BM25 alone           : @1={bm25_test_m['@1']:.4f} "
          f"@3={bm25_test_m['@3']:.4f} MRR={bm25_test_m['MRR']:.4f}")
    print(f"E5-large alone       : @1={e5_test_m['@1']:.4f} "
          f"@3={e5_test_m['@3']:.4f} MRR={e5_test_m['MRR']:.4f}")
    print(f"Standard RRF (1/1)   : @1={std_rrf_m['@1']:.4f} "
          f"@3={std_rrf_m['@3']:.4f} MRR={std_rrf_m['MRR']:.4f}")
    print(f"Adaptive RRF (tuned) : @1={test_overall['@1']:.4f} "
          f"@3={test_overall['@3']:.4f} MRR={test_overall['MRR']:.4f}")

    # ============================================================
    # HONEST COMPARISON
    # ============================================================
    print()
    print("=" * 100)
    print("HONEST COMPARISON (Test Set, No Tuning Leakage)")
    print("=" * 100)

    baseline_best = max([bm25_test_m["@1"], e5_test_m["@1"], std_rrf_m["@1"]])
    adaptive_delta = test_overall["@1"] - baseline_best

    print(f"\nBest baseline @1     : {baseline_best:.4f}")
    print(f"Adaptive RRF @1      : {test_overall['@1']:.4f}")
    print(f"Delta                : {adaptive_delta:+.4f}")

    if adaptive_delta > 0.01:
        print("\n→ Adaptive RRF shows meaningful improvement on test set ✅")
    elif adaptive_delta > 0:
        print("\n→ Adaptive RRF shows marginal improvement ⚠️")
    else:
        print("\n→ Adaptive RRF does NOT improve over baselines ❌")

    # Save
    output = {
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "seed": SEED,
        "best_weights_from_train": {lvl: list(w) for lvl, w in best_train_weights.items()},
        "val_overall": val_overall,
        "test_overall": test_overall,
        "baselines_on_test": {
            "bm25": bm25_test_m,
            "e5_large": e5_test_m,
            "standard_rrf": std_rrf_m,
        },
        "honest_delta": adaptive_delta,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()