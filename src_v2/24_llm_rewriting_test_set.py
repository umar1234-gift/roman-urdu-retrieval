"""
LLM Rewriting — Train/Val/Test Split Evaluation
================================================
Proper evaluation without tuning leakage.

Uses SAME split as script 23 (seed=42) for comparability.

On held-out TEST set (90 queries):
- Original queries: BM25, E5-large, Std RRF, Adaptive RRF
- Rewritten queries: BM25, E5-large, Std RRF, Adaptive RRF

Adaptive RRF weights are tuned ONLY on train set (no test leakage).

Run: python src_v2/24_llm_rewriting_test_set.py
"""

from pathlib import Path
import json
import math
import re
import numpy as np
from collections import Counter, defaultdict
import random
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"
ORIGINAL_QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"
REWRITTEN_QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "rewritten_queries.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "llm_rewriting_test_set_results.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "llm_rewriting_test_set_report.txt"

MODEL_NAME = "intfloat/multilingual-e5-large"
RRF_K = 10
LEVELS = ["L0", "L1", "L2", "L3", "L4"]

# Same split as script 23
SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Weight grid
WEIGHT_GRID = [
    (1.0, 1.0), (0.7, 0.3), (0.5, 0.5), (0.4, 0.6),
    (0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.0, 1.0),
]


# ============================================================
# HELPERS
# ============================================================

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


def metrics(records):
    """Compute @1, @3, @5, MRR from a list of records with 'gold_rank'."""
    total = len(records)
    if total == 0:
        return {"@1": 0, "@3": 0, "@5": 0, "MRR": 0, "n": 0}

    g1 = sum(1 for r in records if r["gold_rank"] == 1)
    g3 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 3)
    g5 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 5)
    mrr = sum(1 / r["gold_rank"] for r in records if r["gold_rank"]) / total

    return {"@1": g1/total, "@3": g3/total, "@5": g5/total, "MRR": mrr, "n": total}


def stratified_split(queries, seed=42):
    """Stratified split by level. Same logic as script 23."""
    random.seed(seed)
    by_level = defaultdict(list)
    for q in queries:
        by_level[q["level"]].append(q)

    train, val, test = [], [], []

    for lvl in LEVELS:
        items = list(by_level[lvl])  # copy
        random.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    return train, val, test


# ============================================================
# RANKING COMPUTATION
# ============================================================

def compute_rankings(queries, text_key, bm25, model, doc_emb, filenames, label=""):
    """
    Compute BM25 and E5-large rankings for a list of queries.
    text_key: field name in each query dict to use as query text.
    Returns two dicts: {query_id: {doc_idx: rank}}.
    """
    bm25_ranks = {}
    e5_ranks = {}

    print(f"Computing rankings for {len(queries)} queries [{label}]...")

    for idx, q in enumerate(queries):
        text = q[text_key]
        qid = q["query_id"]

        # BM25 ranking
        bm25_ranking = bm25.rank(text)
        bm25_ranks[qid] = {
            doc_idx: rank
            for rank, (doc_idx, _) in enumerate(bm25_ranking, start=1)
        }

        # E5-large ranking
        q_emb = model.encode(
            ["query: " + text],
            normalize_embeddings=True
        )[0]
        sims = np.dot(doc_emb, q_emb)
        e5_order = np.argsort(sims)[::-1]
        e5_ranks[qid] = {
            doc_idx: rank
            for rank, doc_idx in enumerate(e5_order, start=1)
        }

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(queries)}")

    print(f"  Done [{label}]")
    return bm25_ranks, e5_ranks


# ============================================================
# RETRIEVAL WITH WEIGHTS
# ============================================================

def retrieve_with_weights(items, weight, bm25_ranks, e5_ranks, filenames):
    """
    Run retrieval for each item using the given RRF weight.
    Returns list of records: {query_id, level, gold_rank}.
    """
    w_b, w_e = weight
    n_docs = len(filenames)
    results = []

    for item in items:
        qid = item["query_id"]
        gold_doc = item["gold_document"]
        bm25_rank = bm25_ranks[qid]
        e5_rank = e5_ranks[qid]

        # RRF scores for all docs
        scores = {}
        for doc_idx in range(n_docs):
            r_b = bm25_rank[doc_idx]
            r_e = e5_rank[doc_idx]
            scores[doc_idx] = (w_b / (RRF_K + r_b)) + (w_e / (RRF_K + r_e))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        gold_rank = None
        for rank, (doc_idx, _) in enumerate(ranked, start=1):
            if filenames[doc_idx] == gold_doc:
                gold_rank = rank
                break

        results.append({
            "query_id": qid,
            "level": item["level"],
            "gold_rank": gold_rank,
        })

    return results


def retrieve_single_system(items, rank_source, filenames):
    """
    Run retrieval using a single system's ranking.
    rank_source: {query_id: {doc_idx: rank}} dict.
    """
    results = []
    for item in items:
        qid = item["query_id"]
        gold_doc = item["gold_document"]
        rank_map = rank_source[qid]

        ranked = sorted(rank_map.items(), key=lambda x: x[1])

        gold_rank = None
        for rank, (doc_idx, _) in enumerate(ranked, start=1):
            if filenames[doc_idx] == gold_doc:
                gold_rank = rank
                break

        results.append({
            "query_id": qid,
            "level": item["level"],
            "gold_rank": gold_rank,
        })
    return results


# ============================================================
# TUNING
# ============================================================

def tune_weights_per_level(train_items, bm25_ranks, e5_ranks, filenames):
    """Tune best weight per level on training set."""
    best_weights = {}

    for lvl in LEVELS:
        lvl_train = [x for x in train_items if x["level"] == lvl]
        if not lvl_train:
            best_weights[lvl] = (1.0, 1.0)
            continue

        best_w = None
        best_p1 = -1.0
        for w in WEIGHT_GRID:
            recs = retrieve_with_weights(lvl_train, w, bm25_ranks, e5_ranks, filenames)
            m = metrics(recs)
            if m["@1"] > best_p1:
                best_p1 = m["@1"]
                best_w = w

        best_weights[lvl] = best_w

    return best_weights


def eval_adaptive(items, weights_per_level, bm25_ranks, e5_ranks, filenames):
    """Evaluate adaptive RRF (per-level weights) on a set of items."""
    all_records = []
    for lvl in LEVELS:
        lvl_items = [x for x in items if x["level"] == lvl]
        if not lvl_items:
            continue
        w = weights_per_level[lvl]
        recs = retrieve_with_weights(lvl_items, w, bm25_ranks, e5_ranks, filenames)
        all_records.extend(recs)
    return all_records


def eval_standard_rrf(items, bm25_ranks, e5_ranks, filenames):
    """Standard RRF with equal weights (1.0, 1.0)."""
    return retrieve_with_weights(items, (1.0, 1.0), bm25_ranks, e5_ranks, filenames)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("LLM REWRITING — TRAIN/VAL/TEST EVALUATION")
    print("=" * 100)
    print(f"Seed: {SEED}")
    print(f"Split: {int(TRAIN_RATIO*100)}% / {int(VAL_RATIO*100)}% / {int(TEST_RATIO*100)}%")
    print(f"RRF k: {RRF_K}\n")

    # ============================================================
    # 1. Load data
    # ============================================================
    print("Loading data...")
    chunks = load_json(CHUNKS_FILE)
    orig_queries = load_json(ORIGINAL_QUERIES_FILE)
    rw_queries = load_json(REWRITTEN_QUERIES_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]
    n_docs = len(filenames)

    print(f"  Chunks          : {len(chunks)}")
    print(f"  Original queries: {len(orig_queries)}")
    print(f"  Rewritten       : {len(rw_queries)}")

    # Map rewritten queries by query_id
    rw_by_id = {q["query_id"]: q for q in rw_queries}

    # Build unified query list
    # Note: "original_text" always from orig_queries["query"]
    #       "rewritten_text" from rw_queries["query"]
    unified = []
    for q_o in orig_queries:
        qid = q_o["query_id"]
        if qid not in rw_by_id:
            raise ValueError(f"Missing rewritten version for {qid}")
        q_r = rw_by_id[qid]
        unified.append({
            "query_id": qid,
            "doc_id": q_o["doc_id"],
            "level": q_o["level"],
            "variant": q_o["variant"],
            "original_text": q_o["query"],
            "rewritten_text": q_r["query"],
            "gold_document": q_o["gold_document"],
            "was_rewritten": q_r["was_rewritten"],
        })

    print(f"  Unified queries : {len(unified)}")

    # Sanity check: L0/L1 not rewritten
    l0_l1_not_rw = all(not u["was_rewritten"] for u in unified if u["level"] in ["L0", "L1"])
    l2_l3_l4_rw = all(u["was_rewritten"] for u in unified if u["level"] in ["L2", "L3", "L4"])
    print(f"  L0/L1 unchanged : {l0_l1_not_rw}")
    print(f"  L2/L3/L4 rewritten: {l2_l3_l4_rw}")

    # ============================================================
    # 2. Split (same seed as script 23)
    # ============================================================
    print()
    print("=" * 100)
    print("SPLIT")
    print("=" * 100)

    train, val, test = stratified_split(unified, seed=SEED)

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
    # 3. Build indexes
    # ============================================================
    print()
    print("=" * 100)
    print("BUILDING INDEXES")
    print("=" * 100)

    print("Building BM25...")
    bm25 = BM25(texts)

    print(f"Loading E5-large ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding documents...")
    doc_texts = ["passage: " + t for t in texts]
    doc_emb = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=True)

    # ============================================================
    # 4. Compute rankings for both original and rewritten
    # ============================================================
    print()
    print("=" * 100)
    print("COMPUTING RANKINGS")
    print("=" * 100)

    bm25_orig, e5_orig = compute_rankings(
        unified, "original_text", bm25, model, doc_emb, filenames, label="ORIGINAL"
    )
    bm25_rw, e5_rw = compute_rankings(
        unified, "rewritten_text", bm25, model, doc_emb, filenames, label="REWRITTEN"
    )

    # ============================================================
    # 5. Tune adaptive weights on TRAIN
    # ============================================================
    print()
    print("=" * 100)
    print("TUNING ADAPTIVE WEIGHTS ON TRAIN")
    print("=" * 100)

    print("\n--- Original queries ---")
    weights_orig = tune_weights_per_level(train, bm25_orig, e5_orig, filenames)
    for lvl in LEVELS:
        print(f"  {lvl}: {weights_orig[lvl]}")

    print("\n--- Rewritten queries ---")
    weights_rw = tune_weights_per_level(train, bm25_rw, e5_rw, filenames)
    for lvl in LEVELS:
        print(f"  {lvl}: {weights_rw[lvl]}")

    # ============================================================
    # 6. Evaluate on TEST set
    # ============================================================
    print()
    print("=" * 100)
    print("EVALUATION ON HELD-OUT TEST SET")
    print("=" * 100)

    # --- Original queries ---
    print("\n--- ORIGINAL QUERIES ---")
    orig_bm25 = retrieve_single_system(test, bm25_orig, filenames)
    orig_e5 = retrieve_single_system(test, e5_orig, filenames)
    orig_std = eval_standard_rrf(test, bm25_orig, e5_orig, filenames)
    orig_adapt = eval_adaptive(test, weights_orig, bm25_orig, e5_orig, filenames)

    # --- Rewritten queries ---
    print("--- REWRITTEN QUERIES ---")
    rw_bm25 = retrieve_single_system(test, bm25_rw, filenames)
    rw_e5 = retrieve_single_system(test, e5_rw, filenames)
    rw_std = eval_standard_rrf(test, bm25_rw, e5_rw, filenames)
    rw_adapt = eval_adaptive(test, weights_rw, bm25_rw, e5_rw, filenames)

    # ============================================================
    # 7. Report
    # ============================================================
    print()
    print("=" * 100)
    print("RESULTS ON TEST SET (n=90)")
    print("=" * 100)

    header = f"{'System':<35} {'@1':>8} {'@3':>8} {'@5':>8} {'MRR':>8}"
    print(header)
    print("-" * 100)

    # Original
    for name, recs in [
        ("ORIG: BM25", orig_bm25),
        ("ORIG: E5-large", orig_e5),
        ("ORIG: Standard RRF", orig_std),
        ("ORIG: Adaptive RRF", orig_adapt),
    ]:
        m = metrics(recs)
        print(f"{name:<35} {m['@1']:>8.4f} {m['@3']:>8.4f} {m['@5']:>8.4f} {m['MRR']:>8.4f}")

    print()
    # Rewritten
    for name, recs in [
        ("REWRITTEN: BM25 + LLM", rw_bm25),
        ("REWRITTEN: E5-large + LLM", rw_e5),
        ("REWRITTEN: Standard RRF + LLM", rw_std),
        ("REWRITTEN: Adaptive RRF + LLM", rw_adapt),
    ]:
        m = metrics(recs)
        print(f"{name:<35} {m['@1']:>8.4f} {m['@3']:>8.4f} {m['@5']:>8.4f} {m['MRR']:>8.4f}")

    # ============================================================
    # 8. Per-level breakdown
    # ============================================================
    print()
    print("=" * 100)
    print("PER-LEVEL @1 ON TEST SET")
    print("=" * 100)

    print(f"\n{'System':<35} " + " ".join(f"{lvl:>8}" for lvl in LEVELS))
    print("-" * 100)

    for name, recs in [
        ("ORIG: BM25", orig_bm25),
        ("ORIG: E5-large", orig_e5),
        ("ORIG: Adaptive RRF", orig_adapt),
        ("REWRITTEN: BM25 + LLM", rw_bm25),
        ("REWRITTEN: E5-large + LLM", rw_e5),
        ("REWRITTEN: Adaptive RRF + LLM", rw_adapt),
    ]:
        row = []
        for lvl in LEVELS:
            lvl_recs = [r for r in recs if r["level"] == lvl]
            m = metrics(lvl_recs)
            row.append(m["@1"])
        print(f"{name:<35} " + " ".join(f"{v:>8.4f}" for v in row))

    # ============================================================
    # 9. Key comparisons
    # ============================================================
    print()
    print("=" * 100)
    print("KEY COMPARISONS (TEST SET)")
    print("=" * 100)

    m_orig_e5 = metrics(orig_e5)
    m_rw_e5 = metrics(rw_e5)
    m_orig_adapt = metrics(orig_adapt)
    m_rw_adapt = metrics(rw_adapt)

    print()
    print("1. Best ORIGINAL system:")
    orig_systems = {
        "BM25": metrics(orig_bm25),
        "E5-large": m_orig_e5,
        "Standard RRF": metrics(orig_std),
        "Adaptive RRF": m_orig_adapt,
    }
    best_orig = max(orig_systems.items(), key=lambda x: x[1]["@1"])
    print(f"   {best_orig[0]}: @1={best_orig[1]['@1']:.4f}")

    print()
    print("2. Best REWRITTEN system:")
    rw_systems = {
        "BM25 + LLM": metrics(rw_bm25),
        "E5-large + LLM": m_rw_e5,
        "Standard RRF + LLM": metrics(rw_std),
        "Adaptive RRF + LLM": m_rw_adapt,
    }
    best_rw = max(rw_systems.items(), key=lambda x: x[1]["@1"])
    print(f"   {best_rw[0]}: @1={best_rw[1]['@1']:.4f}")

    print()
    print("3. LLM rewriting impact (same system, original vs rewritten):")
    print(f"   E5-large alone  : {m_orig_e5['@1']:.4f} → {m_rw_e5['@1']:.4f}  "
          f"(Δ = {m_rw_e5['@1'] - m_orig_e5['@1']:+.4f})")
    print(f"   Adaptive RRF    : {m_orig_adapt['@1']:.4f} → {m_rw_adapt['@1']:.4f}  "
          f"(Δ = {m_rw_adapt['@1'] - m_orig_adapt['@1']:+.4f})")

    print()
    print("4. Best-to-best:")
    print(f"   Original best   : {best_orig[0]} ({best_orig[1]['@1']:.4f})")
    print(f"   Rewritten best  : {best_rw[0]} ({best_rw[1]['@1']:.4f})")
    print(f"   Overall delta   : {best_rw[1]['@1'] - best_orig[1]['@1']:+.4f}")

    # ============================================================
    # 10. Save
    # ============================================================
    output = {
        "split_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "seed": SEED,
        "weights_tuned_on_train": {
            "original": {lvl: list(w) for lvl, w in weights_orig.items()},
            "rewritten": {lvl: list(w) for lvl, w in weights_rw.items()},
        },
        "test_original": {
            "bm25": metrics(orig_bm25),
            "e5_large": m_orig_e5,
            "standard_rrf": metrics(orig_std),
            "adaptive_rrf": m_orig_adapt,
        },
        "test_rewritten": {
            "bm25_llm": metrics(rw_bm25),
            "e5_large_llm": m_rw_e5,
            "standard_rrf_llm": metrics(rw_std),
            "adaptive_rrf_llm": m_rw_adapt,
        },
        "best_original": {
            "system": best_orig[0],
            "metrics": best_orig[1],
        },
        "best_rewritten": {
            "system": best_rw[0],
            "metrics": best_rw[1],
        },
    }

    save_json(RESULTS_FILE, output)

    print()
    print("=" * 100)
    print(f"Saved: {RESULTS_FILE}")
    print("=" * 100)


if __name__ == "__main__":
    main()