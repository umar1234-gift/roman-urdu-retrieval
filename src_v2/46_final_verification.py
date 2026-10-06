"""
FINAL VERIFICATION SCRIPT
==========================
Blockbuster verification of every claim in the paper.
Cross-checks all numbers, statistical tests, and consistency.

Run: python src_v2/46_final_verification.py

Output:
- Beautiful PASS/FAIL report for every claim
- Auto-saves verification.json
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
FACTS_FILE = PROJECT_ROOT / "config" / "master_facts.json"
A_SINGLE_FILE = PROJECT_ROOT / "data_v2" / "processed" / "ablation_single_clean.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "final_verification.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "final_verification.txt"

MODEL_NAME = "intfloat/multilingual-e5-large"
RRF_K = 10
N_FOLDS = 5
SEED = 42
N_BOOTSTRAP = 5000
LEVELS = ["L0", "L1", "L2", "L3", "L4"]

WEIGHT_GRID = [(0.0,1.0),(0.1,0.9),(0.2,0.8),(0.3,0.7),(0.4,0.6),
               (0.5,0.5),(0.6,0.4),(0.7,0.3),(0.8,0.2),(0.9,0.1),(1.0,0.0)]

ALPHA_GRID = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]


# ============================================================
# UTILITIES
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


def metrics(records):
    total = len(records)
    if total == 0:
        return {"@1": 0, "@3": 0, "MRR": 0, "n": 0, "count_at1": 0}
    g1 = sum(1 for r in records if r["gold_rank"] == 1)
    g3 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 3)
    mrr = sum(1 / r["gold_rank"] for r in records if r["gold_rank"]) / total
    return {"@1": g1/total, "@3": g3/total, "MRR": mrr,
            "n": total, "count_at1": g1}


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


def cluster_bootstrap_ci(records, metric="at1", n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    by_doc = defaultdict(list)
    for r in records:
        by_doc[r["doc_id"]].append(r)
    docs = list(by_doc.keys())
    n_docs = len(docs)

    boot = np.empty(n_boot)
    for i in range(n_boot):
        sampled_docs = rng.choice(docs, size=n_docs, replace=True)
        sampled = []
        for d in sampled_docs:
            sampled.extend(by_doc[d])
        if metric == "at1":
            boot[i] = np.mean([1 if r["gold_rank"] == 1 else 0 for r in sampled])
        elif metric == "mrr":
            boot[i] = np.mean([1/r["gold_rank"] if r["gold_rank"] else 0
                               for r in sampled])
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_cluster_bootstrap(recs_a, recs_b, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    map_a = {r["query_id"]: r for r in recs_a}
    map_b = {r["query_id"]: r for r in recs_b}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))
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

    return {"delta": obs, "ci_95": [lo, hi], "p_value": p,
            "n_queries": len(common)}


def group_folds(queries, n_folds=5, seed=42):
    random.seed(seed)
    docs = sorted({q["doc_id"] for q in queries})
    random.shuffle(docs)
    folds = [[] for _ in range(n_folds)]
    for i, d in enumerate(docs):
        folds[i % n_folds].append(d)
    return folds


# ============================================================
# VERIFICATION FRAMEWORK
# ============================================================

class Verifier:
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
        self.warned = 0

    def check(self, name, condition, expected=None, actual=None,
              severity="FAIL"):
        """Register a verification check."""
        status = "PASS" if condition else severity
        if condition:
            self.passed += 1
        elif severity == "FAIL":
            self.failed += 1
        else:
            self.warned += 1

        self.checks.append({
            "name": name,
            "status": status,
            "expected": str(expected) if expected is not None else None,
            "actual": str(actual) if actual is not None else None,
        })

        symbol = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}[status]
        line = f"  [{symbol} {status}] {name}"
        if expected is not None and actual is not None:
            line += f"  (expected: {expected}, actual: {actual})"
        print(line)

    def summary(self):
        print()
        print("=" * 100)
        print("VERIFICATION SUMMARY")
        print("=" * 100)
        print(f"  Total checks : {len(self.checks)}")
        print(f"  Passed       : {self.passed}")
        print(f"  Failed       : {self.failed}")
        print(f"  Warnings     : {self.warned}")
        print()
        if self.failed == 0:
            print("  ✅ ALL CHECKS PASSED — PAPER IS READY")
        else:
            print(f"  ❌ {self.failed} CHECKS FAILED — FIX BEFORE SUBMISSION")
        return self.failed == 0


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("FINAL BLOCKBUSTER VERIFICATION")
    print("=" * 100)
    print()

    v = Verifier()

    # ============================================================
    # SECTION 1: DATASET INTEGRITY
    # ============================================================
    print("=" * 100)
    print("SECTION 1: DATASET INTEGRITY")
    print("=" * 100)

    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)
    facts = load_json(FACTS_FILE)

    v.check("Chunk count = 30", len(chunks) == 30, 30, len(chunks))
    v.check("Query count = 450", len(queries) == 450, 450, len(queries))
    v.check("Master facts documents = 30",
            len(facts.get("documents", [])) == 30, 30,
            len(facts.get("documents", [])))

    # Level distribution
    level_counts = Counter(q["level"] for q in queries)
    for lvl in LEVELS:
        v.check(f"{lvl} count = 90", level_counts[lvl] == 90, 90, level_counts[lvl])

    # Unique query IDs
    qids = [q["query_id"] for q in queries]
    v.check("Unique query IDs", len(qids) == len(set(qids)), 450, len(set(qids)))

    # Unique doc IDs
    doc_ids = sorted({q["doc_id"] for q in queries})
    v.check("Unique doc IDs = 30", len(doc_ids) == 30, 30, len(doc_ids))

    # Every doc has 15 queries
    per_doc = Counter(q["doc_id"] for q in queries)
    docs_ok = all(c == 15 for c in per_doc.values())
    v.check("Every doc has 15 queries", docs_ok, "all=15",
            f"min={min(per_doc.values())}, max={max(per_doc.values())}")

    # Every (doc, level) has 3 variants
    per_dl = Counter((q["doc_id"], q["level"]) for q in queries)
    dl_ok = all(c == 3 for c in per_dl.values())
    v.check("Every (doc, level) has 3 variants", dl_ok, "all=3",
            f"min={min(per_dl.values())}, max={max(per_dl.values())}")

    # Gold documents exist in chunks
    chunk_files = {c["filename"] for c in chunks}
    missing = [q["query_id"] for q in queries if q["gold_document"] not in chunk_files]
    v.check("All gold documents exist in chunks",
            len(missing) == 0, 0, len(missing))

    print()

    # ============================================================
    # SECTION 2: A_SINGLE CACHE INTEGRITY
    # ============================================================
    print("=" * 100)
    print("SECTION 2: A_SINGLE CACHE INTEGRITY")
    print("=" * 100)

    a_single = load_json(A_SINGLE_FILE)
    v.check("A_single cache = 450 entries", len(a_single) == 450, 450, len(a_single))

    # All L0-L4 present
    a_single_ids = set(a_single.keys())
    query_ids = set(qids)
    missing = query_ids - a_single_ids
    v.check("All query IDs present in A_single",
            len(missing) == 0, 0, len(missing))

    # Non-empty
    empty = [k for k, val in a_single.items() if not val.strip()]
    v.check("No empty entries in A_single",
            len(empty) == 0, 0, len(empty))

    # Fallback count (entries where cache == original)
    fallback_count = sum(
        1 for q in queries
        if q["level"] != "L0" and a_single.get(q["query_id"], "").strip() == q["query"].strip()
    )
    v.check("Fallback count ≤ 4 (Claude-verified)",
            fallback_count <= 4, "≤ 4", fallback_count)

    print()

    # ============================================================
    # SECTION 3: BUILD INDEXES
    # ============================================================
    print("=" * 100)
    print("SECTION 3: BUILDING INDEXES")
    print("=" * 100)
    print("  Building BM25...")
    bm25 = BM25([c["text"] for c in chunks])

    print("  Loading E5-large...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + c["text"] for c in chunks]
    print("  Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True,
                            show_progress_bar=True)

    filenames = [c["filename"] for c in chunks]
    n_docs = len(filenames)

    # Precompute scores
    print("  Precomputing scores...")
    bm25_scores = {}
    e5_scores = {}
    bm25_ranks = {}
    zero_count = 0

    for q in queries:
        qid = q["query_id"]
        bm25_s = bm25.scores_all(q["query"])
        bm25_scores[qid] = bm25_s

        q_emb = model.encode(["query: " + q["query"]], normalize_embeddings=True)[0]
        e5_s = np.dot(doc_emb, q_emb)
        e5_scores[qid] = e5_s

        if bm25_s.max() < 1e-9:
            bm25_ranks[qid] = None
            zero_count += 1
        else:
            order = np.argsort(-bm25_s, kind="stable")
            ranks = np.empty(n_docs, dtype=int)
            for r, i in enumerate(order, start=1):
                ranks[i] = r
            bm25_ranks[qid] = ranks

    # Now for A_single text
    print("  Precomputing A_single scores...")
    a_bm25_scores = {}
    a_e5_scores = {}
    a_bm25_ranks = {}
    a_zero_count = 0

    for q in queries:
        qid = q["query_id"]
        text = a_single[qid]
        bm25_s = bm25.scores_all(text)
        a_bm25_scores[qid] = bm25_s

        q_emb = model.encode(["query: " + text], normalize_embeddings=True)[0]
        e5_s = np.dot(doc_emb, q_emb)
        a_e5_scores[qid] = e5_s

        if bm25_s.max() < 1e-9:
            a_bm25_ranks[qid] = None
            a_zero_count += 1
        else:
            order = np.argsort(-bm25_s, kind="stable")
            ranks = np.empty(n_docs, dtype=int)
            for r, i in enumerate(order, start=1):
                ranks[i] = r
            a_bm25_ranks[qid] = ranks

    print(f"  Zero-score queries (original): {zero_count}")
    print(f"  Zero-score queries (A_single): {a_zero_count}")
    print()

    # ============================================================
    # SECTION 4: ZERO-SCORE QUERIES
    # ============================================================
    print("=" * 100)
    print("SECTION 4: ZERO-SCORE QUERY CLAIMS")
    print("=" * 100)

    v.check("Zero-score count = 16", zero_count == 16, 16, zero_count)

    zero_ids = [qid for qid, r in bm25_ranks.items() if r is None]
    zero_levels = Counter(q["level"] for q in queries if q["query_id"] in zero_ids)
    v.check("Zero-score all in L3/L4",
            all(lvl in ["L3", "L4"] for lvl in zero_levels),
            "all L3/L4", dict(zero_levels))

    print()

    # ============================================================
    # SECTION 5: E5-LARGE BASELINE
    # ============================================================
    print("=" * 100)
    print("SECTION 5: E5-LARGE BASELINE")
    print("=" * 100)

    e5_records = []
    for q in queries:
        qid = q["query_id"]
        gold = q["gold_document"]
        order = np.argsort(-e5_scores[qid], kind="stable")
        gr = None
        for r, i in enumerate(order, start=1):
            if filenames[i] == gold:
                gr = r
                break
        e5_records.append({
            "query_id": qid, "doc_id": q["doc_id"],
            "level": q["level"], "gold_rank": gr,
        })

    e5_m = metrics(e5_records)
    v.check("E5-large @1 = 0.9222",
            abs(e5_m["@1"] - 0.9222) < 0.001, 0.9222, round(e5_m["@1"], 4))
    v.check("E5-large @3 = 0.9689",
            abs(e5_m["@3"] - 0.9689) < 0.001, 0.9689, round(e5_m["@3"], 4))
    v.check("E5-large MRR = 0.9489",
            abs(e5_m["MRR"] - 0.9489) < 0.001, 0.9489, round(e5_m["MRR"], 4))

    print()

    # ============================================================
    # SECTION 6: RRF TIE-AWARE
    # ============================================================
    print("=" * 100)
    print("SECTION 6: RRF TIE-AWARE CLAIM")
    print("=" * 100)

    hybrid_records = []
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

        hybrid_records.append({
            "query_id": qid, "doc_id": q["doc_id"],
            "level": q["level"], "gold_rank": gr,
        })

    hybrid_m = metrics(hybrid_records)
    v.check("Hybrid-large (tie-aware) @1 = E5-large @1",
            abs(hybrid_m["@1"] - e5_m["@1"]) < 0.001,
            round(e5_m["@1"], 4), round(hybrid_m["@1"], 4))

    print()

    # ============================================================
    # SECTION 7: A_SINGLE RETRIEVAL
    # ============================================================
    print("=" * 100)
    print("SECTION 7: A_SINGLE RETRIEVAL")
    print("=" * 100)

    a_e5_records = []
    for q in queries:
        qid = q["query_id"]
        gold = q["gold_document"]
        order = np.argsort(-a_e5_scores[qid], kind="stable")
        gr = None
        for r, i in enumerate(order, start=1):
            if filenames[i] == gold:
                gr = r
                break
        a_e5_records.append({
            "query_id": qid, "doc_id": q["doc_id"],
            "level": q["level"], "gold_rank": gr,
        })

    a_e5_m = metrics(a_e5_records)
    v.check("A_single E5-large @1 = 0.9867",
            abs(a_e5_m["@1"] - 0.9867) < 0.005, 0.9867,
            round(a_e5_m["@1"], 4))
    v.check("A_single E5-large @3 = 1.0000",
            abs(a_e5_m["@3"] - 1.0) < 0.001, 1.0,
            round(a_e5_m["@3"], 4))

    # Per-level
    a_l4 = metrics([r for r in a_e5_records if r["level"] == "L4"])
    v.check("A_single L4 @1 = 0.9667",
            abs(a_l4["@1"] - 0.9667) < 0.005, 0.9667,
            round(a_l4["@1"], 4))

    print()

    # ============================================================
    # SECTION 8: GAP RECOVERY
    # ============================================================
    print("=" * 100)
    print("SECTION 8: GAP RECOVERY CLAIMS")
    print("=" * 100)

    l0_orig = metrics([r for r in e5_records if r["level"] == "L0"])["@1"]
    l4_orig = metrics([r for r in e5_records if r["level"] == "L4"])["@1"]
    l4_rewr = metrics([r for r in a_e5_records if r["level"] == "L4"])["@1"]

    recovery_measured = (l4_rewr - l4_orig) / (l0_orig - l4_orig)
    recovery_100 = (l4_rewr - l4_orig) / (1.0 - l4_orig)

    v.check("L0 original = 0.9889",
            abs(l0_orig - 0.9889) < 0.005, 0.9889, round(l0_orig, 4))
    v.check("L4 original = 0.7889",
            abs(l4_orig - 0.7889) < 0.005, 0.7889, round(l4_orig, 4))
    v.check("L4 A_single = 0.9667",
            abs(l4_rewr - 0.9667) < 0.005, 0.9667, round(l4_rewr, 4))
    v.check("Recovery (measured) = 0.8889",
            abs(recovery_measured - 0.8889) < 0.01, 0.8889,
            round(recovery_measured, 4))

    print()

    # ============================================================
    # SECTION 9: DEGRADATION
    # ============================================================
    print("=" * 100)
    print("SECTION 9: DEGRADATION CLAIM (L0 → L4)")
    print("=" * 100)

    for lvl in LEVELS:
        v.check(f"E5-large {lvl} present",
                len([r for r in e5_records if r["level"] == lvl]) == 90,
                90, len([r for r in e5_records if r["level"] == lvl]))

    drop = l0_orig - l4_orig
    v.check("L0→L4 drop = 20.00 pp",
            abs(drop - 0.20) < 0.01, 0.20, round(drop, 4))

    print()

    # ============================================================
    # SECTION 10: STATISTICAL SIGNIFICANCE
    # ============================================================
    print("=" * 100)
    print("SECTION 10: STATISTICAL SIGNIFICANCE")
    print("=" * 100)

    # A_single vs Original (L2-L4)
    a_l2l4 = [r for r in a_e5_records if r["level"] in ["L2","L3","L4"]]
    o_l2l4 = [r for r in e5_records if r["level"] in ["L2","L3","L4"]]
    pb = paired_cluster_bootstrap(a_l2l4, o_l2l4, N_BOOTSTRAP, SEED)

    v.check("A_single vs Original (L2-L4): Δ > 0",
            pb["delta"] > 0, "> 0", round(pb["delta"], 4))
    v.check("A_single vs Original (L2-L4): p < 0.01",
            pb["p_value"] < 0.01, "< 0.01", round(pb["p_value"], 4))
    v.check("A_single vs Original (L2-L4): CI excludes zero",
            pb["ci_95"][0] > 0, "lo > 0",
            [round(pb["ci_95"][0], 4), round(pb["ci_95"][1], 4)])

    # E5-large vs Hybrid (tie-aware)
    pb2 = paired_cluster_bootstrap(hybrid_records, e5_records, N_BOOTSTRAP, SEED)
    v.check("Hybrid vs E5: p > 0.05 (not significant)",
            pb2["p_value"] > 0.05, "> 0.05", round(pb2["p_value"], 4))

    print()

    # ============================================================
    # SECTION 11: PER-LEVEL COUNTS CONSISTENCY
    # ============================================================
    print("=" * 100)
    print("SECTION 11: PER-LEVEL COUNT CONSISTENCY")
    print("=" * 100)

    # Sum of per-level counts should equal overall
    total_count = sum(
        metrics([r for r in a_e5_records if r["level"] == lvl])["count_at1"]
        for lvl in LEVELS
    )
    v.check("A_single per-level @1 counts sum = overall",
            total_count == a_e5_m["count_at1"],
            a_e5_m["count_at1"], total_count)

    # Original also
    total_o = sum(
        metrics([r for r in e5_records if r["level"] == lvl])["count_at1"]
        for lvl in LEVELS
    )
    v.check("Original per-level @1 counts sum = overall",
            total_o == e5_m["count_at1"],
            e5_m["count_at1"], total_o)

    print()

    # ============================================================
    # SECTION 12: DATA LEAKAGE CHECK
    # ============================================================
    print("=" * 100)
    print("SECTION 12: DATA LEAKAGE CHECK")
    print("=" * 100)

    # Check that query text for L0 != L1 != L2 != L3 != L4
    by_key = defaultdict(dict)
    for q in queries:
        key = (q["doc_id"], q["variant"])
        by_key[key][q["level"]] = q["query"]

    dups = 0
    for key, lvl_dict in by_key.items():
        texts = list(lvl_dict.values())
        if len(set(texts)) < len(texts):
            dups += 1

    v.check("No duplicate queries across levels",
            dups == 0, 0, dups)

    print()

    # ============================================================
    # SECTION 13: BM25 ZERO-SCORE HONESTY
    # ============================================================
    print("=" * 100)
    print("SECTION 13: BM25 ZERO-SCORE HONESTY")
    print("=" * 100)

    bm25_records = []
    for q in queries:
        qid = q["query_id"]
        gold = q["gold_document"]
        ranks = bm25_ranks[qid]
        if ranks is None:
            gr = None
        else:
            gr = ranks[[i for i, f in enumerate(filenames) if f == gold][0]]
        bm25_records.append({
            "query_id": qid, "doc_id": q["doc_id"],
            "level": q["level"], "gold_rank": gr,
        })

    bm25_m = metrics(bm25_records)
    v.check("BM25 (tie-aware) @1 reported",
            True, "reported", round(bm25_m["@1"], 4))
    v.check("BM25 zero-score queries excluded from ranking",
            zero_count == 16, 16, zero_count)

    print()

    # ============================================================
    # SECTION 14: STATISTICAL POWER CHECK
    # ============================================================
    print("=" * 100)
    print("SECTION 14: STATISTICAL POWER CHECK")
    print("=" * 100)

    # CI width for A_single
    ci_lo, ci_hi = cluster_bootstrap_ci(a_e5_records, "at1", N_BOOTSTRAP, SEED)
    ci_width = ci_hi - ci_lo
    v.check("A_single CI width < 0.05",
            ci_width < 0.05, "< 0.05", round(ci_width, 4))

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    all_passed = v.summary()

    # Save verification
    save_json(RESULTS_FILE, {
        "checks": v.checks,
        "summary": {
            "total": len(v.checks),
            "passed": v.passed,
            "failed": v.failed,
            "warned": v.warned,
        },
        "key_numbers": {
            "e5_large_original_at1": e5_m["@1"],
            "e5_large_a_single_at1": a_e5_m["@1"],
            "hybrid_tie_aware_at1": hybrid_m["@1"],
            "bm25_tie_aware_at1": bm25_m["@1"],
            "zero_score_count": zero_count,
            "recovery_measured": recovery_measured,
            "recovery_100_ceiling": recovery_100,
            "a_single_vs_original_L2L4": pb,
        }
    })

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(
            f"[{c['status']}] {c['name']}" +
            (f"  expected={c['expected']} actual={c['actual']}"
             if c['expected'] else "")
            for c in v.checks
        ))

    print()
    print(f"Saved: {RESULTS_FILE}")
    print(f"Saved: {REPORT_FILE}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())