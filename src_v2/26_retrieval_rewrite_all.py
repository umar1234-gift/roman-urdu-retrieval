"""
Retrieval on ALL-Rewritten Queries (No Oracle Labels)
=====================================================
Runs BM25, E5-large, Standard RRF, and Weighted RRF on 
all_rewritten_queries.json.

Key difference from previous pipeline:
- No level labels used for routing
- Single global weight for Weighted RRF (not per-level)
- Realistic deployable scenario

Run: python src_v2/26_retrieval_rewrite_all.py
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
REWRITTEN_FILE = PROJECT_ROOT / "data_v2" / "processed" / "all_rewritten_queries.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "retrieval_all_rewritten.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "retrieval_all_rewritten.txt"

MODEL_NAME = "intfloat/multilingual-e5-large"
RRF_K = 10
LEVELS = ["L0", "L1", "L2", "L3", "L4"]

# Single global weight for Weighted RRF (no oracle labels)
GLOBAL_WEIGHT = (0.2, 0.8)  # from previous experiment


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
    total = len(records)
    if total == 0:
        return {"@1": 0, "@3": 0, "@5": 0, "MRR": 0, "n": 0}
    g1 = sum(1 for r in records if r["gold_rank"] == 1)
    g3 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 3)
    g5 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 5)
    mrr = sum(1 / r["gold_rank"] for r in records if r["gold_rank"]) / total
    return {"@1": g1/total, "@3": g3/total, "@5": g5/total, "MRR": mrr, "n": total}


def main():
    print("=" * 100)
    print("RETRIEVAL ON ALL-REWRITTEN QUERIES (NO ORACLE LABELS)")
    print("=" * 100)

    chunks = load_json(CHUNKS_FILE)
    queries = load_json(REWRITTEN_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]
    n_docs = len(filenames)

    print(f"Chunks          : {len(chunks)}")
    print(f"Queries         : {len(queries)}")
    print(f"RRF k           : {RRF_K}")
    print(f"Global weight   : {GLOBAL_WEIGHT}")

    # BM25
    print("\nBuilding BM25 index...")
    bm25 = BM25(texts)

    # E5-large
    print(f"Loading E5-large...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    print("Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=True)

    # Pre-compute rankings for rewritten queries
    print("\nComputing rankings for rewritten queries...")
    bm25_ranks = {}
    e5_ranks = {}

    for idx, q in enumerate(queries):
        qid = q["query_id"]
        text = q["rewritten_query"]

        bm25_ranking = bm25.rank(text)
        bm25_ranks[qid] = {i: r for r, (i, _) in enumerate(bm25_ranking, start=1)}

        q_emb = model.encode(["query: " + text], normalize_embeddings=True)[0]
        sims = np.dot(doc_emb, q_emb)
        e5_order = np.argsort(sims)[::-1]
        e5_ranks[qid] = {i: r for r, i in enumerate(e5_order, start=1)}

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(queries)}")

    print("Done.\n")

    # Evaluate each system
    systems = {
        "BM25": [],
        "E5-large": [],
        "Standard RRF (1.0/1.0)": [],
        "Weighted RRF (0.2/0.8)": [],
    }

    for q in queries:
        qid = q["query_id"]
        level = q["level"]
        gold_doc = q["gold_document"]
        bm25_rank = bm25_ranks[qid]
        e5_rank = e5_ranks[qid]

        # BM25 alone
        ranked_bm25 = sorted(bm25_rank.items(), key=lambda x: x[1])
        for r, (i, _) in enumerate(ranked_bm25, start=1):
            if filenames[i] == gold_doc:
                systems["BM25"].append({"query_id": qid, "level": level, "gold_rank": r})
                break

        # E5-large alone
        ranked_e5 = sorted(e5_rank.items(), key=lambda x: x[1])
        for r, (i, _) in enumerate(ranked_e5, start=1):
            if filenames[i] == gold_doc:
                systems["E5-large"].append({"query_id": qid, "level": level, "gold_rank": r})
                break

        # Standard RRF
        rrf_scores = {}
        for i in range(n_docs):
            rrf_scores[i] = (1.0/(RRF_K + bm25_rank[i])) + (1.0/(RRF_K + e5_rank[i]))
        ranked_std = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        for r, (i, _) in enumerate(ranked_std, start=1):
            if filenames[i] == gold_doc:
                systems["Standard RRF (1.0/1.0)"].append({"query_id": qid, "level": level, "gold_rank": r})
                break

        # Weighted RRF (single global weight)
        w_b, w_e = GLOBAL_WEIGHT
        rrf_scores_w = {}
        for i in range(n_docs):
            rrf_scores_w[i] = (w_b/(RRF_K + bm25_rank[i])) + (w_e/(RRF_K + e5_rank[i]))
        ranked_w = sorted(rrf_scores_w.items(), key=lambda x: x[1], reverse=True)
        for r, (i, _) in enumerate(ranked_w, start=1):
            if filenames[i] == gold_doc:
                systems["Weighted RRF (0.2/0.8)"].append({"query_id": qid, "level": level, "gold_rank": r})
                break

    # Report
    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out()
    out("=" * 100)
    out("RESULTS ON ALL-REWRITTEN QUERIES (n=450)")
    out("=" * 100)
    out(f"\n{'System':<30} {'@1':>8} {'@3':>8} {'@5':>8} {'MRR':>8}")
    out("-" * 100)

    for name, recs in systems.items():
        m = metrics(recs)
        out(f"{name:<30} {m['@1']:>8.4f} {m['@3']:>8.4f} {m['@5']:>8.4f} {m['MRR']:>8.4f}")

    # Per-level
    out()
    out("=" * 100)
    out("PER-LEVEL @1")
    out("=" * 100)
    out(f"{'System':<30} " + " ".join(f"{lvl:>8}" for lvl in LEVELS))
    out("-" * 100)

    for name, recs in systems.items():
        row = []
        for lvl in LEVELS:
            lvl_recs = [r for r in recs if r["level"] == lvl]
            row.append(metrics(lvl_recs)["@1"])
        out(f"{name:<30} " + " ".join(f"{v:>8.4f}" for v in row))

    # Comparison with previous (oracle) results
    out()
    out("=" * 100)
    out("COMPARISON: PREVIOUS (ORACLE) vs NEW (ALL-REWRITTEN)")
    out("=" * 100)
    out()
    out("Previous (L2/L3/L4 only, oracle labels):")
    out("  Best on test set: E5-large + LLM = 98.89% @1")
    out()
    out(f"New (all 450 rewritten, no oracle labels):")
    best = max(systems.items(), key=lambda x: metrics(x[1])["@1"])
    m = metrics(best[1])
    out(f"  Best: {best[0]} = {m['@1']:.4f} @1")
    out()
    out("This is the HONEST, deployable result.")
    out("It reflects what a real system (without level labels) would achieve.")

    # Save
    save_json(RESULTS_FILE, {
        "systems": {name: metrics(recs) for name, recs in systems.items()},
        "per_query": systems,
    })

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    out()
    out(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()