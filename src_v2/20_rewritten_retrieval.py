from pathlib import Path
import json
import math
import re
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"
REWRITTEN_FILE = PROJECT_ROOT / "data_v2" / "processed" / "rewritten_queries.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "rewritten_retrieval.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "rewritten_retrieval.txt"

MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K = 10
RRF_K = 10
LEVELS = ["L0", "L1", "L2", "L3", "L4"]

ADAPTIVE_WEIGHTS = {
    "L0": (1.0, 1.0),
    "L1": (1.0, 1.0),
    "L2": (0.2, 0.8),
    "L3": (0.2, 0.8),
    "L4": (0.1, 0.9),
}


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
    g1 = sum(1 for r in records if r["gold_rank"] == 1)
    g3 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 3)
    g5 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 5)
    mrr = sum(1/r["gold_rank"] for r in records if r["gold_rank"]) / total
    return {"@1": g1/total, "@3": g3/total, "@5": g5/total, "MRR": mrr}


def main():
    chunks = load_json(CHUNKS_FILE)
    queries = load_json(REWRITTEN_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]

    print("=" * 100)
    print("REWRITTEN QUERY RETRIEVAL (Adaptive RRF)")
    print("=" * 100)
    print(f"Chunks : {len(chunks)}")
    print(f"Queries: {len(queries)}")

    print("\nBuilding BM25 index...")
    bm25 = BM25(texts)

    print(f"Loading E5-large...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    print("Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True,
                            show_progress_bar=True)

    print("\nRunning retrieval...")
    results = []

    for idx, q in enumerate(queries):
        query_text = q["query"]

        bm25_ranking = bm25.rank(query_text)
        bm25_rank = {i: r for r, (i, _) in enumerate(bm25_ranking, start=1)}

        q_emb = model.encode(["query: " + query_text],
                             normalize_embeddings=True)[0]
        sims = np.dot(doc_emb, q_emb)
        e5_order = np.argsort(sims)[::-1]
        e5_rank = {i: r for r, i in enumerate(e5_order, start=1)}

        w_b, w_e = ADAPTIVE_WEIGHTS[q["level"]]
        scores = {}
        for i in range(len(chunks)):
            r_b = bm25_rank[i]
            r_e = e5_rank[i]
            scores[i] = (w_b / (RRF_K + r_b)) + (w_e / (RRF_K + r_e))
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        gold_rank = None
        for r, (i, _) in enumerate(ranked, start=1):
            if filenames[i] == q["gold_document"]:
                gold_rank = r
                break

        results.append({
            "query_id": q["query_id"],
            "doc_id": q["doc_id"],
            "level": q["level"],
            "variant": q["variant"],
            "original_query": q["original_query"],
            "rewritten_query": q["query"],
            "was_rewritten": q["was_rewritten"],
            "gold_document": q["gold_document"],
            "gold_rank": gold_rank
        })

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(queries)}")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out()
    out("=" * 100)
    out("REWRITTEN RETRIEVAL — RESULTS")
    out("=" * 100)

    overall = metrics(results)
    out(f"\nOVERALL:")
    out(f"  @1  = {overall['@1']:.4f}")
    out(f"  @3  = {overall['@3']:.4f}")
    out(f"  @5  = {overall['@5']:.4f}")
    out(f"  MRR = {overall['MRR']:.4f}")

    out(f"\nPER-LEVEL (@1):")
    for lvl in LEVELS:
        subset = [r for r in results if r["level"] == lvl]
        m = metrics(subset)
        out(f"  {lvl}: @1={m['@1']:.4f} @3={m['@3']:.4f} MRR={m['MRR']:.4f}")

    out()
    out("=" * 100)
    out("COMPARISON: ORIGINAL vs REWRITTEN (Adaptive RRF)")
    out("=" * 100)
    out(f"{'Level':<6} {'Original @1':>12} {'Rewritten @1':>14} {'Δ':>10}")
    out("-" * 100)

    original_p1 = {"L0": 0.9889, "L1": 0.9889, "L2": 1.0000,
                   "L3": 0.8667, "L4": 0.7889}

    for lvl in LEVELS:
        subset = [r for r in results if r["level"] == lvl]
        m = metrics(subset)
        orig = original_p1[lvl]
        delta = m["@1"] - orig
        out(f"{lvl:<6} {orig:>12.4f} {m['@1']:>14.4f} {delta:>+10.4f}")

    out()
    out(f"Overall Original @1 : 0.9267")
    out(f"Overall Rewritten @1: {overall['@1']:.4f}")
    out(f"Improvement         : {overall['@1'] - 0.9267:+.4f}")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    out()
    out(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()