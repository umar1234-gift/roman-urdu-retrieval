from pathlib import Path
import json
import math
import re
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "rrf_sensitivity.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "rrf_sensitivity.txt"

MODEL_NAME = "intfloat/multilingual-e5-base"
TOP_K = 10
K_VALUES = [10, 20, 30, 40, 50, 60, 80, 100]
LEVELS = ["L0", "L1", "L2", "L3", "L4"]


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
    g10 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 10)
    mrr = sum(1/r["gold_rank"] for r in records if r["gold_rank"]) / total
    return {"@1": g1/total, "@3": g3/total, "@5": g5/total,
            "@10": g10/total, "MRR": mrr}


def main():
    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]

    print("=" * 100)
    print("RRF k SENSITIVITY ANALYSIS")
    print("=" * 100)
    print(f"Chunks : {len(chunks)}")
    print(f"Queries: {len(queries)}")
    print(f"k test : {K_VALUES}\n")

    # BM25
    print("Building BM25 index...")
    bm25 = BM25(texts)

    # E5
    print(f"Loading E5...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    print("Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True,
                            show_progress_bar=True)

    # Pre-compute BM25 and E5 rankings per query (only once)
    print("\nPre-computing rankings for all queries...")
    q_data = []
    for idx, q in enumerate(queries):
        bm25_ranking = bm25.rank(q["query"])
        bm25_rank_by_idx = {i: r for r, (i, _) in enumerate(bm25_ranking, start=1)}

        q_emb = model.encode(["query: " + q["query"]],
                             normalize_embeddings=True)[0]
        sims = np.dot(doc_emb, q_emb)
        e5_order = np.argsort(sims)[::-1]
        e5_rank_by_idx = {i: r for r, i in enumerate(e5_order, start=1)}

        q_data.append({
            "query": q,
            "bm25_rank": bm25_rank_by_idx,
            "e5_rank": e5_rank_by_idx
        })

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(queries)}")

    print("Done.\n")

    # Now evaluate for each k
    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out("=" * 100)
    out("RRF k SENSITIVITY RESULTS")
    out("=" * 100)

    all_results = {}

    for k in K_VALUES:
        print(f"Evaluating k = {k}...")

        results = []
        for item in q_data:
            q = item["query"]
            bm25_rank_by_idx = item["bm25_rank"]
            e5_rank_by_idx = item["e5_rank"]

            rrf_scores = {}
            for i in range(len(chunks)):
                r_b = bm25_rank_by_idx[i]
                r_e = e5_rank_by_idx[i]
                rrf_scores[i] = (1.0 / (k + r_b)) + (1.0 / (k + r_e))

            ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            gold_rank = None
            for r, (i, _) in enumerate(ranked, start=1):
                if filenames[i] == q["gold_document"]:
                    gold_rank = r
                    break

            results.append({
                "query_id": q["query_id"],
                "level": q["level"],
                "doc_id": q["doc_id"],
                "gold_rank": gold_rank
            })

        all_results[k] = results

    # Print table
    out()
    out("OVERALL METRICS BY k")
    out("-" * 100)
    out(f"{'k':<6} {'@1':>8} {'@3':>8} {'@5':>8} {'@10':>8} {'MRR':>8}")
    out("-" * 100)

    for k in K_VALUES:
        m = metrics(all_results[k])
        out(f"{k:<6} {m['@1']:>8.4f} {m['@3']:>8.4f} {m['@5']:>8.4f} {m['@10']:>8.4f} {m['MRR']:>8.4f}")

    # Per level @1
    out()
    out("PER-LEVEL @1 BY k")
    out("-" * 100)
    out(f"{'k':<6} " + " ".join(f"{lvl:>8}" for lvl in LEVELS))
    out("-" * 100)

    for k in K_VALUES:
        row = []
        for lvl in LEVELS:
            subset = [r for r in all_results[k] if r["level"] == lvl]
            row.append(metrics(subset)["@1"])
        out(f"{k:<6} " + " ".join(f"{v:>8.4f}" for v in row))

    # Find best k
    best_k_mrr = max(K_VALUES, key=lambda k: metrics(all_results[k])["MRR"])
    best_k_p1 = max(K_VALUES, key=lambda k: metrics(all_results[k])["@1"])

    out()
    out("=" * 100)
    out("BEST VALUES")
    out("=" * 100)
    out(f"Best k by MRR : {best_k_mrr}")
    out(f"Best k by @1  : {best_k_p1}")
    out(f"Current k     : 60")

    # Save
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({str(k): all_results[k] for k in K_VALUES}, f, indent=2, ensure_ascii=False)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    out()
    out(f"Saved: {RESULTS_FILE}")
    out(f"Saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()