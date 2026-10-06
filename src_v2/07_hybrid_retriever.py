from pathlib import Path
import json
import math
import re
import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "hybrid_results.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "hybrid_report.txt"

MODEL_NAME = "intfloat/multilingual-e5-base"
TOP_K = 10
RRF_K = 60
LEVELS = ["L0", "L1", "L2", "L3", "L4"]


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- BM25 ----------
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


def compute_metrics(records):
    total = len(records)
    g1 = sum(1 for r in records if r["gold_rank"] == 1)
    g3 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 3)
    g5 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 5)
    g10 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 10)
    nf = sum(1 for r in records if r["gold_rank"] is None)
    mrr = sum(1/r["gold_rank"] for r in records if r["gold_rank"]) / total
    return {
        "total": total,
        "gold_at_1": g1, "gold_at_3": g3, "gold_at_5": g5, "gold_at_10": g10,
        "not_found": nf,
        "p1": g1/total, "p3": g3/total, "p5": g5/total, "p10": g10/total,
        "mrr": mrr
    }


def main():
    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]

    print("=" * 100)
    print("HYBRID RETRIEVAL (BM25 + E5, RRF)")
    print("=" * 100)
    print(f"Chunks : {len(chunks)}")
    print(f"Queries: {len(queries)}")
    print(f"RRF k  : {RRF_K}\n")

    # BM25
    print("Building BM25 index...")
    bm25 = BM25(texts)

    # E5
    print(f"Loading E5 ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    print("Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True,
                            show_progress_bar=True)
    print("Ready.\n")

    results = []
    report_lines = ["=" * 100, "HYBRID RETRIEVAL (RRF)", "=" * 100]
    print("-" * 100)
    print("PER-QUERY RESULTS")
    print("-" * 100)

    for q in queries:
        # BM25 ranking (by chunk index)
        bm25_ranking = bm25.rank(q["query"])
        bm25_rank_by_idx = {idx: r for r, (idx, _) in
                            enumerate(bm25_ranking, start=1)}

        # E5 ranking
        q_emb = model.encode(["query: " + q["query"]],
                             normalize_embeddings=True)[0]
        sims = np.dot(doc_emb, q_emb)
        e5_order = np.argsort(sims)[::-1]
        e5_rank_by_idx = {idx: r for r, idx in enumerate(e5_order, start=1)}

        # RRF over chunk indices
        rrf_scores = {}
        for idx in range(len(chunks)):
            r_b = bm25_rank_by_idx[idx]
            r_e = e5_rank_by_idx[idx]
            rrf_scores[idx] = (1.0 / (RRF_K + r_b)) + (1.0 / (RRF_K + r_e))

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        gold_rank = None
        for r, (idx, _) in enumerate(ranked, start=1):
            if filenames[idx] == q["gold_document"]:
                gold_rank = r
                break

        top_results = [
            {"rank": r, "filename": filenames[i],
             "rrf_score": round(s, 8)}
            for r, (i, s) in enumerate(ranked[:TOP_K], start=1)
        ]
        top1 = top_results[0]["filename"]
        correct_top1 = (top1 == q["gold_document"])

        record = {
            "query_id": q["query_id"], "doc_id": q["doc_id"],
            "level": q["level"], "variant": q["variant"],
            "query": q["query"], "gold_answer": q["gold_answer"],
            "gold_document": q["gold_document"],
            "gold_rank": gold_rank,
            "top1_filename": top1, "top1_correct": correct_top1,
            "top_results": top_results
        }
        results.append(record)

        line = (f"{q['query_id']} {q['doc_id']} {q['level']} V{q['variant']} | "
                f"GoldRank={gold_rank if gold_rank else 'NF'} | "
                f"Top1={top1} | {'✓' if correct_top1 else '✗'}")
        print(line)
        report_lines.append(line)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # OVERALL
    overall = compute_metrics(results)
    print()
    print("=" * 100)
    print("OVERALL METRICS")
    print("=" * 100)
    print(f"Total queries : {overall['total']}")
    print(f"Gold @1       : {overall['gold_at_1']}/{overall['total']} = {overall['p1']:.4f}")
    print(f"Gold @3       : {overall['gold_at_3']}/{overall['total']} = {overall['p3']:.4f}")
    print(f"Gold @5       : {overall['gold_at_5']}/{overall['total']} = {overall['p5']:.4f}")
    print(f"Gold @10      : {overall['gold_at_10']}/{overall['total']} = {overall['p10']:.4f}")
    print(f"Not found     : {overall['not_found']}")
    print(f"MRR           : {overall['mrr']:.4f}")

    # PER LEVEL
    print()
    print("=" * 100)
    print("PER-LEVEL METRICS")
    print("=" * 100)
    print(f"{'Level':<6} {'@1':>8} {'@3':>8} {'@5':>8} {'@10':>8} {'MRR':>8} {'NF':>4}")
    print("-" * 60)
    per_level = {}
    for lvl in LEVELS:
        subset = [r for r in results if r["level"] == lvl]
        m = compute_metrics(subset)
        per_level[lvl] = m
        print(f"{lvl:<6} {m['p1']:>8.4f} {m['p3']:>8.4f} {m['p5']:>8.4f} {m['p10']:>8.4f} {m['mrr']:>8.4f} {m['not_found']:>4}")

    # DEGRADATION
    l0_p1 = per_level["L0"]["p1"]
    l4_p1 = per_level["L4"]["p1"]
    drop = l0_p1 - l4_p1
    pct = (drop / l0_p1 * 100) if l0_p1 else 0
    print()
    print("=" * 100)
    print("DEGRADATION L0 → L4")
    print("=" * 100)
    print(f"L0 @1 = {l0_p1:.4f}")
    print(f"L4 @1 = {l4_p1:.4f}")
    print(f"Drop  = {drop:.4f} ({pct:+.1f}%)")

    # WRONG BY LEVEL
    print()
    print("=" * 100)
    print("WRONG-ABOVE-GOLD BY LEVEL")
    print("=" * 100)
    wbl = defaultdict(int)
    tbl = defaultdict(int)
    for r in results:
        tbl[r["level"]] += 1
        if not r["top1_correct"]:
            wbl[r["level"]] += 1
    for lvl in LEVELS:
        w = wbl[lvl]
        t = tbl[lvl]
        print(f"{lvl}: {w}/{t} = {w/t:.4f}")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print()
    print(f"Saved: {RESULTS_FILE}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()