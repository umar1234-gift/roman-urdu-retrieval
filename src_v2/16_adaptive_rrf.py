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

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "adaptive_rrf_results.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "adaptive_rrf_report.txt"

MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K = 10
RRF_K = 10
LEVELS = ["L0", "L1", "L2", "L3", "L4"]

# Per-level weight candidates to test
WEIGHT_GRID = [
    (1.0, 1.0), (0.7, 0.3), (0.5, 0.5), (0.4, 0.6),
    (0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.0, 1.0)
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
    g1 = sum(1 for r in records if r["gold_rank"] == 1)
    g3 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 3)
    g5 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 5)
    mrr = sum(1/r["gold_rank"] for r in records if r["gold_rank"]) / total
    return {"@1": g1/total, "@3": g3/total, "@5": g5/total, "MRR": mrr}


def main():
    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]

    print("=" * 100)
    print("ADAPTIVE RRF — Per-Level Weight Optimization")
    print("=" * 100)
    print(f"Chunks        : {len(chunks)}")
    print(f"Queries       : {len(queries)}")
    print(f"Weight grid   : {len(WEIGHT_GRID)} configs")
    print(f"RRF k         : {RRF_K}\n")

    # Build BM25
    print("Building BM25 index...")
    bm25 = BM25(texts)

    # Load E5-large
    print(f"Loading E5-large...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    print("Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True,
                            show_progress_bar=True)

    # Pre-compute rankings
    print("\nPre-computing rankings...")
    q_data = []
    for idx, q in enumerate(queries):
        bm25_ranking = bm25.rank(q["query"])
        bm25_rank = {i: r for r, (i, _) in enumerate(bm25_ranking, start=1)}

        q_emb = model.encode(["query: " + q["query"]],
                             normalize_embeddings=True)[0]
        sims = np.dot(doc_emb, q_emb)
        e5_order = np.argsort(sims)[::-1]
        e5_rank = {i: r for r, i in enumerate(e5_order, start=1)}

        q_data.append({
            "query": q,
            "bm25_rank": bm25_rank,
            "e5_rank": e5_rank
        })

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(queries)}")

    print("Done.\n")

    # --------------------------------------------------------
    # Grid search per level
    # --------------------------------------------------------
    print("=" * 100)
    print("GRID SEARCH — FIND BEST WEIGHTS PER LEVEL")
    print("=" * 100)

    best_per_level = {}

    for lvl in LEVELS:
        lvl_items = [item for item in q_data if item["query"]["level"] == lvl]
        print(f"\n{lvl} ({len(lvl_items)} queries)")

        best_w = None
        best_score = -1
        results_by_w = {}

        for w_b, w_e in WEIGHT_GRID:
            results = []
            for item in lvl_items:
                q = item["query"]
                bm25_rank = item["bm25_rank"]
                e5_rank = item["e5_rank"]

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

                results.append({"gold_rank": gold_rank})

            m = metrics(results)
            results_by_w[(w_b, w_e)] = m

            if m["@1"] > best_score:
                best_score = m["@1"]
                best_w = (w_b, w_e)

        # Print all weights for this level
        for (w_b, w_e), m in results_by_w.items():
            marker = " ← BEST" if (w_b, w_e) == best_w else ""
            print(f"  ({w_b:.1f}/{w_e:.1f}) @1={m['@1']:.4f} "
                  f"@3={m['@3']:.4f} MRR={m['MRR']:.4f}{marker}")

        best_per_level[lvl] = best_w

    # --------------------------------------------------------
    # Apply adaptive weights
    # --------------------------------------------------------
    print("\n" + "=" * 100)
    print("ADAPTIVE RRF — BEST WEIGHTS PER LEVEL")
    print("=" * 100)

    for lvl in LEVELS:
        print(f"{lvl}: w_bm25={best_per_level[lvl][0]:.1f}, "
              f"w_e5={best_per_level[lvl][1]:.1f}")

    # Evaluate full adaptive
    adaptive_results = []
    for item in q_data:
        q = item["query"]
        lvl = q["level"]
        w_b, w_e = best_per_level[lvl]

        bm25_rank = item["bm25_rank"]
        e5_rank = item["e5_rank"]

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

        adaptive_results.append({
            "query_id": q["query_id"],
            "doc_id": q["doc_id"],
            "level": q["level"],
            "variant": q["variant"],
            "query": q["query"],
            "gold_document": q["gold_document"],
            "gold_rank": gold_rank
        })

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------
    print("\n" + "=" * 100)
    print("ADAPTIVE RRF — FINAL RESULTS")
    print("=" * 100)

    overall = metrics(adaptive_results)
    print(f"\nOVERALL:")
    print(f"  @1  = {overall['@1']:.4f}")
    print(f"  @3  = {overall['@3']:.4f}")
    print(f"  @5  = {overall['@5']:.4f}")
    print(f"  MRR = {overall['MRR']:.4f}")

    print(f"\nPER-LEVEL:")
    for lvl in LEVELS:
        subset = [r for r in adaptive_results if r["level"] == lvl]
        m = metrics(subset)
        print(f"  {lvl}: @1={m['@1']:.4f} @3={m['@3']:.4f} MRR={m['MRR']:.4f}")

    # Compare
    print("\n" + "=" * 100)
    print("COMPARISON")
    print("=" * 100)
    print(f"{'System':<30} {'@1':>8} {'@3':>8} {'MRR':>8}")
    print("-" * 100)
    print(f"{'E5-large alone':<30} {0.9222:>8.4f} {0.9689:>8.4f} {0.9489:>8.4f}")
    print(f"{'Weighted RRF (0.2/0.8)':<30} {0.9244:>8.4f} {0.9711:>8.4f} {0.9505:>8.4f}")
    print(f"{'Adaptive RRF (per-level)':<30} {overall['@1']:>8.4f} {overall['@3']:>8.4f} {overall['MRR']:>8.4f}")

    # Save
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "best_weights_per_level": {lvl: list(w) for lvl, w in best_per_level.items()},
            "overall_metrics": overall,
            "per_query": adaptive_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()