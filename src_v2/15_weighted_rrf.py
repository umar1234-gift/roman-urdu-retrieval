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

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "weighted_rrf_results.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "weighted_rrf_report.txt"

MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K = 10
LEVELS = ["L0", "L1", "L2", "L3", "L4"]

# Weight configurations (w_bm25, w_e5)
WEIGHT_CONFIGS = [
    ("RRF-equal (1.0/1.0)", 1.0, 1.0),
    ("BM25-dominant (0.7/0.3)", 0.7, 0.3),
    ("E5-slight-dominant (0.4/0.6)", 0.4, 0.6),
    ("E5-dominant (0.3/0.7)", 0.3, 0.7),
    ("E5-strong (0.2/0.8)", 0.2, 0.8),
    ("E5-very-strong (0.1/0.9)", 0.1, 0.9),
    ("E5-only (0.0/1.0)", 0.0, 1.0),
]

# k value (from sensitivity analysis, k=10 marginally best)
RRF_K = 10


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
    return {
        "@1": g1/total, "@3": g3/total, "@5": g5/total,
        "@10": g10/total, "MRR": mrr, "total": total
    }


def main():
    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]

    print("=" * 100)
    print("WEIGHTED RRF ANALYSIS (BM25 + E5-LARGE)")
    print("=" * 100)
    print(f"Chunks        : {len(chunks)}")
    print(f"Queries       : {len(queries)}")
    print(f"RRF k         : {RRF_K}")
    print(f"Configurations: {len(WEIGHT_CONFIGS)}\n")

    # --------------------------------------------------------
    # BM25 index
    # --------------------------------------------------------
    print("Building BM25 index...")
    bm25 = BM25(texts)

    # --------------------------------------------------------
    # E5-large document embeddings
    # --------------------------------------------------------
    print(f"Loading E5-large ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    print("Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True,
                            show_progress_bar=True)
    print("Ready.\n")

    # --------------------------------------------------------
    # Pre-compute per-query rankings (BM25 + E5-large)
    # --------------------------------------------------------
    print("Pre-computing per-query rankings...")
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
            print(f"  Processed {idx + 1}/{len(queries)}")

    print("Done.\n")

    # --------------------------------------------------------
    # Evaluate each weight config
    # --------------------------------------------------------
    report = []
    def out(s=""):
        print(s)
        report.append(s)

    all_results = {}

    for config_name, w_b, w_e in WEIGHT_CONFIGS:
        print(f"Evaluating: {config_name}...")

        results = []
        for item in q_data:
            q = item["query"]
            bm25_rank = item["bm25_rank"]
            e5_rank = item["e5_rank"]

            # Weighted RRF
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
                "query": q["query"],
                "gold_document": q["gold_document"],
                "gold_rank": gold_rank
            })

        all_results[config_name] = results

    # ========================================================
    # REPORT
    # ========================================================
    out("=" * 110)
    out("WEIGHTED RRF — RESULTS")
    out("=" * 110)

    # Overall
    out()
    out("OVERALL METRICS (n=450)")
    out("-" * 110)
    out(f"{'Config':<28} {'@1':>8} {'@3':>8} {'@5':>8} {'@10':>8} {'MRR':>8}")
    out("-" * 110)

    for config_name, _, _ in WEIGHT_CONFIGS:
        m = metrics(all_results[config_name])
        out(f"{config_name:<28} {m['@1']:>8.4f} {m['@3']:>8.4f} "
            f"{m['@5']:>8.4f} {m['@10']:>8.4f} {m['MRR']:>8.4f}")

    # Per level @1
    out()
    out("PER-LEVEL @1 ACCURACY")
    out("-" * 110)
    out(f"{'Config':<28} " + " ".join(f"{lvl:>8}" for lvl in LEVELS) + "   Drop")
    out("-" * 110)

    for config_name, _, _ in WEIGHT_CONFIGS:
        row = []
        for lvl in LEVELS:
            subset = [r for r in all_results[config_name] if r["level"] == lvl]
            row.append(metrics(subset)["@1"])
        drop = row[0] - row[-1]
        out(f"{config_name:<28} " + " ".join(f"{v:>8.4f}" for v in row)
            + f"  {drop:>+7.4f}")

    # Best config
    out()
    out("=" * 110)
    out("BEST CONFIGURATION")
    out("=" * 110)

    best_p1 = max(WEIGHT_CONFIGS, key=lambda c: metrics(all_results[c[0]])["@1"])
    best_mrr = max(WEIGHT_CONFIGS, key=lambda c: metrics(all_results[c[0]])["MRR"])

    out(f"Best by @1 : {best_p1[0]} → {metrics(all_results[best_p1[0]])['@1']:.4f}")
    out(f"Best by MRR: {best_mrr[0]} → {metrics(all_results[best_mrr[0]])['MRR']:.4f}")

    # Reference: E5-large alone
    out()
    out("REFERENCE (E5-large alone, from previous run):")
    out("  @1 = 0.9222  @3 = 0.9689  @5 = 0.9778  @10 = 0.9933  MRR = 0.9489")

    # Compare to equal RRF
    out()
    out("IMPROVEMENT OVER EQUAL RRF:")
    eq = metrics(all_results["RRF-equal (1.0/1.0)"])
    for config_name, _, _ in WEIGHT_CONFIGS:
        m = metrics(all_results[config_name])
        d_p1 = m["@1"] - eq["@1"]
        d_mrr = m["MRR"] - eq["MRR"]
        out(f"{config_name:<28} Δ@1 = {d_p1:+.4f}   ΔMRR = {d_mrr:+.4f}")

    # Save
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "configs": {name: {"w_bm25": wb, "w_e5": we,
                                "overall": metrics(all_results[name])}
                        for name, wb, we in WEIGHT_CONFIGS},
            "per_query": all_results
        }, f, indent=2, ensure_ascii=False)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    out()
    out(f"Saved: {RESULTS_FILE}")
    out(f"Saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()