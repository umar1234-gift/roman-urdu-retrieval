"""
Weight Sweep for Weighted RRF
==============================
Sweeps w_bm25 and w_e5 across the full grid to visualize
how fusion weight affects performance.

Run: python src_v2/31_weight_sweep.py
"""

from pathlib import Path
import json
import math
import re
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "all_rewritten_queries.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "weight_sweep.json"
FIG_FILE = PROJECT_ROOT / "paper" / "figures" / "fig8_weight_sweep.png"

MODEL_NAME = "intfloat/multilingual-e5-large"
RRF_K = 10

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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


def main():
    print("=" * 100)
    print("WEIGHT SWEEP FOR WEIGHTED RRF")
    print("=" * 100)

    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]
    n_docs = len(filenames)

    print(f"Chunks : {len(chunks)}")
    print(f"Queries: {len(queries)}\n")

    # Build
    bm25 = BM25(texts)
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + t for t in texts]
    doc_emb = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=True)

    # Precompute rankings for all queries
    print("\nPrecomputing rankings...")
    bm25_ranks = {}
    e5_ranks = {}

    for idx, q in enumerate(queries):
        qid = q["query_id"]
        text = q["rewritten_query"]

        bm25_r = bm25.rank(text)
        bm25_ranks[qid] = {i: r for r, (i, _) in enumerate(bm25_r, start=1)}

        q_emb = model.encode(["query: " + text], normalize_embeddings=True)[0]
        sims = np.dot(doc_emb, q_emb)
        e5_order = np.argsort(sims)[::-1]
        e5_ranks[qid] = {i: r for r, i in enumerate(e5_order, start=1)}

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(queries)}")

    print("Done.\n")

    # Sweep
    print("Sweeping weights...")
    matrix = np.zeros((len(WEIGHTS), len(WEIGHTS)))

    for i, w_b in enumerate(WEIGHTS):
        for j, w_e in enumerate(WEIGHTS):
            if w_b == 0 and w_e == 0:
                matrix[i, j] = 0
                continue

            # Normalize so weights sum to 1 (avoid scaling artifacts)
            total = w_b + w_e
            wb_n = w_b / total
            we_n = w_e / total

            correct = 0
            for q in queries:
                qid = q["query_id"]
                gold = q["gold_document"]
                bm25_rank = bm25_ranks[qid]
                e5_rank = e5_ranks[qid]

                scores = {k: (wb_n/(RRF_K + bm25_rank[k])) + (we_n/(RRF_K + e5_rank[k]))
                          for k in range(n_docs)}
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

                for r, (k, _) in enumerate(ranked, start=1):
                    if filenames[k] == gold:
                        if r == 1:
                            correct += 1
                        break

            matrix[i, j] = correct / len(queries)

        print(f"  w_bm25={w_b:.1f} done")

    # Report
    print()
    print("=" * 100)
    print("WEIGHT SWEEP RESULTS (normalized weights)")
    print("=" * 100)
    print()
    print(f"{'w_bm25':<8} " + " ".join(f"{w:>6.1f}" for w in WEIGHTS))
    print("-" * 100)

    for i, w_b in enumerate(WEIGHTS):
        row = " ".join(f"{matrix[i, j]:>6.3f}" for j in range(len(WEIGHTS)))
        print(f"{w_b:<8} {row}")

    # Best
    idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    best_wb = WEIGHTS[idx[0]]
    best_we = WEIGHTS[idx[1]]
    best_score = matrix[idx]
    print()
    print(f"Best: w_bm25={best_wb:.1f}, w_e5={best_we:.1f} -> @1={best_score:.4f}")

    # Figure
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.9, vmax=1.0)

    ax.set_xticks(np.arange(len(WEIGHTS)))
    ax.set_xticklabels([f"{w:.1f}" for w in WEIGHTS])
    ax.set_yticks(np.arange(len(WEIGHTS)))
    ax.set_yticklabels([f"{w:.1f}" for w in WEIGHTS])

    ax.set_xlabel("w_e5 (E5 weight)", fontsize=12, fontweight="bold")
    ax.set_ylabel("w_bm25 (BM25 weight)", fontsize=12, fontweight="bold")
    ax.set_title("Weighted RRF: @1 Accuracy Across Weight Combinations",
                 fontsize=13, fontweight="bold")

    for i in range(len(WEIGHTS)):
        for j in range(len(WEIGHTS)):
            v = matrix[i, j]
            if v > 0:
                color = "white" if v < 0.95 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color=color, fontsize=7)

    plt.colorbar(im, ax=ax, label="@1 Accuracy")
    plt.tight_layout()
    FIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_FILE, dpi=200, bbox_inches="tight")
    plt.close()

    save_json(RESULTS_FILE, {
        "weights": WEIGHTS,
        "matrix": matrix.tolist(),
        "best": {"w_bm25": best_wb, "w_e5": best_we, "score": float(best_score)},
    })

    print(f"Saved: {RESULTS_FILE}")
    print(f"Saved: {FIG_FILE}")


if __name__ == "__main__":
    main()