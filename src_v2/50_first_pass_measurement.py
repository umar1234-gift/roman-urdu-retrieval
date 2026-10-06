"""
Direct First-Pass Measurement
================================
Measures A_single first-pass @1 directly from cache,
WITHOUT any retry or correction.

Run: python src_v2/50_first_pass_measurement.py
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
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"
CACHE_FILE = PROJECT_ROOT / "data_v2" / "processed" / "ablation_single.json"

MODEL_NAME = "intfloat/multilingual-e5-large"
LEVELS = ["L0", "L1", "L2", "L3", "L4"]


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def main():
    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)
    with open(QUERIES_FILE) as f:
        queries = json.load(f)
    with open(CACHE_FILE) as f:
        cache = json.load(f)

    print("=" * 100)
    print("DIRECT FIRST-PASS MEASUREMENT")
    print("=" * 100)
    print(f"Cache entries: {len(cache)}")

    # Count empty entries
    empty = [qid for qid, val in cache.items() if not val.strip()]
    print(f"Empty entries: {len(empty)}")
    for e in empty:
        print(f"  {e}")

    # Retrieve
    filenames = [c["filename"] for c in chunks]
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + c["text"] for c in chunks]
    doc_emb = model.encode(doc_texts, normalize_embeddings=True,
                            show_progress_bar=True)

    results = []
    for q in queries:
        qid = q["query_id"]
        text = cache.get(qid, "").strip()

        # Empty text → retrieval on empty
        q_emb = model.encode(["query: " + text],
                              normalize_embeddings=True)[0]
        sims = np.dot(doc_emb, q_emb)
        order = np.argsort(sims)[::-1]

        gr = None
        for r, i in enumerate(order, start=1):
            if filenames[i] == q["gold_document"]:
                gr = r
                break

        results.append({
            "query_id": qid, "level": q["level"],
            "doc_id": q["doc_id"], "gold_rank": gr,
            "text_used": text[:50],
            "was_empty": not text,
        })

    # Metrics
    total = len(results)
    correct = sum(1 for r in results if r["gold_rank"] == 1)
    print()
    print(f"First-pass @1 : {correct}/{total} = {correct/total:.4f}")

    # Per level
    print()
    for lvl in LEVELS:
        sub = [r for r in results if r["level"] == lvl]
        c = sum(1 for r in sub if r["gold_rank"] == 1)
        print(f"  {lvl}: {c}/{len(sub)} = {c/len(sub):.4f}")

    # Save
    out = PROJECT_ROOT / "data_v2" / "results" / "first_pass_direct.json"
    with open(out, "w") as f:
        json.dump({"first_pass_at1": correct/total,
                    "correct": correct, "total": total,
                    "empty_count": len(empty)},
                   f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()