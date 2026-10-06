"""
Correct First-Pass Measurement
================================
Fixes script 50 bug: uses original query for L0 (not in cache).

Run: python src_v2/51_correct_first_pass.py
"""

from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"
CACHE_FILE = PROJECT_ROOT / "data_v2" / "processed" / "ablation_single.json"

MODEL_NAME = "intfloat/multilingual-e5-large"
LEVELS = ["L0", "L1", "L2", "L3", "L4"]


def main():
    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)
    with open(QUERIES_FILE) as f:
        queries = json.load(f)
    with open(CACHE_FILE) as f:
        cache = json.load(f)

    print("=" * 100)
    print("CORRECT FIRST-PASS MEASUREMENT")
    print("=" * 100)

    # L0 not in cache -> use original query
    # L1-L4 in cache -> use cache (first-pass)

    filenames = [c["filename"] for c in chunks]
    model = SentenceTransformer(MODEL_NAME)
    doc_texts = ["passage: " + c["text"] for c in chunks]
    doc_emb = model.encode(doc_texts, normalize_embeddings=True,
                            show_progress_bar=True)

    results = []
    for q in queries:
        qid = q["query_id"]
        lvl = q["level"]

        if lvl == "L0":
            # L0 not in cache -> use original query
            text = q["query"]
        else:
            # L1-L4: use cache, fallback to original if empty
            cached = cache.get(qid, "").strip()
            text = cached if cached else q["query"]

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
            "query_id": qid, "level": lvl,
            "gold_rank": gr,
            "text_used": text[:50],
        })

    total = len(results)
    correct = sum(1 for r in results if r["gold_rank"] == 1)
    print(f"\nFirst-pass @1 : {correct}/{total} = {correct/total:.4f}")

    print()
    for lvl in LEVELS:
        sub = [r for r in results if r["level"] == lvl]
        c = sum(1 for r in sub if r["gold_rank"] == 1)
        print(f"  {lvl}: {c}/{len(sub)} = {c/len(sub):.4f}")

    # MQ0006 detail
    mq6 = [r for r in results if r["query_id"] == "MQ0006"][0]
    print(f"\nMQ0006: rank={mq6['gold_rank']}, text='{mq6['text_used']}'")


if __name__ == "__main__":
    main()