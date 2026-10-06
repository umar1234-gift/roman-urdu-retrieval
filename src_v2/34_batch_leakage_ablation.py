"""
Batch Leakage Ablation
=======================
Tests whether LLM sees L0 English context when translating L4 queries
in the same batch (batch size = 15 = queries per document).

Variants:
A. One query per call (no batching) — baseline, slow
B. Shuffled batches across documents (no doc-sequential context)
C. L3/L4-only batches (no L0/L1 English context)

For each variant, compute @1 (E5-large) and chrF (vs L0 references).

Run: python src_v2/34_batch_leakage_ablation.py
"""

from pathlib import Path
import json
import os
import re
import time
import random
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sacrebleu.metrics import CHRF


PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"
CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"

OUT_DIR = PROJECT_ROOT / "data_v2" / "processed"
RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "batch_leakage_ablation.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "batch_leakage_ablation.txt"

MODEL = "openai/gpt-oss-120b"
E5_MODEL = "intfloat/multilingual-e5-large"
RRF_K = 10
SEED = 42


PROMPT_SINGLE = """Translate the following query into natural English.

Rules:
1. Preserve the EXACT information need.
2. If it's already English, return it unchanged.
3. Return ONLY the translation, no explanation.

Query: {query}

Translation:"""


PROMPT_BATCH = """You are a professional translator. Translate each numbered query into natural English.

Rules:
1. Preserve the EXACT information need.
2. If the query is already in English, return it essentially unchanged.
3. If it's Roman Urdu, translate to natural English.
4. Output ONLY a numbered list — no explanations.

Queries:
{queries}

Output:"""


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def init_groq():
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY environment variable")
    return Groq(api_key=api_key)


def rewrite_single(client, text):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT_SINGLE.format(query=text)}],
        temperature=0.0,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def rewrite_batch(client, texts):
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT_BATCH.format(queries=numbered)}],
        temperature=0.0,
        max_tokens=2048,
    )
    text = response.choices[0].message.content.strip()

    out = []
    for line in text.split("\n"):
        line = line.strip()
        m = re.match(r"^\d+\.\s*(.+)$", line)
        if m:
            out.append(m.group(1).strip())

    while len(out) < len(texts):
        out.append(texts[len(out)])

    return out[:len(texts)]


def main():
    print("=" * 100)
    print("BATCH LEAKAGE ABLATION")
    print("=" * 100)

    queries = load_json(QUERIES_FILE)
    chunks = load_json(CHUNKS_FILE)

    # Only L1-L4 (L0 stays English)
    to_rewrite = [q for q in queries if q["level"] in ["L1", "L2", "L3", "L4"]]
    print(f"Total queries to rewrite: {len(to_rewrite)}\n")

    client = init_groq()

    variants = {}

    # =========================================================
    # VARIANT A: One query per call (no batching)
    # =========================================================
    print("=" * 100)
    print("VARIANT A: One query per call (no batch context)")
    print("=" * 100)

    cache_a_file = OUT_DIR / "ablation_single.json"
    if cache_a_file.exists():
        cache_a = load_json(cache_a_file)
        print(f"Loaded cache: {len(cache_a)} queries")
    else:
        cache_a = {}

    for idx, q in enumerate(to_rewrite):
        qid = q["query_id"]
        if qid in cache_a:
            continue

        try:
            result = rewrite_single(client, q["query"])
            cache_a[qid] = result
        except Exception as e:
            print(f"  Error {qid}: {e}")
            cache_a[qid] = q["query"]

        if (idx + 1) % 50 == 0:
            save_json(cache_a_file, cache_a)
            print(f"  {idx+1}/{len(to_rewrite)} done")

        time.sleep(0.3)  # rate limit

    save_json(cache_a_file, cache_a)
    variants["A_single"] = cache_a
    print(f"Variant A complete: {len(cache_a)} queries\n")

    # =========================================================
    # VARIANT B: Shuffled batches across documents
    # =========================================================
    print("=" * 100)
    print("VARIANT B: Shuffled batches (no doc-sequential context)")
    print("=" * 100)

    cache_b_file = OUT_DIR / "ablation_shuffle.json"
    if cache_b_file.exists():
        cache_b = load_json(cache_b_file)
        print(f"Loaded cache: {len(cache_b)} queries")
    else:
        cache_b = {}

    remaining = [q for q in to_rewrite if q["query_id"] not in cache_b]
    random.seed(SEED)
    random.shuffle(remaining)

    BATCH = 15
    n_batches = (len(remaining) + BATCH - 1) // BATCH

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH
        end = min(start + BATCH, len(remaining))
        batch = remaining[start:end]

        qids = [q["query_id"] for q in batch]
        texts = [q["query"] for q in batch]

        try:
            results = rewrite_batch(client, texts)
            for qid, r in zip(qids, results):
                cache_b[qid] = r
        except Exception as e:
            print(f"  Error batch {batch_idx}: {e}")
            for qid, t in zip(qids, texts):
                cache_b[qid] = t

        if (batch_idx + 1) % 5 == 0:
            save_json(cache_b_file, cache_b)
            print(f"  Batch {batch_idx+1}/{n_batches} done")

        time.sleep(2)

    save_json(cache_b_file, cache_b)
    variants["B_shuffle"] = cache_b
    print(f"Variant B complete: {len(cache_b)} queries\n")

    # =========================================================
    # VARIANT C: L3/L4-only batches (no L0/L1 English context)
    # =========================================================
    print("=" * 100)
    print("VARIANT C: L3/L4-only batches")
    print("=" * 100)

    cache_c_file = OUT_DIR / "ablation_l34only.json"
    if cache_c_file.exists():
        cache_c = load_json(cache_c_file)
        print(f"Loaded cache: {len(cache_c)} queries")
    else:
        cache_c = {}

    l34 = [q for q in to_rewrite if q["level"] in ["L3", "L4"]]
    remaining_c = [q for q in l34 if q["query_id"] not in cache_c]
    n_batches_c = (len(remaining_c) + BATCH - 1) // BATCH

    print(f"L3/L4 queries: {len(l34)}, remaining: {len(remaining_c)}")

    for batch_idx in range(n_batches_c):
        start = batch_idx * BATCH
        end = min(start + BATCH, len(remaining_c))
        batch = remaining_c[start:end]

        qids = [q["query_id"] for q in batch]
        texts = [q["query"] for q in batch]

        try:
            results = rewrite_batch(client, texts)
            for qid, r in zip(qids, results):
                cache_c[qid] = r
        except Exception as e:
            print(f"  Error batch {batch_idx}: {e}")
            for qid, t in zip(qids, texts):
                cache_c[qid] = t

        time.sleep(2)
        print(f"  Batch {batch_idx+1}/{n_batches_c} done")

    # For L1/L2, use variant A results (fallback)
    for q in to_rewrite:
        qid = q["query_id"]
        if q["level"] in ["L1", "L2"] and qid not in cache_c:
            cache_c[qid] = cache_a.get(qid, q["query"])

    save_json(cache_c_file, cache_c)
    variants["C_l34only"] = cache_c
    print(f"Variant C complete: {len(cache_c)} queries\n")

    # =========================================================
    # EVALUATE EACH VARIANT
    # =========================================================
    print("=" * 100)
    print("EVALUATING VARIANTS (E5-large retrieval + chrF)")
    print("=" * 100)

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]

    print("\nLoading E5-large...")
    model = SentenceTransformer(E5_MODEL)
    doc_texts = ["passage: " + t for t in texts]
    doc_emb = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=True)

    # chrF references
    ref_lookup = {}
    for q in queries:
        if q["level"] == "L0":
            ref_lookup[(q["doc_id"], q["variant"])] = q["query"]

    chrf = CHRF(word_order=2)

    results = {}

    for variant_name, cache in variants.items():
        print(f"\n--- {variant_name} ---")

        correct = 0
        total = 0
        chrf_scores = []

        for q in queries:
            qid = q["query_id"]
            text = cache.get(qid, q["query"])
            gold = q["gold_document"]

            # Retrieval
            q_emb = model.encode(["query: " + text], normalize_embeddings=True)[0]
            sims = np.dot(doc_emb, q_emb)
            order = np.argsort(sims)[::-1]

            gold_rank = None
            for r, i in enumerate(order, start=1):
                if filenames[i] == gold:
                    gold_rank = r
                    break

            if gold_rank == 1:
                correct += 1
            total += 1

            # chrF (only for L1-L4)
            if q["level"] in ["L1", "L2", "L3", "L4"]:
                key = (q["doc_id"], q["variant"])
                if key in ref_lookup:
                    chrf_scores.append(chrf.sentence_score(text, [ref_lookup[key]]).score)

        p1 = correct / total
        mean_chrf = float(np.mean(chrf_scores))

        results[variant_name] = {
            "@1": p1,
            "chrf": mean_chrf,
            "n_correct": correct,
            "n_total": total,
            "n_chrf": len(chrf_scores)
        }

        print(f"  @1   : {p1:.4f} ({correct}/{total})")
        print(f"  chrF : {mean_chrf:.2f}")

    # =========================================================
    # COMPARISON
    # =========================================================
    print()
    print("=" * 100)
    print("COMPARISON")
    print("=" * 100)
    print()
    print(f"{'Variant':<20} {'@1':>10} {'chrF':>10}")
    print("-" * 100)
    for name in ["A_single", "B_shuffle", "C_l34only"]:
        if name in results:
            r = results[name]
            print(f"{name:<20} {r['@1']:>10.4f} {r['chrf']:>10.2f}")

    # Interpretation
    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    if "A_single" in results and "C_l34only" in results:
        gap = results["A_single"]["@1"] - results["C_l34only"]["@1"]
        print(f"\n@1 gap (single vs L3/L4-only): {gap:+.4f}")

        if abs(gap) > 0.02:
            print("  -> SIGNIFICANT leakage from L0 English context in batches")
        else:
            print("  -> Batch composition does NOT materially affect @1")

    save_json(RESULTS_FILE, results)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2))

    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
