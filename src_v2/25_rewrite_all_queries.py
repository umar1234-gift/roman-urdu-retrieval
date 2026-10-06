"""
Rewrite ALL 450 queries (L0-L4) — Fixes oracle routing issue.
=============================================================
Previous pipeline only rewrote L2-L4 (270 queries), keeping L0/L1 
unchanged. This used hidden level labels — unrealistic for real systems.

This script rewrites ALL 450 queries, so evaluation reflects a 
deployable system that doesn't know the query language level.

Run: python src_v2/25_rewrite_all_queries.py
"""

from pathlib import Path
import json
import os
import re
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"
OUTPUT_FILE = PROJECT_ROOT / "data_v2" / "processed" / "all_rewritten_queries.json"
CACHE_FILE = PROJECT_ROOT / "data_v2" / "processed" / "rewrite_cache_all.json"

MODEL = "openai/gpt-oss-120b"
BATCH_SIZE = 15
BATCH_DELAY = 2.0


PROMPT_TEMPLATE = """You are a professional translator. Translate each numbered query into natural English.

Rules:
1. Preserve the EXACT information need — do not add or remove details.
2. If the query is already in English, return it essentially unchanged 
   (you may fix minor grammar only).
3. If it's Roman Urdu, translate to natural English.
4. Output ONLY a numbered list — no explanations.
5. Every input line MUST have a corresponding output line.

Example:
Input:
1. What is the minimum attendance?
2. Minimum attendance kitni honi chahiye?

Output:
1. What is the minimum attendance?
2. What is the minimum attendance required?

Now translate these:

{queries}

Output:"""


def init_groq():
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY environment variable")
    return Groq(api_key=api_key)


def rewrite_batch(client, queries):
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
    prompt = PROMPT_TEMPLATE.format(queries=numbered)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2048,
    )
    text = response.choices[0].message.content.strip()

    translations = []
    for line in text.split("\n"):
        line = line.strip()
        m = re.match(r"^\d+\.\s*(.+)$", line)
        if m:
            translations.append(m.group(1).strip())

    while len(translations) < len(queries):
        translations.append(queries[len(translations)])

    return translations[:len(queries)]


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 100)
    print("REWRITE ALL 450 QUERIES (ORACLE ROUTING FIX)")
    print("=" * 100)

    queries = load_json(QUERIES_FILE)
    print(f"Loaded {len(queries)} queries")

    cache = {}
    if CACHE_FILE.exists():
        cache = load_json(CACHE_FILE)
        print(f"Cache: {len(cache)} entries")

    print("\nInitializing Groq...")
    client = init_groq()
    print("Ready.\n")

    # ALL queries need rewriting (no oracle label)
    to_rewrite = [
        (q["query_id"], q["query"])
        for q in queries
        if q["query_id"] not in cache
    ]

    total_batches = (len(to_rewrite) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Queries to rewrite : {len(to_rewrite)}")
    print(f"Total batches      : {total_batches}\n")

    start_time = time.time()

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(to_rewrite))
        batch = to_rewrite[start:end]

        query_ids = [b[0] for b in batch]
        query_texts = [b[1] for b in batch]

        elapsed = time.time() - start_time
        print(f"Batch {batch_idx + 1}/{total_batches} ({len(batch)} queries) "
              f"[elapsed {elapsed:.0f}s]...")

        try:
            translations = rewrite_batch(client, query_texts)
            for qid, trans in zip(query_ids, translations):
                cache[qid] = trans

            save_json(CACHE_FILE, cache)
            print(f"  Done. Cache: {len(cache)} entries")

            if batch_idx < total_batches - 1:
                time.sleep(BATCH_DELAY)

        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(10)

    # Build output with all levels rewritten
    print("\nBuilding final output...")
    output = []
    for q in queries:
        rewritten = cache.get(q["query_id"], q["query"])
        output.append({
            "query_id": q["query_id"],
            "doc_id": q["doc_id"],
            "level": q["level"],
            "variant": q["variant"],
            "original_query": q["query"],
            "rewritten_query": rewritten,
            "gold_document": q["gold_document"],
            "gold_answer": q.get("gold_answer", ""),
        })

    save_json(OUTPUT_FILE, output)

    total_time = time.time() - start_time

    print()
    print("=" * 100)
    print(f"Total rewritten : {len(output)}")
    print(f"Total time      : {total_time:.1f} seconds")
    print(f"Saved: {OUTPUT_FILE}")
    print("=" * 100)

    # Show sample for each level
    print("\nSAMPLE REWRITES (all levels):")
    for lvl in ["L0", "L1", "L2", "L3", "L4"]:
        samples = [o for o in output if o["level"] == lvl][:2]
        print(f"\n--- {lvl} ---")
        for s in samples:
            print(f"  {s['query_id']}")
            print(f"    Original : {s['original_query']}")
            print(f"    Rewritten: {s['rewritten_query']}")


if __name__ == "__main__":
    main()