"""
LLM Query Rewriting using Groq (2026 updated)
==============================================
Groq decommissioned llama-3.1-8b-instant and llama-3.3-70b-versatile
on 2026-08-16. Now using openai/gpt-oss-120b.
"""

from pathlib import Path
import json
import os
import re
import time


# ============================================================
# CONFIG — UPDATED FOR 2026 GROQ MODELS
# ============================================================

# Current Groq production models (as of Sept 2026):
#   "openai/gpt-oss-120b"  — best quality, 1K req/day
#   "openai/gpt-oss-20b"   — fastest, more requests
MODEL = "openai/gpt-oss-120b"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"
OUTPUT_FILE = PROJECT_ROOT / "data_v2" / "processed" / "rewritten_queries.json"
CACHE_FILE = PROJECT_ROOT / "data_v2" / "processed" / "rewrite_cache_groq.json"

LEVELS_TO_REWRITE = ["L2", "L3", "L4"]
BATCH_SIZE = 15
BATCH_DELAY = 2.0


PROMPT_TEMPLATE = """You are a professional translator specializing in Roman Urdu (Urdu written in Latin script) to English translation.

Your task: Translate each numbered Roman Urdu query below into natural, fluent English.

CRITICAL RULES:
1. Preserve the EXACT information need — do not add or remove details.
2. Use natural English question phrasing.
3. If the query asks for a number, date, or percentage, keep it as a question.
4. Output ONLY a numbered list of English translations — no explanations.
5. Every input line MUST have a corresponding output line.

Example:
Input:
1. Fall 2026 ki classes kab shuru hongi?
2. Minimum attendance kitni honi chahiye?

Output:
1. When will Fall 2026 classes begin?
2. What is the minimum attendance required?

Now translate these:

{queries}

Output:"""


def init_groq():
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("Run: pip install groq")

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
    print(f"LLM QUERY REWRITING (GROQ) — Model: {MODEL}")
    print("=" * 100)

    queries = load_json(QUERIES_FILE)
    print(f"Loaded {len(queries)} queries")

    cache = {}
    if CACHE_FILE.exists():
        cache = load_json(CACHE_FILE)
        print(f"Cache: {len(cache)} entries already rewritten")

    print(f"\nInitializing Groq...")
    client = init_groq()
    print("Ready.\n")

    to_rewrite = [
        (q["query_id"], q["query"], q["level"])
        for q in queries
        if q["level"] in LEVELS_TO_REWRITE and q["query_id"] not in cache
    ]

    total_batches = (len(to_rewrite) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Queries to rewrite : {len(to_rewrite)}")
    print(f"Batch size         : {BATCH_SIZE}")
    print(f"Total batches      : {total_batches}\n")

    start_time = time.time()

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(to_rewrite))
        batch = to_rewrite[start:end]

        query_ids = [b[0] for b in batch]
        query_texts = [b[1] for b in batch]

        elapsed = time.time() - start_time
        print(f"Batch {batch_idx + 1}/{total_batches} "
              f"({len(batch)} queries) [elapsed {elapsed:.0f}s]...")

        try:
            translations = rewrite_batch(client, query_texts)
            for qid, trans in zip(query_ids, translations):
                cache[qid] = trans

            save_json(CACHE_FILE, cache)
            print(f"  ✓ Done. Cache: {len(cache)} entries")

            if batch_idx < total_batches - 1:
                time.sleep(BATCH_DELAY)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            time.sleep(10)

    print(f"\nBuilding final output...")
    rewritten = []
    for q in queries:
        if q["level"] in LEVELS_TO_REWRITE and q["query_id"] in cache:
            rewritten_query = cache[q["query_id"]]
        else:
            rewritten_query = q["query"]

        rewritten.append({
            **q,
            "original_query": q["query"],
            "query": rewritten_query,
            "was_rewritten": (q["level"] in LEVELS_TO_REWRITE)
        })

    save_json(OUTPUT_FILE, rewritten)

    total_time = time.time() - start_time

    print()
    print("=" * 100)
    print(f"Total queries    : {len(queries)}")
    print(f"Rewritten        : {sum(1 for r in rewritten if r['was_rewritten'])}")
    print(f"Total time       : {total_time:.1f} seconds")
    print(f"Saved: {OUTPUT_FILE}")
    print("=" * 100)

    print("\nSAMPLE REWRITES:")
    samples = [r for r in rewritten if r["was_rewritten"]]
    for lvl in ["L2", "L3", "L4"]:
        lvl_samples = [r for r in samples if r["level"] == lvl][:2]
        print(f"\n--- {lvl} ---")
        for r in lvl_samples:
            print(f"  {r['query_id']}")
            print(f"    Original : {r['original_query']}")
            print(f"    Rewritten: {r['query']}")


if __name__ == "__main__":
    main()