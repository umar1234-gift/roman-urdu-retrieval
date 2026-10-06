"""
Clean Fallback Policy — Honest A_single Numbers
================================================
Removes manual fixes from cache, applies deterministic fallback.

Run: python src_v2/39_clean_fallback.py
"""

from pathlib import Path
import json
import os
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"
CACHE_FILE = PROJECT_ROOT / "data_v2" / "processed" / "ablation_single.json"
CLEAN_CACHE_FILE = PROJECT_ROOT / "data_v2" / "processed" / "ablation_single_clean.json"

MODEL = "openai/gpt-oss-120b"

# Queries with known manual fixes — reset to empty
MANUALLY_FIXED = ["MQ0169", "MQ0364", "MQ0389"]

# Query that was retried
RETRIED = ["MQ0021"]

PROMPT_SINGLE = """Translate the following query into natural English.

Rules:
1. Preserve the EXACT information need.
2. If it's already English, return it unchanged.
3. Return ONLY the translation, no explanation.

Query: {query}

Translation:"""


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


def call_llm(client, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )
    # Check finish reason
    finish = response.choices[0].finish_reason
    content = response.choices[0].message.content
    return content.strip() if content else "", finish


def main():
    print("=" * 100)
    print("CLEAN FALLBACK POLICY — HONEST NUMBERS")
    print("=" * 100)

    queries = load_json(ORIGINAL_FILE)
    cache = load_json(CACHE_FILE)

    print(f"Total queries : {len(queries)}")
    print(f"Cache entries : {len(cache)}")

    # Start with fresh cache
    clean_cache = {}
    report = []

    def log(s):
        print(s)
        report.append(s)

    # Manually fixed queries need re-query with higher max_tokens
    log(f"\n--- Step 1: Re-query manually-fixed queries with max_tokens=300 ---")
    log(f"IDs: {MANUALLY_FIXED}")

    client = init_groq()

    for qid in MANUALLY_FIXED:
        q = next((x for x in queries if x["query_id"] == qid), None)
        if not q:
            continue

        text = q["query"]
        log(f"\n{qid} ({q['level']}): {text[:80]}")

        result, finish = call_llm(client, PROMPT_SINGLE.format(query=text))
        log(f"  LLM output: {repr(result[:80])}")
        log(f"  Finish reason: {finish}")

        if result and result.strip():
            clean_cache[qid] = result
            log(f"  [OK] Stored LLM output")
        else:
            clean_cache[qid] = text  # fallback
            log(f"  [FALLBACK] Empty → using original text")

        time.sleep(1)

    # Copy all other cached values (excluding manually fixed)
    log(f"\n--- Step 2: Copy remaining cache entries ---")
    for q in queries:
        qid = q["query_id"]
        if qid in clean_cache:
            continue  # already processed
        if qid in MANUALLY_FIXED:
            continue

        cached = cache.get(qid, "")
        if cached and cached.strip():
            clean_cache[qid] = cached
        else:
            clean_cache[qid] = q["query"]  # fallback

    save_json(CLEAN_CACHE_FILE, clean_cache)

    # Analysis
    log(f"\n--- Step 3: Analysis ---")
    log(f"Total entries in clean cache: {len(clean_cache)}")

    fallback_count = 0
    fallback_ids = []

    for q in queries:
        qid = q["query_id"]
        if q["level"] == "L0":
            continue
        cached = clean_cache.get(qid, "").strip()
        orig = q["query"].strip()
        if cached == orig:
            fallback_count += 1
            fallback_ids.append(qid)

    log(f"\nFallback entries (cache == original): {fallback_count}")
    for qid in fallback_ids:
        log(f"  {qid}")

    with open(PROJECT_ROOT / "data_v2" / "results" / "clean_fallback_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    log(f"\nSaved: {CLEAN_CACHE_FILE}")
    log(f"\nNow run script 36 with CLEAN_CACHE_FILE to get honest numbers.")


if __name__ == "__main__":
    main()