"""
Fix Fallback Cache Entries in A_single Ablation (v3)
=====================================================
Detects entries where cache == original text (fallback), and retries.

Run: python src_v2/37_fix_empty_cache.py
"""

from pathlib import Path
import json
import os
import re
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"
CACHE_FILE = PROJECT_ROOT / "data_v2" / "processed" / "ablation_single.json"

MODEL = "openai/gpt-oss-120b"


PROMPT_1 = """Translate the following query into natural English.

Rules:
1. Preserve the EXACT information need.
2. If it's already English, return it unchanged.
3. Return ONLY the translation, no explanation.

Query: {query}

Translation:"""


PROMPT_2 = """Translate this query to English:

{query}

English:"""


PROMPT_3 = """You are a translator. Translate the Roman Urdu query below into English.

Example:
Input: "Kitab khana kitne baje khulta hai?"
Output: "What time does the library open?"

Now translate this:
Input: "{query}"
Output:"""


PROMPT_4 = """Rewrite the following query in fluent English. The output must be non-empty and must preserve the information need.

Input query: {query}

English rewrite (must not be empty):"""


PROMPT_5 = """Task: Translate Roman Urdu to English.

Input: "{query}"

Output (English only, must not be empty):"""


PROMPTS = [PROMPT_1, PROMPT_2, PROMPT_3, PROMPT_4, PROMPT_5]


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
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def rewrite_with_retry(client, text, qid):
    for i, prompt_template in enumerate(PROMPTS, start=1):
        try:
            result = call_llm(client, prompt_template.format(query=text))
            print(f"    Strategy {i}: {repr(result[:70])}")

            if result and result.strip():
                return result, i
        except Exception as e:
            print(f"    Strategy {i}: ERROR {e}")

        time.sleep(0.3)

    return None, None


def main():
    print("=" * 100)
    print("FIX FALLBACK CACHE ENTRIES (v3)")
    print("=" * 100)

    queries = load_json(ORIGINAL_FILE)
    cache = load_json(CACHE_FILE)

    print(f"Total queries : {len(queries)}")
    print(f"Cache entries : {len(cache)}")

    # Detect entries where cache == original (likely fallback) for L1-L4
    to_fix = []
    for q in queries:
        qid = q["query_id"]
        lvl = q["level"]
        if lvl == "L0":
            continue

        cached = cache.get(qid, "").strip()
        orig = q["query"].strip()

        if not cached or cached == orig:
            to_fix.append(q)

    print(f"\nQueries with fallback (cache == original): {len(to_fix)}")
    for q in to_fix:
        print(f"  {q['query_id']} ({q['level']}): {q['query'][:70]}")

    if not to_fix:
        print("\nNo fixes needed.")
        return

    print("\nInitializing Groq...")
    client = init_groq()
    print("Ready.\n")

    fixed = 0
    still_fallback = []

    for q in to_fix:
        qid = q["query_id"]
        text = q["query"]

        print(f"\n{'=' * 80}")
        print(f"Retrying {qid} ({q['level']})")
        print(f"  Original : {text[:90]}")

        result, strategy = rewrite_with_retry(client, text, qid)

        if result and result.strip() != text.strip():
            cache[qid] = result
            print(f"  [OK] Fixed with strategy {strategy}")
            print(f"  Result: {result[:90]}")
            fixed += 1
        else:
            cache[qid] = text  # keep fallback
            still_fallback.append(qid)
            print(f"  [FAIL] All strategies failed or same as original.")

        time.sleep(0.5)

    save_json(CACHE_FILE, cache)

    print(f"\n{'=' * 100}")
    print(f"Fixed        : {fixed}/{len(to_fix)}")
    print(f"Still fallback: {len(still_fallback)}")
    if still_fallback:
        print(f"Unfixed: {still_fallback}")
    print(f"Saved: {CACHE_FILE}")

    print(f"\n{'=' * 100}")
    print("Now re-run script 36 for updated numbers.")
    print("=" * 100)


if __name__ == "__main__":
    main()