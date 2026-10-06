"""
Manual Fix for 3 Broken Entries
================================
"""

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = PROJECT_ROOT / "data_v2" / "processed" / "ablation_single.json"

MANUAL_FIXES = {
    "MQ0169": "What is the hostel curfew time for residents?",
    "MQ0364": "How many counselling sessions is a student entitled to per semester?",
    "MQ0389": "Do we get the name of the person who gave the feedback?",
}

def main():
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    for qid, text in MANUAL_FIXES.items():
        old = cache.get(qid, "")
        cache[qid] = text
        print(f"{qid}:")
        print(f"  Old: {old[:80]}")
        print(f"  New: {text}")

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {CACHE_FILE}")

if __name__ == "__main__":
    main()