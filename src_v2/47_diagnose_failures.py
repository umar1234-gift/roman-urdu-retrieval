"""
Diagnose Verification Failures
================================
Shows exactly which queries are duplicated and which are unchanged.

Run: python src_v2/47_diagnose_failures.py
"""

from pathlib import Path
import json
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"
A_SINGLE_FILE = PROJECT_ROOT / "data_v2" / "processed" / "ablation_single_clean.json"

REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "diagnose_failures.txt"

LEVELS = ["L0", "L1", "L2", "L3", "L4"]


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    queries = load_json(QUERIES_FILE)
    a_single = load_json(A_SINGLE_FILE)

    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out("=" * 100)
    out("DIAGNOSE VERIFICATION FAILURES")
    out("=" * 100)

    # ---- Failure 1: Unchanged queries ----
    out()
    out("=" * 100)
    out("FAILURE 1: 'Fallback' analysis")
    out("=" * 100)
    out()
    out("For each non-L0 query, compare cache text to original text.")
    out("Classify by level:")
    out()

    by_level = defaultdict(lambda: {"total": 0, "unchanged": 0, "changed": 0})
    unchanged_ids = []

    for q in queries:
        qid = q["query_id"]
        lvl = q["level"]
        if lvl == "L0":
            continue

        by_level[lvl]["total"] += 1
        cached = a_single.get(qid, "").strip()
        original = q["query"].strip()

        if cached == original:
            by_level[lvl]["unchanged"] += 1
            unchanged_ids.append({"query_id": qid, "level": lvl,
                                   "text": original[:80]})
        else:
            by_level[lvl]["changed"] += 1

    out(f"{'Level':<8} {'Total':>8} {'Unchanged':>12} {'Changed':>10}")
    out("-" * 50)
    for lvl in LEVELS:
        d = by_level[lvl]
        if d["total"] > 0:
            out(f"{lvl:<8} {d['total']:>8} {d['unchanged']:>12} {d['changed']:>10}")

    out()
    out("INTERPRETATION:")
    out("  - L1 unchanged: EXPECTED (queries already ~English; LLM correctly left as-is)")
    out("  - L2/L3/L4 unchanged: potential REAL fallbacks")
    out()

    l2l4_unchanged = [u for u in unchanged_ids if u["level"] in ["L2", "L3", "L4"]]
    out(f"L2/L3/L4 unchanged (real fallback candidates): {len(l2l4_unchanged)}")
    for u in l2l4_unchanged:
        out(f"  {u['query_id']} ({u['level']}): {u['text']}")

    # ---- Failure 2: Duplicate queries ----
    out()
    out("=" * 100)
    out("FAILURE 2: Duplicate queries across levels")
    out("=" * 100)
    out()

    by_dv = defaultdict(dict)
    for q in queries:
        key = (q["doc_id"], q["variant"])
        by_dv[key][q["level"]] = q["query"].strip()

    duplicates = []
    for key, lvl_dict in by_dv.items():
        texts = {}
        for lvl, text in lvl_dict.items():
            texts.setdefault(text, []).append(lvl)

        for text, lvls in texts.items():
            if len(lvls) > 1:
                duplicates.append({
                    "doc_id": key[0],
                    "variant": key[1],
                    "levels": lvls,
                    "text": text,
                })

    out(f"Total (doc, variant) groups with duplicates: {len(duplicates)}/90")
    out()
    out("Level-pair distribution:")
    pair_counts = defaultdict(int)
    for d in duplicates:
        pair = tuple(sorted(d["levels"]))
        pair_counts[pair] += 1

    for pair, count in sorted(pair_counts.items(), key=lambda x: -x[1]):
        out(f"  {pair}: {count}")
    out()
    out("First 15 duplicates:")
    out("-" * 100)
    for d in duplicates[:15]:
        out(f"  {d['doc_id']} V{d['variant']} levels={d['levels']}")
        out(f"    text: {d['text'][:90]}")

    # ---- Recommendations ----
    out()
    out("=" * 100)
    out("RECOMMENDATIONS")
    out("=" * 100)
    out()

    if len(l2l4_unchanged) <= 2:
        out("FAILURE 1: Check logic was too strict.")
        out(f"  Real L2/L3/L4 fallbacks: {len(l2l4_unchanged)}")
        out(f"  L1 unchanged (expected): {by_level['L1']['unchanged']}")
        out("  → Fix verification check: only count L2/L3/L4 as fallbacks")
    else:
        out(f"FAILURE 1: {len(l2l4_unchanged)} L2/L3/L4 unchanged — investigate")

    out()
    if len(duplicates) == 0:
        out("FAILURE 2: No duplicates — verification check bug")
    else:
        out(f"FAILURE 2: {len(duplicates)} duplicates across levels.")
        out("  Options:")
        out("   a) Disclose in paper Limitations: 'N queries are surface-identical")
        out("      across adjacent levels, which may amplify similarity between L1/L2'")
        out("   b) Fix L1 queries to include Urdu markers")
        out("   c) Accept as-is: real-world queries sometimes look identical to English")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nSaved: {REPORT_FILE}")


if __name__ == "__main__":
    main()