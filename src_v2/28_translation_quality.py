"""
Translation Quality Assessment
================================
Computes BLEU and chrF for LLM translations.

Requires: pip install sacrebleu

For each level (L2/L3/L4), compares:
- Source: Roman Urdu query
- Hypothesis: LLM-rewritten query
- Reference: Human-written English equivalent (L0 query of same doc/variant)

Run: python src_v2/28_translation_quality.py
"""

from pathlib import Path
import json
from collections import defaultdict

try:
    from sacrebleu.metrics import BLEU, CHRF
except ImportError:
    print("Install: pip install sacrebleu")
    raise


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"
REWRITTEN_FILE = PROJECT_ROOT / "data_v2" / "processed" / "all_rewritten_queries.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "translation_quality.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "translation_quality.txt"

LEVELS = ["L0", "L1", "L2", "L3", "L4"]


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 100)
    print("TRANSLATION QUALITY ASSESSMENT (BLEU + chrF)")
    print("=" * 100)

    original = load_json(ORIGINAL_FILE)
    rewritten = load_json(REWRITTEN_FILE)

    # Build index for lookup
    orig_by_id = {q["query_id"]: q for q in original}
    rw_by_id = {q["query_id"]: q for q in rewritten}

    # Build reference lookup: for each (doc_id, variant), find the L0 query
    # (Human-written English equivalent)
    ref_lookup = {}
    for q in original:
        if q["level"] == "L0":
            key = (q["doc_id"], q["variant"])
            ref_lookup[key] = q["query"]

    print(f"Loaded {len(original)} original, {len(rewritten)} rewritten")
    print(f"Reference L0 queries: {len(ref_lookup)}\n")

    # BLEU and chrF
    bleu = BLEU(effective_order=True)
    chrf = CHRF(word_order=2)

    # Collect by level
    by_level = defaultdict(lambda: {"hyps": [], "refs": [], "ids": []})

    for q in rewritten:
        lvl = q["level"]
        if lvl == "L0":
            continue  # Skip L0 (unchanged)

        key = (q["doc_id"], q["variant"])
        if key not in ref_lookup:
            continue

        hyps = q["rewritten_query"]
        refs = ref_lookup[key]

        by_level[lvl]["hyps"].append(hyps)
        by_level[lvl]["refs"].append(refs)
        by_level[lvl]["ids"].append(q["query_id"])

    # Report
    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out()
    out("=" * 100)
    out("TRANSLATION QUALITY PER LEVEL")
    out("=" * 100)
    out(f"\n{'Level':<8} {'N':>6} {'BLEU':>10} {'chrF':>10}")
    out("-" * 100)

    results = {}
    for lvl in ["L1", "L2", "L3", "L4"]:
        data = by_level[lvl]
        if not data["hyps"]:
            continue

        bleu_score = bleu.corpus_score(data["hyps"], [data["refs"]]).score
        chrf_score = chrf.corpus_score(data["hyps"], [data["refs"]]).score

        results[lvl] = {
            "n": len(data["hyps"]),
            "bleu": bleu_score,
            "chrf": chrf_score,
        }

        out(f"{lvl:<8} {len(data['hyps']):>6} {bleu_score:>10.2f} {chrf_score:>10.2f}")

    # Overall (L2-L4 only, since those are the real translations)
    all_hyps = []
    all_refs = []
    for lvl in ["L2", "L3", "L4"]:
        all_hyps.extend(by_level[lvl]["hyps"])
        all_refs.extend(by_level[lvl]["refs"])

    bleu_overall = bleu.corpus_score(all_hyps, [all_refs]).score
    chrf_overall = chrf.corpus_score(all_hyps, [all_refs]).score

    out()
    out("=" * 100)
    out("OVERALL (L2 + L3 + L4 only — real translations)")
    out("=" * 100)
    out(f"N     : {len(all_hyps)}")
    out(f"BLEU  : {bleu_overall:.2f}")
    out(f"chrF  : {chrf_overall:.2f}")

    results["overall_L2_L4"] = {
        "n": len(all_hyps),
        "bleu": bleu_overall,
        "chrf": chrf_overall,
    }

    # Interpretation
    out()
    out("=" * 100)
    out("INTERPRETATION")
    out("=" * 100)
    out()
    out("BLEU scale (typical MT literature):")
    out("  < 10  : Poor translation")
    out("  10-20 : Low quality")
    out("  20-30 : Understandable")
    out("  30-40 : Good quality")
    out("  > 40  : High quality")
    out()
    out("chrF scale:")
    out("  < 30  : Poor")
    out("  30-50 : Moderate")
    out("  50-70 : Good")
    out("  > 70  : High")
    out()

    if bleu_overall >= 30:
        out(f"BLEU {bleu_overall:.1f} indicates GOOD translation quality.")
    elif bleu_overall >= 20:
        out(f"BLEU {bleu_overall:.1f} indicates UNDERSTANDABLE translation.")
    else:
        out(f"BLEU {bleu_overall:.1f} indicates LOW translation quality — a limitation.")

    out()
    out("Note: BLEU against human L0 references may underestimate quality")
    out("because L0 references are not strict translations — they're")
    out("independently written English queries with the same intent.")

    # Sample side-by-side (best and worst chrF)
    out()
    out("=" * 100)
    out("SAMPLE COMPARISONS (L4)")
    out("=" * 100)

    samples = []
    for qid, hyp, ref in zip(by_level["L4"]["ids"],
                              by_level["L4"]["hyps"],
                              by_level["L4"]["refs"]):
        # Sentence-level chrF
        s_chrf = chrf.sentence_score(hyp, [ref]).score
        samples.append((s_chrf, qid, hyp, ref))

    samples.sort(reverse=True)

    out()
    out("--- TOP 3 (best translation quality) ---")
    for score, qid, hyp, ref in samples[:3]:
        out(f"\n{qid} (chrF={score:.1f})")
        out(f"  Rewritten: {hyp}")
        out(f"  Reference: {ref}")

    out()
    out("--- BOTTOM 3 (worst translation quality) ---")
    for score, qid, hyp, ref in samples[-3:]:
        out(f"\n{qid} (chrF={score:.1f})")
        out(f"  Rewritten: {hyp}")
        out(f"  Reference: {ref}")

    # Save
    save_json(RESULTS_FILE, results)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    out()
    out(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()