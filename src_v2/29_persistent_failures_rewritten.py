"""
Persistent Failures Analysis (After LLM Rewriting)
===================================================
Identifies queries that fail even after LLM rewriting.

Run: python src_v2/29_persistent_failures_rewritten.py
"""

from pathlib import Path
import json
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RETRIEVAL_FILE = PROJECT_ROOT / "data_v2" / "results" / "retrieval_all_rewritten.json"
REWRITTEN_FILE = PROJECT_ROOT / "data_v2" / "processed" / "all_rewritten_queries.json"
CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "persistent_failures.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "persistent_failures.txt"

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
    print("PERSISTENT FAILURES ANALYSIS (AFTER LLM REWRITING)")
    print("=" * 100)

    retrieval = load_json(RETRIEVAL_FILE)
    rewritten = load_json(REWRITTEN_FILE)
    chunks = load_json(CHUNKS_FILE)

    text_by_filename = {c["filename"]: c["text"] for c in chunks}
    rewritten_by_id = {q["query_id"]: q for q in rewritten}

    # retrieval has structure: {"systems": {...}, "per_query": {system: [records]}}
    per_query = defaultdict(lambda: {})

    for system_name, recs in retrieval["per_query"].items():
        for r in recs:
            qid = r["query_id"]
            per_query[qid][system_name] = r["gold_rank"]

    # Identify persistent failures (all systems fail)
    persistent = []
    for qid, systems in per_query.items():
        if all(rank != 1 for rank in systems.values()):
            record = {
                "query_id": qid,
                "ranks": dict(systems),
            }
            if qid in rewritten_by_id:
                q = rewritten_by_id[qid]
                record["doc_id"] = q["doc_id"]
                record["level"] = q["level"]
                record["original_query"] = q["original_query"]
                record["rewritten_query"] = q["rewritten_query"]
                record["gold_document"] = q["gold_document"]

            persistent.append(record)

    persistent.sort(key=lambda x: (x.get("level", "?"), x.get("doc_id", "?"), x["query_id"]))

    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out()
    out("=" * 100)
    out(f"PERSISTENT FAILURES: {len(persistent)} queries")
    out("=" * 100)

    # By level
    by_level = defaultdict(int)
    for p in persistent:
        by_level[p.get("level", "?")] += 1

    out()
    out("By Level:")
    for lvl in LEVELS:
        out(f"  {lvl}: {by_level[lvl]}")

    # By doc
    by_doc = defaultdict(int)
    for p in persistent:
        by_doc[p.get("doc_id", "?")] += 1

    out()
    out("By Document:")
    for doc_id in sorted(by_doc.keys()):
        out(f"  {doc_id}: {by_doc[doc_id]}")

    # Detailed list
    out()
    out("=" * 100)
    out("DETAILED LIST")
    out("=" * 100)

    for p in persistent:
        out()
        out(f"{p['query_id']} | {p.get('doc_id', '?')} | {p.get('level', '?')}")
        out(f"  Gold doc      : {p.get('gold_document', '?')}")
        out(f"  Original      : {p.get('original_query', '?')}")
        out(f"  Rewritten     : {p.get('rewritten_query', '?')}")
        ranks_str = " / ".join(f"{k}={v}" for k, v in p["ranks"].items())
        out(f"  Ranks         : {ranks_str}")

    # Classification
    out()
    out("=" * 100)
    out("FAILURE CLASSIFICATION")
    out("=" * 100)

    for p in persistent:
        gold = p.get("gold_document", "")
        rewritten_text = p.get("rewritten_query", "").lower()

        if gold in text_by_filename:
            gold_text = text_by_filename[gold].lower()
            rw_words = set(w for w in rewritten_text.split() if len(w) > 3)
            gold_words = set(gold_text.split())
            overlap = rw_words & gold_words
            overlap_pct = len(overlap) / len(rw_words) if rw_words else 0

            out()
            out(f"{p['query_id']} ({p.get('level', '?')})")
            out(f"  Rewritten: {p.get('rewritten_query', '')[:80]}")
            out(f"  Overlap with gold doc: {overlap_pct:.1%}")

            if overlap_pct < 0.3:
                out(f"  -> Likely VOCABULARY GAP")
            elif overlap_pct < 0.6:
                out(f"  -> Likely CROSS-DOCUMENT COLLISION")
            else:
                out(f"  -> Likely SEMANTIC DRIFT")

    # Summary
    out()
    out("=" * 100)
    out("SUMMARY")
    out("=" * 100)
    out(f"Total persistent failures: {len(persistent)} / 450 ({len(persistent)/450:.2%})")
    out()
    out("Comparison:")
    out(f"  Before LLM rewriting: 34 / 450 (7.56%)")
    out(f"  After  LLM rewriting: {len(persistent)} / 450 ({len(persistent)/450:.2%})")
    out(f"  Reduction: {34 - len(persistent)} queries fixed")

    save_json(RESULTS_FILE, {
        "persistent_failures": persistent,
        "by_level": dict(by_level),
        "by_doc": dict(by_doc),
        "total": len(persistent),
    })

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    out()
    out(f"Saved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()