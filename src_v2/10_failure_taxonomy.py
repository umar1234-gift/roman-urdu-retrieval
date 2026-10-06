from pathlib import Path
import json
from collections import Counter, defaultdict


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BM25_FILE = PROJECT_ROOT / "data_v2" / "results" / "bm25_results.json"
DENSE_FILE = PROJECT_ROOT / "data_v2" / "results" / "dense_results.json"
HYBRID_FILE = PROJECT_ROOT / "data_v2" / "results" / "hybrid_results.json"
CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"

OUTPUT_FILE = PROJECT_ROOT / "data_v2" / "results" / "failure_taxonomy.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "failure_taxonomy.txt"


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_failure(query, top1_doc, gold_doc, top1_text, gold_text):
    """
    Heuristic classification of why retrieval failed.
    """
    q_lower = query.lower()
    top1_lower = top1_text.lower()
    gold_lower = gold_text.lower()

    # Check if query terms appear in top1 or gold
    q_words = set(w for w in q_lower.split() if len(w) > 3)

    top1_overlap = sum(1 for w in q_words if w in top1_lower)
    gold_overlap = sum(1 for w in q_words if w in gold_lower)

    # Lexical collision: top1 shares many terms with query but not gold
    if top1_overlap > gold_overlap:
        return "lexical_collision"
    elif top1_overlap == gold_overlap and top1_overlap > 0:
        return "tied_lexical"
    else:
        return "semantic_drift"


def main():
    bm25 = load_json(BM25_FILE)
    dense = load_json(DENSE_FILE)
    hybrid = load_json(HYBRID_FILE)
    chunks = load_json(CHUNKS_FILE)

    text_by_filename = {c["filename"]: c["text"] for c in chunks}

    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out("=" * 100)
    out("FAILURE TAXONOMY ANALYSIS")
    out("=" * 100)

    # --------------------------------------------------------
    # 1. Failure types by system
    # --------------------------------------------------------
    out()
    out("FAILURE TYPE BY SYSTEM (top-1 wrong cases)")
    out("-" * 100)

    taxonomy = {}

    for name, records in [("BM25", bm25), ("E5", dense), ("Hybrid", hybrid)]:
        types = Counter()
        failures = []
        for r in records:
            if r["top1_correct"]:
                continue
            top1_doc = r["top1_filename"]
            gold_doc = r["gold_document"]
            top1_text = text_by_filename.get(top1_doc, "")
            gold_text = text_by_filename.get(gold_doc, "")

            ftype = classify_failure(r["query"], top1_doc, gold_doc,
                                      top1_text, gold_text)
            types[ftype] += 1
            failures.append({
                "query_id": r["query_id"],
                "doc_id": r["doc_id"],
                "level": r["level"],
                "query": r["query"],
                "gold": gold_doc,
                "top1": top1_doc,
                "gold_rank": r["gold_rank"],
                "failure_type": ftype
            })

        taxonomy[name] = {
            "total_failures": len(failures),
            "by_type": dict(types),
            "failures": failures
        }

        out()
        out(f"--- {name} ---")
        for ftype, count in types.most_common():
            out(f"  {ftype:20s}: {count}")

    # --------------------------------------------------------
    # 2. Failures by level
    # --------------------------------------------------------
    out()
    out("FAILURE COUNT BY LEVEL (per system)")
    out("-" * 100)
    out(f"{'System':<10} {'L0':>6} {'L1':>6} {'L2':>6} {'L3':>6} {'L4':>6}")
    out("-" * 100)

    for name in ["BM25", "E5", "Hybrid"]:
        failures = taxonomy[name]["failures"]
        by_level = Counter(f["level"] for f in failures)
        out(f"{name:<10} " + " ".join(f"{by_level.get(l, 0):>6}" for l in ["L0","L1","L2","L3","L4"]))

    # --------------------------------------------------------
    # 3. Hardest docs (fails per doc)
    # --------------------------------------------------------
    out()
    out("FAILS PER DOCUMENT (sum across all 3 systems, max=45)")
    out("-" * 100)
    out(f"{'Doc':<6} {'BM25':>6} {'E5':>6} {'Hybrid':>6} {'Total':>6}")
    out("-" * 100)

    fails_by_doc = defaultdict(lambda: {"BM25": 0, "E5": 0, "Hybrid": 0})
    for name in ["BM25", "E5", "Hybrid"]:
        for f in taxonomy[name]["failures"]:
            fails_by_doc[f["doc_id"]][name] += 1

    sorted_docs = sorted(fails_by_doc.items(),
                          key=lambda x: -(x[1]["BM25"] + x[1]["E5"] + x[1]["Hybrid"]))

    for doc_id, counts in sorted_docs:
        total = counts["BM25"] + counts["E5"] + counts["Hybrid"]
        if total > 0:
            out(f"{doc_id:<6} {counts['BM25']:>6} {counts['E5']:>6} {counts['Hybrid']:>6} {total:>6}")

    # --------------------------------------------------------
    # 4. Persistent failures (all 3 systems fail)
    # --------------------------------------------------------
    out()
    out("PERSISTENT FAILURES (all 3 systems fail)")
    out("-" * 100)

    bm25_by_id = {r["query_id"]: r for r in bm25}
    e5_by_id = {r["query_id"]: r for r in dense}
    hyb_by_id = {r["query_id"]: r for r in hybrid}

    persistent = []
    for qid in bm25_by_id:
        b = bm25_by_id[qid]
        e = e5_by_id[qid]
        h = hyb_by_id[qid]

        if (not b["top1_correct"] and not e["top1_correct"]
                and not h["top1_correct"]):
            persistent.append({
                "query_id": qid,
                "doc_id": b["doc_id"],
                "level": b["level"],
                "query": b["query"],
                "gold": b["gold_document"],
                "bm25_rank": b["gold_rank"],
                "e5_rank": e["gold_rank"],
                "hybrid_rank": h["gold_rank"],
                "bm25_top": b["top1_filename"],
                "e5_top": e["top1_filename"],
                "hybrid_top": h["top1_filename"]
            })

    out(f"Total persistent failures: {len(persistent)}")
    out()
    for p in persistent[:30]:
        out(f"{p['query_id']} {p['doc_id']} {p['level']} | "
            f"BM25_r={p['bm25_rank']} E5_r={p['e5_rank']} Hyb_r={p['hybrid_rank']} | "
            f"gold={p['gold']}")
        out(f"   Q: {p['query'][:80]}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    output = {
        "taxonomy": taxonomy,
        "persistent_failures": persistent,
        "fails_by_doc": {k: dict(v) for k, v in fails_by_doc.items()}
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    out()
    out(f"Saved: {OUTPUT_FILE}")
    out(f"Saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()