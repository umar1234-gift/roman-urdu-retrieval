from pathlib import Path
import json
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "dense_results.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "dense_report.txt"

MODEL_NAME = "intfloat/multilingual-e5-base"
TOP_K = 10
LEVELS = ["L0", "L1", "L2", "L3", "L4"]


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(records):
    total = len(records)
    g1 = sum(1 for r in records if r["gold_rank"] == 1)
    g3 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 3)
    g5 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 5)
    g10 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 10)
    nf = sum(1 for r in records if r["gold_rank"] is None)
    mrr = sum(1/r["gold_rank"] for r in records if r["gold_rank"]) / total
    return {
        "total": total,
        "gold_at_1": g1, "gold_at_3": g3, "gold_at_5": g5, "gold_at_10": g10,
        "not_found": nf,
        "p1": g1/total, "p3": g3/total, "p5": g5/total, "p10": g10/total,
        "mrr": mrr
    }


def main():
    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)

    print("=" * 100)
    print("DENSE RETRIEVAL (Multilingual-E5) — COMPREHENSIVE")
    print("=" * 100)
    print(f"Chunks loaded : {len(chunks)}")
    print(f"Queries loaded: {len(queries)}")

    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.\n")

    # Encode documents
    doc_texts = ["passage: " + c["text"] for c in chunks]
    print("Encoding documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True,
                            show_progress_bar=True)
    filenames = [c["filename"] for c in chunks]
    print("Document encoding complete.\n")

    results = []
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("DENSE RETRIEVAL (E5) — FULL REPORT")
    report_lines.append("=" * 100)

    print("-" * 100)
    print("PER-QUERY RESULTS")
    print("-" * 100)

    for q in queries:
        q_emb = model.encode(["query: " + q["query"]],
                              normalize_embeddings=True)[0]
        sims = np.dot(doc_emb, q_emb)
        order = np.argsort(sims)[::-1]

        gold_rank = None
        gold_score = None
        for rank, idx in enumerate(order, start=1):
            if filenames[idx] == q["gold_document"]:
                gold_rank = rank
                gold_score = float(sims[idx])
                break

        top_results = [
            {"rank": r, "filename": filenames[i],
             "score": round(float(sims[i]), 6)}
            for r, i in enumerate(order[:TOP_K], start=1)
        ]
        top1 = top_results[0]["filename"]
        correct_top1 = (top1 == q["gold_document"])

        record = {
            "query_id": q["query_id"], "doc_id": q["doc_id"],
            "level": q["level"], "variant": q["variant"],
            "query": q["query"], "gold_answer": q["gold_answer"],
            "gold_document": q["gold_document"],
            "gold_rank": gold_rank, "gold_score": gold_score,
            "top1_filename": top1, "top1_correct": correct_top1,
            "top_results": top_results
        }
        results.append(record)

        line = (f"{q['query_id']} {q['doc_id']} {q['level']} V{q['variant']} | "
                f"GoldRank={gold_rank if gold_rank else 'NF'} | "
                f"Top1={top1} | {'✓' if correct_top1 else '✗'}")
        print(line)
        report_lines.append(line)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- OVERALL ----
    overall = compute_metrics(results)
    print()
    print("=" * 100)
    print("OVERALL METRICS")
    print("=" * 100)
    print(f"Total queries : {overall['total']}")
    print(f"Gold @1       : {overall['gold_at_1']}/{overall['total']} = {overall['p1']:.4f}")
    print(f"Gold @3       : {overall['gold_at_3']}/{overall['total']} = {overall['p3']:.4f}")
    print(f"Gold @5       : {overall['gold_at_5']}/{overall['total']} = {overall['p5']:.4f}")
    print(f"Gold @10      : {overall['gold_at_10']}/{overall['total']} = {overall['p10']:.4f}")
    print(f"Not found     : {overall['not_found']}")
    print(f"MRR           : {overall['mrr']:.4f}")

    report_lines += ["", "=" * 100, "OVERALL METRICS", "=" * 100,
                     f"Total queries : {overall['total']}",
                     f"Gold @1       : {overall['gold_at_1']}/{overall['total']} = {overall['p1']:.4f}",
                     f"Gold @3       : {overall['gold_at_3']}/{overall['total']} = {overall['p3']:.4f}",
                     f"Gold @5       : {overall['gold_at_5']}/{overall['total']} = {overall['p5']:.4f}",
                     f"Gold @10      : {overall['gold_at_10']}/{overall['total']} = {overall['p10']:.4f}",
                     f"Not found     : {overall['not_found']}",
                     f"MRR           : {overall['mrr']:.4f}"]

    # ---- PER LEVEL ----
    print()
    print("=" * 100)
    print("PER-LEVEL METRICS")
    print("=" * 100)
    print(f"{'Level':<6} {'@1':>8} {'@3':>8} {'@5':>8} {'@10':>8} {'MRR':>8} {'NF':>4}")
    print("-" * 60)

    report_lines += ["", "=" * 100, "PER-LEVEL METRICS", "=" * 100]
    per_level = {}
    for lvl in LEVELS:
        subset = [r for r in results if r["level"] == lvl]
        m = compute_metrics(subset)
        per_level[lvl] = m
        line = f"{lvl:<6} {m['p1']:>8.4f} {m['p3']:>8.4f} {m['p5']:>8.4f} {m['p10']:>8.4f} {m['mrr']:>8.4f} {m['not_found']:>4}"
        print(line)
        report_lines.append(line)

    # ---- DEGRADATION ----
    l0_p1 = per_level["L0"]["p1"]
    l4_p1 = per_level["L4"]["p1"]
    drop = l0_p1 - l4_p1
    pct = (drop / l0_p1 * 100) if l0_p1 else 0

    print()
    print("=" * 100)
    print("DEGRADATION L0 → L4")
    print("=" * 100)
    print(f"L0 @1 = {l0_p1:.4f}")
    print(f"L4 @1 = {l4_p1:.4f}")
    print(f"Drop  = {drop:.4f} ({pct:+.1f}%)")

    report_lines += ["", "=" * 100, "DEGRADATION L0 → L4", "=" * 100,
                     f"L0 @1 = {l0_p1:.4f}",
                     f"L4 @1 = {l4_p1:.4f}",
                     f"Drop  = {drop:.4f} ({pct:+.1f}%)"]

    # ---- PER DOCUMENT ----
    print()
    print("=" * 100)
    print("PER-DOCUMENT METRICS")
    print("=" * 100)
    print(f"{'Doc':<6} {'avg_rank':>10} {'@1':>6} {'@1_rate':>10}")

    report_lines += ["", "=" * 100, "PER-DOCUMENT METRICS", "=" * 100]
    for doc_id in sorted({r["doc_id"] for r in results}):
        subset = [r for r in results if r["doc_id"] == doc_id]
        ranks = [r["gold_rank"] for r in subset if r["gold_rank"]]
        avg_rank = sum(ranks) / len(ranks) if ranks else None
        g1 = sum(1 for r in subset if r["gold_rank"] == 1)
        rate = g1 / len(subset)
        line = f"{doc_id:<6} {avg_rank:>10.2f} {g1:>6} {rate:>10.2f}"
        print(line)
        report_lines.append(line)

    # ---- WRONG-ABOVE-GOLD ----
    print()
    print("=" * 100)
    print("WRONG-ABOVE-GOLD")
    print("=" * 100)
    wrong = [r for r in results if not r["top1_correct"]]
    print(f"Count: {len(wrong)}/{len(results)} = {len(wrong)/len(results):.4f}")
    print()
    for r in wrong:
        line = (f"{r['query_id']} {r['doc_id']} {r['level']} V{r['variant']} | "
                f"gold={r['gold_document']} (rank {r['gold_rank']}) | "
                f"top1={r['top1_filename']}")
        print(line)
        report_lines.append(line)

    # ---- WRONG BY LEVEL ----
    print()
    print("=" * 100)
    print("WRONG-ABOVE-GOLD BY LEVEL")
    print("=" * 100)
    report_lines += ["", "=" * 100, "WRONG-ABOVE-GOLD BY LEVEL", "=" * 100]
    wrong_by_level = defaultdict(int)
    total_by_level = defaultdict(int)
    for r in results:
        total_by_level[r["level"]] += 1
        if not r["top1_correct"]:
            wrong_by_level[r["level"]] += 1
    for lvl in LEVELS:
        w = wrong_by_level[lvl]
        t = total_by_level[lvl]
        rate = w / t if t else 0
        line = f"{lvl}: {w}/{t} = {rate:.4f}"
        print(line)
        report_lines.append(line)

    # ---- HARDEST ----
    print()
    print("=" * 100)
    print("HARDEST QUERIES (gold rank > 5)")
    print("=" * 100)
    report_lines += ["", "=" * 100, "HARDEST QUERIES (gold rank > 5)", "=" * 100]
    hard = sorted([r for r in results if r["gold_rank"] and r["gold_rank"] > 5],
                  key=lambda r: -r["gold_rank"])
    for r in hard[:20]:
        line = (f"rank={r['gold_rank']:>3} | "
                f"{r['query_id']} {r['doc_id']} {r['level']} | "
                f"{r['query'][:70]}")
        print(line)
        report_lines.append(line)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print()
    print("=" * 100)
    print(f"Results saved : {RESULTS_FILE}")
    print(f"Report saved  : {REPORT_FILE}")
    print("=" * 100)


if __name__ == "__main__":
    main()