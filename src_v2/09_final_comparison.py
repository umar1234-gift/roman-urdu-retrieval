from pathlib import Path
import json
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BM25_FILE = PROJECT_ROOT / "data_v2" / "results" / "bm25_results.json"
DENSE_FILE = PROJECT_ROOT / "data_v2" / "results" / "dense_results.json"
HYBRID_FILE = PROJECT_ROOT / "data_v2" / "results" / "hybrid_results.json"

OUTPUT_TXT = PROJECT_ROOT / "data_v2" / "results" / "final_comparison.txt"
OUTPUT_JSON = PROJECT_ROOT / "data_v2" / "results" / "final_comparison.json"

LEVELS = ["L0", "L1", "L2", "L3", "L4"]


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def metrics(records):
    total = len(records)
    g1 = sum(1 for r in records if r["gold_rank"] == 1)
    g3 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 3)
    g5 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 5)
    g10 = sum(1 for r in records if r["gold_rank"] and r["gold_rank"] <= 10)
    nf = sum(1 for r in records if r["gold_rank"] is None)
    mrr = sum(1/r["gold_rank"] for r in records if r["gold_rank"]) / total
    return {
        "@1": g1/total, "@3": g3/total, "@5": g5/total, "@10": g10/total,
        "MRR": mrr, "NF": nf, "total": total
    }


def main():
    bm25 = load_json(BM25_FILE)
    dense = load_json(DENSE_FILE)
    hybrid = load_json(HYBRID_FILE)

    systems = {
        "BM25": bm25,
        "E5": dense,
        "Hybrid": hybrid
    }

    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out("=" * 100)
    out("FINAL COMPARISON: BM25 vs E5 vs Hybrid RRF")
    out("=" * 100)

    # --------------------------------------------------------
    # TABLE 1: OVERALL
    # --------------------------------------------------------
    out()
    out("TABLE 1: OVERALL PERFORMANCE (n=450)")
    out("-" * 100)
    out(f"{'System':<10} {'@1':>8} {'@3':>8} {'@5':>8} {'@10':>8} {'MRR':>8}")
    out("-" * 100)

    table1 = {}
    for name, recs in systems.items():
        m = metrics(recs)
        table1[name] = m
        out(f"{name:<10} {m['@1']:>8.4f} {m['@3']:>8.4f} {m['@5']:>8.4f} {m['@10']:>8.4f} {m['MRR']:>8.4f}")

    # --------------------------------------------------------
    # TABLE 2: PER LEVEL
    # --------------------------------------------------------
    out()
    out("TABLE 2: PER-LEVEL PERFORMANCE (@1)")
    out("-" * 100)
    out(f"{'System':<10} " + " ".join(f"{lvl:>8}" for lvl in LEVELS) + "  drop_L0_L4")
    out("-" * 100)

    table2 = {}
    for name, recs in systems.items():
        row = {}
        for lvl in LEVELS:
            subset = [r for r in recs if r["level"] == lvl]
            row[lvl] = metrics(subset)["@1"]
        drop = row["L0"] - row["L4"]
        table2[name] = row
        out(f"{name:<10} " + " ".join(f"{row[lvl]:>8.4f}" for lvl in LEVELS) + f"   {drop:>+8.4f}")

    # --------------------------------------------------------
    # TABLE 3: PER-LEVEL MRR
    # --------------------------------------------------------
    out()
    out("TABLE 3: PER-LEVEL PERFORMANCE (MRR)")
    out("-" * 100)
    out(f"{'System':<10} " + " ".join(f"{lvl:>8}" for lvl in LEVELS))
    out("-" * 100)

    for name, recs in systems.items():
        row = {}
        for lvl in LEVELS:
            subset = [r for r in recs if r["level"] == lvl]
            row[lvl] = metrics(subset)["MRR"]
        out(f"{name:<10} " + " ".join(f"{row[lvl]:>8.4f}" for lvl in LEVELS))

    # --------------------------------------------------------
    # TABLE 4: WRONG ABOVE GOLD
    # --------------------------------------------------------
    out()
    out("TABLE 4: WRONG-ABOVE-GOLD RATE (top-1 incorrect)")
    out("-" * 100)
    out(f"{'System':<10} {'Overall':>10} " + " ".join(f"{lvl:>8}" for lvl in LEVELS))
    out("-" * 100)

    table4 = {}
    for name, recs in systems.items():
        total = len(recs)
        wrong = sum(1 for r in recs if not r["top1_correct"])
        row = {"overall": wrong/total}
        for lvl in LEVELS:
            subset = [r for r in recs if r["level"] == lvl]
            w = sum(1 for r in subset if not r["top1_correct"])
            row[lvl] = w/len(subset)
        table4[name] = row
        out(f"{name:<10} {row['overall']:>10.4f} " + " ".join(f"{row[lvl]:>8.4f}" for lvl in LEVELS))

    # --------------------------------------------------------
    # TABLE 5: PER-DOCUMENT (avg gold rank)
    # --------------------------------------------------------
    out()
    out("TABLE 5: PER-DOCUMENT AVERAGE GOLD RANK (lower = better)")
    out("-" * 100)
    out(f"{'Doc':<6} {'BM25':>8} {'E5':>8} {'Hybrid':>8}  {'Best':>8}")
    out("-" * 100)

    by_doc = defaultdict(dict)
    for name, recs in systems.items():
        for r in recs:
            by_doc[r["doc_id"]].setdefault(name, []).append(r)

    worst_docs = []
    for doc_id in sorted(by_doc.keys()):
        ranks = {}
        for name in systems:
            recs = by_doc[doc_id].get(name, [])
            vals = [r["gold_rank"] for r in recs if r["gold_rank"]]
            ranks[name] = sum(vals)/len(vals) if vals else 0
        best = min(ranks, key=ranks.get)
        out(f"{doc_id:<6} {ranks['BM25']:>8.2f} {ranks['E5']:>8.2f} {ranks['Hybrid']:>8.2f}  {best:>8}")
        worst_docs.append((doc_id, ranks))

    # --------------------------------------------------------
    # TABLE 6: HYBRID WINS OVER INDIVIDUAL SYSTEMS
    # --------------------------------------------------------
    out()
    out("TABLE 6: HYBRID ADVANTAGE OVER INDIVIDUAL SYSTEMS")
    out("-" * 100)

    bm25_by_id = {r["query_id"]: r for r in bm25}
    e5_by_id = {r["query_id"]: r for r in dense}
    hyb_by_id = {r["query_id"]: r for r in hybrid}

    wins_over_bm25 = 0
    losses_to_bm25 = 0
    wins_over_e5 = 0
    losses_to_e5 = 0

    for qid in bm25_by_id:
        h = hyb_by_id[qid]
        b = bm25_by_id[qid]
        e = e5_by_id[qid]

        h_r = h["gold_rank"] or 999
        b_r = b["gold_rank"] or 999
        e_r = e["gold_rank"] or 999

        if h_r < b_r:
            wins_over_bm25 += 1
        elif h_r > b_r:
            losses_to_bm25 += 1

        if h_r < e_r:
            wins_over_e5 += 1
        elif h_r > e_r:
            losses_to_e5 += 1

    out(f"Hybrid vs BM25: wins={wins_over_bm25}, losses={losses_to_bm25}, ties=450-wins-losses")
    out(f"Hybrid vs E5:   wins={wins_over_e5}, losses={losses_to_e5}, ties=450-wins-losses")

    # --------------------------------------------------------
    # TABLE 7: COMPLEMENTARY CASE ANALYSIS
    # --------------------------------------------------------
    out()
    out("TABLE 7: COMPLEMENTARY STRENGTHS (L3+L4 only)")
    out("-" * 100)

    bm25_better_than_e5 = 0
    e5_better_than_bm25 = 0
    both_correct = 0
    both_wrong = 0

    for qid, b in bm25_by_id.items():
        if b["level"] not in ("L3", "L4"):
            continue
        e = e5_by_id[qid]
        b_ok = b["gold_rank"] == 1
        e_ok = e["gold_rank"] == 1

        if b_ok and not e_ok:
            bm25_better_than_e5 += 1
        elif e_ok and not b_ok:
            e5_better_than_bm25 += 1
        elif b_ok and e_ok:
            both_correct += 1
        else:
            both_wrong += 1

    out(f"BM25 correct, E5 wrong : {bm25_better_than_e5}")
    out(f"E5 correct, BM25 wrong : {e5_better_than_bm25}")
    out(f"Both correct          : {both_correct}")
    out(f"Both wrong            : {both_wrong}")
    out(f"Total L3+L4 queries   : {bm25_better_than_e5 + e5_better_than_bm25 + both_correct + both_wrong}")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    result_json = {
        "table1_overall": table1,
        "table2_per_level_p1": table2,
        "table4_wrong_above_gold": table4,
        "hybrid_wins": {
            "vs_bm25": {"wins": wins_over_bm25, "losses": losses_to_bm25},
            "vs_e5": {"wins": wins_over_e5, "losses": losses_to_e5}
        },
        "complementary_L3_L4": {
            "bm25_better": bm25_better_than_e5,
            "e5_better": e5_better_than_bm25,
            "both_correct": both_correct,
            "both_wrong": both_wrong
        }
    }

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    out()
    out(f"Saved: {OUTPUT_TXT}")
    out(f"Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()