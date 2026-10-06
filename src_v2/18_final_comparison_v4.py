from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "BM25": PROJECT_ROOT / "data_v2" / "results" / "bm25_results.json",
    "E5-base": PROJECT_ROOT / "data_v2" / "results" / "dense_results.json",
    "E5-large": PROJECT_ROOT / "data_v2" / "results" / "dense_model_comparison.json",
    "Hybrid-base": PROJECT_ROOT / "data_v2" / "results" / "hybrid_results.json",
    "Hybrid-large": PROJECT_ROOT / "data_v2" / "results" / "hybrid_e5large_results.json",
    "Adaptive-RRF": PROJECT_ROOT / "data_v2" / "results" / "adaptive_rrf_results.json",
}

OUTPUT_TXT = PROJECT_ROOT / "data_v2" / "results" / "final_comparison_v4.txt"
OUTPUT_JSON = PROJECT_ROOT / "data_v2" / "results" / "final_comparison_v4.json"

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
    mrr = sum(1/r["gold_rank"] for r in records if r["gold_rank"]) / total
    return {"@1": g1/total, "@3": g3/total, "@5": g5/total,
            "@10": g10/total, "MRR": mrr, "total": total}


def main():
    systems = {}

    for name, path in FILES.items():
        data = load_json(path)
        if name == "E5-large":
            systems[name] = data["e5-large"]
        elif name == "Adaptive-RRF":
            systems[name] = data["per_query"]
        else:
            systems[name] = data

    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out("=" * 130)
    out("FINAL COMPARISON v4 — 6 SYSTEMS (n=450 queries)")
    out("=" * 130)

    # ============================================================
    # TABLE 1: OVERALL PERFORMANCE
    # ============================================================
    out()
    out("TABLE 1: OVERALL PERFORMANCE")
    out("-" * 130)
    out(f"{'System':<15} {'@1':>8} {'@3':>8} {'@5':>8} {'@10':>8} {'MRR':>8}")
    out("-" * 130)

    table1 = {}
    # Sort by @1 descending
    sorted_systems = sorted(systems.items(),
                            key=lambda x: metrics(x[1])["@1"],
                            reverse=True)

    for name, recs in sorted_systems:
        m = metrics(recs)
        table1[name] = m
        out(f"{name:<15} {m['@1']:>8.4f} {m['@3']:>8.4f} "
            f"{m['@5']:>8.4f} {m['@10']:>8.4f} {m['MRR']:>8.4f}")

    # ============================================================
    # TABLE 2: PER-LEVEL @1
    # ============================================================
    out()
    out("TABLE 2: PER-LEVEL @1 ACCURACY")
    out("-" * 130)
    out(f"{'System':<15} " + " ".join(f"{lvl:>8}" for lvl in LEVELS) + "   Drop")
    out("-" * 130)

    table2 = {}
    for name, recs in sorted_systems:
        row = {}
        for lvl in LEVELS:
            subset = [r for r in recs if r["level"] == lvl]
            row[lvl] = metrics(subset)["@1"]
        drop = row["L0"] - row["L4"]
        table2[name] = row
        out(f"{name:<15} " + " ".join(f"{row[lvl]:>8.4f}" for lvl in LEVELS)
            + f"  {drop:>+7.4f}")

    # ============================================================
    # TABLE 3: PER-LEVEL MRR
    # ============================================================
    out()
    out("TABLE 3: PER-LEVEL MRR")
    out("-" * 130)
    out(f"{'System':<15} " + " ".join(f"{lvl:>8}" for lvl in LEVELS))
    out("-" * 130)

    for name, recs in sorted_systems:
        row = []
        for lvl in LEVELS:
            subset = [r for r in recs if r["level"] == lvl]
            row.append(metrics(subset)["MRR"])
        out(f"{name:<15} " + " ".join(f"{v:>8.4f}" for v in row))

    # ============================================================
    # TABLE 4: WRONG-ABOVE-GOLD
    # ============================================================
    out()
    out("TABLE 4: WRONG-ABOVE-GOLD RATE (lower = better)")
    out("-" * 130)
    out(f"{'System':<15} {'Overall':>10} " + " ".join(f"{lvl:>8}" for lvl in LEVELS))
    out("-" * 130)

    table4 = {}
    for name, recs in sorted_systems:
        total = len(recs)
        wrong = sum(1 for r in recs if r["gold_rank"] != 1)
        row = {"overall": wrong/total}
        for lvl in LEVELS:
            subset = [r for r in recs if r["level"] == lvl]
            w = sum(1 for r in subset if r["gold_rank"] != 1)
            row[lvl] = w/len(subset)
        table4[name] = row
        out(f"{name:<15} {row['overall']:>10.4f} "
            + " ".join(f"{row[lvl]:>8.4f}" for lvl in LEVELS))

    # ============================================================
    # TABLE 5: BEST PER LEVEL
    # ============================================================
    out()
    out("TABLE 5: BEST SYSTEM PER LEVEL (@1)")
    out("-" * 130)
    for lvl in LEVELS:
        best_name = None
        best_score = -1
        for name, recs in systems.items():
            subset = [r for r in recs if r["level"] == lvl]
            m = metrics(subset)
            if m["@1"] > best_score:
                best_score = m["@1"]
                best_name = name
        out(f"{lvl}: {best_name} ({best_score:.4f})")

    # ============================================================
    # TABLE 6: DEGRADATION (sorted, low=stable)
    # ============================================================
    out()
    out("TABLE 6: DEGRADATION L0 → L4 (sorted by stability)")
    out("-" * 130)
    out(f"{'System':<15} {'L0 @1':>10} {'L4 @1':>10} {'Drop':>10} {'Drop %':>10}")
    out("-" * 130)

    deg_list = []
    for name, recs in systems.items():
        l0 = metrics([r for r in recs if r["level"] == "L0"])["@1"]
        l4 = metrics([r for r in recs if r["level"] == "L4"])["@1"]
        drop = l0 - l4
        pct = (drop / l0 * 100) if l0 else 0
        deg_list.append((name, l0, l4, drop, pct))

    deg_list.sort(key=lambda x: x[3])

    for name, l0, l4, drop, pct in deg_list:
        out(f"{name:<15} {l0:>10.4f} {l4:>10.4f} {drop:>+10.4f} {pct:>+9.1f}%")

    # ============================================================
    # TABLE 7: WINNER SUMMARY
    # ============================================================
    out()
    out("=" * 130)
    out("TABLE 7: WINNER SUMMARY")
    out("=" * 130)

    best_p1 = max(systems.items(), key=lambda x: metrics(x[1])["@1"])
    best_mrr = max(systems.items(), key=lambda x: metrics(x[1])["MRR"])
    best_l4 = max(systems.items(),
                  key=lambda x: metrics([r for r in x[1] if r["level"] == "L4"])["@1"])
    most_stable = deg_list[0]

    out(f"🏆 Best overall @1     : {best_p1[0]:<15} ({metrics(best_p1[1])['@1']*100:.2f}%)")
    out(f"🏆 Best overall MRR    : {best_mrr[0]:<15} ({metrics(best_mrr[1])['MRR']:.4f})")
    out(f"🏆 Best L4 @1          : {best_l4[0]:<15} "
        f"({metrics([r for r in best_l4[1] if r['level']=='L4'])['@1']*100:.2f}%)")
    out(f"🏆 Most stable         : {most_stable[0]:<15} (drop {most_stable[3]*100:+.1f}%)")

    # ============================================================
    # TABLE 8: IMPROVEMENT OVER BASELINES
    # ============================================================
    out()
    out("=" * 130)
    out("TABLE 8: IMPROVEMENT OF ADAPTIVE-RRF OVER BASELINES")
    out("=" * 130)

    adapt = metrics(systems["Adaptive-RRF"])

    baselines = ["BM25", "E5-base", "E5-large", "Hybrid-base", "Hybrid-large"]
    out(f"{'Baseline':<15} {'Baseline @1':>12} {'Adaptive @1':>12} {'Δ @1':>10}")
    out("-" * 130)

    for b in baselines:
        bm = metrics(systems[b])
        d = adapt["@1"] - bm["@1"]
        out(f"{b:<15} {bm['@1']:>12.4f} {adapt['@1']:>12.4f} {d:>+10.4f}")

    # ============================================================
    # SAVE
    # ============================================================
    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    out()
    out(f"Saved: {OUTPUT_TXT}")
    out(f"Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()