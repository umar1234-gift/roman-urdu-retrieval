from pathlib import Path
import json
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "BM25": PROJECT_ROOT / "data_v2" / "results" / "bm25_results.json",
    "E5-base": PROJECT_ROOT / "data_v2" / "results" / "dense_results.json",
    "E5-large": PROJECT_ROOT / "data_v2" / "results" / "dense_model_comparison.json",
    "Hybrid-base": PROJECT_ROOT / "data_v2" / "results" / "hybrid_results.json",
    "Hybrid-large": PROJECT_ROOT / "data_v2" / "results" / "hybrid_e5large_results.json",
}

OUTPUT_TXT = PROJECT_ROOT / "data_v2" / "results" / "final_comparison_v3.txt"
OUTPUT_JSON = PROJECT_ROOT / "data_v2" / "results" / "final_comparison_v3.json"

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
    return {
        "@1": g1/total,
        "@3": g3/total,
        "@5": g5/total,
        "@10": g10/total,
        "MRR": mrr,
        "total": total,
        "gold_at_1": g1
    }


def main():
    systems = {}

    # Load standard files
    for name, path in FILES.items():
        if name == "E5-large":
            # dense_model_comparison.json has nested structure
            data = load_json(path)
            if "e5-large" in data:
                systems[name] = data["e5-large"]
            else:
                print(f"WARNING: 'e5-large' key not found in {path}")
            continue
        systems[name] = load_json(path)

    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out("=" * 120)
    out("FINAL COMPARISON v3 — 5 SYSTEMS")
    out("=" * 120)

    # ============================================================
    # TABLE 1: OVERALL PERFORMANCE
    # ============================================================
    out()
    out("TABLE 1: OVERALL PERFORMANCE (n=450)")
    out("-" * 120)
    out(f"{'System':<15} {'@1':>8} {'@3':>8} {'@5':>8} {'@10':>8} {'MRR':>8}")
    out("-" * 120)

    table1 = {}
    for name, recs in systems.items():
        m = metrics(recs)
        table1[name] = m
        out(f"{name:<15} {m['@1']:>8.4f} {m['@3']:>8.4f} "
            f"{m['@5']:>8.4f} {m['@10']:>8.4f} {m['MRR']:>8.4f}")

    # ============================================================
    # TABLE 2: PER-LEVEL @1 ACCURACY
    # ============================================================
    out()
    out("TABLE 2: PER-LEVEL @1 ACCURACY")
    out("-" * 120)
    out(f"{'System':<15} " + " ".join(f"{lvl:>8}" for lvl in LEVELS) + "   Drop")
    out("-" * 120)

    table2 = {}
    for name, recs in systems.items():
        row = {}
        for lvl in LEVELS:
            subset = [r for r in recs if r["level"] == lvl]
            row[lvl] = metrics(subset)["@1"]
        drop = row["L0"] - row["L4"]
        table2[name] = row
        out(f"{name:<15} " + " ".join(f"{row[lvl]:>8.4f}" for lvl in LEVELS) + f"  {drop:>+7.4f}")

    # ============================================================
    # TABLE 3: PER-LEVEL MRR
    # ============================================================
    out()
    out("TABLE 3: PER-LEVEL MRR")
    out("-" * 120)
    out(f"{'System':<15} " + " ".join(f"{lvl:>8}" for lvl in LEVELS))
    out("-" * 120)

    table3 = {}
    for name, recs in systems.items():
        row = {}
        for lvl in LEVELS:
            subset = [r for r in recs if r["level"] == lvl]
            row[lvl] = metrics(subset)["MRR"]
        table3[name] = row
        out(f"{name:<15} " + " ".join(f"{row[lvl]:>8.4f}" for lvl in LEVELS))

    # ============================================================
    # TABLE 4: WRONG-ABOVE-GOLD RATE (FIXED)
    # ============================================================
    out()
    out("TABLE 4: WRONG-ABOVE-GOLD RATE (top-1 is not gold)")
    out("-" * 120)
    out(f"{'System':<15} {'Overall':>10} " + " ".join(f"{lvl:>8}" for lvl in LEVELS))
    out("-" * 120)

    table4 = {}
    for name, recs in systems.items():
        total = len(recs)
        # FIXED: use gold_rank instead of top1_correct
        wrong = sum(1 for r in recs if r["gold_rank"] != 1)
        row = {"overall": wrong/total}
        for lvl in LEVELS:
            subset = [r for r in recs if r["level"] == lvl]
            w = sum(1 for r in subset if r["gold_rank"] != 1)
            row[lvl] = w/len(subset)
        table4[name] = row
        out(f"{name:<15} {row['overall']:>10.4f} " + " ".join(f"{row[lvl]:>8.4f}" for lvl in LEVELS))

    # ============================================================
    # TABLE 5: BEST SYSTEM PER LEVEL
    # ============================================================
    out()
    out("TABLE 5: BEST SYSTEM PER LEVEL (@1)")
    out("-" * 120)

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
    # TABLE 6: BEST SYSTEM OVERALL
    # ============================================================
    out()
    out("TABLE 6: BEST SYSTEM OVERALL")
    out("-" * 120)

    best_overall_p1 = max(systems.items(), key=lambda x: metrics(x[1])["@1"])
    best_overall_mrr = max(systems.items(), key=lambda x: metrics(x[1])["MRR"])

    out(f"Best by @1 : {best_overall_p1[0]} ({metrics(best_overall_p1[1])['@1']:.4f})")
    out(f"Best by MRR: {best_overall_mrr[0]} ({metrics(best_overall_mrr[1])['MRR']:.4f})")

    # ============================================================
    # TABLE 7: DEGRADATION COMPARISON (sorted)
    # ============================================================
    out()
    out("TABLE 7: DEGRADATION L0 → L4 (sorted, low=stable)")
    out("-" * 120)
    out(f"{'System':<15} {'L0 @1':>10} {'L4 @1':>10} {'Drop':>10} {'Drop %':>10}")
    out("-" * 120)

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
    # SUMMARY HIGHLIGHTS
    # ============================================================
    out()
    out("=" * 120)
    out("SUMMARY HIGHLIGHTS")
    out("=" * 120)

    # Winner overall
    out(f"• Best overall @1     : {best_overall_p1[0]} ({metrics(best_overall_p1[1])['@1']*100:.2f}%)")
    out(f"• Best L4 @1          : ", )
    best_l4 = max(systems.items(), key=lambda x: metrics([r for r in x[1] if r["level"] == "L4"])["@1"])
    out(f"  → {best_l4[0]} ({metrics([r for r in best_l4[1] if r['level']=='L4'])['@1']*100:.2f}%)")

    # Most stable
    out(f"• Most stable (L0→L4): {deg_list[0][0]} (drop {deg_list[0][3]*100:+.1f}%)")

    # RRF observation
    out()
    out("RRF OBSERVATION:")
    out(f"  BM25 alone           : {metrics(systems['BM25'])['@1']*100:.2f}%")
    out(f"  E5-base alone        : {metrics(systems['E5-base'])['@1']*100:.2f}%")
    out(f"  E5-large alone       : {metrics(systems['E5-large'])['@1']*100:.2f}%")
    out(f"  Hybrid-base          : {metrics(systems['Hybrid-base'])['@1']*100:.2f}%")
    out(f"  Hybrid-large         : {metrics(systems['Hybrid-large'])['@1']*100:.2f}%")
    out()
    out("  → RRF helps when base retrievers are comparable (BM25 ≈ E5-base)")
    out("  → RRF hurts when one dominates (E5-large >> BM25)")

    # ============================================================
    # SAVE
    # ============================================================
    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    output_json = {
        "table1_overall": {k: {kk: vv for kk, vv in v.items()} for k, v in table1.items()},
        "table2_per_level_p1": table2,
        "table3_per_level_mrr": table3,
        "table4_wrong_above_gold": table4,
        "degradation": [
            {"system": name, "l0": l0, "l4": l4, "drop": drop, "drop_pct": pct}
            for name, l0, l4, drop, pct in deg_list
        ]
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    out()
    out(f"Saved: {OUTPUT_TXT}")
    out(f"Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()