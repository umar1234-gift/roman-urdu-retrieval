from pathlib import Path
import json
import math
import re
import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"

RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "dense_model_comparison.json"
REPORT_FILE = PROJECT_ROOT / "data_v2" / "results" / "dense_model_comparison.txt"

TOP_K = 10
LEVELS = ["L0", "L1", "L2", "L3", "L4"]

# Models to compare
MODELS = [
    {
        "name": "e5-small",
        "hf_id": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: "
    },
    {
        "name": "e5-base",
        "hf_id": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: "
    },
    {
        "name": "e5-large",
        "hf_id": "intfloat/multilingual-e5-large",
        "query_prefix": "query: ",
        "passage_prefix": "passage: "
    },
    {
        "name": "labse",
        "hf_id": "sentence-transformers/LaBSE",
        "query_prefix": "",
        "passage_prefix": ""
    },
    {
        "name": "mpnet-multi",
        "hf_id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "query_prefix": "",
        "passage_prefix": ""
    },
]


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
        "@1": g1/total, "@3": g3/total, "@5": g5/total,
        "@10": g10/total, "MRR": mrr,
        "gold_at_1": g1, "total": total
    }


def run_model(model_config, chunks, queries):
    """Run one dense model on all queries. Returns results list."""

    name = model_config["name"]
    hf_id = model_config["hf_id"]
    q_prefix = model_config["query_prefix"]
    p_prefix = model_config["passage_prefix"]

    print(f"\n{'=' * 100}")
    print(f"Running model: {name} ({hf_id})")
    print(f"{'=' * 100}")

    texts = [c["text"] for c in chunks]
    filenames = [c["filename"] for c in chunks]

    print("Loading model...")
    model = SentenceTransformer(hf_id)

    doc_texts = [p_prefix + t for t in texts]
    print(f"Encoding {len(doc_texts)} documents...")
    doc_emb = model.encode(doc_texts, normalize_embeddings=True,
                            show_progress_bar=True)

    results = []
    print(f"Evaluating {len(queries)} queries...")

    for q in queries:
        q_emb = model.encode([q_prefix + q["query"]],
                              normalize_embeddings=True)[0]
        sims = np.dot(doc_emb, q_emb)
        order = np.argsort(sims)[::-1]

        gold_rank = None
        for rank, idx in enumerate(order, start=1):
            if filenames[idx] == q["gold_document"]:
                gold_rank = rank
                break

        results.append({
            "query_id": q["query_id"],
            "doc_id": q["doc_id"],
            "level": q["level"],
            "variant": q["variant"],
            "query": q["query"],
            "gold_document": q["gold_document"],
            "gold_rank": gold_rank
        })

    return results


def main():
    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)

    print("=" * 100)
    print("DENSE MODEL COMPARISON")
    print("=" * 100)
    print(f"Chunks : {len(chunks)}")
    print(f"Queries: {len(queries)}")
    print(f"Models : {len(MODELS)}")

    all_results = {}

    for model_config in MODELS:
        name = model_config["name"]

        try:
            results = run_model(model_config, chunks, queries)
            all_results[name] = results
            print(f"[OK] {name} complete")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            continue

    # ============================================================
    # REPORT
    # ============================================================

    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out()
    out("=" * 100)
    out("DENSE MODEL COMPARISON — OVERALL")
    out("=" * 100)
    out(f"{'Model':<15} {'@1':>8} {'@3':>8} {'@5':>8} {'@10':>8} {'MRR':>8}")
    out("-" * 100)

    for name in all_results:
        m = metrics(all_results[name])
        out(f"{name:<15} {m['@1']:>8.4f} {m['@3']:>8.4f} "
            f"{m['@5']:>8.4f} {m['@10']:>8.4f} {m['MRR']:>8.4f}")

    # Per level
    out()
    out("=" * 100)
    out("PER-LEVEL @1 ACCURACY")
    out("=" * 100)
    out(f"{'Model':<15} " + " ".join(f"{lvl:>8}" for lvl in LEVELS) + "  L0→L4 Drop")
    out("-" * 100)

    for name in all_results:
        row = []
        for lvl in LEVELS:
            subset = [r for r in all_results[name] if r["level"] == lvl]
            row.append(metrics(subset)["@1"])
        drop = row[0] - row[-1]
        out(f"{name:<15} " + " ".join(f"{v:>8.4f}" for v in row) + f"   {drop:>+8.4f}")

    # Per level MRR
    out()
    out("=" * 100)
    out("PER-LEVEL MRR")
    out("=" * 100)
    out(f"{'Model':<15} " + " ".join(f"{lvl:>8}" for lvl in LEVELS))
    out("-" * 100)

    for name in all_results:
        row = []
        for lvl in LEVELS:
            subset = [r for r in all_results[name] if r["level"] == lvl]
            row.append(metrics(subset)["MRR"])
        out(f"{name:<15} " + " ".join(f"{v:>8.4f}" for v in row))

    # Best model per level
    out()
    out("=" * 100)
    out("BEST MODEL PER LEVEL (@1)")
    out("=" * 100)

    for lvl in LEVELS:
        best = None
        best_score = -1
        for name in all_results:
            subset = [r for r in all_results[name] if r["level"] == lvl]
            m = metrics(subset)
            if m["@1"] > best_score:
                best_score = m["@1"]
                best = name
        out(f"{lvl}: {best} ({best_score:.4f})")

    # Per-document comparison on hardest docs
    out()
    out("=" * 100)
    out("PERFORMANCE ON HARDEST DOCS (avg gold rank, lower=better)")
    out("=" * 100)

    hard_docs = ["D21", "D30", "D28", "D14", "D05", "D02", "D20"]
    out(f"{'Model':<15} " + " ".join(f"{d:>8}" for d in hard_docs))
    out("-" * 100)

    for name in all_results:
        row = []
        for doc_id in hard_docs:
            subset = [r for r in all_results[name] if r["doc_id"] == doc_id]
            ranks = [r["gold_rank"] for r in subset if r["gold_rank"]]
            avg = sum(ranks)/len(ranks) if ranks else 0
            row.append(avg)
        out(f"{name:<15} " + " ".join(f"{v:>8.2f}" for v in row))

    # Save
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    out()
    out(f"Saved: {RESULTS_FILE}")
    out(f"Saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()