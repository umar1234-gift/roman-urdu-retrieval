from pathlib import Path
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"
QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"

OUTPUT_FILE = PROJECT_ROOT / "data_v2" / "processed" / "validation_report.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(t):
    t = str(t).lower()
    t = re.sub(r"[^\w\s\.%:]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)
    doc_text = {c["filename"]: c["text"] for c in chunks}

    print("=" * 100)
    print("FULL VALIDATION")
    print("=" * 100)

    doc_fails = 0
    print("\n--- DOC VALIDATION ---")
    for c in chunks:
        ok = normalize(c["fact"]) in normalize(c["text"])
        if not ok:
            doc_fails += 1
        print(f"[{'OK ' if ok else 'FAIL'}] {c['chunk_id']} | fact='{c['fact']}'")

    query_fails = 0
    level_counts = {}
    doc_counts = {}

    print("\n--- QUERY VALIDATION ---")
    for q in queries:
        gold_doc = q["gold_document"]
        gold_ans = q["gold_answer"]

        if gold_doc not in doc_text:
            ok = False
        else:
            ok = normalize(gold_ans) in normalize(doc_text[gold_doc])

        if not ok:
            query_fails += 1

        level_counts[q["level"]] = level_counts.get(q["level"], 0) + 1
        doc_counts[q["doc_id"]] = doc_counts.get(q["doc_id"], 0) + 1

    print()
    print("=" * 100)
    print(f"Documents: {len(chunks)} | Failed: {doc_fails}")
    print(f"Queries:   {len(queries)} | Failed: {query_fails}")
    print(f"Level distribution: {level_counts}")
    print(f"Docs with queries: {len(doc_counts)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "doc_fails": doc_fails,
            "query_fails": query_fails,
            "level_counts": level_counts
        }, f, indent=2)


if __name__ == "__main__":
    main()