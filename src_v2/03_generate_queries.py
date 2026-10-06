from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FACTS_FILE = PROJECT_ROOT / "config" / "master_facts.json"
QUERIES_FILE = PROJECT_ROOT / "config" / "master_queries.json"

OUTPUT_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def slugify(topic):
    return topic.lower().replace(" ", "_").replace("&", "and")


def main():
    print("=" * 100)
    print("GENERATING FINAL QUERY DATASET")
    print("=" * 100)

    facts = load_json(FACTS_FILE)
    queries = load_json(QUERIES_FILE)

    records = []
    counter = 1

    for doc in facts["documents"]:
        doc_id = doc["doc_id"]
        topic = doc["topic"]
        fact = doc["fact"]
        filename = f"{doc_id}_{slugify(topic)}.txt"

        if doc_id not in queries:
            raise ValueError(f"No queries for {doc_id}")

        for level in ["L0", "L1", "L2", "L3", "L4"]:
            variants = queries[doc_id][level]
            if len(variants) != 3:
                raise ValueError(f"{doc_id} {level}: {len(variants)} variants")

            for v_num, q_text in enumerate(variants, start=1):
                records.append({
                    "query_id": f"MQ{counter:04d}",
                    "doc_id": doc_id,
                    "topic": topic,
                    "level": level,
                    "variant": v_num,
                    "query": q_text,
                    "gold_answer": fact,
                    "gold_document": filename
                })
                counter += 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Total queries: {len(records)}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()