from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FACTS_FILE = PROJECT_ROOT / "config" / "master_facts.json"

OUTPUT_DIR = PROJECT_ROOT / "data_v2" / "raw" / "policies"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def slugify(text):
    return text.lower().replace(" ", "_").replace("&", "and")


def main():

    print("=" * 100)
    print("GENERATING POLICY DOCUMENTS FROM MASTER FACTS")
    print("=" * 100)

    config = load_json(FACTS_FILE)
    documents = config["documents"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []

    for doc in documents:

        doc_id = doc["doc_id"]
        topic = doc["topic"]
        fact = doc["fact"]
        fact_type = doc["fact_type"]
        unique_terms = doc["unique_terms"]
        paragraphs = doc["paragraphs"]

        # Build filename: D01_academic_calendar.txt
        slug = slugify(topic)
        filename = f"{doc_id}_{slug}.txt"
        filepath = OUTPUT_DIR / filename

        # Document header + paragraphs
        lines = []
        lines.append(f"# {topic}")
        lines.append("")
        lines.append(f"Document ID: {doc_id}")
        lines.append(f"Policy Area: {topic}")
        lines.append("")
        lines.append("-" * 80)
        lines.append("")

        for para in paragraphs:
            lines.append(para)
            lines.append("")

        lines.append("-" * 80)
        lines.append("")
        lines.append(f"End of {topic} policy.")

        content = "\n".join(lines)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        word_count = len(content.split())

        manifest.append({
            "doc_id": doc_id,
            "topic": topic,
            "filename": filename,
            "fact": fact,
            "fact_type": fact_type,
            "unique_terms": unique_terms,
            "word_count": word_count
        })

        print(f"[OK] {filename:<45} fact='{fact}' words={word_count}")

    # Save manifest
    manifest_file = OUTPUT_DIR.parent / "documents_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print()
    print("-" * 100)
    print(f"Total documents: {len(manifest)}")
    print(f"Saved to: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_file}")


if __name__ == "__main__":
    main()