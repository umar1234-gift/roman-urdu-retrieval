from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent

POLICY_DIR = PROJECT_ROOT / "data_v2" / "raw" / "policies"
MANIFEST_FILE = PROJECT_ROOT / "data_v2" / "raw" / "documents_manifest.json"

OUTPUT_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_corpus_chunks.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    print("=" * 100)
    print("CHUNKING POLICY DOCUMENTS")
    print("=" * 100)

    manifest = load_json(MANIFEST_FILE)

    chunks = []

    for entry in manifest:

        doc_id = entry["doc_id"]
        filename = entry["filename"]
        filepath = POLICY_DIR / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Missing: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # 1 chunk per document (decision D2)
        chunk_id = f"{doc_id}_C1"

        chunks.append({
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "filename": filename,
            "topic": entry["topic"],
            "fact": entry["fact"],
            "fact_type": entry["fact_type"],
            "text": text
        })

        print(f"[OK] {chunk_id} | {filename}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print()
    print("-" * 100)
    print(f"Total chunks: {len(chunks)}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()