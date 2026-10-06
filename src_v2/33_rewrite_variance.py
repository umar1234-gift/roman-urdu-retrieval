"""
Rewrite Variance — 3 Runs of LLM Rewriting
===========================================
Runs LLM rewriting 3 times with same prompt (temperature=0)
to measure run-to-run variance.

Note: temperature=0 should give deterministic output. If we see variance,
it means the LLM is non-deterministic.

Run: python src_v2/33_rewrite_variance.py
"""

from pathlib import Path
import json
import os
import re
import time
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUERIES_FILE = PROJECT_ROOT / "data_v2" / "processed" / "main_queries.json"

OUT_DIR = PROJECT_ROOT / "data_v2" / "processed"
RESULTS_FILE = PROJECT_ROOT / "data_v2" / "results" / "rewrite_variance.json"

MODEL = "openai/gpt-oss-120b"
BATCH_SIZE = 15
N_RUNS = 3


PROMPT_TEMPLATE = """You are a professional translator. Translate each numbered query into natural English.

Rules:
1. Preserve the EXACT information need.
2. If the query is already in English, return it essentially unchanged.
3. If it's Roman Urdu, translate to natural English.
4. Output ONLY a numbered list — no explanations.

Queries:
{queries}

Output:"""


def init_groq():
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Set GROQ_API_KEY environment variable")
    return Groq(api_key=api_key)


def rewrite_batch(client, queries):
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
    prompt = PROMPT_TEMPLATE.format(queries=numbered)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2048,
    )
    text = response.choices[0].message.content.strip()

    translations = []
    for line in text.split("\n"):
        line = line.strip()
        m = re.match(r"^\d+\.\s*(.+)$", line)
        if m:
            translations.append(m.group(1).strip())

    while len(translations) < len(queries):
        translations.append(queries[len(translations)])

    return translations[:len(queries)]


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 100)
    print(f"REWRITE VARIANCE ({N_RUNS} runs)")
    print("=" * 100)

    queries = load_json(QUERIES_FILE)

    # Only rewrite L1-L4 (L0 stays English)
    to_rewrite = [q for q in queries if q["level"] in ["L1", "L2", "L3", "L4"]]
    print(f"Total queries : {len(queries)}")
    print(f"L1-L4 queries : {len(to_rewrite)}")
    print(f"Runs          : {N_RUNS}\n")

    client = init_groq()

    all_runs = {}

    for run_idx in range(N_RUNS):
        print(f"\n{'=' * 100}")
        print(f"RUN {run_idx + 1}/{N_RUNS}")
        print(f"{'=' * 100}")

        run_output = {}
        total_batches = (len(to_rewrite) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(total_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(to_rewrite))
            batch = to_rewrite[start:end]

            qids = [q["query_id"] for q in batch]
            texts = [q["query"] for q in batch]

            print(f"  Batch {batch_idx + 1}/{total_batches}...")

            try:
                translations = rewrite_batch(client, texts)
                for qid, t in zip(qids, translations):
                    run_output[qid] = t
                time.sleep(2)
            except Exception as e:
                print(f"    Error: {e}")
                time.sleep(5)

        all_runs[f"run_{run_idx + 1}"] = run_output
        print(f"  Run {run_idx + 1} complete: {len(run_output)} queries")

    # Compare across runs
    print()
    print("=" * 100)
    print("COMPARISON ACROSS RUNS")
    print("=" * 100)

    report = []
    def out(s=""):
        print(s)
        report.append(s)

    out()
    out(f"{'Query ID':<12} " + " ".join(f"{'Run ' + str(i+1):>30}" for i in range(N_RUNS)))
    out("-" * 100)

    # Compare 100 random queries
    sample_ids = [q["query_id"] for q in to_rewrite][:20]

    identical_count = 0
    total_count = 0

    for qid in sample_ids:
        texts = [all_runs[f"run_{i+1}"].get(qid, "") for i in range(N_RUNS)]
        all_same = len(set(texts)) == 1
        if all_same:
            identical_count += 1
        total_count += 1
        status = "SAME" if all_same else "DIFF"
        out()
        out(f"{qid}  [{status}]")
        for i, t in enumerate(texts):
            out(f"  Run {i+1}: {t[:80]}")

    out()
    out(f"Identical across all {N_RUNS} runs: {identical_count}/{total_count} queries")

    # Full overlap
    all_ids = [q["query_id"] for q in to_rewrite]
    full_identical = 0
    for qid in all_ids:
        texts = [all_runs[f"run_{i+1}"].get(qid, "") for i in range(N_RUNS)]
        if len(set(texts)) == 1:
            full_identical += 1

    out()
    out(f"Full dataset: {full_identical}/{len(all_ids)} identical across all {N_RUNS} runs")
    out(f"Variance rate: {1 - full_identical/len(all_ids):.2%}")

    if full_identical == len(all_ids):
        out()
        out("Conclusion: LLM is DETERMINISTIC at temperature=0.")
        out("No batch composition variance observed.")
    else:
        out()
        out(f"Conclusion: LLM is NON-DETERMINISTIC.")
        out(f"{len(all_ids) - full_identical} queries differ across runs.")

    save_json(RESULTS_FILE, {
        "n_runs": N_RUNS,
        "total_queries": len(all_ids),
        "identical_all_runs": full_identical,
        "variance_rate": 1 - full_identical/len(all_ids),
        "runs": all_runs,
    })

    print(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()