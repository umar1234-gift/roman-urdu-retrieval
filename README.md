# Roman Urdu Cross-Lingual Retrieval

A reproducible benchmark for cross-lingual information retrieval 
on Roman Urdu queries against English documents.

## Overview

- **Corpus**: 30 English policy documents
- **Queries**: 450 across 5 linguistic levels (L0-L4)
- **Systems**: BM25, Multilingual-E5, RRF, Convex fusion
- **Best result**: 98.67% @1 with LLM query rewriting

## Key Findings

1. Cross-lingual degradation is severe (20-28 pp, p < 0.0002)
2. LLM query rewriting recovers 88.89% of the gap
3. Zero-score tie-breaking artifact in RRF (previously unreported)
4. Convex fusion helps on hard subgroups (L3-L4, p = 0.028)

## Contents

- `DOCUMENTATION.md` — Complete project documentation
- `PAPER_DRAFT.md` — Full paper draft
- `src_v2/` — 49 reproducible Python scripts
- `config/` — Master facts and queries (JSON)
- `data_v2/` — Processed data and results
- `paper/figures/` — 8 publication-quality figures

## Reproducibility

```bash
pip install -r requirements.txt
python src_v2/01_generate_documents.py
# ... see DOCUMENTATION.md for full pipeline