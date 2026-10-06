# Roman Urdu Cross-Lingual Retrieval - Complete Documentation

Project: Cross-Lingual Information Retrieval for Roman Urdu Queries
Author: Umar Farooq
Institution: Thal University Bhakkar
Semester: 5th (BS Software Engineering)
Date: October 6, 2026
Status: Draft ready for arXiv preprint

---

## Table of Contents

1. Overview
2. Research Questions
3. Dataset
4. Systems Evaluated
5. Methodology
6. Results
7. Statistical Analysis
8. Fusion Analysis
9. Failure Analysis
10. Key Findings
11. Limitations
12. Project Structure
13. How to Reproduce
14. Future Work
15. References

---

## 1. Overview

This project presents a systematic evaluation of sparse (BM25), dense (Multilingual-E5), and hybrid retrieval systems on Roman Urdu queries against English documents. The scenario reflects a real cross-lingual setting common across South Asia.

Roman Urdu is a code-mixed language variety where Urdu grammar and vocabulary are written in Latin script, spoken by over 100 million South Asians. In Pakistan, most formal documents are authored in English, while many users search in Roman Urdu.

Example:
Document says: "The minimum attendance is 75%."
User queries: "Talib-e-ilm ki kam az kam hazri kitni honi zaroori hai?"

### 1.1 Contributions

1. A reproducible benchmark of 30 documents and 450 queries across 5 linguistic levels.
2. Cluster-aware evaluation with grouped 5-fold cross-validation and document-level bootstrap confidence intervals.
3. Two strong findings: severe cross-lingual degradation and significant LLM rewriting improvement.
4. A zero-score tie-breaking artifact in rank-based fusion, a previously unreported issue in cross-lingual hybrid retrieval.
5. Reproducibility artifacts: 51 scripts, 8 figures, full documentation, all raw results.

### 1.2 Best Result

Honest deployment scenario, single-query LLM rewriting with deterministic fallback:

System: E5-large with LLM rewriting
@1: 98.67 percent
@3: 100.00 percent
MRR: 0.9930
95 percent CI for @1: [0.9733, 0.9978]

### 1.3 Broader Applicability

Potentially applicable to any English-Roman Urdu mixed retrieval scenario: healthcare, government, e-commerce, legal, news. This work does not empirically verify these domains.

---

## 2. Research Questions

RQ1: How much does retrieval performance degrade from formal English (L0) to natural Roman Urdu (L4)?

RQ2: Can LLM-based query rewriting recover the cross-lingual gap?

RQ3: Do hybrid fusion methods improve cross-lingual retrieval?

---

## 3. Dataset

### 3.1 Corpus: 30 Policy Documents

Domain: University policies
Language: Formal English
Length: 80 to 100 words per document
Facts: Exactly 1 unique fact per document
Vocabulary: Intentionally distinct terms per document

Design caveat: Vocabulary is intentionally distinct to reduce cross-document lexical overlap. Real-world corpora will have more naturalistic challenges.

### 3.2 Queries: 450 Total

Each document has 15 queries: 5 linguistic levels times 3 variants.

Level 0, formal English. Example: "What minimum attendance is required per course?"

Level 1, English-dominant with Urdu genitive markers. Example: "What minimum attendance is required per course ki classes mein?"

Level 2, balanced code-mixing. Example: "Minimum attendance kitni honi chahiye?"

Level 3, Urdu-dominant. Example: "Talib-e-ilm ki kam az kam hazri kitni honi zaroori hai?"

Level 4, natural Roman Urdu. Example: "Agar talib-e-ilm imtihan dena chahta hai to minimum hazri kitni honi chahiye?"

Query authorship: All 450 queries were written by the author, a native Roman Urdu speaker. No LLM was used to generate query text. The script 03_generate_queries.py assembles queries from a manually curated JSON file named master_queries.json. No independent validation was performed.

Known limitation: 32 out of 90 L1 queries are surface-identical to their L0 counterparts, with no Urdu markers added. One L3 to L4 pair and one L2 to L3 pair are also identical. This may amplify L0 to L1 similarity in results.

Level distribution:
L0: 90 queries
L1: 90 queries
L2: 90 queries
L3: 90 queries
L4: 90 queries

---

## 4. Systems Evaluated

BM25 (sparse retrieval): k1 = 1.5, b = 0.75.

E5-base (dense retrieval): 278 million parameters.

E5-large (dense retrieval): 560 million parameters.

Hybrid-base: Reciprocal Rank Fusion of BM25 and E5-base. k = 10, equal weights 1.0 and 1.0.

Hybrid-large: Reciprocal Rank Fusion of BM25 and E5-large. k = 10, equal weights 1.0 and 1.0.

Weighted RRF: score equals w_bm25 divided by (k plus r_bm25) plus w_e5 divided by (k plus r_e5).

Convex combination (score-based fusion): score equals alpha times normalized BM25 score plus (1 minus alpha) times normalized E5 score.

LLM Query Rewriting: Groq model openai/gpt-oss-120b. Temperature 0.0. max_tokens 300 for first pass, 1500 for retry.

### 4.1 LLM Rewriting Prompt

All 450 queries including L0 were rewritten using the same prompt:

You are a professional translator. Translate each numbered query into natural English.
Rules:
1. Preserve the EXACT information need.
2. If the query is already in English, return it essentially unchanged.
3. If it is Roman Urdu, translate to natural English.
4. Output ONLY a numbered list, no explanations.
Queries: {queries}
Output:

No level labels are used. This is a deployable pipeline.

---

## 5. Methodology

### 5.1 Evaluation Metrics

P@K: fraction of queries with the gold document in top-K.

MRR: mean reciprocal rank.

Gap recovery: (L4_rewritten minus L4_original) divided by (L0_measured minus L4_original).

### 5.2 Cluster-Aware Evaluation

Because 15 queries share each document, standard independence assumptions are violated. We use:

Grouped 5-fold cross-validation. Documents split into 5 folds, 6 documents per fold, 90 queries per fold.

Cluster bootstrap. Documents sampled with replacement, 5000 iterations.

Paired cluster bootstrap. Paired at document level for system comparisons.

Multiple-comparison note: With 10 comparisons, the Bonferroni threshold at alpha equal to 0.05 is 0.005. Only L0 versus L4 degradation and A_single rewriting tests survive this threshold. Others are labeled exploratory.

### 5.3 Batch Leakage Ablation

Three variants:
A_single: one query per API call, no batch context. Primary deployment scenario.
B_shuffle: shuffled batches across documents.
C_l34only: L3 and L4 only batches.

### 5.4 Parsing-Failure Policy

First pass used max_tokens 300. Four outputs required correction on first pass:

MQ0169, level L1. First pass empty.
MQ0364, level L1. First pass empty.
MQ0389, level L4. First pass empty.
MQ0006, level L1. First pass no-op, meaning the LLM returned the input unchanged because the query is already English.

So the four cases comprise 3 empty outputs and 1 no-op output.

Two deterministic policies were tested:

Policy 1, fallback-to-original: if LLM output is empty, use original query text for retrieval.

Policy 2, retry then fallback: re-query with max_tokens 1500; if still empty, fallback to original.

Results:

Raw first-pass, empty outputs treated as failures: 97.78 percent @1, that is 440 out of 450.

First-pass plus fallback-to-original: 98.67 percent @1, that is 444 out of 450.

Retry then fallback: 98.67 percent @1, that is 444 out of 450.

Both policies achieve the same result. This means the empty outputs are fixed by retrieving on the original query text, not by improved translations. The incremental value of retry over fallback is zero on this benchmark.

Note on L0: For L0 queries, the original text is used in first-pass evaluation because L0 rewrites were observed to be unchanged. The LLM correctly treated them as already English. This means script 51 bypasses the L0 cache and reads the original L0 query directly.

---

## 6. Results

### 6.1 Original Queries (Full 450, Tie-Aware)

System BM25: @1 equals 88.44 percent, @3 equals 94.00 percent, MRR equals 0.9148.

System E5-large: @1 equals 92.22 percent, @3 equals 96.89 percent, MRR equals 0.9489.

System Hybrid-large (RRF tie-aware): @1 equals 92.22 percent, @3 equals 97.33 percent, MRR equals 0.9497.

System Weighted-RRF (in-fold, tie-aware): @1 equals 92.67 percent, @3 equals 97.11 percent, MRR equals 0.9518.

System Convex (in-fold): @1 equals 93.11 percent, @3 equals 97.56 percent, MRR equals 0.9551.

### 6.2 Per-Level @1 (Original)

BM25: L0 equals 96.67, L1 equals 97.78, L2 equals 97.78, L3 equals 75.56, L4 equals 74.44.

E5-large: L0 equals 98.89, L1 equals 98.89, L2 equals 100.0, L3 equals 84.44, L4 equals 78.89.

Hybrid-large (tie-aware): L0 equals 98.89, L1 equals 98.89, L2 equals 98.89, L3 equals 84.44, L4 equals 80.00.

Weighted-RRF (in-fold): L0 equals 98.89, L1 equals 98.89, L2 equals 98.89, L3 equals 86.67, L4 equals 80.00.

Convex (in-fold): L0 equals 98.89, L1 equals 98.89, L2 equals 98.89, L3 equals 86.67, L4 equals 82.22.

Degradation from L0 to L4:
BM25: 22.22 percentage points.
E5-base: 27.78 percentage points.
E5-large: 20.00 percentage points.
Hybrid-base: 23.33 percentage points.
Hybrid-large: 18.89 percentage points.
Weighted-RRF: 18.89 percentage points.
Convex: 16.67 percentage points.

Four systems drop 20 to 28 percentage points: BM25, E5-base, E5-large, Hybrid-base.

### 6.3 A_single, Deployment Scenario

Raw first-pass, empty outputs as failures:
@1 equals 97.78 percent, that is 440 out of 450.
@3 equals 99.33 percent.
MRR equals 0.9851.
95 percent CI for @1 equals [0.9622, 0.9911].

Fallback or retry:
@1 equals 98.67 percent, that is 444 out of 450.
@3 equals 100.00 percent.
MRR equals 0.9930.
95 percent CI for @1 equals [0.9733, 0.9978].

### 6.4 Per-Level @1 (A_single, Fallback or Retry)

L0: 98.89 percent, @3 equals 100.0, MRR equals 0.9944.
L1: 98.89 percent, @3 equals 100.0, MRR equals 0.9944.
L2: 100.00 percent, @3 equals 100.0, MRR equals 1.0000.
L3: 98.89 percent, @3 equals 100.0, MRR equals 0.9944.
L4: 96.67 percent, @3 equals 100.0, MRR equals 0.9815.

### 6.5 Gap Recovery (A_single, Fallback or Retry)

L0 original equals 98.89 percent.
L4 original equals 78.89 percent.
L4 A_single equals 96.67 percent.

Recovery against measured L0 ceiling equals (96.67 minus 78.89) divided by (98.89 minus 78.89), which equals 88.89 percent.

Recovery against assumed 100 percent ceiling equals (96.67 minus 78.89) divided by (100 minus 78.89), which equals 84.21 percent.

95 percent CI for recovery against the 100 percent ceiling is [0.5556, 1.0000].

### 6.6 Zero-Score Tie-Breaking Artifact

16 out of 450 queries, which is 3.6 percent, have all-zero BM25 scores. All 16 are in L3 and L4, specifically 9 in L3 and 7 in L4. These are queries where no term matches any document because they use Urdu-only vocabulary such as parhai, sawari, and shikayat, which have no English overlap.

Impact on rank-based fusion:

Variant A, arbitrary tie with stable argsort: @1 equals 90.67 percent, @3 equals 96.44 percent, MRR equals 0.9384.

Variant B, skip BM25 when all-zero: @1 equals 92.22 percent, @3 equals 97.33 percent, MRR equals 0.9497.

Variant C, random tie with fixed seed: @1 equals 91.11 percent, @3 equals 96.67 percent, MRR equals 0.9414.

E5-large alone: @1 equals 92.22 percent, @3 equals 96.89 percent, MRR equals 0.9489.

Finding: Variant B recovers RRF to E5-large parity. The apparent RRF failure is a tie-breaking artifact, not a fusion limitation. This is a previously unreported issue in cross-lingual hybrid retrieval.

### 6.7 Fusion: Convex Combination, In-Fold Alpha

Fold 1: best alpha equals 0.2, train @1 equals 0.9250, test @1 equals 0.9778, oracle @1 equals 0.9889.

Fold 2: best alpha equals 0.5, train @1 equals 0.9361, test @1 equals 0.9222, oracle @1 equals 0.9444.

Fold 3: best alpha equals 0.2, train @1 equals 0.9694, test @1 equals 0.8000, oracle @1 equals 0.8000.

Fold 4: best alpha equals 0.2, train @1 equals 0.9194, test @1 equals 1.0000, oracle @1 equals 1.0000.

Fold 5: best alpha equals 0.2, train @1 equals 0.9306, test @1 equals 0.9556, oracle @1 equals 0.9667.

Mean train @1 equals 0.9361. Mean test @1 equals 0.9311. Mean oracle @1 equals 0.9400.

Test @1 equals 93.11 percent on held-out folds.

### 6.8 E5-large Remaining Failures (A_single)

6 out of 450 failures.

MQ0018, document D02, level L0, rank 2. Query: "How long is the drop window after semester start?"

MQ0021, document D02, level L1, rank 2. Query: "How long is the drop window after the semester starts?"

MQ0104, document D07, level L4, rank 3.

MQ0150, document D10, level L4, rank 2.

MQ0445, document D30, level L3, rank 2.

MQ0448, document D30, level L4, rank 2.

Analysis: MQ0018 and MQ0021 relate to D02 Add-Drop versus D20 Withdrawal vocabulary collision, a benchmark ambiguity, not a translation issue. MQ0445 and MQ0448 have the word Sawari not translated by the LLM. MQ0104 and MQ0150 involve semantic drift.

Excluding the two ambiguous D02 and D20 cases: 444 out of 448 equals 99.11 percent @1.

---

## 7. Statistical Analysis

### 7.1 Cluster-Aware Paired Bootstrap Tests

L0 versus L4 (BM25): delta equals +22.22 percentage points, 95 percent CI [+10.00, +35.56], p less than 0.0002. Significant.

L0 versus L4 (E5-base): delta equals +27.78 percentage points, 95 percent CI [+15.56, +41.11], p less than 0.0002. Significant.

L0 versus L4 (E5-large): delta equals +20.00 percentage points, 95 percent CI [+8.89, +32.22], p less than 0.0002. Significant.

L0 versus L4 (Hybrid-base): delta equals +23.33 percentage points, 95 percent CI [+11.11, +36.67], p less than 0.0002. Significant.

A_single versus Original (L2 to L4): delta equals +10.74 percentage points, 95 percent CI [+4.07, +18.52], p equals 0.0004. Significant.

A_single versus Original (all 450): delta equals +6.44 percentage points, 95 percent CI [+2.22, +11.11], p equals 0.0044. Significant.

Convex (in-fold) versus E5-large, overall: delta equals +0.89 percentage points, 95 percent CI [minus 0.22, +2.22], p equals 0.18. Not significant.

Convex (in-fold) versus E5-large, L3 to L4 subgroup: delta equals +2.78 percentage points, 95 percent CI [+0.56, +6.11], p equals 0.028. Significant but does not survive strict Bonferroni correction of alpha equal to 0.005. Framed as hypothesis-driven.

E5-large versus E5-base: delta equals +3.56 percentage points, 95 percent CI [0.00, +7.33], p equals 0.0504. Borderline.

E5-large versus Hybrid-large (tie-aware): delta equals 0.00 percentage points, identical performance.

p less than 0.0002 is the minimum bootstrap resolution with 5000 samples. It is not literally p equals zero.

Bonferroni threshold with 10 tests is alpha equal to 0.005. Only L0 versus L4 tests and A_single rewriting survive.

---

## 8. Fusion Analysis

### 8.1 Rank-Based Fusion, RRF

Naive RRF with arbitrary tie-breaking achieves 90.67 percent @1, below E5-large alone at 92.22 percent.

Proper zero-score handling, meaning skip BM25 when all-zero, recovers RRF to 92.22 percent @1, exactly E5-large parity.

Conclusion: The apparent RRF failure was entirely a tie-breaking artifact on 16 zero-score queries, not a fusion limitation.

### 8.2 Score-Based Fusion, Convex Combination

In-fold convex combination achieves 93.11 percent @1 overall, a +0.89 percentage point improvement over E5-large alone. Not statistically significant at p equals 0.18.

Per-level analysis:
L0: delta equals 0.
L1: delta equals 0.
L2: delta equals minus 1.11 percentage points.
L3: delta equals +2.22 percentage points, p equals 0.254.
L4: delta equals +3.33 percentage points, p equals 0.079.
L3 to L4 subgroup: delta equals +2.78 percentage points, p equals 0.028.

Finding: Fusion improves accuracy specifically on the hardest levels, Roman Urdu L3 and L4, where single-retriever performance is weakest. On L0 to L2, E5-large already achieves 98.89 to 100 percent, a ceiling effect that prevents any fusion improvement.

Caveat: With 8 or more subgroup comparisons, strict Bonferroni threshold is 0.005. The L3 to L4 result at p equals 0.028 does not survive this correction. Reported as a hypothesis-driven finding.

---

## 9. Failure Analysis

### 9.1 Zero-Score BM25 Queries

16 out of 450 queries, 3.6 percent, have all-zero BM25 scores.

Level distribution: 9 in L3, 7 in L4.

Cause: Urdu-only vocabulary such as parhai, sawari, shikayat has no English overlap with the English documents. BM25 produces no signal for these queries.

Implication: Naive rank-based fusion injects arbitrary tie order for these queries, causing a 1.55 percentage point drop in @1.

### 9.2 E5-large Remaining Failures (A_single)

Total: 6 out of 450.

Categorized:
Benchmark ambiguity, 2 queries: MQ0018 and MQ0021 due to D02 and D20 vocabulary collision.
Vocabulary gap, 2 queries: MQ0445 and MQ0448 because Sawari was not translated.
Semantic drift, 2 queries: MQ0104 and MQ0150.

### 9.3 Query Duplication Across Levels

32 out of 90 L1 queries are surface-identical to their L0 counterparts.

1 L3 to L4 pair is identical.

1 L2 to L3 pair is identical.

This is a benchmark design limitation. It may amplify L0 to L1 similarity and partially explain identical L0 and L1 results.

---

## 10. Key Findings

Finding 1: Cross-lingual degradation is severe and significant. Four systems drop 20 to 28 percentage points from L0 to L4. Cluster bootstrap gives p less than 0.0002. Caveat: L4 queries are longer and more indirect than L0 queries, so degradation is partly stylistic.

Finding 2: LLM query rewriting significantly improves retrieval. Raw first-pass gives 97.78 percent @1 when empty outputs are treated as failures. With deterministic fallback-to-original for empty outputs, accuracy rises to 98.67 percent @1. On L2 to L4, the improvement is +10.74 percentage points with p equals 0.0004. L4 accuracy improves from 78.89 percent to 96.67 percent. Gap recovery is 88.89 percent against measured L0 ceiling.

Finding 3: Zero-score tie-breaking artifact in rank-based fusion. 16 out of 450 queries have all-zero BM25 scores. Naive RRF causes a 1.55 percentage point drop in @1. Proper handling restores E5-large parity. This is a previously unreported artifact in cross-lingual hybrid retrieval.

Finding 4: Fusion improves retrieval specifically on hard subgroups. Convex combination overall gives +0.89 percentage points, not significant at p equals 0.18. On the L3 to L4 subgroup, Roman Urdu, the improvement is +2.78 percentage points with p equals 0.028. This is hypothesis-driven and does not survive strict Bonferroni correction. It suggests fusion helps when single-retriever performance is weak.

Finding 5: Model size shows a trend. E5-large versus E5-base is +3.56 percentage points at p equals 0.0504. Borderline, does not survive multiple-comparison correction.

Finding 6: LLM non-determinism is an observation. 25.83 percent of queries differ across 3 runs at temperature 0 on the batch pipeline. Per-run retrieval was not measured. A_single 3-run variance is future work.

---

## 11. Limitations

1. Corpus size is 30 documents, small for generalization.

2. Documents are synthetic and templated. Real-world corpora will differ.

3. Only 30 candidates. Ceiling effect around 98 percent.

4. Single domain, university policies only.

5. Query authorship is a single author, a native Roman Urdu speaker. No LLM was used to generate query text. No independent validation was performed.

6. Query duplication across levels. 32 out of 90 L1 queries are surface-identical to L0 counterparts. One L3 to L4 pair and one L2 to L3 pair are also identical. This may amplify L0 to L1 similarity.

7. Parsing-failure policy. 3 of 450 LLM outputs were empty on first pass, and 1 was a no-op. Raw first-pass equals 97.78 percent. Fallback and retry both yield 98.67 percent. No real translation improvement from retry over fallback.

8. Single LLM tested. Only openai/gpt-oss-120b.

9. Potential training-data contamination. Roman Urdu and English parallel text may be in pretraining data.

10. No human evaluation. L0 to L4 naturalness was not verified by native speakers.

11. No independent translation quality assessment. BLEU reported, no COMET or human judgment.

12. Roman Urdu spelling variation is not modeled.

13. Weight selection is leakage-free. Other choices such as k equals 10, E5-large, and prompt use full-data knowledge.

14. Fusion evaluated on saturated rewritten data.

15. Broader applicability is untested. Healthcare, government, legal, and e-commerce are potential but not verified.

16. LLM non-determinism. 25.83 percent variance on batch pipeline only.

17. D02 and D20 benchmark ambiguity. 2 queries have ambiguous gold labels due to vocabulary collision between Add-Drop and Withdrawal policies.

18. BM25 zero-score queries. 16 out of 450. No signal for these queries. Tie-handling disclosed.

19. Multiple comparisons in subgroup analysis. L3 to L4 subgroup p equals 0.028 does not survive strict Bonferroni correction. Framed as hypothesis-driven.

20. Fallback versus retry equivalence. Both policies give identical results on this benchmark. The incremental value of retry over fallback is zero here, though it may differ on other benchmarks.

21. L0 cache bypass. For L0 queries, script 51 reads the original query directly rather than the rewritten cache, since L0 rewrites were unchanged. This is stated for methodological clarity.

---

## 12. Project Structure

Research/
  DOCUMENTATION.md
  requirements.txt
  LICENSE
  config/
    master_facts.json
    master_queries.json
  data_v2/
    raw/policies/
    processed/
      main_corpus_chunks.json
      main_queries.json
      ablation_single_clean.json
      ablation_single.json
      ablation_shuffle.json
      ablation_l34only.json
    results/
      retrieval_all_rewritten.json
      kfold_cluster_eval.json
      translation_quality.json
      persistent_failures.json
      paired_cluster_bootstrap.json
      weight_sweep.json
      in_fold_weight_tuning.json
      rewrite_variance.json
      batch_leakage_ablation.json
      missing_cluster_tests.json
      a_single_exact.json
      in_fold_a_single.json
      zero_score_analysis.json
      rrf_tie_fixed.json
      clean_pipeline_rerun.json
      final_verification.json
      diagnose_failures.json
      convex_significance.json
      first_pass_direct.json
  paper/
    PAPER_DRAFT.md
    figures/
      fig1_per_level_p1.png
      fig2_degradation_curve.png
      fig3_llm_rewriting_effect.png
      fig4_zero_score_queries.png
      fig5_rrf_tie_artifact.png
      fig6_convex_per_level.png
      fig7_significance.png
      fig8_overall_comparison.png
  src_v2/
    01_generate_documents.py
    02_chunk_documents.py
    03_generate_queries.py
    04_validate_all.py
    05_bm25_retriever.py
    06_dense_retriever.py
    07_hybrid_retriever.py
    08_statistical_tests.py
    09_final_comparison.py
    10_failure_taxonomy.py
    11_rrf_sensitivity.py
    12_dense_model_comparison.py
    13_hybrid_e5large.py
    14_final_comparison_v3.py
    15_weighted_rrf.py
    16_adaptive_rrf.py
    17_statistical_tests_v3.py
    18_final_comparison_v4.py
    19_llm_query_rewriting_groq.py
    20_rewritten_retrieval.py
    21_rewriting_significance.py
    22_generate_figures.py
    23_train_val_test_split.py
    24_llm_rewriting_test_set.py
    25_rewrite_all_queries.py
    26_retrieval_rewrite_all.py
    27_kfold_cluster_eval.py
    28_translation_quality.py
    29_persistent_failures_rewritten.py
    30_paired_cluster_bootstrap.py
    31_weight_sweep.py
    32_in_fold_weight_tuning.py
    33_rewrite_variance.py
    34_batch_leakage_ablation.py
    35_missing_cluster_tests.py
    36_a_single_exact_numbers.py
    37_fix_empty_cache.py
    38_in_fold_a_single.py
    39_clean_fallback.py
    40_convex_combination.py
    41_debug_bm25.py
    42_in_fold_convex.py
    43_zero_score_analysis.py
    44_rrf_tie_fixed.py
    45_clean_pipeline_rerun.py
    46_final_verification.py
    47_diagnose_failures.py
    48_convex_significance.py
    49_final_figures.py
    50_first_pass_measurement.py
    51_correct_first_pass.py

---

## 13. How to Reproduce

### 13.1 Install

pip install sentence-transformers scipy numpy matplotlib groq sacrebleu

### 13.2 Full Pipeline

Step 1, data generation:
python src_v2/01_generate_documents.py
python src_v2/02_chunk_documents.py
python src_v2/03_generate_queries.py
python src_v2/04_validate_all.py

Step 2, baseline retrieval:
python src_v2/05_bm25_retriever.py
python src_v2/06_dense_retriever.py
python src_v2/12_dense_model_comparison.py

Step 3, rewrite all queries:
set GROQ_API_KEY environment variable
python src_v2/25_rewrite_all_queries.py

Step 4, clean fallback:
python src_v2/39_clean_fallback.py

Step 5, evaluation:
python src_v2/36_a_single_exact_numbers.py
python src_v2/51_correct_first_pass.py
python src_v2/28_translation_quality.py
python src_v2/29_persistent_failures_rewritten.py

Step 6, statistical tests:
python src_v2/30_paired_cluster_bootstrap.py
python src_v2/35_missing_cluster_tests.py

Step 7, fusion analysis:
python src_v2/31_weight_sweep.py
python src_v2/32_in_fold_weight_tuning.py
python src_v2/38_in_fold_a_single.py
python src_v2/40_convex_combination.py
python src_v2/42_in_fold_convex.py
python src_v2/44_rrf_tie_fixed.py
python src_v2/45_clean_pipeline_rerun.py
python src_v2/48_convex_significance.py

Step 8, verification:
python src_v2/43_zero_score_analysis.py
python src_v2/46_final_verification.py
python src_v2/47_diagnose_failures.py

Step 9, figures:
python src_v2/49_final_figures.py

Total runtime approximately 95 minutes, excluding model downloads.

### 13.3 Requirements

sentence-transformers version 2.2.0 or higher
scipy version 1.10.0 or higher
numpy version 1.24.0 or higher
matplotlib version 3.7.0 or higher
groq version 0.4.0 or higher
sacrebleu version 2.4.0 or higher

---

## 14. Future Work

1. Native-speaker query authoring. 100 or more queries written by 3 or more native speakers without L0 reference.

2. Corpus expansion to 300 or more documents with natural vocabulary overlap.

3. Multiple LLMs. GPT-4, Gemini, Aya-101 for comparison.

4. Three runs of A_single for variance estimation.

5. Human translation quality assessment with COMET and human judgment on 50 to 100 queries.

6. Natural distractors. Realistic overlapping-vocabulary documents.

7. Cross-domain evaluation. Healthcare, government, legal, e-commerce pilots.

8. Spelling normalization. Handle Roman Urdu spelling variants such as hazri and hazree.

9. Query re-authoring to fix L0 and L1 duplicates.

10. In-fold fusion on original queries where benchmark is not saturated.

11. Zero-score handling in other fusion methods. Test whether the artifact affects learned fusion.

12. Larger max_tokens on first pass to reduce empty rate.

---

## 15. References

[1] Robertson, S., and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4), 333 to 389.

[2] Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., and Wei, F. (2024). Multilingual E5 Text Embeddings: A Technical Report. arXiv:2402.05672.

[3] Cormack, G. V., Clarke, C. L. A., and Buettcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. Proceedings of SIGIR, 758 to 759.

[4] Grefenstette, G. (Ed.) (1998). Cross-Language Information Retrieval. Kluwer Academic Publishers.

[5] Peters, C., Braschler, M., Gonzalo, J., and Kluck, M. (Eds.) (2002). Evaluation of Cross-Language Information Retrieval Systems: CLEF 2001. Lecture Notes in Computer Science, volume 2406. Springer.

[6] Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., and Liu, Z. (2024). BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity. arXiv:2402.03216.

[7] Zhang, X., Thakur, N., Ogundepo, O., Kamalloo, E., Alfonso-Hermelo, D., Li, X., Liu, Q., Rezagholizadeh, M., and Lin, J. (2023). MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages. Transactions of the Association for Computational Linguistics, 11, 1114 to 1131.

[8] Zhang, X., Ma, X., Shi, P., and Lin, J. (2021). Mr. TyDi: A Multi-lingual Benchmark for Dense Retrieval. arXiv:2108.08787.

[9] Bruch, S., Gai, S., and Ingber, A. (2023). An Analysis of Fusion Functions for Hybrid Retrieval. ACM Transactions on Information Systems, 42(1). DOI: 10.1145/3596512.

[10] Banerjee, S., Choudhury, M., Chakma, K., Naskar, S. K., Das, A., Bandyopadhyay, S., and Rosso, P. (2020). MSIR at FIRE: A Comprehensive Report from 2013 to 2016. SN Computer Science, 1(1).

[11] Gao, L., Ma, X., Lin, J., and Callan, J. (2023). Precise Zero-Shot Dense Retrieval without Relevance Labels. Proceedings of ACL.

[12] Wang, L., Yang, N., and Wei, F. (2023). Query2doc: Query Expansion with Large Language Models. Proceedings of EMNLP.

[13] Butt, M. U. T., Varanasi, S., and Neumann, G. (2025). Roman Urdu as a Low-Resource Language: Building the First IR Dataset and Baseline. Proceedings of the First Workshop on Advancing NLP for Low-Resource Languages, LowResNLP 2025. ACL Anthology 2025.lowresnlp-1.9.

[14] Rana, T. A., Shahzadi, K., Rana, T., Arshad, A., and Tubishat, M. (2021). An Unsupervised Approach for Sentiment Analysis on Social Media Short Text Classification in Roman Urdu. ACM Transactions on Asian and Low-Resource Language Information Processing, 21(2), 1 to 16. DOI: 10.1145/3474119.

[15] Alam, M., and Hussain, S. (2022). Roman-Urdu-Parl: Roman-Urdu and Urdu Parallel Corpus for Urdu Language Understanding. ACM Transactions on Asian and Low-Resource Language Information Processing, 21(1), 13:1 to 13:20.

---

## Citation

@misc{roman_urdu_retrieval_2026,
  title={Quantifying the Cross-Lingual Gap in Roman Urdu Retrieval and Closing It with LLM Query Rewriting},
  author={Umar Farooq},
  year={2026},
  note={Independent Research Project, Thal University Bhakkar}
}

---

## License

Code: MIT License
Data: CC-BY-4.0

---

End of Documentation

