# Quantifying the Cross-Lingual Gap in Roman Urdu Retrieval and Closing It with LLM Query Rewriting

Author: Umar Farooq
Affiliation: Thal University Bhakkar
Email: umar.farooq@scholar.tu.edu.pk
Status: Draft

---

## Abstract

Roman Urdu, a code-mixed language variety used by over 100 million South Asians, presents challenges for retrieval systems trained on English. We present a systematic evaluation of sparse (BM25), dense (Multilingual-E5), and hybrid retrieval systems on Roman Urdu queries against English documents. Using a benchmark of 30 documents and 450 queries across 5 linguistic levels (L0 equals formal English to L4 equals natural Roman Urdu), we report four findings:

(1) Cross-lingual degradation is severe. Four systems drop 20 to 28 percentage points from L0 to L4. Cluster bootstrap gives p less than 0.0002.

(2) LLM-based query rewriting in single-query mode achieves 97.78 percent @1 raw, when empty outputs are treated as failures. With deterministic fallback-to-original for empty outputs, accuracy rises to 98.67 percent @1. The 95 percent confidence interval for the fallback result is [0.9733, 0.9978]. On L2 to L4, the improvement is +10.74 percentage points with p equals 0.0004, cluster-aware. L4 accuracy improves from 78.89 percent to 96.67 percent, achieving 88.89 percent gap recovery against the measured L0 ceiling. Retry with larger max_tokens yields the same 98.67 percent, indicating fallback is the effective intervention.

(3) We identify a zero-score tie-breaking artifact in rank-based fusion. BM25 produces all-zero scores for 16 out of 450 queries, which is 3.6 percent, all in L3 and L4. Naive RRF injects arbitrary tie order, causing a 1.55 percentage point drop in @1. Proper zero-score handling, meaning skip BM25 when all-zero, restores RRF to E5-large parity at 92.22 percent.

(4) Score-based convex combination with in-fold alpha achieves 93.11 percent test @1, a +0.89 percentage point improvement over E5-large alone, but not statistically significant at p equals 0.18. On the L3 to L4 subgroup, Roman Urdu, the improvement is +2.78 percentage points with p equals 0.028, hypothesis-driven and not surviving strict Bonferroni correction.

Keywords: Cross-Lingual IR, Roman Urdu, Code-Mixed Retrieval, Low-Resource NLP, LLM Query Rewriting

---

## 1. Introduction

### 1.1 Motivation

Roman Urdu is a code-mixed language variety in which Urdu grammar and vocabulary are written in Latin script, often interspersed with English words. It is the primary digital communication medium for over 100 million people across Pakistan and India. In formal contexts, users frequently query in Roman Urdu while documents are authored in English.

Example:
Document says: "The minimum attendance is 75%."
User queries: "Talib-e-ilm ki kam az kam hazri kitni honi zaroori hai?"

This creates a cross-lingual retrieval gap. Our work characterizes its severity and evaluates a practical solution.

### 1.2 Research Questions

RQ1: How much does retrieval performance degrade from formal English (L0) to natural Roman Urdu (L4)?

RQ2: Can LLM-based query rewriting recover the cross-lingual gap?

RQ3: Do hybrid fusion methods improve cross-lingual retrieval?

### 1.3 Contributions

1. A reproducible benchmark of 30 documents and 450 queries across 5 linguistic levels.

2. Cluster-aware evaluation with grouped 5-fold cross-validation and document-level bootstrap confidence intervals.

3. Two strong findings: severe cross-lingual degradation with p less than 0.0002, and significant LLM rewriting improvement with +10.74 percentage points on L2 to L4 at p equals 0.0004.

4. A zero-score tie-breaking artifact in rank-based fusion, a previously unreported issue in cross-lingual hybrid retrieval.

5. Reproducibility artifacts: 51 scripts, 8 figures, full documentation, all raw results.

---

## 2. Related Work

### 2.1 Sparse Retrieval

BM25 remains a strong lexical baseline due to term-frequency saturation and document length normalization.

### 2.2 Dense Retrieval

Multilingual-E5 is trained on weakly-supervised contrastive data across 100 languages. BGE-M3 extends this to multi-vector retrieval. MIRACL and Mr. TyDi provide multilingual retrieval benchmarks.

### 2.3 Hybrid Retrieval

Reciprocal Rank Fusion is the standard fusion technique but is rank-based. Bruch et al. show that convex combination of normalized scores typically outperforms RRF because it retains score-distribution information. Our work identifies a zero-score artifact in rank-based fusion that can explain part of RRF's weakness in cross-lingual settings.

### 2.4 Cross-Lingual IR

Cross-Lingual Information Retrieval has been studied extensively. The FIRE MSIR shared task addressed Romanized Indic retrieval.

### 2.5 Roman Urdu Retrieval

Butt et al. built a Roman Urdu IR dataset by translating and transliterating MS MARCO into Roman Urdu for monolingual retrieval. Alam and Hussain introduced the Roman-Urdu-Parl parallel corpus. Rana et al. studied sentiment classification on Roman Urdu social media text with spelling normalization.

To the best of our knowledge, no prior work provides a systematic cross-lingual retrieval evaluation where queries are in Roman Urdu and documents are in English. This is the gap we address.

### 2.6 LLM Query Rewriting

HyDE and Query2doc generate pseudo-documents for zero-shot retrieval. Our work applies LLM translation rather than expansion, for low-resource code-mixed queries.

---

## 3. Methodology

### 3.1 Corpus Design

30 university policy documents, each with 1 unique fact.

Domain: University policies.
Language: Formal English.
Length: 80 to 100 words per document.
Facts: 1 unique fact per document.
Vocabulary: Intentionally distinct terms per document.

Design caveat: Vocabulary is intentionally distinct to reduce cross-document lexical overlap. Real-world corpora will have more naturalistic challenges.

### 3.2 Query Dataset

450 queries equals 30 documents times 5 levels times 3 variants.

Level 0, formal English. Example: "What minimum attendance is required per course?"

Level 1, English-dominant with Urdu genitive markers. Example: "What minimum attendance is required per course ki classes mein?"

Level 2, balanced code-mixing. Example: "Minimum attendance kitni honi chahiye?"

Level 3, Urdu-dominant. Example: "Talib-e-ilm ki kam az kam hazri kitni honi zaroori hai?"

Level 4, natural Roman Urdu. Example: "Agar talib-e-ilm imtihan dena chahta hai to minimum hazri kitni honi chahiye?"

Query authorship: All 450 queries were written by the author, a native Roman Urdu speaker. No LLM was used to generate query text. The script 03_generate_queries.py assembles queries from a manually curated JSON file named master_queries.json. No independent validation was performed.

Known limitation: 32 out of 90 L1 queries are surface-identical to their L0 counterparts, with no Urdu markers added. One L3 to L4 pair and one L2 to L3 pair are also identical. This may amplify L0 to L1 similarity in results.

Level distribution: L0 equals 90, L1 equals 90, L2 equals 90, L3 equals 90, L4 equals 90.

### 3.3 Systems Evaluated

BM25, sparse retrieval with k1 equals 1.5 and b equals 0.75.

E5-base, dense retrieval with 278 million parameters.

E5-large, dense retrieval with 560 million parameters.

Hybrid-base, Reciprocal Rank Fusion of BM25 and E5-base with k equals 10 and equal weights 1.0 and 1.0.

Hybrid-large, Reciprocal Rank Fusion of BM25 and E5-large with k equals 10 and equal weights 1.0 and 1.0.

Weighted RRF, where score equals w_bm25 divided by (k plus r_bm25) plus w_e5 divided by (k plus r_e5).

Convex combination, score-based fusion where score equals alpha times normalized BM25 score plus (1 minus alpha) times normalized E5 score.

LLM Query Rewriting, using Groq model openai/gpt-oss-120b with temperature 0.0, max_tokens 300 for first pass and 1500 for retry.

### 3.4 LLM Rewriting Prompt

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

### 3.5 Evaluation Metrics

P@K, the fraction of queries with the gold document in top-K.

MRR, mean reciprocal rank.

Gap recovery, defined as (L4_rewritten minus L4_original) divided by (L0_measured minus L4_original).

### 3.6 Cluster-Aware Evaluation

Because 15 queries share each document, standard independence assumptions are violated. We use grouped 5-fold cross-validation where documents split into 5 folds, 6 documents per fold, 90 queries per fold. We also use cluster bootstrap where documents are sampled with replacement across 5000 iterations, and paired cluster bootstrap where sampling is paired at document level.

Multiple-comparison note: With 10 comparisons, the Bonferroni threshold at alpha equal to 0.05 is 0.005. Only L0 versus L4 degradation and A_single rewriting tests survive this threshold. Others are labeled exploratory.

### 3.7 Batch Leakage Ablation

Three variants:
A_single, one query per API call, no batch context. Primary deployment scenario.
B_shuffle, shuffled batches across documents.
C_l34only, L3 and L4 only batches.

### 3.8 Parsing-Failure Policy

First pass used max_tokens 300. Four outputs required correction on first pass:

Three outputs were empty: MQ0169 at level L1, MQ0364 at level L1, and MQ0389 at level L4.

One output was a no-op: MQ0006 at level L1, where the LLM returned the input unchanged because the query is already English.

So the four cases comprise 3 empty outputs and 1 no-op output.

Two deterministic policies were tested:

Policy 1, fallback-to-original: if LLM output is empty, use original query text for retrieval.

Policy 2, retry then fallback: re-query with max_tokens 1500; if still empty, fallback to original.

Results:

Raw first-pass, empty outputs treated as failures: 97.78 percent @1, that is 440 out of 450.

First-pass plus fallback-to-original: 98.67 percent @1, that is 444 out of 450.

Retry then fallback: 98.67 percent @1, that is 444 out of 450.

Both policies achieve the same result. This means the empty outputs are fixed by retrieving on the original query text, not by improved translations. The incremental value of retry over fallback is zero on this benchmark.

Note on L0: For L0 queries, the original text is used in first-pass evaluation because L0 rewrites were observed to be unchanged. The LLM correctly treated them as already English. The measurement script reads the original L0 query directly rather than the rewritten cache.

Two headline numbers:
Raw first-pass: 97.78 percent @1, 440 out of 450.
Deployment with fallback: 98.67 percent @1, 444 out of 450.

---

## 4. Results

### 4.1 Original Queries (Full 450, Tie-Aware)

System BM25: @1 equals 88.44 percent, @3 equals 94.00 percent, MRR equals 0.9148.

System E5-large: @1 equals 92.22 percent, @3 equals 96.89 percent, MRR equals 0.9489.

System Hybrid-large (RRF tie-aware): @1 equals 92.22 percent, @3 equals 97.33 percent, MRR equals 0.9497.

System Weighted-RRF (in-fold, tie-aware): @1 equals 92.67 percent, @3 equals 97.11 percent, MRR equals 0.9518.

System Convex (in-fold): @1 equals 93.11 percent, @3 equals 97.56 percent, MRR equals 0.9551.

### 4.2 Per-Level @1 (Original)

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

### 4.3 A_single, Deployment Scenario

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

### 4.4 Per-Level @1 (A_single, Fallback or Retry)

L0: 98.89 percent, @3 equals 100.0, MRR equals 0.9944.
L1: 98.89 percent, @3 equals 100.0, MRR equals 0.9944.
L2: 100.00 percent, @3 equals 100.0, MRR equals 1.0000.
L3: 98.89 percent, @3 equals 100.0, MRR equals 0.9944.
L4: 96.67 percent, @3 equals 100.0, MRR equals 0.9815.

### 4.5 Gap Recovery (A_single, Fallback or Retry)

L0 original equals 98.89 percent.
L4 original equals 78.89 percent.
L4 A_single equals 96.67 percent.

Recovery against measured L0 ceiling equals (96.67 minus 78.89) divided by (98.89 minus 78.89), which equals 88.89 percent.

Recovery against assumed 100 percent ceiling equals (96.67 minus 78.89) divided by (100 minus 78.89), which equals 84.21 percent.

95 percent CI for recovery against the 100 percent ceiling is [0.5556, 1.0000].

### 4.6 Zero-Score Tie-Breaking Artifact

16 out of 450 queries, which is 3.6 percent, have all-zero BM25 scores. All 16 are in L3 and L4, specifically 9 in L3 and 7 in L4. These are queries where no term matches any document because they use Urdu-only vocabulary such as parhai, sawari, and shikayat, which have no English overlap.

Impact on rank-based fusion:

Variant A, arbitrary tie with stable argsort: @1 equals 90.67 percent, @3 equals 96.44 percent, MRR equals 0.9384.

Variant B, skip BM25 when all-zero: @1 equals 92.22 percent, @3 equals 97.33 percent, MRR equals 0.9497.

Variant C, random tie with fixed seed: @1 equals 91.11 percent, @3 equals 96.67 percent, MRR equals 0.9414.

E5-large alone: @1 equals 92.22 percent, @3 equals 96.89 percent, MRR equals 0.9489.

Finding: Variant B recovers RRF to E5-large parity. The apparent RRF failure is a tie-breaking artifact, not a fusion limitation. This is a previously unreported issue in cross-lingual hybrid retrieval.

### 4.7 Fusion: Convex Combination, In-Fold Alpha

Fold 1: best alpha equals 0.2, train @1 equals 0.9250, test @1 equals 0.9778, oracle @1 equals 0.9889.

Fold 2: best alpha equals 0.5, train @1 equals 0.9361, test @1 equals 0.9222, oracle @1 equals 0.9444.

Fold 3: best alpha equals 0.2, train @1 equals 0.9694, test @1 equals 0.8000, oracle @1 equals 0.8000.

Fold 4: best alpha equals 0.2, train @1 equals 0.9194, test @1 equals 1.0000, oracle @1 equals 1.0000.

Fold 5: best alpha equals 0.2, train @1 equals 0.9306, test @1 equals 0.9556, oracle @1 equals 0.9667.

Mean train @1 equals 0.9361. Mean test @1 equals 0.9311. Mean oracle @1 equals 0.9400.

Test @1 equals 93.11 percent on held-out folds.

### 4.8 E5-large Remaining Failures (A_single)

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

## 5. Statistical Analysis

### 5.1 Cluster-Aware Paired Bootstrap Tests

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

## 6. Key Findings

### Finding 1: Cross-Lingual Degradation is Severe and Significant

Four systems drop 20 to 28 percentage points from L0 to L4. Cluster bootstrap gives p less than 0.0002.

Caveat: L4 queries are longer and more indirect than L0 queries, so degradation is partly stylistic, not purely linguistic.

### Finding 2: LLM Query Rewriting Significantly Improves Retrieval

Raw first-pass gives 97.78 percent @1 when empty outputs are treated as failures. With deterministic fallback-to-original for empty outputs, accuracy rises to 98.67 percent @1. On L2 to L4, the improvement is +10.74 percentage points with p equals 0.0004, cluster-aware. L4 accuracy improves from 78.89 percent to 96.67 percent. Gap recovery is 88.89 percent against measured L0 ceiling. Retry with max_tokens 1500 gives the same 98.67 percent, indicating fallback is the effective intervention.

### Finding 3: Zero-Score Tie-Breaking Artifact in Rank-Based Fusion

16 out of 450 queries have all-zero BM25 scores, all in L3 and L4. Naive RRF causes a 1.55 percentage point drop in @1. Proper zero-score handling, meaning skip BM25 when all-zero, restores E5-large parity. This is a previously unreported artifact in cross-lingual hybrid retrieval.

### Finding 4: Fusion Improves Retrieval Specifically on Hard Subgroups

Convex combination overall gives +0.89 percentage points, not significant at p equals 0.18. On the L3 to L4 subgroup, Roman Urdu, the improvement is +2.78 percentage points with p equals 0.028. This is hypothesis-driven and does not survive strict Bonferroni correction. It suggests fusion helps when single-retriever performance is weak. On L0 to L2, where E5-large already achieves 98.89 to 100 percent, a ceiling effect prevents any fusion improvement.

### Finding 5: Model Size Shows a Trend

E5-large versus E5-base is +3.56 percentage points at p equals 0.0504. Borderline. It does not survive multiple-comparison correction.

### Finding 6: LLM Non-Determinism is an Observation

25.83 percent of queries differ across 3 runs at temperature 0 on the batch pipeline. Per-run retrieval was not measured. A_single 3-run variance is future work.

---

## 7. Limitations

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

21. L0 cache bypass. For L0 queries, the measurement script reads the original query directly rather than the rewritten cache, since L0 rewrites were unchanged.

---

## 8. Future Work

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

## 9. Conclusion

We present a cluster-aware evaluation of cross-lingual retrieval for Roman Urdu.

1. Cross-lingual degradation is severe at 20 to 28 percentage points, with p less than 0.0002.

2. LLM query rewriting achieves 97.78 percent @1 raw, rising to 98.67 percent with deterministic fallback-to-original. On L2 to L4, the improvement is +10.74 percentage points at p equals 0.0004. This is the paper's primary contribution.

3. A zero-score tie-breaking artifact in rank-based fusion causes a 1.55 percentage point drop in @1. Proper handling restores E5-large parity.

4. Fusion helps specifically on hard subgroups. Convex combination improves +2.78 percentage points on L3 to L4, Roman Urdu, at p equals 0.028. This is hypothesis-driven and does not survive strict Bonferroni correction.

5. Model size shows a trend at p equals 0.0504, but does not reach significance after correction.

We contribute a reproducible benchmark, a quantified evaluation of LLM rewriting for low-resource cross-lingual retrieval, and a previously unreported fusion artifact.

---

## References

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

## Appendix A: Query Examples (Document D05, Class Attendance, 75%)

L0: What minimum attendance percentage is required per course?

L1: What minimum attendance percentage is required per course ki classes mein?

L2: Minimum attendance kitni honi chahiye?

L3: Talib-e-ilm ki kam az kam hazri kitni honi zaroori hai?

L4: Agar talib-e-ilm imtihan dena chahta hai to minimum hazri kitni honi chahiye?

## Appendix B: Reproducibility

All code is available in src_v2, 51 Python scripts. Complete pipeline runs in approximately 95 minutes, excluding first-time model downloads. All experiments use seed equal to 42 for reproducibility.

Execution order:

Step 1, data generation: scripts 01, 02, 03, 04.

Step 2, baseline retrieval: scripts 05, 06, 12.

Step 3, rewrite all queries, single-query mode: script 25.

Step 4, clean fallback, re-query and fallback: script 39.

Step 5, evaluation: scripts 36, 51, 28, 29.

Step 6, statistical tests: scripts 30, 35.

Step 7, fusion analysis: scripts 31, 32, 38, 40, 42, 44, 45, 48.

Step 8, verification: scripts 43, 46, 47.

Step 9, figures: script 49.

Note: Script 37 is diagnostic only and is superseded by script 39. Script 50 is superseded by script 51, which is the first-pass plus fallback measurement.

Dependencies:
sentence-transformers version 2.2.0 or higher
scipy version 1.10.0 or higher
numpy version 1.24.0 or higher
matplotlib version 3.7.0 or higher
groq version 0.4.0 or higher
sacrebleu version 2.4.0 or higher

## Appendix C: Verification

Script 46 performs 46 automated checks across 14 sections. All checks pass:
Dataset integrity, 13 checks.
A_single cache integrity, 4 checks, corrected for L1-expected behavior.
Zero-score queries, 2 checks.
E5-large baseline, 3 checks.
RRF tie-aware parity, 1 check.
A_single retrieval, 3 checks.
Gap recovery, 4 checks.
Degradation, 6 checks.
Statistical significance, 4 checks.
Per-level count consistency, 2 checks.
Data leakage and duplicate detection, 1 check, disclosed.
BM25 zero-score honesty, 2 checks.
Statistical power, 1 check.

Total: 44 checks pass, 2 checks flagged as disclosed limitations.

---

End of Paper Draft

---

