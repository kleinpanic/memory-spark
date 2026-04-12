# memory-spark v0.4.0 — Full BEIR Benchmark Results

**Run completed:** 2026-04-03 02:35 EDT  
**Duration:** ~33 hours (SciFact + FiQA + NFCorpus × 36 configs)  
**Datasets:** SciFact (300 queries), FiQA (648 queries), NFCorpus (323 queries)  
**Embedding model:** `spark/nvidia/llama-embed-nemotron-8b` (dims=4096)  
**Reranker model:** `spark/nvidia/llama-nemotron-rerank-1b-v2`  
**Infrastructure:** DGX Spark node (local); Spark embed port 18091, reranker port 18096

---

## 📊 Comparison with Published BEIR Baselines (SOTA)

The following table compares our best configurations against published NDCG@10 numbers from Thakur et al. (2021) *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models* (NeurIPS 2021). All BEIR baselines are zero-shot (no dataset-specific fine-tuning), same as our evaluation.

| System | SciFact | FiQA | NFCorpus | Avg |
|--------|---------|------|----------|-----|
| BM25 (Thakur et al. 2021) | 0.665 | 0.236 | 0.325 | 0.409 |
| DPR (Thakur et al. 2021) | 0.318 | 0.167 | 0.182 | 0.222 |
| ANCE (Thakur et al. 2021) | 0.507 | 0.295 | 0.237 | 0.346 |
| TAS-B (Thakur et al. 2021) | 0.643 | 0.300 | 0.319 | 0.421 |
| Contriever (Izacard et al. 2021) | 0.677 | 0.329 | 0.328 | 0.445 |
| **memory-spark: Vector-Only (Config A)** | **0.7709** | **0.5469** | **0.4443** | **0.5874** |
| **memory-spark: Best (Config U)** | **0.7889** | **0.5526** | 0.4344 | **0.5920** |
| **memory-spark: GATE-A (Default)** | **0.7802** | **0.5479** | 0.4256 | **0.5846** |

**Context:** All baselines above are from Thakur et al. (2021). Our evaluation uses 3 datasets (SciFact, FiQA, NFCorpus) — not the full 18-dataset BEIR benchmark. Modern dense retrievers (E5-Large-v2: ~56% avg, BGE-Large-EN-v1.5: ~54% avg on full BEIR) outperform these 2021 baselines significantly. Our relative improvements are against the 2021 SOTA for comparability; a full BEIR 2.0 comparison is planned.

**Our embedding model:** `llama-embed-nemotron-8b` (8B parameter, instruction-tuned, 4096-dim) — a modern large model vs the 2021-era 110M–330M parameter baselines above.

### Relative Improvement vs. Contriever (2021-era strongest baseline)

> **⚠️ Caveat:** These improvements are against 2021-era baselines. E5-Large-v2 and BGE-Large-EN-v1.5 achieve ~56% and ~54% average NDCG@10 on full 18-dataset BEIR. Our 59.2% average on 3 favorable datasets is not directly comparable.

| Dataset | Contriever (2021) | Our Best (3 datasets) | Improvement |
|---------|------------------|----------------------|-------------|
| SciFact | 0.677 | **0.7889** | **+16.5%** |
| FiQA | 0.329 | **0.5526** | **+68.0%** |
| NFCorpus | 0.328 | **0.4443** (Vec-A) | **+35.5%** |
| **Average** | 0.445 | **0.5920** | **+33.1%** |

> **Note on NFCorpus:** Our Vector-Only baseline (0.4443) outperforms our best reranked configuration (Config U: 0.4344) on NFCorpus. This is a structural finding — see [Cross-Dataset Variance Analysis](#cross-dataset-variance-analysis) below.

---

## 🏆 Top-Line Results (Ranked by 3-Dataset Average NDCG@10)

| Rank | Config | Label | SciFact | FiQA | NFCorpus | **Avg** | p50 (ms) |
|------|--------|-------|---------|------|----------|---------|----------|
| 1 | **U** | Vector → Logit Blend (α=0.4) | 0.7889 | 0.5526 | 0.4344 | **0.5920** | 1506 |
| 2 | V | Vector → Logit Blend (α=0.6) | 0.7885 | 0.5526 | 0.4344 | 0.5918 | 1457 |
| 3 | W | Vector → Logit Blend (α=0.7) | 0.7883 | 0.5526 | 0.4344 | 0.5918 | 1467 |
| 4 | N | Vector → Blended Reranker (α=0.5) | 0.7863 | 0.5526 | 0.4344 | 0.5911 | 1466 |
| 5 | Q | Vector → Logit-Recovered Blend (α=0.5) | 0.7863 | 0.5526 | 0.4344 | 0.5911 | 1456 |
| 6 | X | Vector → Logit Blend (α=0.8) | 0.7847 | 0.5526 | 0.4344 | 0.5906 | 1454 |
| 7 | O | Conditional Rerank (skip confident, α=0.3) | 0.7792 | 0.5526 | 0.4344 | 0.5887 | 930 |
| 8 | S | Vector → Logit Blend + Conditional (α=0.3) | 0.7792 | 0.5526 | 0.4344 | 0.5887 | 1066 |
| 9 | M | Vector → Blended Reranker (α=0.3) | 0.7756 | 0.5526 | 0.4344 | 0.5875 | 1525 |
| 10 | **A** | **Vector-Only** (baseline) | 0.7709 | 0.5469 | **0.4443** | 0.5874 | **516** |
| 11 | MQ-B | Multi-Query (3) → Logit Blend (α=0.4) | **0.7889** | 0.5418 | 0.4254 | 0.5854 | 8690 |
| 12 | **GATE-A** | RRF + Hard Gate (spread > 0.08 or < 0.02) | 0.7802 | 0.5479 | 0.4256 | 0.5846 | 626 |
| 13 | RRF-D | Vector ⊕ Reranker via RRF (k=20, equal) | 0.7798 | 0.5479 | 0.4255 | 0.5844 | 1502 |
| 14 | RRF-A | Vector ⊕ Reranker via RRF (k=60, equal) | 0.7797 | 0.5479 | 0.4256 | 0.5844 | 1458 |
| 15 | RRF-B | Vector ⊕ Reranker via RRF (k=60, vec=1.5) | 0.7788 | 0.5465 | 0.4264 | 0.5839 | 1468 |

---

## Full Results by Dataset

### SciFact (300 queries — Scientific Claim Verification)

| Config | Label | NDCG@10 | Recall@10 | MAP@10 | p50 (ms) | p95 (ms) |
|--------|-------|---------|-----------|--------|----------|----------|
| U | Vector → Logit Blend (α=0.4) | **0.7889** | — | — | 1506 | — |
| MQ-B | Multi-Query (3) → Logit Blend (α=0.4) | **0.7889** | — | — | 8690 | — |
| V | Vector → Logit Blend (α=0.6) | 0.7885 | — | — | 1457 | — |
| W | Vector → Logit Blend (α=0.7) | 0.7883 | — | — | 1467 | — |
| N | Vector → Blended Reranker (α=0.5) | 0.7863 | — | — | 1466 | — |
| Q | Vector → Logit-Recovered Blend (α=0.5) | 0.7863 | — | — | 1456 | — |
| X | Vector → Logit Blend (α=0.8) | 0.7847 | — | — | 1454 | — |
| MQ-D | Multi-Query (3) → Conditional Logit Blend | 0.7827 | — | — | 7626 | — |
| GATE-D | RRF (vec=1.5) + Soft Gate | 0.7803 | — | — | 1452 | — |
| GATE-A | RRF + Hard Gate | 0.7802 | — | — | 626 | — |
| GATE-B | RRF + Soft Gate (dynamic vector weight) | 0.7802 | — | — | 1465 | — |
| O | Conditional Rerank (α=0.3) | 0.7792 | — | — | 930 | — |
| P | Full 9A+9B Pipeline | 0.7797 | — | — | 1539 | — |
| RRF-D | Vector ⊕ Reranker via RRF (k=20) | 0.7798 | — | — | 1502 | — |
| RRF-A | Vector ⊕ Reranker via RRF (k=60) | 0.7797 | — | — | 1458 | — |
| S | Logit Blend + Conditional (α=0.3) | 0.7792 | — | — | 1066 | — |
| GATE-C | RRF + Soft Gate (threshold=0.12) | 0.7782 | — | — | 1490 | — |
| RRF-B | Vector ⊕ Reranker RRF (vec=1.5) | 0.7788 | — | — | 1468 | — |
| MQ-C | Multi-Query (3) → Logit Blend (α=0.5) | 0.7843 | — | — | 7945 | — |
| T | Vector → Logit Blend (α=0.3) | 0.7756 | — | — | 1624 | — |
| M | Vector → Blended Reranker (α=0.3) | 0.7756 | — | — | 1525 | — |
| RRF-C | Vector ⊕ Reranker RRF (reranker=1.5) | 0.7770 | — | — | 1454 | — |
| A | Vector-Only | 0.7709 | 0.9037 | 0.7231 | 516 | 657 |
| K | Vector → Adaptive MMR | 0.7622 | — | — | 699 | — |
| MQ-A | Multi-Query (3) → Vector-Only | 0.7619 | — | — | 7550 | — |
| L | Full Adaptive (RRF → Reranker → MMR) | 0.7515 | — | — | 2057 | — |
| D | Hybrid + Reranker | 0.7525 | — | — | 1486 | — |
| G | Full Pipeline | 0.7525 | — | — | 1628 | — |
| I | Adaptive Hybrid (overlap-aware RRF) | 0.7557 | — | — | 1069 | — |
| H | Vector → Reranker (no RRF) | 0.7395 | — | — | 1543 | — |
| R | Vector → Pure Logit Reranker (α=0) | 0.7395 | — | — | 2381 | — |
| C | Hybrid | 0.7307 | — | — | 950 | — |
| E | Hybrid + MMR | 0.7290 | — | — | 1179 | — |
| F | Hybrid + HyDE | 0.7278 | — | — | 8828 | — |
| J | Reranker-as-Fusioner (union → rerank) | 0.7262 | — | — | 2113 | — |
| B | FTS-Only | 0.6587 | — | — | 588 | — |

### FiQA (648 queries — Financial Q&A)

| Config | NDCG@10 | p50 (ms) |
|--------|---------|----------|
| M/N/O/Q/S/T/U/V/W/X | **0.5526** | ~1000–1600 |
| H/R/RRF-A/RRF-D/GATE-A | 0.5479 | ~626–2381 |
| RRF-C | 0.5477 | 1454 |
| RRF-B | 0.5465 | 1468 |
| A (Vector-Only) | 0.5469 | 516 |
| K | 0.5394 | — |
| MQ-D | 0.5393 | 7626 |
| MQ-C | 0.5374 | 7945 |
| GATE-D | 0.5409 | 1452 |
| GATE-B | 0.5371 | 1465 |
| GATE-C | 0.5362 | 1490 |
| MQ-B | 0.5418 | 8690 |
| MQ-A | 0.5289 | 7550 |
| L | 0.5133 | 2057 |
| P | 0.5134 | 1539 |
| I | 0.5045 | 1069 |
| D/G | 0.4505 | ~1500 |
| C | 0.4364 | 950 |
| E | 0.4247 | 1179 |
| F (HyDE) | 0.4196 | 8828 |
| J | 0.2796 | 2113 |
| B (FTS-Only) | 0.2421 | 588 |

### NFCorpus (323 queries — Medical/Scientific)

| Config | NDCG@10 | p50 (ms) |
|--------|---------|----------|
| A (Vector-Only) | **0.4443** | 516 |
| I (Adaptive Hybrid) | 0.4356 | 1069 |
| M/N/O/P/Q/S/T/U/V/W/X | 0.4344 | ~1000–1600 |
| MQ-A | 0.4341 | 7550 |
| K | 0.4277 | 699 |
| MQ-C | 0.4292 | 7945 |
| RRF-B | 0.4264 | 1468 |
| H/R/RRF-A/GATE-A | 0.4256 | ~626–2381 |
| RRF-D | 0.4255 | 1502 |
| MQ-D | 0.4273 | 7626 |
| L | 0.4290 | 2057 |
| RRF-C | 0.4236 | 1454 |
| MQ-B | 0.4254 | 8690 |
| F (HyDE) | 0.4171 | 8828 |
| C (Hybrid) | 0.4142 | 950 |
| G (Full Pipeline) | 0.4105 | 1628 |
| D (Hybrid + Reranker) | 0.4113 | 1486 |
| E (Hybrid + MMR) | 0.4044 | 1179 |
| J (Reranker Fusioner) | 0.4072 | 2113 |
| GATE-D | 0.4006 | 1452 |
| GATE-B | 0.3996 | 1465 |
| GATE-C | 0.3992 | 1490 |
| B (FTS-Only) | 0.3146 | 588 |

---

## Key Findings

### 1. Vector + Logit Blend is the clear winner (Configs U, V, W, N, Q)
Configs U–X and M/N/Q all achieve NDCG@10 of 0.5875–0.5920 average. The sweet spot is α=0.4–0.5 for logit blending. Very robust — multiple configs cluster here, suggesting this range is stable.

### 2. Pure Vector-Only (Config A) is surprisingly competitive
Average 0.5874 — virtually tied with the complex reranking configs (~0.001 gap), but at **516ms p50** vs 1400–1600ms for blended configs. On NFCorpus specifically, Vector-Only **wins outright** (0.4443 vs 0.4344). The simpler the domain, the harder it is to beat.

### 3. GATE-A is the best latency/accuracy tradeoff
NDCG avg 0.5846 at **626ms p50** — 60% faster than equivalent reranking configs (1450ms) while losing only 0.007 NDCG points vs Config U. Skips reranking when vector confidence is high (78% of queries on SciFact). **Best for production under latency constraints.**

### 4. Multi-Query expansion is expensive and doesn't pay off overall
MQ configs run at 7500–8700ms p50 (10–17x slower than vector-only). MQ-B matches Config U on SciFact (0.7889) but loses significantly on FiQA (0.5418 vs 0.5526) and NFCorpus (0.4254 vs 0.4344). Net average loss vs Config U. **Not worth it for the cost.**

### 5. HyDE (Config F) is the worst latency/accuracy combo
8828ms p50 for a 0.5215 average. Worst of any reranking approach. HyDE's timeout sensitivity is too high for production use with the current Spark setup.

### 6. FTS-Only (Config B) and Reranker-as-Fusioner (Config J) are catastrophically bad on FiQA
B: 0.2421 NDCG (vs 0.5469 vector). J: 0.2796. FiQA requires semantic understanding; pure keyword or wrong-pipeline fusion destroys performance. These are only viable as fallback/diagnostic tools.

### 7. GATE-B/C/D hurt NFCorpus significantly
The Soft Gate variants drop NFCorpus to 0.3992–0.4006 vs GATE-A's 0.4256. The dynamic weight adjustment in soft gating is counterproductive on medical/scientific queries where the distinction between "confident" and "uncertain" is less clear. Hard gating wins.

### 8. Reranker provides small positive lift on SciFact, neutral on others
Going from Config A (0.7709) to Config U (0.7889) = +0.018 NDCG on SciFact. On FiQA, A=0.5469 vs U=0.5526 (+0.006). On NFCorpus, A **beats** U (0.4443 vs 0.4344). Reranker lift is dataset-dependent.

---

## Cross-Dataset Variance Analysis

**Deep audit conducted 2026-04-03.** This section explains the structural reasons why performance varies across datasets and why reranking behaves differently per domain.

### Dataset Structural Differences

| Property | SciFact | FiQA | NFCorpus |
|----------|---------|------|----------|
| Queries | 300 | 648 | 323 |
| Corpus size | 5,183 docs | 57,638 docs | 3,633 docs |
| Avg relevant docs/query | **1.1** | **2.6** | **38.2** |
| Relevance scale | Binary (0/1) | Binary (0/1) | Graded (0/1/2) |
| Query avg word length | 12.4 words | 10.8 words | **3.3 words** |
| Query format | Declarative scientific claims | Financial questions | Nutrition **video titles** |
| Corpus format | PubMed abstracts | Forum answers | PubMed abstracts |
| Retrieval task | Same-domain claim→paper | Same-domain Q&A | **Cross-domain** title→abstract |

### Why NFCorpus Scores Lower (and Why That's Fine)

NFCorpus is structurally the hardest dataset for any dense retrieval system:

1. **Cross-domain retrieval**: Queries are 3-word video titles ("Breast Cancer Cells Feed on Cholesterol") from nutritionfacts.org. Documents are PubMed abstracts. The semantic gap between a colloquial video title and a dense technical abstract is fundamentally larger than same-domain retrieval.

2. **Graded relevance with 38 avg relevant docs**: With 38 relevant documents per query and a top-10 return limit, the maximum achievable Recall@10 under *perfect* retrieval is only **0.61**. Our Vector-Only Config A achieves Recall@10 of 0.2212 — which represents good performance given this structural ceiling.

3. **Max achievable Recall@10 is ~0.61**: 129 of 323 queries have ≤10 relevant docs (Recall@10 = 1.0 is achievable). The remaining 194 queries have 11–475 relevant docs, capping Recall@10 below 1.0 even with perfect retrieval.

Our NFCorpus NDCG@10 of 0.4443 (Vector-Only) **beats every published 2021 BEIR baseline** including BM25 (0.325) and Contriever (0.328) by 35.5%.

### Why Reranking Hurts NFCorpus

Per-query analysis of Config A (vector) vs Config H (vector + reranker):

| Outcome | Count | % |
|---------|-------|---|
| Reranker helped (first-relevant moved up) | 16 | 5.0% |
| Reranker hurt (first-relevant moved down) | 22 | 6.8% |
| No change | 285 | 88.2% |
| **Net** | | **-1.9%** |

The reranker is a cross-encoder trained on Q&A pairs. A 3-word video title ("Breast Cancer Cells Feed on Cholesterol") is out-of-distribution for the reranker. It produces near-zero discrimination signal, causing 88.2% of queries to see no change, with the remaining 11.8% split slightly in favor of harm.

### Why Reranking Hurts Specific SciFact Queries Despite Overall Gain

Config U (blended reranking) helped 49 SciFact queries and hurt 34 (net +15). The hurt cases are instructive:

- **Pattern**: Relevant doc ranked #3 by vector at weak score (0.23–0.41). Reranker also scores it low. The blend (α=0.4) pulls its combined score below the 10th-place cutoff. Doc disappears from results. NDCG → 0.0.
- **Specific cases**: Query 1 (relevant doc 31715818 at vector rank #3, score 0.2285 → gone after blend), Query 431 (relevant at rank #3, score 0.4114 → gone), and ~32 others.
- **Root cause**: Score-floor problem with blending. When both vector and reranker agree a doc is weak, blending amplifies that weakness and drops the doc out of the final pool.
- **Planned fix**: Phase 13 — position preservation guarantee (tracked in oc-task b770298b).

### Reranker Score Distribution Issue (Confirmed)

The `llama-nemotron-rerank-1b-v2` model shows tight score compression:
- 58% of top results score ≥ 0.999
- Score range for top results: 0.83–1.0
- Discrimination is weak at the high end — the model cannot meaningfully separate rank 1 from rank 5 when all score ≥ 0.99

This is why the Hard Gate (GATE-A) is effective: it skips the reranker when vector confidence is high (78% of queries), preventing the reranker from introducing noise where it has no meaningful signal.

---

## Production Recommendations

| Use Case | Recommended Config | Rationale |
|----------|-------------------|-----------|
| **General purpose** | **Config U** (α=0.4 logit blend) | Best avg NDCG, stable across all domains |
| **Latency-sensitive** | **GATE-A** | 626ms p50, only -0.007 avg vs Config U |
| **Scientific/medical** | **Config A** (Vector-Only) | NFCorpus winner; reranking hurts this domain |
| **Financial/factual** | **Config U or N** | Strong FiQA performance (0.5526) |
| **Current system default** | GATE-A | Confirmed in code: `hardGate=true, spread=[0.02,0.08]` |

---

## Config Reference

| ID | Description |
|----|-------------|
| A | Vector-Only (baseline) |
| B | FTS-Only |
| C | Hybrid (RRF vector+FTS) |
| D | Hybrid + Reranker |
| E | Hybrid + MMR |
| F | Hybrid + HyDE |
| G | Full Pipeline (C+D+E) |
| H | Vector → Reranker (no RRF) |
| I | Adaptive Hybrid (overlap-aware RRF) |
| J | Reranker-as-Fusioner (union then rerank) |
| K | Vector → Adaptive MMR |
| L | Full Adaptive (overlap RRF → Reranker → Adaptive MMR) |
| M | Vector → Blended Reranker (α=0.3) |
| N | Vector → Blended Reranker (α=0.5) |
| O | Conditional Rerank (skip if confident, α=0.3) |
| P | Full 9A+9B (Adaptive RRF → Conditional Blended Reranker → MMR) |
| Q | Vector → Logit-Recovered Blend (α=0.5) |
| R | Vector → Pure Logit Reranker (α=0) |
| S | Vector → Logit Blend + Conditional (α=0.3) |
| T | Vector → Logit Blend (α=0.3) |
| U | Vector → Logit Blend (α=0.4) ⭐ Best Overall |
| V | Vector → Logit Blend (α=0.6) |
| W | Vector → Logit Blend (α=0.7) |
| X | Vector → Logit Blend (α=0.8) |
| MQ-A | Multi-Query (3 expansions) → Vector-Only |
| MQ-B | Multi-Query (3 expansions) → Logit Blend (α=0.4) |
| MQ-C | Multi-Query (3 expansions) → Logit Blend (α=0.5) |
| MQ-D | Multi-Query (3 expansions) → Conditional Logit Blend |
| RRF-A | Vector ⊕ Reranker via RRF (k=60, equal weight) |
| RRF-B | Vector ⊕ Reranker via RRF (k=60, vector weight=1.5) |
| RRF-C | Vector ⊕ Reranker via RRF (k=60, reranker weight=1.5) |
| RRF-D | Vector ⊕ Reranker via RRF (k=20, equal weight) |
| GATE-A | RRF + Hard Gate (skip if spread > 0.08 or < 0.02) ⭐ Best Latency |
| GATE-B | RRF + Soft Gate (dynamic vector weight from spread) |
| GATE-C | RRF + Soft Gate (threshold=0.12, wider confidence band) |
| GATE-D | RRF (vector=1.5) + Soft Gate (best of Fix 1 + Fix 2) |

---

*Raw logs: `evaluation/results/full-run-{scifact,fiqa,nfcorpus}-20260401-172216.log`*  
*JSON results: `evaluation/results/beir-{dataset}-{config}-*.json`*
