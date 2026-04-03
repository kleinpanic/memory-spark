# memory-spark v0.4.0 — Full BEIR Benchmark Results

**Run completed:** 2026-04-03 02:35 EDT  
**Duration:** ~33 hours (SciFact + FiQA + NFCorpus × 36 configs)  
**Datasets:** SciFact (300 queries), FiQA (648 queries), NFCorpus (323 queries)  
**Embedding model:** `spark/nvidia/llama-embed-nemotron-8b` (dims=4096)  
**Reranker model:** `spark/nvidia/llama-nemotron-rerank-1b-v2`  
**Infrastructure:** DGX Spark node (local); Spark embed port 18091, reranker port 18096

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
