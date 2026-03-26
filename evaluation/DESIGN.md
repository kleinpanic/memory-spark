# Evaluation Framework Design

## Overview

This evaluation framework measures retrieval quality of memory-spark using
established Information Retrieval (IR) metrics adapted for the agent memory
domain. We follow BEIR (Thakur et al., 2021) methodology where possible,
with domain-specific adaptations for agent workspace retrieval.

## Metrics

### Primary Metrics (per-query, then macro-averaged)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **NDCG@10** | DCG@10 / IDCG@10 | Position-aware relevance (rewards correct results ranked higher) |
| **MRR** | 1/rank_of_first_relevant | How quickly we find the first relevant result |
| **Recall@K** | \|relevant ∩ retrieved_K\| / \|relevant\| | Coverage at cutoff K (K=1,3,5,10) |
| **MAP@10** | Mean of precision@k for each relevant doc | Overall precision-recall balance |
| **Precision@5** | \|relevant ∩ retrieved_5\| / 5 | Fraction of top-5 that are relevant |

### Secondary Metrics

| Metric | Purpose |
|--------|---------|
| **Latency (p50, p95, p99)** | End-to-end retrieval speed |
| **Freshness Score** | Mean temporal decay score of retrieved results |
| **Category Accuracy** | Per-category breakdown (safety, infra, workflow, etc.) |

## Ground Truth Design

### Categories (based on real agent memory use cases)

1. **Safety & Policy** — Agent safety rules, restart protocols, banned commands
2. **Infrastructure** — IPs, ports, tunnel configs, machine topology
3. **Workflow & Process** — Task tracking, delegation, approval flows
4. **Historical Knowledge** — Past incidents, postmortems, migration logs
5. **Configuration** — Model settings, provider configs, cron schedules
6. **Reference Documentation** — API docs, framework guides, user manuals
7. **Cross-Agent Knowledge** — Information spanning multiple agent contexts
8. **Mistake Prevention** — Known errors, anti-patterns, learned lessons

### Relevance Grading (BEIR-compatible)

- **3** = Perfectly relevant (directly answers the query)
- **2** = Highly relevant (contains key information)
- **1** = Marginally relevant (related but incomplete)
- **0** = Not relevant

### Query Set Size Target

- 60 queries minimum (≥7 per category)
- Each query has 1-5 relevant documents with graded relevance
- Queries range from keyword-like to natural language questions

## Ablation Study Design

### Components Under Test

| Component | Toggle Method | Expected Impact |
|-----------|---------------|-----------------|
| Hybrid Search (FTS+Vector) | Vector-only fallback | -5-15% NDCG |
| Cross-Encoder Reranking | Skip rerank step | -10-20% NDCG |
| Temporal Decay | Set all decay=1.0 | -2-5% on recent queries |
| Quality Filter | Disable quality scoring | -5-10% (more noise) |
| Contextual Prefixes | Index raw text only | -8-15% NDCG |
| Mistake Weighting | Remove 1.6x boost | -10-20% on safety queries |
| IVF_PQ Indexing | Brute-force kNN | +1-2% quality, 5-10x slower |

### Ablation Protocol

1. Run full pipeline → record baseline metrics
2. For each component: disable → re-run same query set → record metrics
3. Compute delta from baseline
4. Run 5 iterations per configuration for statistical significance
5. Report mean ± std with 95% CI

## Comparison Baselines

1. **No Memory** — Agent responds without any retrieved context (0% expected)
2. **Vanilla Vector Search** — Cosine similarity only, no rerank/decay/hybrid
3. **memory-spark (full)** — Complete pipeline with all components

## Output Format

All results are saved as JSON in `evaluation/results/` for chart generation:

```json
{
  "run_id": "2026-03-26T04:00:00Z",
  "config": { "rerank": true, "decay": true, ... },
  "metrics": {
    "ndcg_at_10": { "mean": 0.82, "std": 0.03, "ci_95": [0.79, 0.85] },
    "mrr": { "mean": 0.91, "std": 0.02, "ci_95": [0.89, 0.93] },
    ...
  },
  "per_category": { ... },
  "per_query": [ ... ]
}
```

## References

- Thakur et al. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models."
- Muennighoff et al. (2023). "MTEB: Massive Text Embedding Benchmark."
- Anthropic (2024). "Contextual Retrieval."
- Gao et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval."
- Asai et al. (2024). "Self-RAG: Learning to Retrieve, Generate, and Critique."
