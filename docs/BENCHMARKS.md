# Benchmarks

## Methodology

memory-spark is evaluated using the [BEIR (Benchmarking Information Retrieval)](https://github.com/beir-cellar/beir) benchmark suite. BEIR provides standardized datasets with queries, documents, and relevance judgments across multiple domains.

### Datasets

| Dataset | Domain | Queries | Docs | Avg Relevant/Query |
|---------|--------|---------|------|--------------------|
| **SciFact** | Scientific claims | 300 | 5,183 | 1.1 |
| **FiQA** | Financial Q&A | 648 | 57,638 | 2.6 |
| **NFCorpus** | Medical/nutrition | 323 | 3,633 | 38.2 |

### Metrics

- **NDCG@10**: Normalized Discounted Cumulative Gain at 10 — the primary metric. Measures ranking quality, weighting higher-ranked relevant documents more.
- **MRR**: Mean Reciprocal Rank — how early the first relevant document appears.
- **Recall@10**: Fraction of all relevant documents found in the top 10.
- **MAP@10**: Mean Average Precision at 10 — precision at each relevant document rank.
- **Precision@10**: Fraction of top 10 results that are relevant.
- **Latency**: Mean per-query latency including embedding, search, and reranking.

### Pipeline

Each query runs through:
1. **Embed** — query text → 4096-dim vector via `nvidia/llama-embed-nemotron-8b` with instruction-aware prefixing
2. **Vector Search** — LanceDB ANN search (IVF_PQ index)
3. **FTS Search** — LanceDB full-text search (tantivy)
4. **Hybrid Merge** — Reciprocal Rank Fusion (RRF) combining vector and FTS ranks
5. **Reranker Gate** — Dynamic gate decides whether cross-encoder reranking is worthwhile
6. **Cross-Encoder Rerank** — `nvidia/llama-nemotron-rerank-1b-v2` rescores candidates
7. **RRF Blend** — Fuses original vector ranks with reranker ranks
8. **MMR** — Maximal Marginal Relevance for diversity

## Results: SciFact (300 queries)

### Phase 12 Configurations

| Config | Description | NDCG@10 | Δ Baseline | MRR | Recall@10 | Latency |
|--------|-------------|---------|------------|-----|-----------|---------|
| **A** | Vector-Only (baseline) | 0.7709 | — | 0.7365 | 0.9037 | 528ms |
| **RRF-A** | RRF k=60, equal weight | 0.7797 | +0.88% | 0.7511 | 0.8924 | 1540ms |
| **RRF-B** | RRF k=60, vec=1.5× | 0.7788 | +0.79% | 0.7505 | 0.8924 | 1446ms |
| **RRF-C** | RRF k=60, rerank=1.5× | 0.7770 | +0.61% | 0.7476 | 0.8924 | 1548ms |
| **RRF-D** | RRF k=20, equal weight | 0.7798 | +0.90% | 0.7514 | 0.8924 | 1452ms |
| **GATE-A** ★ | Hard gate (0.08/0.02) | **0.7802** | **+0.94%** | 0.7455 | **0.9137** | **732ms** |
| **GATE-B** | Soft gate | 0.7802 | +0.93% | 0.7518 | 0.8924 | 2297ms |
| **GATE-C** | Soft gate, wide (0.15) | 0.7782 | +0.74% | 0.7493 | 0.8924 | 2131ms |
| **GATE-D** | Soft+RRF k=20, vec=1.5× | **0.7803** | **+0.94%** | **0.7525** | 0.8924 | 1413ms |

### Key Findings

1. **GATE-A is the production winner**: Best recall (0.9137, +1.1%), best latency (732ms), near-best NDCG (0.7802). The hard gate skipped 236/300 queries (78%), only firing the reranker on 64 queries where vector spread was in the productive [0.02, 0.08] range.

2. **Every Phase 12 config beats the baseline**: The reranker finally provides a consistent positive lift after fixing the Arrow vector bug (Phase 7), implementing RRF (Phase 12 Fix 1), and adding the dynamic gate (Phase 12 Fix 2).

3. **RRF is scale-invariant**: The old score-based hybrid merging mixed BM25 scores (5–20+) with cosine similarities (0.2–0.6), causing "rank washout." RRF uses rank positions only — no normalization needed.

4. **Hard gate preserves recall**: All RRF and soft-gate configs dropped recall from 0.9037 → 0.8924 because the reranker narrows the result set. GATE-A is the *only* config that improved recall because it skips reranking on most queries, preserving the full vector result set.

### Gate Skip Statistics (GATE-A)

| Skip Reason | Count | % |
|-------------|-------|---|
| Vector confident (spread > 0.08) | 234 | 78% |
| Tied set (spread < 0.02) | 2 | 0.7% |
| Reranker fired (spread 0.02–0.08) | 64 | 21.3% |

## Evolution: Phase-by-Phase

| Phase | What Changed | NDCG Impact |
|-------|-------------|-------------|
| Phase 7 | Arrow Vector `.toArray()` fix | MMR no longer a no-op |
| Phase 8 | Adaptive pipeline, overlap-aware merge | Stable baseline |
| Phase 9-10 | Score interpolation, unified reranker | Better blending |
| Phase 11B | Multi-query expansion | +recall on ambiguous queries |
| Phase 12 | RRF + dynamic reranker gate | +0.94% NDCG, +1.1% recall |

## Bug Impact Analysis

| Bug | Impact | Fix |
|-----|--------|-----|
| Arrow Vector type mismatch | MMR returned `NaN` for all pairs — no-op | `.toArray()` conversion |
| BM25 sigmoid saturation | FTS scores all ≈1.0, drowning semantic signal | RRF (rank-based, no scores) |
| Reranker score compression | 0.83–1.0 range, arbitrary reshuffling | Dynamic gate bypasses |
| HyDE averaging vs. replacement | Diluted semantic focus | Full replacement |
| Precision@k denominator | Score inflation on small result sets | Fixed to use k |

## Running Benchmarks

```bash
# Single dataset, specific configs
npx tsx scripts/run-beir-bench.ts --dataset scifact --config A,GATE-A

# Full suite (all configs × all datasets, ~7-8 hours)
bash scripts/run-full-benchmark.sh

# Results are written to evaluation/results/
```
