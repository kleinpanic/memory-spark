# Changelog

## [Unreleased] — Phase 11B: Multi-Query Expansion (2026-04-01)

### Added
- **Multi-Query Expansion module** (`src/query/expander.ts`)
  - Generates 3 LLM reformulations per query via Nemotron-Super
  - Quality gates: length filtering, dedup, meta-commentary rejection, format stripping
  - Graceful degradation: any failure falls back to original query only
  - Configurable: numReformulations, temperature, timeout, model, auth
- **33 unit tests** (`tests/expander.test.ts`)
  - parseReformulations: 13 tests (parsing, filtering, dedup, edge cases)
  - buildExpansionPrompt: 3 tests
  - expandQuery: 17 tests (success, failures, auth, timeouts, garbage handling)
- **4 new benchmark configs** (MQ-A through MQ-D)
  - MQ-A: Multi-Query → Vector-Only
  - MQ-B: Multi-Query → Logit Blend α=0.4
  - MQ-C: Multi-Query → Logit Blend α=0.5
  - MQ-D: Multi-Query → Conditional Logit Blend α=0.4
- **Multi-vector recall** in production pipeline (`src/auto/recall.ts`)
  - poolSearch supports multiple query vectors with parallel search + union
  - Deduplicates by chunk ID, keeps highest score per document
- **Diagnostic script** (`scripts/diag-multi-query.ts`)
  - Verified +14.7 new docs/query (37% candidate pool increase)
- **queryExpansion config** in `AutoRecallConfig` (`src/config.ts`)
- **STATE.md** — comprehensive project state documentation
- **CHANGELOG.md** — this file

### Changed
- `RetrievalConfig` interface extended with `useMultiQuery` and `multiQueryN`
- `QueryTelemetry` interface extended with `multiQuery` stage data
- `runRetrieval()` now accepts optional `QueryExpansionConfig` parameter

## Phase 11A: Alpha Sweep (2026-04-01)

### Added
- Configs U (α=0.4), V (α=0.6), W (α=0.7), X (α=0.8)
- `docs/PLAN-phase11-next-improvements.md`

### Results
- **New best: Config U (α=0.4) = 0.7889 NDCG@10** (+2.3% over old baseline)
- Sweet spot at α ∈ [0.4, 0.6] with remarkably flat plateau

## Phase 10B: Unified Reranker Pipeline (2026-04-01)

### Fixed
- **Critical:** Blended reranker configs bypassed `normalizeQueryForReranker()` by using direct `fetch()` — raw declarative claims went to a Q&A-trained cross-encoder, causing score compression
- Unified all reranker paths through single `reranker.rerank()` with `alphaOverride`

### Results
- **First time reranker beat baseline:** Config Q (α=0.5) = 0.7863 (+2.0%)

## Phase 10A: Logit-Space Calibration (2026-04-01)

### Added
- `recoverLogit()` — maps sigmoid-compressed scores back to logit space
- `normalizeQueryForReranker()` — declarative → interrogative conversion
- Logit-space blending in `blendScores()`
- Spread guard updated for logit-space spread

## Phase 9: Score Interpolation & Conditional Routing (2026-04-01)

### Added
- `scoreBlendAlpha` — interpolates vector + reranker scores
- Conditional routing — skips reranker when vector top-1 is confident (spread > threshold)
- Configs M-P (blending experiments)

## Phase 8: Adaptive Remediation (2026-04-01)

### Added
- Adaptive RRF (overlap-aware dynamic weighting)
- Union Reranking / Reranker-as-Fusioner
- Adaptive MMR (score-spread-aware λ)
- Configs H-L

## Phase 7: Forensic Diagnostics & Pipeline Fixes (2026-03-31)

### Fixed
- **Critical:** Arrow Vector type mismatch — `.toArray()` fix in `lancedb.ts`
- MMR was a complete no-op (NaN cosine similarity from Arrow Vectors)
- Weighted RRF to combat rank-washout
- Reranker spread guard (bypass on tight score distributions)
- HyDE hardening (timeouts + retry logic)

### Added
- Feature branch `fix/phase7-pipeline-bugs`
- `docs/PLAN-phase7-remediation.md`
- Diagnostic scripts: `diag-arrow.ts`, `diag-full.ts`, `diag-mmr.ts`, `diag-rrf.ts`, `diag-vectors.ts`

## Phase 1-6: Foundation (2026-03-30 – 2026-03-31)

### Added
- BEIR benchmark pipeline (`scripts/run-beir-bench.ts`)
- Configs A-G (Vector, FTS, Hybrid, Reranker, MMR, HyDE, Full Pipeline)
- Metrics: NDCG@k, MAP@k, Recall@k, Precision@k, MRR@k
- Instruction-aware asymmetric embeddings
- Reciprocal Rank Fusion (RRF) replacing score-based hybrid merge
- Pipeline reordering: Retrieve → Rerank → MMR

### Fixed
- BM25 sigmoid saturation (hardcoded midpoint 3.0)
- MAP@k and Precision@k denominator bugs
- SciFact zero-score issues via dataset-scoped filtering
- HyDE quality gates (reject LLM refusals/thinking traces)

### Results
- **Baseline: Config A (Vector-Only) = 0.7709 NDCG@10**
