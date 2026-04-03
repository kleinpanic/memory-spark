# Changelog

All notable changes to memory-spark are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [0.4.1] — 2026-04-03

### Fixed

- **P1-A: Score clamping killed reranker gate signal** (`recall.ts`): `applySourceWeighting` was calling `Math.min(1.0, score * weight)`, clamping boosted mistake/capture chunks to 1.0. With 2+ relevant chunks clamped, the gate's top-5 spread collapsed to 0.0, triggering the "tied set → skip reranker" branch and bypassing reranking exactly when it would help most. Fix: removed clamp, allow scores to exceed 1.0, added `normalizeScores()` post-weighting to rescale to [0,1] while preserving real spread.
- **P1-C: HyDE enabled by default with 15s timeout**: HyDE fired on every recall query, blocking for up to 15s when Spark LLM was unavailable. During benchmarks HyDE failed 100% of the time. Fix: default `enabled: false`, timeout reduced from 15000ms → 4000ms.
- **P2-B: Source dedup ran after weighting (wrong order)**: `deduplicateSources` compared inflated scores rather than raw cosine similarities, producing arbitrary dedup outcomes. Fix: dedup now runs on raw scores before `applySourceWeighting`.
- **P2-A: MAX_RERANK_CANDIDATES hardcoded at 30**: Silently dropped candidates ranked 31-40 after scoring. Fix: now uses `cfg.rerank.topN` (default raised from 20 → 40).
- **P3-C: Integration tests failed with ECONNREFUSED in dev**: Tests used `describe.skipIf(SKIP_INTEGRATION)` but ECONNREFUSED errors still surfaced without that env var. Fix: added runtime Spark reachability probe at test startup — suite skips cleanly if Spark is unreachable.

### Changed

- `normalizeScores()` added as exported utility in `recall.ts` (rescales result array top score to 1.0)
- `rerank.topN` default: 20 → 40
- `hyde.timeoutMs` default: 15000 → 4000
- Pipeline stage order: dedup → source weighting → temporal decay → normalize → reranker (was: source weighting → temporal decay → dedup)

### Documentation

- `docs/PLAN-phase13.md`: Full Phase 13 audit (11 issues, root causes, fix complexity, recommended order)
- `docs/BENCHMARKS.md`: SOTA comparison vs BM25/DPR/ANCE/TAS-B/Contriever; cross-dataset variance analysis
- `paper/memory-spark.tex`: §6.2 SOTA table, §6.3 cross-dataset variance, Contriever citation
- `README.md`: SOTA comparison table inline

## [0.4.0] — 2026-04-02

### Added

- **Dynamic Reranker Gate (GATE-A)**: Skips reranker on 78% of queries, +0.94% NDCG, 50% latency reduction
- **Reciprocal Rank Fusion (RRF)**: Scale-invariant hybrid merging (replaces sigmoid-based fusion)
- **Multi-Query Expansion**: 3-way LLM reformulation for recall improvement
- **5 new plugin tools**: `memory_recall_debug`, `memory_bulk_ingest`, `memory_temporal`, `memory_related`, `memory_gate_status`
- **10-page technical paper**: `paper/memory-spark.pdf`
- **36-configuration BEIR benchmark suite**: SciFact, FiQA, NFCorpus

### Fixed

- **FTS WHERE filter bypass (C1/Security)**: Cross-agent data leakage via immutable LanceDB query API
- **Init deadlock (C2)**: Plugin permanently bricked if Spark was down at cold start
- **MAP@k metric (C4)**: Reverted to BEIR-standard `totalRelevant` denominator
- **Arrow Vector NaN (M1)**: `getByIds()` now converts Arrow vectors to JS arrays
- **~80 silent test failures (M6)**: Converted `return bool` patterns to proper `expect()` assertions
- **Capture error logging (H3)**: Silent catch blocks now log warnings
- **Embed error handling (H2)**: 7 tool calls wrapped in try/catch with graceful errors

### Changed

- Port numbers corrected: embed 18091, rerank 18096
- Version bumped from 0.1.0 → 0.4.0
- NER tagging parallelized via `Promise.all()`
- Vitest coverage thresholds raised from 15% → 35%, `coverage.all: true`
- "Bug Archaeology" → "Failure Modes in Production RAG" in paper

### Removed

- Dead code: `evaluation/run.ts`, `evaluation/charts.ts` (978 lines)

### Security

- Sanitized PII from `evaluation/golden-dataset.json` (emails, IPs, names, locations)

---

## [Unreleased] — v1.0.0 Development

### 2026-03-27 — Session: Architecture Overhaul

#### Added

- **Pool column** on MemoryChunk — logical sections within single LanceDB table
  - `agent_memory` — per-agent workspace files, captures (auto-injected)
  - `agent_tools` — per-agent tool definitions (auto-injected for tool context)
  - `shared_knowledge` — cross-agent facts (auto-injected, 0.8x weight)
  - `shared_mistakes` — cross-agent mistakes (auto-injected, 1.6x boost)
  - `agent_mistakes` — per-agent mistakes (auto-injected, 1.6x boost)
  - `shared_rules` — global rules & preferences (always injected)
  - `reference_library` — PDFs, documentation (tool-call only)
  - `reference_code` — code examples (tool-call only)
- **`resolvePool()`** — auto-routes chunks to correct pool based on content_type and path
- **`memory_mistakes_search` tool** — search mistakes across all agents
- **`memory_mistakes_store` tool** — log mistakes with severity, root cause, fix, lessons; `shared: boolean` to promote to cross-agent
- **`memory_rules_store` tool** — store global rules/preferences with scope + category
- **`memory_rules_search` tool** — search stored rules by query
- ~~**TableManager** (`src/storage/table-manager.ts`)~~ — removed: pool column + native WHERE replaces multi-table
- ~~**MultiTableBackend** (`src/storage/multi-table-backend.ts`)~~ — removed: pool column + native WHERE replaces multi-table
- **FTS+WHERE validation test** (`tests/fts-where-test.ts`) — proves LanceDB 0.27 fix
- **3 regression tests** for Precision@k, MAP@k, temporal decay NaN guard
- **ARCHITECTURE.md** — comprehensive v1.0 design document

#### Fixed

- **FTS scoring** — detect `_score` vs `_distance`, normalize BM25 via sigmoid (Bug #1)
- **NaN poison** — guard temporal decay against invalid timestamps (Bug #2)
- **Concurrent search** — sequential Vector→FTS to prevent connection corruption (Bug #3)
- **Misleading contextual embeddings comment** — now honest TODO (Bug #4)
- **Chunk size threading** — reference.chunkSize now used for reference docs (Bug #5)
- **Precision@k** — divide by k, not ranked.length (Bug #6)
- **MAP@k** — use min(totalRelevant, k) denominator per BEIR standard (Bug #7)
- **Basename fallback** — removed, was inflating Recall/NDCG (Bug #8)
- **Metadata key** — `_meta` not `metadata` in golden dataset (Bug #10)
- **Tier 3 eval** — honestly marked as NOT YET IMPLEMENTED (Bug #11)
- **Capture header** — reflects actual assistant+user behavior (Bug #14)
- **Stale sqlite-vec config** — removed from StorageBackendId type
- **BM25 sigmoid** — extracted constant with calibration docs

#### Changed

- **LanceDB upgraded** 0.14.1 → 0.27.1 (FTS+WHERE panic fixed)
- **FTS search** — replaced 3x overfetch workaround with proper WHERE clauses
- **Pool-based filtering** — both vector search and FTS support pool/pools options
- **tsconfig** — evaluation/, tests/, tools/ now type-checked
- **LanceDBBackend** as sole backend in index.ts (MultiTableBackend removed)
- **mistakes_search** — uses pool-based filtering instead of `instanceof MultiTableBackend`

#### Removed

- `src/storage/multi-table-backend.ts` — ~450 LOC, pool column replaces physical table routing
- `src/storage/table-manager.ts` — ~350 LOC, table lifecycle management no longer needed
- `tests/table-manager.test.ts` — 17 tests removed (tested deleted code)
- `TableNamingConfig` — from config.ts (no longer applicable)
- `cfg.tables` — from MemorySparkConfig (no longer applicable)

#### Closed Investigations

- **Embedding model vision/multimodal** — investigated, determined not viable. Sticking with text-only `llama-embed-nemotron-8b` (4096-dim).

#### Removed

- `src/storage/sqlite-vec.ts` — unused migration adapter (193 LOC)
- `src/sync-rag.ts` — legacy sync code (97 LOC)
- FTS 3x overfetch workaround — replaced with proper WHERE implementation
- Stale `sqlite-vec` references in backend type + config schema

### Pre-Session State (v0.3.0)

- 37k chunks in single `memory_chunks` table
- No pool-based data separation
- FTS broken (3x overfetch workaround)
- 14 known critical bugs
- Benchmark metrics unreliable (wrong formulas)
- LanceDB 0.14.1

## [0.3.0] — 2026-03-26

- Production-grade repo restructure
- Comprehensive pipeline eval (37 tests)
- First real BEIR results (unreliable — bugs found post-eval)

## [0.2.0] — 2026-03-25

- Hybrid search (Vector + FTS)
- Contextual retrieval config
- Auto-capture with quality gates
