# Changelog

All notable changes to memory-spark are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [Unreleased] — v1.0.0 Development

### 2026-03-27 — Session: Architecture Overhaul

#### Added
- **Pool column** on MemoryChunk — logical sections within single LanceDB table
  - `agent_memory` — per-agent workspace files, captures (auto-injected)
  - `agent_tools` — per-agent tool definitions (auto-injected for tool context)
  - `shared_knowledge` — cross-agent facts (auto-injected, 0.8x weight)
  - `shared_mistakes` — cross-agent mistakes (auto-injected, 1.6x boost)
  - `agent_mistakes` — per-agent mistakes (auto-injected, 1.6x boost) ← PENDING
  - `shared_rules` — global rules & preferences (always injected) ← PENDING
  - `reference_library` — PDFs, documentation (tool-call only)
  - `reference_code` — code examples (tool-call only)
- **`resolvePool()`** — auto-routes chunks to correct pool based on content_type and path
- **`memory_mistakes_search` tool** — search mistakes across all agents
- **`memory_mistakes_store` tool** — log mistakes with severity, root cause, fix, lessons
- **TableManager** (`src/storage/table-manager.ts`) — multi-table management foundation
- **MultiTableBackend** (`src/storage/multi-table-backend.ts`) — routing backend (retained for future use)
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
- **MultiTableBackend** as primary backend in index.ts (with legacy fallback)

#### Removed
- `src/storage/sqlite-vec.ts` — unused migration adapter (193 LOC)
- `src/sync-rag.ts` — legacy sync code (97 LOC)
- FTS 3x overfetch workaround — replaced with proper WHERE implementation

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
