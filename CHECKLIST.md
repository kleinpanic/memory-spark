# memory-spark v1.0 — Implementation Checklist

**Last Updated:** 2026-03-27 00:05 EDT  
**Session Start:** 2026-03-26 ~21:30 EDT  
**Commits This Session:** 9 (8c46531 → d7f4a50)  
**Test Status:** 172/172 (159 unit + 13 pool routing)  
**Build Status:** 0 type errors, 0 lint errors  

---

## Session Summary

This session started with a v0.3.0 codebase that had 14 known critical bugs, unreliable benchmarks, no data isolation between agents, and an FTS workaround. We've done:

1. Fixed all 14 bugs (Codex-audited)
2. Upgraded LanceDB 0.14.1 → 0.27.1 (FTS+WHERE panic fixed)
3. Designed and implemented pool-based data architecture (8 pools)
4. Propagated pools through recall, capture, ingest, and manager pipelines
5. Added mistakes tools (search + store)
6. Created architecture doc, changelog, and this checklist
7. Removed the FTS 3x overfetch workaround (proper WHERE implementation)
8. Cleaned up type safety (removed unnecessary casts)

---

## Phase 0: Bug Fixes ✅ COMPLETE (commits 8c46531, 9770281)

- [x] Bug #1: FTS scoring — `_score` vs `_distance` detection
- [x] Bug #2: NaN poison — temporal decay guard
- [x] Bug #3: Concurrent search — sequential Vector→FTS
- [x] Bug #4: Misleading contextual embeddings comment
- [x] Bug #5: Chunk size threading for reference docs
- [x] Bug #6: Precision@k formula (÷k not ÷ranked.length)
- [x] Bug #7: MAP@k formula (BEIR standard)
- [x] Bug #8: Basename fallback removed (inflated metrics)
- [x] Bug #10: Metadata key correction
- [x] Bug #11: Tier 3 eval honestly marked TODO
- [x] Bug #14: Capture header accuracy
- [x] Codex audit #1 — no critical regressions
- [x] BM25 sigmoid midpoint extracted as configurable constant
- [x] Stale sqlite-vec config removed from type

## Phase 1: Pool Architecture ✅ MOSTLY COMPLETE (commits 85e173a, d7f4a50)

### Storage Layer ✅
- [x] LanceDB 0.14.1 → 0.27.1 (FTS+WHERE works now)
- [x] FTS+WHERE validated with 5-test suite (all compound queries pass)
- [x] `pool` column added to MemoryChunk type
- [x] `pool` in seed schema, upsert normalization, row deserialization
- [x] `src/storage/pool.ts` — single source of truth for pool routing
- [x] `resolvePool()` routes by content_type + path + explicit override
- [x] 8 pool values: agent_memory, agent_tools, agent_mistakes, shared_knowledge, shared_mistakes, shared_rules, reference_library, reference_code
- [x] `isAutoInjectPool()`, `isAlwaysInjectPool()` predicates
- [x] Pool filtering in vector search (WHERE pool = ...)
- [x] Pool filtering in FTS search (WHERE pool = ...) — native, no workaround
- [x] `SearchOptions.pool` and `SearchOptions.pools` fields
- [x] Schema evolution check includes `pool`
- [x] FTS 3x overfetch workaround REMOVED

### Pipeline Propagation ✅
- [x] `recall.ts` — pool-aware queries across 5 pool groups
  - [x] Agent memory + tools: `pool IN ('agent_memory', 'agent_tools')`
  - [x] Agent mistakes: `pool = 'agent_mistakes'` filtered by agent_id
  - [x] Shared mistakes: `pool = 'shared_mistakes'` (cross-agent)
  - [x] Shared knowledge: `pool = 'shared_knowledge'` (cross-agent)
  - [x] Shared rules: `pool = 'shared_rules'` (always injected, no relevance gate)
  - [x] Reference pools NEVER auto-injected
- [x] `capture.ts` — captured memories get `pool = 'agent_memory'`
- [x] `pipeline.ts` — `resolvePool()` called on every ingested chunk
- [x] `manager.ts` — search() queries agent + shared pools (excludes references)

### Per-Agent Mistakes ✅ DESIGNED, PARTIALLY IMPLEMENTED
- [x] `agent_mistakes` pool value defined
- [x] Routing: mistakes → `agent_mistakes` by default (not shared)
- [x] Recall: queries agent_mistakes with agent_id filter
- [ ] **`memory_mistakes_store` tool: option to promote to shared_mistakes**
- [ ] **Migration: reclassify existing mistake chunks into agent_mistakes pool**

### Global Rules/Preferences ✅ DESIGNED, PARTIALLY IMPLEMENTED
- [x] `shared_rules` pool value defined
- [x] Recall: shared_rules always injected (minScore=0)
- [ ] **`memory_rules_store` tool — store cross-agent rules/preferences**
- [ ] **`memory_rules_search` tool — search rules**
- [ ] **Rules pinning mechanism (always present regardless of query)**
- [ ] **Agent-specific rules: pool='agent_rules' option**

### Tools ✅ MOSTLY DONE
- [x] `memory_search` — agent memory + shared knowledge
- [x] `memory_get` — read file by path
- [x] `memory_store` — store a new memory
- [x] `memory_forget` — delete by query
- [x] `memory_forget_by_path` — delete by file path
- [x] `memory_reference_search` — search docs/PDFs (NOT auto-injected)
- [x] `memory_index_status` — health check
- [x] `memory_inspect` — debug recall pipeline
- [x] `memory_reindex` — trigger re-scan
- [x] `memory_mistakes_search` — search cross-agent mistakes
- [x] `memory_mistakes_store` — log a mistake with metadata
- [ ] **`memory_rules_store` — store global rules/preferences**
- [ ] **`memory_rules_search` — search rules**

### Retained Infrastructure (not wasted work)
- [x] `TableManager` (`table-manager.ts`) — retained for future multi-table option
- [x] `MultiTableBackend` (`multi-table-backend.ts`) — retained as alternative backend
- [x] Both are importable but not the active path (single-table + pool is primary)

## Phase 2: Reference Library — NOT STARTED

- [ ] PDF text extraction pipeline (pdf-parse or pdfjs)
- [ ] Markdown/HTML reference ingestion
- [ ] Code reference ingestion with language detection
- [ ] `memory_reference_search` wired to `pool = 'reference_library'` filter
- [ ] Version tracking for reference documents
- [ ] Reference re-indexing on boot (configurable)
- [ ] Reference paths from config.reference.paths
- [ ] Reference ingestion should use Spark OCR for scanned PDFs

## Phase 3: LLM Classification — NOT STARTED

- [ ] Integrate Nemotron zero-shot classifier for content routing
- [ ] LLM+heuristic hybrid classification (heuristic fast-path, LLM for ambiguous)
- [ ] Classification → pool routing at ingest time
- [ ] Quality scoring refinement with LLM feedback
- [ ] Use Spark NER endpoint for entity extraction improvement

## Phase 4: Evaluation & Benchmarking — MOSTLY NOT DONE

### Done ✅
- [x] Metric formulas fixed (Precision@k, MAP@k, NDCG, MRR)
- [x] Basename fallback removed
- [x] tsconfig includes evaluation/ for type checking

### Not Done ❌
- [ ] **Benchmark uses pool-aware queries** (currently ignores pools)
- [ ] **Golden dataset per pool** — agent memory, shared knowledge, mistakes, reference queries
- [ ] **BM25 sigmoid calibration** from actual corpus score distribution
- [ ] **Per-pool NDCG/MRR breakdown**
- [ ] **Docker benchmark reproducibility** — fixed seed data, deterministic
- [ ] **Ablation study** — pool isolation vs no-pool baseline
- [ ] **Cross-encoder reranker evaluation** per pool
- [ ] **Tier 3: E2E agent quality** — A/B with memory-spark enabled/disabled
- [ ] **EVALUATION.md** documentation

## Phase 5: Test Suite — PARTIALLY DONE

### Done ✅
- [x] 159 unit tests (pre-pool, still valid)
- [x] 13 pool routing tests (resolvePool, predicates, disjointness)
- [x] 17 table-manager tests
- [x] 5 FTS+WHERE validation tests
- [x] 3 regression tests (Precision@k, MAP@k, temporal decay)

### Not Done ❌
- [ ] **Pool filtering integration tests** — vector + FTS with WHERE pool
- [ ] **Recall pipeline tests** — multi-pool merge, weighting, exclusion, rules always-inject
- [ ] **Capture tests** — pool assignment on auto-capture
- [ ] **Ingest tests** — pool assignment during file ingestion
- [ ] **Mistakes tools tests** — search, store, per-agent vs shared
- [ ] **Rules tools tests** — store, search, always-inject
- [ ] **Reference exclusion tests** — verify reference pool NEVER auto-injected
- [ ] **Integration tests** — full ingest→recall cycle with pool routing
- [ ] **Docker harness tests** — real OpenClaw gateway + Spark backends
- [ ] Target: 250+ unit, 50+ integration

## Phase 6: Documentation — PARTIALLY DONE

### Done ✅
- [x] ARCHITECTURE.md — v1.0 design document
- [x] CHANGELOG.md — session history + pre-session state
- [x] CHECKLIST.md — this file

### Not Done ❌
- [ ] **CONFIGURATION.md** — all config options with defaults, ranges, examples
- [ ] **EVALUATION.md** — how to run and interpret benchmarks
- [ ] **README.md** — updated for v1.0 (pools, tools, setup)
- [ ] **Plugin SDK alignment docs** — lifecycle hooks, tool registration
- [ ] **API reference** for all 11+ tools

## Phase 7: Production Readiness — NOT STARTED

- [ ] Full reindex with pool assignment on all existing chunks
- [ ] Config migration tool (v0.3 → v1.0)
- [ ] Performance profiling (embed latency, search latency, rerank latency per pool)
- [ ] Memory pressure monitoring (Spark GH200 RAM ~90%)
- [ ] Graceful degradation when Spark is down
- [ ] Plugin SDK lifecycle hooks verified against source
- [ ] Docker harness: full E2E test against real OpenClaw + Spark

---

## Commit Log (This Session)

| # | Hash | Description |
|---|------|-------------|
| 1 | `8c46531` | fix: 14 critical bugs — FTS scoring, NaN poison, eval metrics, type safety |
| 2 | `9770281` | fix: Codex audit findings — stale config, BM25 calibration |
| 3 | `de3b8c6` | docs: v1.0 architecture plan (ARCHITECTURE.md) |
| 4 | `9cf412f` | feat: TableManager — multi-table LanceDB foundation (17 tests) |
| 5 | `ed66c6d` | feat: MultiTableBackend — routing logic |
| 6 | `36fdd01` | feat: plugin integration + memory_mistakes tools (11 tools) |
| 7 | `85e173a` | feat: single-table + pool column + LanceDB 0.27.1 upgrade |
| 8 | `ac2f209` | docs: CHANGELOG.md + CHECKLIST.md |
| 9 | `d7f4a50` | feat: pool propagation through all pipelines + pool.ts + 13 tests |

---

## Quick Status

| Area | Status | Notes |
|------|--------|-------|
| Bug fixes | ✅ Done | Codex audited, 14/14 fixed |
| Storage (pool column) | ✅ Done | LanceDB 0.27.1, native WHERE |
| Pool routing | ✅ Done | pool.ts, 8 pools, all predicates |
| Recall pipeline | ✅ Done | 5-group pool-aware queries |
| Capture pipeline | ✅ Done | pool assigned on capture |
| Ingest pipeline | ✅ Done | resolvePool() at ingest |
| Manager search | ✅ Done | pool-filtered queries |
| Mistakes (per-agent) | ⚠️ 80% | Pool + recall done, tool needs share option |
| Rules/preferences | ⚠️ 40% | Pool + recall done, tools not built |
| Reference library | ❌ Not started | PDF/doc ingestion pending |
| LLM classification | ❌ Not started | Nemotron zero-shot integration |
| Benchmarks | ❌ Stale | Need pool-aware golden dataset |
| Test coverage | ⚠️ 60% | Pool unit tests done, integration pending |
| Documentation | ⚠️ 50% | Architecture + changelog done, rest pending |
| Production deploy | ❌ Not started | Reindex + migration pending |
