# memory-spark v1.0 — Implementation Checklist

**Last Updated:** 2026-03-27 ~23:15 EDT
**Status:** Active development — pool infrastructure built, propagation in progress

---

## Phase 0: Bug Fixes ✅ COMPLETE
- [x] Bug #1: FTS scoring — `_score` vs `_distance` detection, BM25 sigmoid normalization
- [x] Bug #2: NaN poison — temporal decay guard for invalid timestamps
- [x] Bug #3: Concurrent search — sequential Vector→FTS to prevent corruption
- [x] Bug #4: Misleading contextual embeddings comment → honest TODO
- [x] Bug #5: Chunk size threading — `reference.chunkSize` used in ingest pipeline
- [x] Bug #6: Precision@k — divide by `k`, not `ranked.length`
- [x] Bug #7: MAP@k — `min(totalRelevant, k)` denominator per BEIR
- [x] Bug #8: Basename fallback removed — was inflating Recall/NDCG
- [x] Bug #10: Metadata key — `_meta` not `metadata` in golden dataset
- [x] Bug #11: Tier 3 eval — honestly marked NOT YET IMPLEMENTED
- [x] Bug #14: Capture header — reflects actual assistant+user behavior
- [x] Codex audit #1 passed (no critical regressions)
- [x] 159 unit tests + 3 regression tests passing

## Phase 1: Storage & Pool Architecture — PARTIALLY COMPLETE
### Done ✅
- [x] LanceDB upgraded 0.14.1 → 0.27.1
- [x] FTS+WHERE confirmed working (5-test validation suite)
- [x] FTS 3x overfetch workaround REMOVED → proper WHERE clauses
- [x] `pool` column added to MemoryChunk type
- [x] `pool` column in seed schema, upsert normalization, row deserialization
- [x] `resolvePool()` function — auto-routes by content_type + path
- [x] Pool filtering in vector search (WHERE pool = ...)
- [x] Pool filtering in FTS search (WHERE pool = ...)
- [x] `SearchOptions.pool` and `SearchOptions.pools` fields
- [x] Schema evolution check includes `pool`
- [x] TableManager + MultiTableBackend (retained as options)
- [x] `memory_mistakes_search` tool
- [x] `memory_mistakes_store` tool
- [x] ARCHITECTURE.md written

### NOT DONE ❌ — Pool propagation through the system
- [ ] **`recall.ts`** — needs pool-aware queries:
  - [ ] Agent memory query: `pool IN ('agent_memory', 'agent_tools') AND agent_id = X`
  - [ ] Shared knowledge query: `pool = 'shared_knowledge'`
  - [ ] Mistakes query: `pool IN ('shared_mistakes', 'agent_mistakes') AND (agent_id = X OR pool = 'shared_mistakes')`
  - [ ] Rules query: `pool = 'shared_rules'` (always inject, no relevance gating)
  - [ ] Reference exclusion: NEVER auto-inject `reference_library` or `reference_code`
- [ ] **`capture.ts`** — captured memories need `pool = 'agent_memory'` set
- [ ] **`pipeline.ts`** — ingested files need `pool` set via `resolvePool()` at ingest time
- [ ] **`workspace.ts`** — workspace scanner needs to set pool on discovered files
- [ ] **`watcher.ts`** — file watcher needs to set pool on new/changed files
- [ ] **`manager.ts`** — search() needs pool-aware queries for tool calls
- [ ] **`mistakes.ts`** — enforceMistakesFiles needs pool awareness
- [ ] **`config.ts`** — pool-related config options (which pools to auto-recall, weights per pool)

### NOT DONE ❌ — Per-agent mistakes
- [ ] Add `agent_mistakes` pool value
- [ ] Mistakes stored with BOTH `pool = 'agent_mistakes'` (per-agent) AND optionally `pool = 'shared_mistakes'` (shared)
- [ ] Recall: query own `agent_mistakes` FIRST, then `shared_mistakes`
- [ ] `memory_mistakes_store` tool: option to share or keep private

### NOT DONE ❌ — Global rules/preferences
- [ ] Add `shared_rules` pool handling in recall (always inject top-N)
- [ ] `memory_rules_store` tool — store cross-agent rules
- [ ] `memory_rules_search` tool — search rules
- [ ] Rules pinning — some rules ALWAYS present regardless of relevance
- [ ] Agent-specific rules: `pool = 'agent_rules'` with `agent_id` filter

## Phase 2: Reference Library — NOT STARTED
- [ ] PDF text extraction pipeline (pdf-parse or pdfjs)
- [ ] Markdown/HTML reference ingestion
- [ ] Code reference ingestion with language detection
- [ ] `memory_reference_search` tool wired to `pool = 'reference_library'` filter
- [ ] Version tracking for reference documents
- [ ] Reference re-indexing on boot (configurable)
- [ ] Reference paths from config.reference.paths

## Phase 3: Classification Pipeline — NOT STARTED
- [ ] Integrate Nemotron zero-shot classifier for content routing
- [ ] LLM+heuristic hybrid classification
- [ ] Classification → pool routing
- [ ] Quality scoring refinement with LLM feedback

## Phase 4: Evaluation & Benchmarking — MOSTLY NOT DONE
### Done ✅
- [x] Metric formulas fixed (Precision@k, MAP@k, NDCG, MRR)
- [x] Basename fallback removed from benchmark
- [x] tsconfig includes evaluation/ for type checking
### NOT DONE ❌
- [ ] **Benchmark uses pool-aware queries** (currently ignores pools)
- [ ] **Golden dataset per pool** (agent memory, shared knowledge, mistakes, reference)
- [ ] **BM25 sigmoid calibration** from actual corpus score distribution
- [ ] **Per-pool NDCG/MRR** — break down metrics by pool type
- [ ] **Docker benchmark reproducibility** — fixed seed, deterministic index
- [ ] **Ablation study** — pool isolation vs no-pool baseline
- [ ] **Cross-encoder reranker** evaluation per pool
- [ ] **Tier 3: E2E agent quality** — A/B with memory-spark enabled/disabled
- [ ] **Evaluation documentation** (EVALUATION.md)

## Phase 5: Tests — MOSTLY NOT DONE
### Done ✅
- [x] 159 unit tests (pre-pool logic)
- [x] 17 table-manager tests
- [x] 5 FTS+WHERE validation tests
- [x] 3 regression tests (Precision@k, MAP@k, temporal decay)
### NOT DONE ❌
- [ ] **Pool routing unit tests** — resolvePool() coverage for all content types
- [ ] **Pool filtering tests** — vector search + FTS with pool WHERE
- [ ] **Recall pipeline tests** — multi-pool merge, weighting, exclusion
- [ ] **Capture tests** — pool assignment on auto-capture
- [ ] **Ingest tests** — pool assignment during file ingestion
- [ ] **Mistakes tools tests** — search, store, per-agent + shared
- [ ] **Rules tools tests** — store, search, always-inject
- [ ] **Reference exclusion tests** — verify reference pool never auto-injected
- [ ] **Integration tests** — full ingest→recall cycle with pool routing
- [ ] **Docker harness tests** — real OpenClaw gateway + Spark
- [ ] Target: 250+ unit tests, 50+ integration tests

## Phase 6: Documentation — PARTIALLY DONE
### Done ✅
- [x] ARCHITECTURE.md
- [x] CHANGELOG.md
- [x] CHECKLIST.md (this file)
### NOT DONE ❌
- [ ] CONFIGURATION.md — all config options documented with defaults
- [ ] EVALUATION.md — how to run and interpret benchmarks
- [ ] README.md — updated for v1.0
- [ ] Plugin SDK alignment docs
- [ ] API reference for all 11+ tools

## Phase 7: Production Readiness — NOT STARTED
- [ ] Full reindex with pool assignment on all 37k existing chunks
- [ ] Config migration tool (v0.3 → v1.0)
- [ ] Performance profiling (embed latency, search latency, rerank latency)
- [ ] Memory pressure monitoring (Spark RAM)
- [ ] Graceful degradation when Spark is down
- [ ] Plugin SDK lifecycle hooks verified

---

## Quick Status Summary

| Area | Status | Confidence |
|------|--------|-----------|
| Bug fixes | ✅ Done | High — Codex audited |
| Storage layer (pool column) | ✅ Done | High — tested |
| LanceDB upgrade | ✅ Done | High — FTS+WHERE proven |
| Recall pipeline | ❌ Broken | Must update for pools |
| Capture pipeline | ❌ Broken | Must set pool on captures |
| Ingest pipeline | ❌ Broken | Must set pool on ingest |
| Per-agent mistakes | ❌ Missing | Not implemented |
| Global rules | ❌ Missing | Not implemented |
| Benchmarks | ❌ Stale | Don't reflect pool changes |
| Test coverage | ⚠️ Partial | Old tests pass but don't test pools |
| Documentation | ⚠️ Partial | Architecture done, rest pending |
| Docker harness | ⚠️ Boots OK | Not tested with pool queries |
