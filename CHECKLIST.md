# memory-spark v1.0 — Implementation Checklist

**Last Updated:** 2026-03-27 01:00 EDT  
**Session Start:** 2026-03-26 ~21:30 EDT  
**Commits This Session:** 15+ (8c46531 → current)  
**Test Status:** 172+/172+ (unit + pool + integration)  
**Build Status:** 0 type errors, 0 lint errors  
**Tools:** 13 registered  
**Parent Task:** `a8ba0510` (20 subtasks: 5 done, 1 in progress, 14 queued)

---

## Session Summary

Started with v0.3.0 codebase: 14 known bugs, unreliable benchmarks, no data isolation, FTS workaround. Now:

1. Fixed all 14 bugs (Codex-audited)
2. Upgraded LanceDB 0.14.1 → 0.27.1 (FTS+WHERE fixed)
3. Pool architecture: 8 pools, single-table (per LanceDB best practices)
4. Pools propagated through recall, capture, ingest, manager
5. Rules tools + mistakes share option
6. Gap review: all plans, memories, oc-tasks audited — no critical gaps
7. ARCHITECTURE.md rewritten for actual implementation
8. CHANGELOG.md, CHECKLIST.md maintained throughout
9. **Rewired index.ts: LanceDBBackend is now the sole backend** (MultiTableBackend deleted)
10. **Removed MultiTableBackend + TableManager** (~850 LOC dead code)

---

## Phase 0: Bug Fixes ✅ COMPLETE (8c46531, 9770281)

- [x] Bugs #1–#14 all fixed (see CHANGELOG.md for full list)
- [x] Codex audit #1 — no critical regressions
- [x] 3 regression tests added (Precision@k, MAP@k, temporal decay NaN)

## Phase 1: Pool Architecture ✅ COMPLETE

### 1A: Storage Layer ✅
- [x] LanceDB 0.14.1 → 0.27.1
- [x] FTS+WHERE validated (5-test suite)
- [x] `pool` column on MemoryChunk
- [x] `src/storage/pool.ts` — resolvePool(), predicates, constants
- [x] Pool filtering in vector search + FTS (native WHERE)
- [x] Schema evolution includes `pool`
- [x] FTS 3x overfetch workaround REMOVED

### 1B: Ingest Pipeline ✅ (subtask `409de264` closed)
- [x] `resolvePool()` called on every chunk at ingest time
- [x] Capture pipeline sets `pool = 'agent_memory'`

### 1C: Search & Recall ✅ (subtask `59fcbf0e` closed)
- [x] Recall queries 5 pool groups sequentially
- [x] Manager search uses pool filtering
- [x] All pools relevance-gated (including rules — per Klein)

### 1D: FTS Fix ✅ (subtask `ee9c95db` closed)
- [x] LanceDB 0.27.1 fixes Arrow panic
- [x] Proper WHERE clauses replace workaround

### 1E: Architecture Docs ✅ (subtask `da3836af`)
- [x] ARCHITECTURE.md rewritten: single-table + pool design

### 1F: Backend Consolidation ✅ (this session)
- [x] **Deleted `multi-table-backend.ts`** (~450 LOC)
- [x] **Deleted `table-manager.ts`** (~350 LOC)
- [x] **Rewired `index.ts`** — `LanceDBBackend` is now the only backend
- [x] **Fixed `mistakes_search`** — uses pool-based filtering (not `instanceof MultiTableBackend`)
- [x] **Cleaned knip** — all unused exports from dead code removed
- [x] **Expanded tests** — backend consolidation coverage

### Per-Agent Mistakes ✅
- [x] `agent_mistakes` pool value + routing
- [x] Recall queries agent_mistakes with agent_id filter
- [x] `memory_mistakes_store` has `shared: boolean` param
- [x] `shared=false` → agent_mistakes, `shared=true` → shared_mistakes

### Global Rules/Preferences ✅
- [x] `shared_rules` pool + routing
- [x] `memory_rules_store` tool (scope: global/agent-specific, category: preference/constraint/workflow/safety)
- [x] `memory_rules_search` tool
- [x] Relevance-gated auto-inject (not always-inject)

### Tools (13 total) ✅
1. `memory_search` — agent memory + shared knowledge
2. `memory_get` — read file by path
3. `memory_store` — store a new memory
4. `memory_forget` — delete by query
5. `memory_forget_by_path` — delete by file path
6. `memory_reference_search` — search docs/PDFs (NOT auto-injected)
7. `memory_index_status` — health check
8. `memory_inspect` — debug recall pipeline
9. `memory_reindex` — trigger re-scan
10. `memory_mistakes_search` — search mistakes (per-agent + shared)
11. `memory_mistakes_store` — log mistake (shared: boolean option)
12. `memory_rules_store` — store rules/preferences/guidelines
13. `memory_rules_search` — search rules

---

## Phase 2: Reference Library — NOT STARTED

- [ ] **2A** (`76bfb098`): PDF/markdown/HTML/code ingestion pipeline
- [ ] **2B** (`23082484`): Wire `memory_reference_search` to `pool = 'reference_library'`
- [ ] Version tracking for reference documents
- [ ] Spark OCR for scanned PDFs

## Phase 3: Quality & Classification — PARTIALLY DONE

- [x] zh-CN/CJK detection in quality.ts (Unicode property escapes)
- [x] Path-based exclusions (zh-CN, zh-TW, ja, ko, fr, de in defaults)
- [x] Session noise patterns (penalty 1.0 → never indexed)
- [ ] **3A** (`2adf04ed`): LLM-assisted classification via Nemotron zero-shot
- [ ] **3B** (`82cb658a`): Capture quality gates + LLM validation
- [ ] **3C** (`aeb1813c`): Dedup duplicate doc versions (installed-v* vs git-latest)

## Phase 4: Evaluation & Benchmarking — MOSTLY NOT DONE

- [x] Metric formulas fixed (Precision@k, MAP@k, NDCG, MRR)
- [x] Basename fallback removed
- [ ] **4A** (`7f013d46`): BM25 sigmoid calibration from corpus score distribution
- [ ] **4B** (`5a2e0c8c`): Per-pool golden datasets
- [ ] **4C** (`8803c3ef`): Docker-based reproducible benchmarking
- [ ] Pool-aware benchmark queries
- [ ] Ablation study: pool isolation vs no-pool
- [ ] EVALUATION.md

## Phase 5: Tests & Docs — PARTIALLY DONE

- [x] 172+ unit tests (core + pool routing + backend consolidation)
- [x] 5 FTS+WHERE validation tests
- [ ] **5A** (`3706e434`): Full test suite overhaul (target: 250+ unit, 50+ integration)
- [ ] **5B** (`04a3b992`): Config schema expansion + CONFIGURATION.md
- [ ] **5C** (`852b462c`): Production migration tool + full reindex

## Phase 6: Plugin SDK — NOT STARTED

- [ ] **6** (`4e0d204f`): Lifecycle hooks verified against OpenClaw source
- [ ] before_prompt_build → prependContext (verified in source, working)
- [ ] agent_end → capture handler (verified in source, working)

## Phase 7: Production & Observability — NOT STARTED

- [ ] **7A** (`d3726fdd`): Context injection analytics (per-turn tracking)
- [ ] **7B** (`d373b58b`): Embed queue resilience (backpressure, circuit breaker)
- [ ] Full reindex with pool assignment on existing chunks
- [ ] Performance profiling per pool

---

## Phase 8: SOTA Research & Scientific Presentation — NOT STARTED

This phase maps our architecture against state-of-the-art RAG papers and produces
a professional-grade presentation (LaTeX paper + polished GitHub).

### 8A: Literature Review & Architecture Mapping
- [ ] Survey SOTA RAG papers: REALM (Guu et al. 2020), RAPTOR (Sarthi et al. 2024), ColBERT (Khattab & Zaharia 2020), RETRO (Borgeaud et al. 2022), Self-RAG (Asai et al. 2023)
- [ ] Map memory-spark architecture against each paper's contributions
- [ ] Identify which SOTA techniques we implement (hybrid search, cross-encoder reranking, temporal decay, pool isolation)
- [ ] Identify which techniques we should adopt (late interaction, hierarchical indexing, self-reflective retrieval)
- [ ] Document trade-offs specific to our use case (agent memory, not general QA)

### 8B: A/B/C/D Testing Framework
- [ ] Design controlled experiment matrix:
  - **A**: Vanilla vector search (no FTS, no reranking, no pools)
  - **B**: Hybrid search + reranking (no pools, no temporal decay)
  - **C**: Full pipeline (pools + hybrid + rerank + temporal decay + MMR)
  - **D**: Full pipeline + SOTA additions (late interaction, self-reflective gating)
- [ ] Implement experiment harness with reproducible seeding
- [ ] Per-configuration NDCG@10, MRR@10, MAP@10, Recall@5, Precision@5
- [ ] Statistical significance testing (paired t-test or bootstrap)

### 8C: LaTeX Paper
- [ ] Structure: Abstract → Intro → Related Work → Architecture → Experiments → Results → Conclusion
- [ ] Figures: pipeline diagram, pool architecture, performance comparison tables
- [ ] Proper citations (BibTeX)
- [ ] Target length: 8-12 pages, conference format (ACL/EMNLP style)

### 8D: Professional GitHub Presentation
- [ ] README rewrite with architecture diagrams
- [ ] Benchmark results table with comparison to baselines
- [ ] Getting started guide
- [ ] Contributing guide
- [ ] API documentation
- [ ] Badge: test status, coverage, npm version

---

## Quick Status

| Area | Status | Notes |
|------|--------|-------|
| Bug fixes | ✅ Done | 14/14, Codex audited |
| Pool architecture | ✅ Done | 8 pools, single-table, LanceDB 0.27.1 |
| Backend consolidation | ✅ Done | MultiTableBackend deleted, LanceDB sole backend |
| Pipeline propagation | ✅ Done | recall, capture, ingest, manager |
| Rules tools | ✅ Done | store + search, relevance-gated |
| Mistakes (per-agent + shared) | ✅ Done | shared: boolean option |
| Gap review | ✅ Done | All plans/memories/tasks audited |
| Architecture docs | ✅ Done | Rewritten for actual implementation |
| Reference library | ❌ Not started | PDF/doc ingestion |
| LLM classification | ❌ Not started | Nemotron zero-shot |
| Benchmarks | ❌ Stale | Need pool-aware golden dataset |
| SOTA research | ❌ Not started | Phase 8 planned |
| Test coverage | ⚠️ 75% | Pool + consolidation tests done, integration pending |
| Production deploy | ❌ Not started | Reindex + migration |
