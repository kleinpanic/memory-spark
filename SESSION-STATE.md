# Session State — 2026-03-27 01:00 EDT

## What's Done
- 14 bugs fixed, Codex audited
- LanceDB 0.14.1 → 0.27.1 (FTS+WHERE works natively)
- Pool architecture: 8 pools, routing in pool.ts, propagated through recall/capture/ingest/manager
- 13 tools (search, get, store, forget, forget_by_path, reference_search, index_status, inspect, reindex, mistakes_search, mistakes_store, rules_store, rules_search)
- **Backend consolidation: MultiTableBackend + TableManager deleted (~800 LOC)**
- **LanceDBBackend is now the sole backend**
- **mistakes_search fixed: uses pool-based filtering, not instanceof check**
- 172+ tests, 0 lint, 0 type errors
- ARCHITECTURE.md, CHANGELOG.md, CHECKLIST.md all updated

## Key Design Decisions
- Single table + pool column (per LanceDB best practices, NOT multi-table)
- ALL pools relevance-gated (including rules — Klein clarified)
- Pool routing: pool.ts resolvePool() is single source of truth
- SDK boundary: `as any` on tool array + hook params is correct (SDK uses any internally)
- Per-agent mistakes default to agent_mistakes pool, optional promote to shared

## Recall Pipeline Flow
1. Agent memory+tools (pool filter, agent_id filter)
2. Agent mistakes (pool filter, agent_id filter)
3. Shared mistakes (pool filter, no agent filter)
4. Shared knowledge (pool filter, no agent filter)
5. Shared rules (pool filter, no agent filter, relevance-gated)
6. Merge → source weighting → temporal decay → MMR → rerank → dedup → budget → inject

## Next Up
- Phase 8: SOTA Research & Scientific Presentation (REALM, RAPTOR, ColBERT mapping)
- Phase 4: Benchmarks (pool-aware golden dataset, BM25 calibration)
- Phase 5: Integration tests (pool filtering, full ingest→recall)
