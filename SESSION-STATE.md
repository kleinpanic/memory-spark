# Session State — 2026-03-27 00:12 EDT

## What's Done (10 commits: 8c46531 → 250727f)
- 14 bugs fixed, Codex audited
- LanceDB 0.14.1 → 0.27.1 (FTS+WHERE works natively)
- Pool architecture: 8 pools, routing in pool.ts, propagated through recall/capture/ingest/manager
- 11 tools (including memory_mistakes_search, memory_mistakes_store)
- 172/172 tests, 0 lint, 0 type errors
- ARCHITECTURE.md, CHANGELOG.md, CHECKLIST.md

## Current Priority Queue
1. **Rules tools** — `memory_rules_store` + `memory_rules_search`, relevance-gated auto-inject (NOT always-inject)
2. **Mistakes share option** — promote agent_mistakes to shared_mistakes
3. **Gap review** — audit ALL plan files, memories, oc-tasks for missed items
4. Benchmarks — pool-aware golden dataset
5. Integration tests

## Key Design Decisions
- Single table + pool column (per LanceDB best practices, NOT multi-table)
- ALL pools relevance-gated (including rules — Klein clarified)
- Pool routing: pool.ts resolvePool() is single source of truth
- SDK boundary: `as any` on tool array + hook params is correct (SDK uses any internally)
- Per-agent mistakes default to agent_mistakes pool, optional promote to shared

## Files Modified This Session
- src/storage/pool.ts (NEW), lancedb.ts, backend.ts, table-manager.ts, multi-table-backend.ts
- src/auto/recall.ts, capture.ts
- src/ingest/pipeline.ts
- src/manager.ts, src/config.ts
- index.ts (11 tools, MultiTableBackend fallback)
- tests/unit.ts (172 tests), tests/fts-where-test.ts, tests/table-manager.test.ts
- ARCHITECTURE.md, CHANGELOG.md, CHECKLIST.md

## Recall Pipeline Flow (current)
1. Agent memory+tools (pool filter, agent_id filter)
2. Agent mistakes (pool filter, agent_id filter)
3. Shared mistakes (pool filter, no agent filter)
4. Shared knowledge (pool filter, no agent filter)
5. Shared rules (pool filter, no agent filter, minScore=0)
6. Merge → source weighting → temporal decay → MMR → rerank → dedup → budget → inject

## Fix Needed
- Rules minScore should NOT be 0. Klein clarified: relevance-gated like everything else.
- Need to build memory_rules_store and memory_rules_search tools
- Need mistakes promote-to-shared option in memory_mistakes_store
