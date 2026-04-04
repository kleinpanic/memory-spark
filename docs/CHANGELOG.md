# Changelog

## [0.4.0] — 2026-04-01

### Highlights
- **Dynamic Reranker Gate (GATE-A)**: Skips the cross-encoder when vector search is confident, cutting latency 50%+ while improving recall by 1.1%. Production default.
- **Reciprocal Rank Fusion (RRF)**: Scale-invariant hybrid merging replaces buggy score-based blending. No more BM25/cosine scale mismatch.
- **5 new plugin tools**: 13 → 18 total. `memory_recall_debug`, `memory_bulk_ingest`, `memory_temporal`, `memory_related`, `memory_gate_status`.
- **Multi-Query Expansion**: LLM-generated query reformulations improve recall for ambiguous queries.
- **704 tests**: Up from 144. Comprehensive coverage across all pipeline stages.

### Added
- **Phase 12: RRF + Dynamic Reranker Gate**
  - `blendByRank()` in `reranker.ts` — standard RRF formula, scale-invariant
  - `computeRerankerGate()` — hard/soft/off gate modes with configurable thresholds
  - 4 RRF benchmark configs (RRF-A through RRF-D)
  - 4 Gate benchmark configs (GATE-A through GATE-D)
  - 47 new tests (25 gate + 22 RRF) in `reranker-gate.test.ts` and `rrf-blend.test.ts`
  - Defaults: `rerankerGate: "hard"`, `blendMode: "rrf"`, thresholds 0.08/0.02

- **Phase 11B: Multi-Query Expansion**
  - `src/query/expander.ts` — generates 3 LLM reformulations per query
  - Quality gates: length filtering, dedup, meta-commentary rejection
  - 33 unit tests for parsing, expansion, failure handling
  - Multi-vector search in `recall.ts` with parallel embedding + union

- **Phase C: New Plugin Tools**
  - `memory_recall_debug` — full 13-stage pipeline trace for a query
  - `memory_bulk_ingest` — batch store 1-100 memories in one call
  - `memory_temporal` — time-windowed memory search (after/before dates)
  - `memory_related` — find semantically similar memories by chunk ID
  - `memory_gate_status` — show reranker gate config and mode

- **Phase D: Production Hardening**
  - Enhanced `getStats()` with pool-level and agent-level breakdown
  - `memory_index_status` now shows chunks by pool, chunks by agent, gate config
  - Full benchmark script (`scripts/run-full-benchmark.sh`)

- **Documentation**
  - `BENCHMARKS.md` — BEIR results with methodology
  - `PLUGIN-API.md` — all 18 tools with examples
  - `TUNING.md` — threshold tuning guide
  - `PLAN-v040-release.md` — release plan
  - README.md completely rewritten for v0.4.0

### Changed
- Default `blendMode`: `"score"` → `"rrf"` (scale-invariant by default)
- Default `rerankerGate`: `"off"` → `"hard"` (GATE-A proven best)
- `RerankConfig` extended with: `blendMode`, `rrfK`, `rrfVectorWeight`, `rrfRerankerWeight`, `rerankerGate`, `rerankerGateThreshold`, `rerankerGateLowThreshold`
- `memory_index_status` output now includes pool/agent breakdown and gate config
- Tests updated to explicitly set `blendMode: "score", rerankerGate: "off"` for legacy score-blending tests

### Fixed
- **Arrow Vector type mismatch** (Phase 7): LanceDB returns Apache Arrow `Vector` objects where bracket indexing returns `undefined`. Fixed with `.toArray()` conversion. This was the root cause of MMR being a no-op.
- **BM25 sigmoid saturation** (Phase 7): FTS scores all mapped to >0.98 via hardcoded sigmoid midpoint, drowning semantic signal. Fixed by switching to RRF.
- **Reranker score compression**: Cross-encoder outputs 0.83–1.0 range with arbitrary reshuffling. Fixed by dynamic gate that bypasses reranking when it can't help.
- **Precision@k denominator**: Was using result set length instead of k, inflating scores on small result sets.
- **HyDE averaging vs. replacement**: Averaging query + hypothetical vectors diluted semantic focus. Switched to full replacement.

## [0.3.0] — 2026-03-25

### Added
- Phase 5A-5C: Reranker GPU fix, candidate limiting, telemetry
- Circuit breaker for embed queue (CLOSED → OPEN → HALF_OPEN pattern)
- Phase 4G-4H: Comprehensive test suite (144 tests)
- Parent-child chunking with context expansion
- Pool-based memory isolation (agent_memory, shared_knowledge, etc.)
- LCM dedup — skip memories that overlap with recent conversation

### Fixed
- Reranker timeout handling
- Stale timestamp bug (using current time instead of file mtime)
- Chunker infinite loop on edge cases

## [0.2.0] — 2026-03-15

### Added
- Initial OpenClaw plugin with 9 tools
- Auto-recall (before_prompt_build hook)
- Auto-capture (agent_end hook)
- LanceDB backend with IVF_PQ + FTS
- Source weighting and temporal decay
- MMR diversity re-ranking
- Prompt injection detection

## [0.1.0] — 2026-03-01

- Initial scaffold and proof of concept
