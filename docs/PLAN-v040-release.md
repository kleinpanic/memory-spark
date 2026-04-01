# memory-spark v0.4.0 Release Plan

> Created: 2026-04-01
> Status: **Active — Phase A in progress**
> Goal: Production-grade release with GATE-A reranker, full BEIR validation, expanded plugin tools, professional documentation, and live agent testing.

---

## Phase A: Full Benchmark + Commit ✅ IN PROGRESS

**Objective:** Lock Phase 12 BEIR results. Merge all feature branches to main. Do NOT push to GitHub until Phase E (docs complete).

### A.1 Full Benchmark Run
- Run all 36 configs × 3 datasets (SciFact, FiQA, NFCorpus)
- Estimated runtime: 7-8 hours in tmux session `beir`
- Command: `npx tsx scripts/run-beir-bench.ts --dataset <dataset>`
- Results written to `evaluation/results/beir-<dataset>-summary-*.json`

### A.2 Lock GATE-A as Default
- Set `rerankerGate: "hard"` in `src/config.ts` defaults
- Thresholds: `gateThreshold: 0.08`, `gateLowThreshold: 0.02`
- Rationale: GATE-A achieved NDCG 0.7802 (+0.94%), Recall 0.9137 (+1.1%), Latency 732ms (best of all configs)

### A.3 Commit Phase 12
- Commit all Phase 12 work on `fix/phase7-pipeline-bugs` branch
- Squash-merge feature branches to `main` in order:
  1. `fix/phase7-pipeline-bugs` — Arrow vector fix, RRF, weighted hybrid, reranker gate
  2. `fix/phase10b-unified-reranker` — unified reranker path, blend modes
  3. `feat/phase11b-multi-query` — multi-query expansion module
- Tag `v0.4.0-rc1` after merge
- Do NOT push until Phase E complete

### A.4 Phase 12 Benchmark Results (SciFact 300q)

| Config | NDCG@10 | Δ Baseline | MRR | Recall@10 | Latency |
|--------|---------|------------|-----|-----------|---------|
| A (Baseline) | 0.7709 | — | 0.7365 | 0.9037 | 528ms |
| RRF-A (k=60, equal) | 0.7797 | +0.88% | 0.7511 | 0.8924 | 1540ms |
| RRF-B (k=60, vec=1.5) | 0.7788 | +0.79% | 0.7505 | 0.8924 | 1446ms |
| RRF-C (k=60, rerank=1.5) | 0.7770 | +0.61% | 0.7476 | 0.8924 | 1548ms |
| RRF-D (k=20, equal) | 0.7798 | +0.90% | 0.7514 | 0.8924 | 1452ms |
| **GATE-A (hard)** | **0.7802** | **+0.94%** | 0.7455 | **0.9137** | **732ms** |
| GATE-B (soft) | 0.7802 | +0.93% | 0.7518 | 0.8924 | 2297ms |
| GATE-C (soft, wide) | 0.7782 | +0.74% | 0.7493 | 0.8924 | 2131ms |
| GATE-D (Fix1+Fix2) | 0.7803 | +0.94% | 0.7525 | 0.8924 | 1413ms |

**Winner: GATE-A** — best recall, best latency, near-best NDCG. Hard gate skipped 236/300 queries (78%), only firing the reranker on 64 queries where vector spread was in the productive [0.02, 0.08] range.

---

## Phase B: Wire Gate into Production Recall

**Objective:** GATE-A working in the live auto-recall pipeline (not just benchmarks).

### B.1 Update `src/auto/recall.ts`
- Import `computeRerankerGate` from `src/rerank/reranker.ts`
- Before `reranker.rerank()`, run gate check on candidate pool
- If `shouldRerank === false`, skip reranker entirely → go to MMR
- If `shouldRerank === true`, pass `vectorWeightMultiplier` to `blendByRank`

### B.2 Add Gate Config
- Add to `RerankConfig`: `gateMode`, `gateThreshold`, `gateLowThreshold`
- Expose in plugin config so agents can tune per-deployment
- Default: `gateMode: "hard"`, thresholds `0.08/0.02`

### B.3 Gate Telemetry
- Log gate decisions: `[recall] gate: skip (vector confident, spread=0.26)`
- Log gate passes: `[recall] gate: pass (spread=0.05, multiplier=0.75)`
- Track skip rate in `memory_stats` tool output

### B.4 End-to-End Tests
- Add integration test in `tests/integration.test.ts` for gate bypass path
- Verify recall returns results when gate skips reranker
- Verify gate telemetry appears in verbose output

---

## Phase C: Plugin Tool Expansion

**Objective:** Expand from 9 to 14 tools for richer agent self-service.

### Current Tools (9)
`memory_search`, `memory_store`, `memory_forget`, `memory_inspect`, `memory_reindex`, `memory_stats`, `memory_config`, `memory_health`, `memory_export`

### New Tools (5)
1. **`memory_recall_debug`** — Full 13-stage pipeline trace for a query. Shows what the agent "sees" behind the scenes: vector scores, hybrid merge, gate decision, reranker impact, MMR diversity, token budget.
2. **`memory_bulk_ingest`** — Ingest multiple documents/notes in one call with batch embedding. Accepts array of `{text, path, source, tags}`.
3. **`memory_temporal`** — Query with explicit time windows. Params: `query`, `after` (ISO date), `before` (ISO date). Uses LanceDB temporal filtering.
4. **`memory_related`** — Given a memory chunk ID, find semantically similar memories (vector neighborhood search). Useful for "what else do I know about this topic?"
5. **`memory_gate_status`** — Expose reranker gate decisions and statistics. Shows: current gate mode, threshold config, skip rate over last N queries, average spread distribution.

---

## Phase D: Production Hardening

**Objective:** Ready for real agentic data at scale.

### D.1 Ingestion Stress Test
- Bulk ingest 10k+ memories, measure embed throughput
- Target: >50 memories/sec sustained with Spark embed service
- LanceDB write latency under load

### D.2 Concurrent Agent Test
- Simulate 3-4 agents hitting memory-spark simultaneously
- Recall + capture interleaved
- Verify no data corruption or race conditions in LanceDB

### D.3 Error Recovery Validation
- Verify circuit breaker handles Spark 502s (existing)
- Test degraded mode (embed down → FTS-only fallback)
- Test reranker timeout → graceful skip

### D.4 Memory Hygiene
- Auto-dedup on ingest (cosine similarity > 0.98 = skip)
- TTL-based expiry for ephemeral memories
- Storage usage reporting in `memory_stats`

### D.5 Security Audit
- Verify `security.ts` cross-agent filters
- Test: agent A stores, agent B with `crossAgent: false` cannot see
- Test: agent B with `crossAgent: true` CAN see

---

## Phase E: Documentation + GitHub Prep

**Objective:** Professional public release documentation. Do NOT push until this is complete.

### E.1 README.md Overhaul
- Architecture diagram (Mermaid)
- Benchmark results table (NDCG/MRR/Recall)
- Quick start guide (OpenClaw plugin install)
- Configuration reference
- Performance tuning guide
- Badge: tests passing, version, license

### E.2 docs/ Directory
- `ARCHITECTURE.md` — pipeline stages, data flow, gate logic, diagrams
- `BENCHMARKS.md` — full BEIR results with methodology, all datasets
- `TUNING.md` — threshold tuning, RRF weights, HyDE, MMR lambda
- `PLUGIN-API.md` — all 14 tool definitions with input/output examples
- `DEPLOYMENT.md` — Spark setup, LanceDB config, Docker support

### E.3 CHANGELOG.md
- Full v0.3.0 → v0.4.0 changelog
- Organized by: Breaking Changes, Features, Bug Fixes, Performance

### E.4 GitHub Release Prep
- Clean commit history on main
- Tag `v0.4.0`
- Release notes with benchmark artifacts
- Do NOT push until Klein reviews docs

---

## Phase F: Live Agent Testing (Post-Push)

**Objective:** Validate with real OpenClaw agents.

1. Enable auto-recall for `main` agent
2. Let it accumulate memories over a day
3. Inspect recall quality — are injected memories relevant?
4. Test cross-agent recall — dev stores, main recalls
5. Monitor p50/p95 latency, gate skip rate, reranker hit rate
6. Tune thresholds based on real query spread distributions

---

## Phase G: RAGAS Integration (Week 2+)

**Objective:** LLM-judged end-to-end RAG evaluation beyond retrieval metrics.

RAGAS adds metrics BEIR can't measure:
- **Context Precision** — are retrieved chunks useful for answering?
- **Faithfulness** — does the LLM answer stick to retrieved facts?
- **Answer Relevancy** — is the final answer helpful?
- **Noise Sensitivity** — does irrelevant context hurt?

Plan:
1. Build RAGAS test harness hooking into recall pipeline
2. Generate test dataset from real agent conversations
3. Use Spark Nemotron as judge LLM (no cloud cost)
4. Periodic regression runs (weekly cron)

---

## Timeline

| Phase | What | ETA | Status |
|-------|------|-----|--------|
| A | Full benchmark + commit | Day 1 | 🔄 In Progress |
| B | Wire gate into recall.ts | Day 1 | ⬜ Next |
| C | Plugin tool expansion | Day 1-2 | ⬜ Queued |
| D | Production hardening | Day 2 | ⬜ Queued |
| E | Documentation + GitHub prep | Day 2 | ⬜ Queued |
| F | Live agent testing | Day 3+ | ⬜ Future |
| G | RAGAS integration | Week 2+ | ⬜ Future |
