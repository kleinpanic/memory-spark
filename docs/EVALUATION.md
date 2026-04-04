# memory-spark Evaluation & Benchmarks

## Methodology

### Test Suites

| Suite | File | What it tests | Pipeline stages |
|-------|------|---------------|-----------------|
| Unit Tests | `tests/unit.test.ts` | Individual functions, edge cases | Chunking, quality gate, weights, cache, security, RRF, gate, reranker |
| Integration Tests | `tests/integration.test.ts` | Full pipeline with Spark backend | Embed, rerank, vector search (skipped in CI: `SKIP_INTEGRATION=1`) |
| FTS+WHERE Validation | `tests/fts-where-test.ts` | LanceDB FTS+WHERE correctness | Pool-scoped FTS queries |
| BEIR Benchmark | `scripts/run-beir-bench.ts` | Standard IR benchmarks (NDCG, MRR) | Full 15-stage pipeline |
| E2E Benchmark | `scripts/e2e-benchmark.ts` | Full injection pipeline | All pipeline stages including EmbedQueue, reranker, LCM dedup |
| Gate Diagnostics | `scripts/diag-full.ts` | Per-query gate/RRF trace | Gate spread, reranker decisions, per-stage scores |

### E2E Benchmark Test Cases

| # | Test | What it validates |
|---|------|-------------------|
| 1 | Basic factual recall | Can it find Klein's timezone from USER.md? |
| 2 | Restart procedure | Does it surface oc-restart from AGENTS.md? |
| 3 | Garbage rejection | No `[media attached:]`, `[System:]`, `HEARTBEAT_OK` in output |
| 4 | Token budget | Injection stays within 2500 tokens |
| 5 | Short message recall | "WireGuard IP?" triggers relevant recall |
| 6 | Empty message | "ok" produces no injection (< 4 char minimum) |
| 7 | LCM dedup | Doesn't duplicate content already in LCM summaries |

---

## Results (v0.4.0)

### Unit Tests
- **704 passing** (17 test files, 0 failures)
- Covers: chunking, hybrid merge (RRF + score), reranker gate (hard/soft/off — 25 tests),
  RRF blending (22 tests), source weighting, temporal decay, garbage detection, quality gate,
  LCM block rejection, cache, prompt injection detection, config defaults (7 tests)

### BEIR Benchmark (SciFact)

36 configurations tested. Production default is **GATE-A** (Hard Gate + RRF).

| Rank | Config | NDCG@10 | Recall@10 | MRR | Strategy |
|------|--------|---------|-----------|-----|----------|
| 1 | U: Logit alpha=0.4 | **0.7889** | 0.9099 | **0.7572** | Best NDCG |
| 2 | V: Logit alpha=0.6 | 0.7885 | **0.9243** | 0.7527 | Best Recall |
| 3 | N: Logit alpha=0.5 | 0.7863 | 0.9143 | 0.7522 | Balanced |
| 4 | MQ-C: Multi-Query | 0.7853 | 0.9177 | 0.7500 | 3 LLM reformulations |
| **P** | **GATE-A (production)** | **0.7802** | **0.9137** | 0.7455 | **78% skip, -50% latency** |

Full 36-config results in `docs/BENCHMARKS.md`.

### SOTA Comparison (SciFact NDCG@10)

| System | NDCG@10 | Notes |
|--------|---------|-------|
| memory-spark v0.4.0 | **0.7889** | Local DGX Spark |
| Contriever | 0.677 | Meta (2022) |
| BM25 | 0.665 | Sparse baseline |

### Latency

| Config | P50 | P95 | Reranker Calls |
|--------|-----|-----|---------------|
| GATE-A (production) | ~732ms | ~1.1s | 21% (gate skips 79%) |
| Full pipeline (no gate) | ~1400ms | ~2.1s | 100% |

### Pipeline Stage Coverage

| Stage | Unit Tested | E2E Tested | Notes |
|-------|------------|------------|-------|
| Query cleaning | yes | yes | Strips oc-tasks, LCM, media paths |
| HyDE generation | yes | yes | Nemotron-120B; silently skips on timeout |
| Multi-query expansion | yes | yes | 3 LLM reformulations, parallel embed |
| Embed (Spark) | -- | yes | Via EmbedQueue (production path) |
| Vector search | yes | yes | IVF_PQ cosine, per-pool |
| FTS search | yes | yes | Tantivy BM25, pool-scoped WHERE |
| Hybrid merge (RRF) | yes | yes | Scale-invariant, k=60 |
| Mistakes injection | -- | yes | Separate relevance-gated search |
| Source weighting | yes | yes | Configurable per-pool weights |
| Temporal decay | yes | yes | 0.8 floor, exp decay (rate=0.03) |
| Score normalization | yes | yes | Rescales to preserve gate spread |
| Reranker gate | yes | yes | Hard gate, spread-based (0.02-0.08) |
| Cross-encoder rerank | -- | yes | llama-nemotron-rerank-1b-v2 |
| MMR diversity | yes | yes | lambda=0.9 (relevance-heavy) |
| Security filter | yes | yes | Prompt injection detection |
| Token budget | -- | yes | Default 2000 tokens |
| Garbage gate | yes | yes | 30+ noise patterns |

### Lint & Code Quality
- ESLint: 0 errors, ~11 pre-existing warnings (max-warnings: 20)
- Knip: 0 unused exports
- TypeScript: clean compilation
- Prettier: all passing

---

## Known Gaps

1. **BEIR coverage is SciFact-only** — FiQA and NFCorpus require more Spark resource headroom
2. **No A/B agent testing** — can't compare agent performance with/without memory-spark in production
3. **OCR/NER coverage** — GLM-OCR and NER services not tested in E2E (no PDF/image test cases)
4. **Integration tests skip in CI** — Spark backend required; excluded via `SKIP_INTEGRATION=1`
5. **Pool routing classification** — heuristic-only currently; LLM-backed routing is planned
