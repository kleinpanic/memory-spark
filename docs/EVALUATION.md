# memory-spark Evaluation & Benchmarks

## Methodology

### Test Suites

| Suite | File | What it tests | Pipeline stages |
|-------|------|---------------|-----------------|
| Unit Tests | `tests/unit.ts` | Individual functions, edge cases | Chunking, quality gate, weights, cache, security |
| Quick Eval v2 | `scripts/quick-eval-v2.ts` | Search relevance (content matching) | Vector + FTS + source weighting + temporal decay |
| E2E Benchmark | `scripts/e2e-benchmark.ts` | Full injection pipeline | All 13 stages including EmbedQueue, reranker, LCM dedup |

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

## Results (v0.2.1)

### Unit Tests
- **144/144 passing** (0 failures)
- Covers: chunking, hybrid merge, source weighting (custom + default), temporal decay, garbage detection, quality gate, LCM block rejection, cache, prompt injection detection

### Quick Eval v2 (Content Relevance)
Tested on 10 queries across 9K+ chunks (standalone index, ~40% complete):

| Metric | Score |
|--------|-------|
| Vector-only relevance | 90% (9/10) |
| Hybrid relevance | 90% (9/10) |

### E2E Benchmark (Full Pipeline)

**Dev index (13.6K chunks):**
| Metric | Score |
|--------|-------|
| Tests passed | 7/7 (100%) |
| Avg memories injected | 4.9 per query |
| Avg injection tokens | ~847 |

**Production copy (22.6K chunks):**
| Metric | Score |
|--------|-------|
| Tests passed | 6/7 (86%) |
| Failed test | Restart procedure (stale data) |

### Lint & Code Quality
- ESLint: 0 errors, 1 warning (intentional `any`)
- Knip: 0 unused exports (2 intentionally kept files)
- TypeScript: clean compilation

## Pipeline Stage Coverage

| Stage | Unit Tested | E2E Tested | Notes |
|-------|------------|------------|-------|
| Query cleaning | ✅ | ✅ | Strips oc-tasks, LCM, media paths |
| Embed (Spark) | — | ✅ | Via EmbedQueue (production path) |
| Vector search | ✅ | ✅ | IVF_PQ cosine |
| FTS search | — | ✅ | tantivy-based, post-filtered |
| Hybrid merge | ✅ (12 tests) | ✅ | Cosine-preserving + rank boost |
| Mistakes injection | — | ✅ | Separate filtered search |
| Source weighting | ✅ (5 tests) | ✅ | Configurable weights |
| Temporal decay | ✅ (2 tests) | ✅ | 0.8 floor, exp decay |
| MMR diversity | — | ✅ | λ=0.7, Jaccard |
| Cross-encoder rerank | — | ✅ | Nemotron-Rerank-1B |
| LCM dedup | — | ✅ | 40% token overlap |
| Security filter | ✅ (6 tests) | ✅ | Prompt injection detection |
| Token budget | — | ✅ | Default 2000 tokens |
| Garbage gate | ✅ (13 tests) | ✅ | 30+ noise patterns |

## Known Gaps

1. **No NDCG/MRR metrics** — quick-eval uses binary relevance (match/no match), not graded
2. **No A/B testing** — can't compare agent performance with/without memory-spark
3. **No latency benchmarks** — no p50/p95/p99 for the full pipeline
4. **Production data needs reindex** — stale garbage affects production E2E results
5. **NER coverage** — NER is used in ingestion but not specifically tested in E2E
6. **OCR coverage** — GLM-OCR/EasyOCR not tested in E2E (no PDF/image test cases)
