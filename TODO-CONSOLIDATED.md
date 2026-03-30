# TODO — Consolidated & Verified

**Generated:** 2026-03-29 20:15 EDT
**Status:** Full audit complete. This file supersedes all other markdown TODO lists.

---

## 🚨 CRITICAL BUG: BEIR All-Zeros

**Root Cause:** `src/config.ts` line 273 — `MEMORY_SPARK_DATA_DIR` env var OVERRIDES `lancedbDir` config parameter.

```typescript
// config.ts:273-277
lancedbDir: dataDir
  ? defaults.lancedbDir // env override takes precedence over user config
  : expandHome(userConfig.lancedbDir ?? defaults.lancedbDir),
```

**Impact:**
- `beir-benchmark.ts` passes `{ lancedbDir: "evaluation/beir-datasets/scifact-index" }`
- But if `MEMORY_SPARK_DATA_DIR` is set in shell, that WINS
- Result: BEIR queries run against workspace index → all zeros because workspace doesn't contain BEIR docs

**Evidence:**
- `beir-scifact-2026-03-27T07-06-06.json`: `indexChunks: 5183` (correct BEIR)
- `beir-scifact-2026-03-29T18-20-12.json`: `indexChunks: 65930` (workspace data)

**Fix:**
```typescript
// Change line 273 to respect user config even when env var is set
lancedbDir: expandHome(userConfig.lancedbDir ?? defaults.lancedbDir),
```
OR
Run BEIR without `MEMORY_SPARK_DATA_DIR` set.

---

## ✅ ALREADY DONE (Verified)

| Item | Location | Status |
|------|----------|--------|
| PDF Batched Processing | `src/ingest/parsers.ts` | ✅ 25-page batches |
| Checkpoint/Resume | `scripts/reindex-benchmark.ts` | ✅ Hash-based |
| Circuit Breaker | `src/embed/queue.ts` | ✅ CLOSED/OPEN/HALF_OPEN |
| HyDE | `src/hyde/generator.ts` | ✅ Implemented |
| LLM-as-Judge | `evaluation/judge.ts` | ✅ Nemotron-Super-120B |
| Pool Column | `test-data/lancedb` | ✅ Present (65,930 chunks) |
| BEIR Datasets | `evaluation/beir-datasets/` | ✅ SciFact, NFCorpus, FiQA |
| BEIR Index | `evaluation/beir-datasets/scifact-index/` | ✅ 5,183 chunks |
| Golden Dataset | `evaluation/golden-dataset.json` | ✅ 139 queries |
| A/B/C/D Testing | `evaluation/benchmark-v2.ts` | ✅ Config ablations |
| A/B/C/D/E/F/G Testing | `evaluation/benchmark-v2.ts` | ✅ Multiple configs |

---

## ❌ NOT IMPLEMENTED (Plan Inaccurate)

### 1. gen-golden-ocmemory-BenchMarkSet.ts (LLM-Generated Queries)

**What the plan says:**
> Use Nemotron-Super-120B to generate questions from chunks for scientific rigor.

**What exists:**
- `evaluation/build-golden-dataset.ts` — MANUALLY authored queries
- Queries are hardcoded based on known content
- NOT LLM-generated from random chunks

**Why this matters (scientific principles):**
- Manual queries introduce author bias
- Known "good" queries don't test real retrieval quality
- LLM-generated queries from random samples = unbiased evaluation
- This is how BEIR and other benchmarks work

**Implementation needed:**
```typescript
// evaluation/gen-golden-ocmemory-BenchMarkSet.ts
// 1. Sample N random chunks from index
// 2. For each chunk, call Nemotron:
//    "Given this text, write 3 questions it directly answers"
// 3. Record: { query, relevant_chunk_ids, agent, file }
// 4. Output: golden-ocmemory.json
```

### 2. benchmark-ocmemory.ts

**Status:** DOES NOT EXIST.

**Confusion:**
- `evaluation/benchmark.ts` — uses `golden-dataset.json` (the manual one)
- `evaluation/benchmark-v2.ts` — uses `golden-dataset.json` (the manual one)
- Neither is called "benchmark-ocmemory.ts"

**What's needed:**
A dedicated runner that:
1. Uses LLM-generated golden dataset
2. Computes NDCG/MAP/Recall for agent memory retrieval
3. Separates from BEIR evaluation cleanly

---

## 🔍 VALIDATED GAPS (Still Need Work)

### 3. BM25 Sigmoid Calibration

**Location:** `src/storage/lancedb.ts:29`

**Current state:**
- Default `BM25_SIGMOID_MIDPOINT = 3.0`
- Configurable via `cfg.fts.sigmoidMidpoint`
- **NEVER CALIBRATED** from actual corpus score distribution

**Impact:**
- On SciFact: vector-only (0.768) > hybrid (0.758)
- FTS is adding noise, not signal
- Wrong midpoint destroys BM25 signal

**Fix:**
1. Run calibration script to collect 100+ BM25 scores from corpus
2. Find median, set as sigmoid midpoint
3. Update config or defaults

**Script:** `evaluation/run-calibration.ts` exists but is stub.

### 4. E2E Plugin Hook Integration Test

**Status:** NOT IMPLEMENTED.

**What exists:**
- `tests/integration.test.ts` — tests backend directly
- `evaluation/pipeline-eval.ts` — tests recall pipeline directly
- Neither tests through OpenClaw plugin hooks

**What's missing:**
Test that:
1. Sends message to OpenClaw gateway
2. Verifies memories injected into system prompt (before_prompt_build)
3. Sends follow-up and verifies capture fired (agent_end)

### 5. Zero-Shot Classifier Validation

**Location:** `src/classify/zero-shot.ts`

**Status:** IMPLEMENTED but UNTESTED.

**What exists:**
- `classifyForCapture()` POSTs to Spark BART-large-MNLI
- Endpoint confirmed UP at `http://10.99.1.1:18113/v1/classify`
- Heuristic fallback if classifier returns "none"

**What's missing:**
- No test exercises the classifier path end-to-end
- No validation that it produces sensible categories
- No calibration of `minConfidence = 0.75` threshold

### 6. Reference Library Benchmark

**Status:** NOT IMPLEMENTED.

**What exists:**
- `reference_library` pool exists in index (35,095 chunks)
- `memory_reference_search` tool queries only this pool
- Parsers support PDF/DOCX/audio for reference files

**What's missing:**
- No benchmark that tests reference retrieval quality
- No test fixtures for reference library evaluation
- No way to measure if reference search is working

---

## 🏗️ ARCHITECTURE QUESTIONS (Need Decisions)

### Q1: Should BEIR be indexed into the main DB or separate?

**Options:**

| Approach | Pros | Cons |
|----------|------|------|
| **Separate DB** (current plan) | Clean separation, comparable to BEIR baselines | Config bug prone, two indexes to maintain |
| **Single DB** (add BEIR to main) | Simpler, one index, reference + agent in one | "Dirty" for paper, cross-contamination |

**Recommendation:** Keep separate for paper rigor. Fix the config bug instead.

### Q2: Is the current golden-dataset.json good enough?

**Current:** 139 manually-authored queries across 12 categories.

**Issues:**
- Author knows the "right" answers
- Biased toward known-good retrieval
- Not reproducible for new content

**Fix:** Implement LLM-generated benchmark (see #1 above).

### Q3: Why isn't Docker being used?

**GAPS.md says:** Docker workspace discovery is broken.

**Reality:** We're not using Docker at all. All benchmarks run on host.

**Decision needed:**
- Fix Docker setup and use it? (cleaner isolation)
- Accept host-based workflow? (simpler, current reality)

---

## 📋 ACTION ITEMS (Priority Order)

### P0: Fix the BEIR Config Bug
- [ ] Fix `src/config.ts` to respect `lancedbDir` config param
- [ ] Run BEIR without `MEMORY_SPARK_DATA_DIR` set
- [ ] Verify BEIR results show real NDCG numbers

### P1: LLM-Generated Benchmark
- [ ] Create `evaluation/gen-golden-ocmemory-BenchMarkSet.ts`
- [ ] Sample 500 random chunks from index
- [ ] Call Nemotron to generate 3 questions per chunk
- [ ] Save as `evaluation/golden-ocmemory-llm.json`
- [ ] Run benchmark against it

### P2: Calibration & Validation
- [ ] Calibrate BM25 sigmoid midpoint
- [ ] Validate zero-shot classifier with real messages
- [ ] Add reference library test fixtures

### P3: E2E Plugin Testing
- [ ] Create `tests/integration/plugin-e2e.ts`
- [ ] Test before_prompt_build hook
- [ ] Test agent_end capture hook

### P4: Documentation
- [ ] Update HEY-OPENCLAW-STOP-BEING-RETARDED.md
- [ ] Archive GAPS.md (superseded by this file)
- [ ] Update PLAN.md with accurate status

---

## 📊 CURRENT INDEX STATUS (Verified 2026-03-29)

**test-data/lancedb (benchmark index):**
- Total chunks: 65,930
- Agents: dev, ghost, immune, main, meta, recovery, research, school, shared, taskmaster
- Pools: reference_library (35k), agent_memory (30k), tools (158), mistakes (121)

**evaluation/beir-datasets/scifact-index (BEIR):**
- Total chunks: 5,183
- Content: SciFact corpus documents
- Status: ✅ Indexed correctly, but config bug prevents use

**~/.openclaw/data/memory-spark/lancedb (production):**
- Total chunks: 24,700
- Agents: 21 agents
- Status: Has pool column

---

## 📚 FILES THIS SUPERSEDES

| File | Status | Action |
|------|--------|--------|
| HEY-OPENCLAW-STOP-BEING-RETARDED.md | OUTDATED | Archive after extracting any remaining valid info |
| GAPS.md | MOSTLY OUTDATED | Archive (most gaps fixed or wrong) |
| PLAN.md | OUTDATED | Archive or rewrite |
| CURRENT-STATE.md | OUTDATED | Archive (this file is the new source of truth) |
| CHECKLIST.md | OUTDATED | Archive |

---

## 🔬 SCIENTIFIC VALIDITY CHECKLIST

For the benchmark to be scientifically rigorous:

- [ ] **Unbiased queries**: LLM-generated from random samples, not manual
- [ ] **Proper relevance judgments**: Query should map to correct chunk(s)
- [ ] **Clean separation**: BEIR vs. agent data in separate indexes
- [ ] **Reproducible**: Scripts can regenerate the benchmark
- [ ] **Comparable metrics**: Standard BEIR metrics (NDCG@10, MAP, Recall)
- [ ] **Statistical significance**: A/B/C/D testing with proper comparison
- [ ] **Calibrated components**: BM25 sigmoid, classifier thresholds tuned

**Current score: 4/7** (clean separation exists but broken, metrics exist, A/B/C/D exists, but no LLM queries, no calibration, no significance testing)

---

**End of Consolidated TODO**
