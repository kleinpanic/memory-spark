# memory-spark v0.2.0 — Offline Development Plan

> Written: 2026-03-26 14:35 EDT | Author: KleinClaw-Meta
> Status: **ACTIVE — executing now**
> Repo: `~/codeWS/TypeScript/memory-spark/`
> Test data: `~/codeWS/TypeScript/memory-spark/test-data/lancedb/`
> Production: **DO NOT TOUCH** until Phase 4 passes

---

## Principle: Zero Production Contact

All development, indexing, evaluation, and benchmarking happens against a **test LanceDB** at `./test-data/lancedb/`. The production path (`~/.openclaw/data/memory-spark/lancedb/`) is never read or written. The production gateway is never restarted. memory-spark stays broken/empty in production until proven good.

Environment variable: `MEMORY_SPARK_DATA_DIR=./test-data` overrides `lancedbDir` in `resolveConfig()`.

---

## Phase 1: Standalone Test Harness

### 1a. Add `MEMORY_SPARK_DATA_DIR` env override to `config.ts`
In `resolveConfig()`, check `process.env.MEMORY_SPARK_DATA_DIR` and use it for `lancedbDir` if set.

### 1b. Write `scripts/standalone-index.ts`
Replicates `watcher.ts:runBootPass()` but runs standalone (no gateway):
- Discovers all agent workspaces via `discoverAllAgents()` + `discoverWorkspaceFiles()`
- Creates a fresh LanceDB at `./test-data/lancedb/`
- Runs the full ingest pipeline (parse → chunk → quality gate → clean → NER → embed → store)
- Indexes reference library paths from config
- Logs progress to stdout
- Run: `MEMORY_SPARK_DATA_DIR=./test-data npx tsx scripts/standalone-index.ts`

### 1c. Write `scripts/standalone-search.ts`
Interactive search against the test index:
- Takes a query string argument
- Runs the full recall pipeline (embed → vector+FTS → hybridMerge → sourceWeighting → decay → MMR → rerank)
- Prints results with scores, paths, snippets
- Run: `MEMORY_SPARK_DATA_DIR=./test-data npx tsx scripts/standalone-search.ts "how to restart gateway"`

---

## Phase 2: Fix the Bugs

### 2a. Manager search parity (`src/manager.ts`)
Import `hybridMerge`, `applySourceWeighting`, `applyTemporalDecay`, `mmrRerank` from `recall.ts`.
Replace the naive merge in `manager.search()` with the production pipeline.

### 2b. Auto-capture thresholds (`src/auto/capture.ts` + `src/config.ts`)
- `capture.ts:56`: Change `cfg.minMessageLength ?? 80` to `cfg.minMessageLength ?? 30`
- `config.ts` default: Change `minConfidence: 0.75` to `minConfidence: 0.60`
- Broaden `containsFactPattern`: add `\b(the issue|root cause|fixed by|the problem|resolved|the answer|the solution)\b`

### 2c. Kill mock eval (`evaluation/run.ts`)
Delete all mock functions. Default to live eval. `--mock` flag removed.

### 2d. Unit tests
Add tests for: hybridMerge preserves cosine, sourceWeighting boosts MISTAKES, temporalDecay floor at 0.8, capture threshold changes.

---

## Phase 3: Eval Against Test Index

### 3a. Run standalone indexer → populate test LanceDB
### 3b. Run live eval: `MEMORY_SPARK_DATA_DIR=./test-data npx tsx evaluation/run.ts`
### 3c. Record baseline: NDCG@10, MRR, Recall@5, p95 latency for all ablations
### 3d. Compare: full pipeline vs vanilla vs no-FTS. If full < vanilla, investigate which stage hurts.

---

## Phase 4: E2E Agent Benchmark

### 4a. Expand golden dataset to 100+ queries with gold answers
### 4b. Write `evaluation/e2e-benchmark.ts`:
- Takes `--model nemotron-super` or `--model codex`
- For each query:
  1. Search test index → get top 5 results
  2. Build prompt: system context + retrieved memories + query
  3. Send to model (Nemotron-Super via Spark vLLM or Codex via API)
  4. Score answer against gold answer using LLM-as-judge
- Produces: accuracy %, per-category breakdown, memory-ON vs memory-OFF comparison
### 4c. A/B test: same queries with empty context vs retrieved context
### 4d. PASS CRITERIA:
- Memory-ON accuracy > Memory-OFF accuracy by ≥10%
- NDCG@10 ≥ 0.60
- p95 latency < 200ms
- Zero prompt injection leakage

---

## Phase 5: Production Deploy (ONLY after Phase 4 passes)
1. Swap test LanceDB → production path
2. Gateway restart via `oc-restart`
3. Verify boot-pass completes
4. Spot-check: agent asks a question → verify recall XML appears
5. Monitor for 24h

---

## Execution Order (starting now)

```
Phase 1a (config env override) → 1b (standalone indexer) → 1c (standalone search)
    ↓
Phase 2a-2d (bug fixes + tests)
    ↓
Phase 3 (eval against test index)
    ↓
Phase 4 (E2E benchmark — the real test)
    ↓
Phase 5 (production deploy — Klein approval required)
```
