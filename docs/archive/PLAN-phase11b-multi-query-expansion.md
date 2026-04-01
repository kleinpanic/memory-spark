# PLAN: Phase 11B — Multi-Query Expansion

**Date:** 2026-04-01
**Author:** KleinClaw-Meta
**Task:** `6b36833e` — memory-spark Phase 11B: Multi-Query Expansion module
**Branch:** `feat/phase11b-multi-query` (off `fix/phase10b-unified-reranker`)
**Baseline:** Config U (α=0.4) = **0.7889 NDCG@10**, Recall@10 = 0.9099

---

## Problem Statement

11% of relevant documents never appear in the top-40 vector results regardless of scoring or reranking — they're invisible to the pipeline. This is a hard retrieval ceiling. No amount of blending, reranking, or MMR can recover documents that aren't in the candidate set.

**Root cause:** A single query embedding captures one semantic interpretation. Different phrasings of the same information need activate completely different regions of the vector space.

**Example:** The query "What causes insulin resistance?" might miss a document titled "Metabolic syndrome and glucose uptake pathways" because the vocabulary overlap is too low. But a reformulation like "How does glucose metabolism become impaired?" would surface it.

---

## Architecture

```
User Query: "What causes insulin resistance?"
     │
     ├──────────────────────────────────────┐
     │                                      │
     ▼                                      ▼
  LLM generates 3 reformulations:        Original query
  1. "How does glucose metabolism          │
      become impaired?"                    │
  2. "Mechanisms of cellular               │
      insulin signaling failure"           │
  3. "Factors leading to reduced           │
      insulin sensitivity"                 │
     │                                      │
     ▼                                      ▼
  Embed each (parallel)                 Embed original
     │                                      │
     ▼                                      ▼
  Vector search × 3                     Vector search
  (top-40 each, parallel)              (top-40)
     │                                      │
     └──────────┬───────────────────────────┘
                │
                ▼
          Union + Dedupe
          (by chunk ID, keep highest score)
                │
                ▼
          Reranker (logit blend, α=0.4)
                │
                ▼
          Top-10 results
```

**Key design decisions:**
1. **Parallel execution:** All 4 embeddings + 4 vector searches run concurrently via `Promise.all`
2. **Score preservation:** When deduping, keep the highest score (document that matched best under any reformulation)
3. **Reranker as quality gate:** The expanded candidate pool feeds into the existing logit-blend reranker, which handles the final ranking
4. **Graceful degradation:** If LLM times out, fall back to original-query-only (no worse than current)

---

## Implementation Stages

### Stage 1: Core Module (`src/query/expander.ts`)

**New file** with the following interface:

```typescript
export interface QueryExpansionConfig {
  /** Whether multi-query expansion is enabled */
  enabled: boolean;
  /** vLLM / OpenAI-compatible chat completions URL */
  llmUrl: string;
  /** Model name */
  model: string;
  /** Number of reformulations to generate (default: 3) */
  numReformulations: number;
  /** Max tokens for generation (default: 150 — enough for 3 reformulations) */
  maxTokens: number;
  /** Temperature (default: 0.7 — high for diversity) */
  temperature: number;
  /** Timeout in ms (default: 15000) */
  timeoutMs: number;
  /** Bearer token (optional) */
  apiKey?: string;
}

/**
 * Generate alternative phrasings of a query.
 * Returns [original, ...reformulations] — always includes the original.
 * On failure, returns [original] (graceful degradation).
 */
export async function expandQuery(
  query: string,
  config: QueryExpansionConfig
): Promise<string[]>;
```

**System prompt:**
```
Generate exactly {n} alternative search queries for the given question or claim.
Each query should use different vocabulary, phrasing, or perspective while preserving the original meaning.
Strategies to use:
- Synonyms and domain-specific terminology
- Active vs. passive voice
- Different abstraction levels (specific ↔ general)
- Question form vs. declarative form
Return one query per line. No numbering, no explanations.
```

**Quality gates (parsed from LLM output):**
- Reject lines < 10 chars (garbage)
- Reject lines > 300 chars (hallucinated paragraphs)
- Reject lines that start with numbers, bullets, or "Note:" (formatting artifacts)
- Reject if output contains < 2 valid reformulations (LLM failure)
- Deduplicate reformulations that are >90% token overlap with original

**Defaults (matching HyDE pattern):**
```typescript
export const QUERY_EXPANSION_DEFAULTS: QueryExpansionConfig = {
  enabled: true,
  llmUrl: `http://${process.env.SPARK_HOST ?? "127.0.0.1"}:18080/v1/chat/completions`,
  model: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
  numReformulations: 3,
  maxTokens: 150,
  temperature: 0.7,
  timeoutMs: 15000,
  apiKey: process.env.SPARK_BEARER_TOKEN,
};
```

**Git checkpoint:** Commit after Stage 1 with unit tests passing.

---

### Stage 2: Unit Tests (`tests/expander.test.ts`)

Comprehensive Vitest suite:

```
describe("expandQuery")
  ✓ returns original query + reformulations
  ✓ always includes original query as first element
  ✓ falls back to [original] on LLM timeout
  ✓ falls back to [original] on empty LLM response
  ✓ rejects short/garbage reformulations (< 10 chars)
  ✓ rejects overly long reformulations (> 300 chars)
  ✓ strips numbering and bullet prefixes
  ✓ deduplicates near-identical reformulations
  ✓ respects numReformulations config
  ✓ handles network errors gracefully
  ✓ skips expansion when query is too short (< 10 chars)
  ✓ passes apiKey as Bearer token when configured

describe("parseReformulations")
  ✓ parses newline-separated output
  ✓ filters empty lines
  ✓ strips common LLM formatting artifacts
```

**Mock strategy:** Mock `fetch` to simulate LLM responses without hitting Spark.

**Git checkpoint:** Commit after Stage 2.

---

### Stage 3: Benchmark Integration (`scripts/run-beir-bench.ts`)

**Extend `RetrievalConfig`:**
```typescript
interface RetrievalConfig {
  // ...existing fields...
  /** Phase 11B: Use multi-query expansion */
  useMultiQuery?: boolean;
  /** Number of reformulations (default: 3) */
  multiQueryN?: number;
}
```

**Integration point in `runRetrieval()`:**
After getting the original `queryVector`, if `config.useMultiQuery`:

1. Call `expandQuery(q.text, expansionConfig)` → `[original, r1, r2, r3]`
2. Embed each reformulation via `embed.embedQuery()` (parallel)
3. Run `backend.vectorSearch()` for each vector (parallel)
4. Union all results: dedupe by chunk ID, keep max score
5. Continue to existing reranker/MMR stages

**Telemetry additions:**
```typescript
tel.stages.multiQuery = {
  reformulations: reformulations.length,
  queries: reformulations.map(r => r.slice(0, 80)),
  perQueryHits: [40, 38, 35, 40], // results per reformulation
  unionSize: 120,                  // total unique candidates
  latencyMs: 2340,                 // total expansion + search time
};
```

**New benchmark configs:**

| Config | Strategy |
|--------|----------|
| MQ-A | Multi-Query (3) → Vector → Top-10 (no reranker) |
| MQ-B | Multi-Query (3) → Vector → Logit Blend α=0.4 |
| MQ-C | Multi-Query (3) → Vector → Logit Blend α=0.5 |
| MQ-D | Multi-Query (3) → Vector → Conditional Logit Blend α=0.4 |

**Git checkpoint:** Commit after Stage 3 with compile check.

---

### Stage 4: Production Recall Integration (`src/auto/recall.ts`)

**Modify the `poolSearch` helper** to accept an optional array of query vectors:

```typescript
// Before: single vector
const vectorResults = await backend.vectorSearch(queryVector, searchOpts);

// After: multi-vector with parallel search + union
if (queryVectors.length > 1) {
  const allResults = await Promise.all(
    queryVectors.map(vec => backend.vectorSearch(vec, searchOpts).catch(() => []))
  );
  vectorResults = unionByChunkId(allResults);
} else {
  vectorResults = await backend.vectorSearch(queryVectors[0], searchOpts);
}
```

**New helper: `unionByChunkId()`:**
```typescript
function unionByChunkId(resultSets: SearchResult[][]): SearchResult[] {
  const best = new Map<string, SearchResult>();
  for (const set of resultSets) {
    for (const r of set) {
      const existing = best.get(r.chunk.id);
      if (!existing || r.score > existing.score) {
        best.set(r.chunk.id, r);
      }
    }
  }
  return [...best.values()].sort((a, b) => b.score - a.score);
}
```

**Config integration** — add to `AutoRecallConfig`:
```typescript
export interface QueryExpansionConfig { ... }
// Add to AutoRecallConfig:
queryExpansion?: QueryExpansionConfig;
```

**Git checkpoint:** Commit after Stage 4.

---

### Stage 5: Integration Tests + Mini-Benchmark

**Integration test (live Spark, 10 queries):**
```
describe("multi-query expansion (live)")
  ✓ generates reformulations for a real query
  ✓ expanded results contain docs not in original top-40
  ✓ union size > original result count
  ✓ latency within acceptable bounds (<5s total)
  ✓ graceful degradation on LLM timeout
```

**Mini-benchmark script: `scripts/diag-multi-query.ts`**
- Run 20 SciFact queries
- Compare: original top-40 IDs vs expanded top-40 IDs
- Report: new unique docs surfaced, overlap percentage, latency breakdown
- Output: `evaluation/results/diag-multi-query-*.json`

**Git checkpoint:** Commit after Stage 5.

---

### Stage 6: Full Benchmark + Analysis

Run configs MQ-A through MQ-D on full SciFact (300 queries).

**Success criteria:**
- Recall@10 improves from 0.9099 → 0.94+
- NDCG@10 improves over 0.7889 (Config U baseline)
- Latency p95 stays under 4000ms (embedding parallelism offsets LLM cost)
- No regressions on individual queries (check per-query telemetry)

**Git checkpoint:** Final commit with results + updated plan docs.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM timeout (Nemotron-Super is slow) | 15s timeout + graceful fallback to original-only |
| Bad reformulations add noise | Reranker filters irrelevant candidates; quality gates on LLM output |
| Latency blowup (4× embed + 4× search) | `Promise.all` parallelism; embed calls are 200ms each |
| Score inconsistency across reformulations | Union keeps max score per doc; reranker re-scores everything |
| Spark rate limits under 4× load | Embed service handles batches well (tested at 300 queries); sequential fallback if needed |

---

## Estimated Latency Budget

| Stage | Current (single query) | Multi-Query (4 queries) |
|-------|----------------------|------------------------|
| LLM expansion | 0ms | ~2000ms (one call, 3 reformulations) |
| Embedding | 200ms × 1 | 200ms × 4 = 200ms (parallel) |
| Vector search | 300ms × 1 | 300ms × 4 = 300ms (parallel) |
| Union + Dedup | 0ms | ~5ms |
| Reranker | 800ms | 800ms (same — reranks top candidates) |
| **Total** | **~1300ms** | **~3300ms** |

The LLM call is the bottleneck. If it's too slow, we can:
1. Cache reformulations for repeated queries (not applicable for BEIR but useful in production)
2. Reduce `numReformulations` from 3 → 2
3. Use a faster model if one becomes available on Spark

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/query/expander.ts` | **CREATE** | Core expansion module |
| `tests/expander.test.ts` | **CREATE** | Unit tests (15+ cases) |
| `scripts/diag-multi-query.ts` | **CREATE** | Mini-benchmark diagnostic |
| `scripts/run-beir-bench.ts` | **MODIFY** | Add MQ configs + integration |
| `src/auto/recall.ts` | **MODIFY** | Multi-vector search + union |
| `src/config.ts` | **MODIFY** | Add QueryExpansionConfig |
| `docs/PLAN-phase11b-multi-query-expansion.md` | **CREATE** | This plan |

---

## Rollback Plan

All work on `feat/phase11b-multi-query`. If performance degrades:
1. `git checkout fix/phase10b-unified-reranker` — instant rollback
2. In production config: `queryExpansion.enabled: false` — runtime disable without code change
3. `useMultiQuery: false` in benchmark configs — skip in evaluation
