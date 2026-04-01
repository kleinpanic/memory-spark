# PLAN.md — Phase 7: Pipeline Bug Remediation

**Date:** 2026-03-31
**Author:** KleinClaw-Meta (forensic diagnostic session)
**Task:** `00889d7b` — Deep recon: BEIR benchmark underperformance
**Status:** Awaiting Klein approval before implementation

---

## Executive Summary

The forensic diagnostic (`scripts/diag-full.ts`) identified **4 confirmed bugs** and **1 architectural issue** that explain why every non-vector pipeline configuration underperforms the vector-only baseline.

| Bug | Severity | Impact on NDCG@10 | File(s) |
|-----|----------|-------------------|---------|
| Arrow Vector Type Mismatch | **CRITICAL** | MMR is 100% dead (NaN cosine) | `src/storage/lancedb.ts` |
| RRF Rank-Washout | HIGH | -5.2% (hybrid vs vector) | `src/auto/recall.ts` |
| Reranker Score Saturation | MEDIUM | Coin-flip on relevant docs | `src/rerank/reranker.ts` |
| HyDE Timeout/Failure | MEDIUM | 0% activation rate | `src/hyde/generator.ts` |
| `hasVectors` Gate False-Positive | HIGH | Cosine path chosen → all NaN | `src/auto/recall.ts` |

**Current Benchmark Baseline (SciFact, 300 queries, `run-beir-bench.ts`):**

| Config | NDCG@10 | Delta vs A |
|--------|---------|------------|
| A: Vector-Only | **0.7709** | — |
| B: FTS-Only | 0.6587 | -14.5% |
| C: Hybrid (RRF) | 0.7307 | -5.2% |
| D: Hybrid + Reranker | 0.7506 | -2.6% |
| E: Hybrid + MMR | 0.7307 | -5.2% (identical to C — MMR dead) |
| F: Hybrid + HyDE | 0.7307 | -5.2% (HyDE failed, fell back to C) |
| G: Full Pipeline | 0.7506 | -2.6% (reranker only, everything else dead) |

**Goal:** Fix all bugs, get Hybrid + Reranker + MMR to **exceed** vector-only baseline.

---

## Pre-Implementation: Git Hygiene

```bash
# 1. Stash any uncommitted work
cd ~/codeWS/TypeScript/memory-spark
git stash push -m "pre-phase7: untracked diagnostic scripts"

# 2. Create a feature branch
git checkout -b fix/phase7-pipeline-bugs

# 3. Verify clean state
git status  # should show nothing

# 4. After each fix: individual commits with descriptive messages
# git add <files> && git commit -m "fix(component): description"
```

**Every fix gets its own commit.** If something goes wrong, we can revert individual fixes without losing others.

---

## Fix 1: Arrow Vector Type Mismatch (CRITICAL)

### Root Cause
`src/storage/lancedb.ts` line 618 casts `row.vector` as `number[]`, but LanceDB returns Apache Arrow `Vector` objects. These have `.length` and `.toArray()` but `vec[0]` returns `undefined`. All downstream cosine similarity computations produce `NaN`.

### The Fix

**File:** `src/storage/lancedb.ts` — `rowToSearchResult()` function (line ~618)

```typescript
// BEFORE (line 618):
vector: (row.vector as number[]) ?? [],

// AFTER:
vector: toJsNumberArray(row.vector) ?? [],
```

**Add helper function** (top of file, after imports):

```typescript
/**
 * Convert a LanceDB vector column value to a plain JS number[].
 * LanceDB returns Apache Arrow Vector objects — bracket indexing
 * returns undefined, breaking all downstream cosine computations.
 * This normalizes to Float64Array → number[] for universal compat.
 */
function toJsNumberArray(vec: unknown): number[] {
  if (!vec) return [];
  if (Array.isArray(vec)) return vec;
  if (typeof (vec as any).toArray === "function") {
    return Array.from((vec as any).toArray() as ArrayLike<number>);
  }
  if (typeof (vec as any)[Symbol.iterator] === "function") {
    return Array.from(vec as Iterable<number>);
  }
  return [];
}
```

**Also fix the `vector` field on `SearchResult`** (line ~635):

```typescript
// BEFORE:
const vector = chunk.vector?.length ? chunk.vector : undefined;

// AFTER:
// chunk.vector is already converted by toJsNumberArray above,
// but guard against empty arrays from FTS-only rows.
const vector = (chunk.vector && chunk.vector.length > 0) ? chunk.vector : undefined;
```

### Why This Fix Is Correct
- Arrow `Vector.toArray()` returns a `Float32Array` or `Float64Array` (typed array)
- `Array.from()` converts to a plain `number[]`
- This is the same fix pattern used in LangChain's LanceDB adapter
- Bracket indexing, `.map()`, `.reduce()` all work on `number[]`

### Commit
```
fix(lancedb): convert Arrow Vector to number[] in rowToSearchResult

LanceDB returns Apache Arrow Vector objects, not JS arrays.
Bracket indexing (vec[0]) returns undefined, causing all cosine
similarity computations to produce NaN. MMR was completely dead.

Adds toJsNumberArray() helper that handles Arrow Vector, TypedArray,
and plain array inputs. Verified with diag-full.ts: 50/50 queries
now produce different MMR output.
```

---

## Fix 2: `hasVectors` Gate False-Positive in MMR

### Root Cause
`src/auto/recall.ts` line 516:

```typescript
const hasVectors = results.every((r) => r.vector && r.vector.length > 0);
```

Arrow `Vector` objects have `.length > 0` (4096), so this gate returns `true`. The code takes the cosine path, but `cosineSimilarity()` receives Arrow objects and produces `NaN` for every pair. The Jaccard fallback (which actually works) is never reached.

After Fix 1, this gate will work correctly because vectors will be `number[]`. But we should add a **runtime type assertion** as defense-in-depth.

### The Fix

**File:** `src/auto/recall.ts` — `mmrRerank()` function

```typescript
// BEFORE (line 516):
const hasVectors = results.every((r) => r.vector && r.vector.length > 0);

// AFTER:
const hasVectors = results.every(
  (r) => r.vector && r.vector.length > 0 && typeof r.vector[0] === "number"
);
```

**Also add a debug log** for pipeline observability:

```typescript
// After the hasVectors check:
if (process.env.MEMORY_SPARK_DEBUG) {
  const strategy = hasVectors ? "cosine" : "jaccard";
  const sampleVec = results[0]?.vector;
  const vecType = sampleVec?.constructor?.name ?? "none";
  console.debug(`[mmrRerank] strategy=${strategy} vectorType=${vecType} candidates=${results.length} limit=${limit} lambda=${lambda}`);
}
```

### Commit
```
fix(recall): harden hasVectors gate with typeof check + debug logging

Arrow Vector objects pass the .length > 0 check but return undefined
for bracket indexing. Adding typeof vec[0] === "number" ensures we only
take the cosine path when vectors are actual JS number arrays.
Falls back to Jaccard (which works) when vectors are non-standard.
```

---

## Fix 3: RRF Rank-Washout Mitigation

### Root Cause
Diagnostic found: when RRF promotes a different doc to #1 than vector search, it's **wrong 9/10 times** (9 irrelevant, 1 relevant). The avg overlap between vector top-40 and FTS top-40 is 13.3 docs — but the overlapping docs aren't necessarily the best ones.

RRF treats all rank positions equally between vector and FTS. But vector search is a much stronger signal on SciFact (NDCG 0.77 vs 0.66). FTS noise dilutes the vector signal.

### The Fix — Weighted RRF

**File:** `src/auto/recall.ts` — `hybridMerge()` function

```typescript
// BEFORE:
export function hybridMerge(
  vectorResults: SearchResult[],
  ftsResults: SearchResult[],
  limit: number,
  k = 60,
): SearchResult[] {

// AFTER — add vectorWeight parameter:
export function hybridMerge(
  vectorResults: SearchResult[],
  ftsResults: SearchResult[],
  limit: number,
  k = 60,
  vectorWeight = 1.0,
  ftsWeight = 1.0,
): SearchResult[] {
```

Apply weights to RRF scores:

```typescript
// Vector results: weighted RRF score
vectorResults.forEach((r, idx) => {
  merged.set(r.chunk.id, {
    result: r,
    rrfScore: vectorWeight * (1 / (k + idx + 1)),
    sources: 1,
  });
});

// FTS results: weighted RRF score
ftsResults.forEach((r, idx) => {
  const id = r.chunk.id;
  const ftsRrf = ftsWeight * (1 / (k + idx + 1));
  // ... rest unchanged
```

**File:** `src/config.ts` — add config fields:

```typescript
// In AutoRecallConfig:
hybridVectorWeight?: number;  // default 1.0
hybridFtsWeight?: number;     // default 1.0
```

**File:** `src/auto/recall.ts` — call site (line ~122):

```typescript
// BEFORE:
const merged = hybridMerge(vectorResults, ftsResults, limit);

// AFTER:
const merged = hybridMerge(
  vectorResults, ftsResults, limit, 60,
  cfg.hybridVectorWeight ?? 1.0,
  cfg.hybridFtsWeight ?? 1.0,
);
```

**Default weights:** `vectorWeight=1.0, ftsWeight=1.0` (backward compatible).
**Tuning:** We'll benchmark `vectorWeight=1.5, ftsWeight=0.5` and `vectorWeight=2.0, ftsWeight=0.5` as new configs H and I.

### Debug Logging

```typescript
// In hybridMerge, before returning:
if (process.env.MEMORY_SPARK_DEBUG) {
  const dualEvidence = sorted.filter(s => s.sources === 2).length;
  const vectorOnly = sorted.filter(s => s.sources === 1).length;
  console.debug(`[hybridMerge] total=${sorted.length} dualEvidence=${dualEvidence} vectorOnly=${vectorOnly} ftsOnly=${sorted.length - dualEvidence - vectorOnly} vectorWeight=${vectorWeight} ftsWeight=${ftsWeight}`);
}
```

### Commit
```
feat(recall): add weighted RRF with configurable vector/FTS weights

RRF rank-washout: FTS noise displaces high-confidence vector matches.
When RRF promotes a different #1 than vector, it's wrong 90% of the time.
Weighted RRF lets us bias toward the stronger signal source.

Adds hybridVectorWeight and hybridFtsWeight to AutoRecallConfig.
Defaults to 1.0/1.0 (no change). Debug logging shows dual-evidence stats.
```

---

## Fix 4: Reranker Score Saturation Diagnostics

### Root Cause
The Nemotron reranker shows tight score distributions (0.83–1.0), with 58% of queries having top-1 score ≥ 0.999. The reranker promotes relevant docs 5 times and demotes them 6 times — essentially a coin flip.

This is **not fixable in our code** — it's a model behavior issue. But we can:
1. Add temperature/logit normalization if the API supports it
2. Add diagnostic logging to track reranker effectiveness
3. Add a config option to bypass reranker when saturation is detected

### The Fix — Observability + Score Spread Guard

**File:** `src/rerank/reranker.ts`

```typescript
// After reranking, before returning results:

// Score spread diagnostic — log when scores are too compressed
if (process.env.MEMORY_SPARK_DEBUG) {
  const scores = reranked.map(r => r.score);
  const spread = Math.max(...scores) - Math.min(...scores);
  const saturated = scores.filter(s => s >= 0.999).length;
  console.debug(`[reranker] topN=${topN} spread=${spread.toFixed(4)} saturated=${saturated}/${scores.length} top3=[${scores.slice(0, 3).map(s => s.toFixed(4)).join(", ")}]`);
}
```

**File:** `src/config.ts` — add to RerankConfig:

```typescript
/** Minimum score spread (max - min) for reranker results to be trusted.
 *  If spread is below this, fall back to input ordering. Default: 0.01 */
minScoreSpread?: number;
```

**File:** `src/rerank/reranker.ts` — add spread guard:

```typescript
// After reranking:
const spread = Math.max(...reranked.map(r => r.score)) - Math.min(...reranked.map(r => r.score));
const minSpread = cfg.spark!.minScoreSpread ?? 0.01;
if (spread < minSpread) {
  // Reranker scores too compressed — not discriminating. Fall back to input order.
  if (process.env.MEMORY_SPARK_DEBUG) {
    console.debug(`[reranker] spread=${spread.toFixed(4)} < minSpread=${minSpread} — falling back to input order`);
  }
  return candidates.slice(0, topN);
}
```

### Commit
```
feat(reranker): add score spread guard + diagnostic logging

Nemotron reranker shows 58% score saturation (top-1 ≥ 0.999).
When score spread is below minScoreSpread (default 0.01), the
reranker isn't discriminating — fall back to input ordering.

Adds MEMORY_SPARK_DEBUG logging for score distributions.
```

---

## Fix 5: HyDE Timeout Hardening

### Root Cause
HyDE fails 5/5 with timeouts. The LLM call for generating hypothetical documents is too slow or the timeout is too aggressive.

### The Fix

**File:** `src/hyde/generator.ts`

```typescript
// 1. Add configurable timeout (currently hardcoded or missing)
// 2. Add retry with exponential backoff (1 attempt currently)
// 3. Add quality gate: reject short/refusal responses
// 4. Add debug logging for generation time + rejection reason
```

I need to read the current generator.ts to be specific here. The fix will:
- Add `timeoutMs` to HydeConfig (default 15000ms, up from whatever it is now)
- Add a single retry on timeout
- Log generation time and rejection reason
- Add a `maxRetries` config (default 1)

### Commit
```
fix(hyde): add timeout config, retry, and quality gate logging

HyDE fails 100% due to timeouts. Adds configurable timeoutMs
(default 15s), single retry on failure, and MEMORY_SPARK_DEBUG
logging for generation time, rejection reason, and LLM response.
```

---

## New Test Cases

### File: `tests/unit.test.ts` — additions

#### Test Group: Arrow Vector Conversion

```typescript
describe("Arrow Vector Handling", () => {
  it("toJsNumberArray: converts Arrow-like Vector to number[]", () => {
    // Simulate Arrow Vector: has .length, .toArray(), but vec[0] = undefined
    const fakeArrow = {
      length: 4,
      toArray: () => new Float32Array([1.0, 2.0, 3.0, 4.0]),
      [Symbol.iterator]: function* () { yield 1; yield 2; yield 3; yield 4; },
    };
    // This simulates what LanceDB actually returns
    expect(fakeArrow[0]).toBeUndefined(); // bracket indexing broken
    
    // Import and test the conversion
    const result = toJsNumberArray(fakeArrow);
    expect(Array.isArray(result)).toBe(true);
    expect(result).toHaveLength(4);
    expect(result[0]).toBeCloseTo(1.0);
    expect(typeof result[0]).toBe("number");
  });

  it("toJsNumberArray: passes through plain number[]", () => {
    const plain = [1.0, 2.0, 3.0];
    const result = toJsNumberArray(plain);
    expect(result).toBe(plain); // same reference, no copy
  });

  it("toJsNumberArray: handles Float32Array", () => {
    const typed = new Float32Array([0.1, 0.2, 0.3]);
    const result = toJsNumberArray(typed);
    expect(Array.isArray(result)).toBe(true);
    expect(result).toHaveLength(3);
  });

  it("toJsNumberArray: returns [] for null/undefined", () => {
    expect(toJsNumberArray(null)).toEqual([]);
    expect(toJsNumberArray(undefined)).toEqual([]);
  });

  it("toJsNumberArray: returns [] for non-vector objects", () => {
    expect(toJsNumberArray("not a vector")).toEqual([]);
    expect(toJsNumberArray(42)).toEqual([]);
    expect(toJsNumberArray({})).toEqual([]);
  });
});
```

#### Test Group: MMR hasVectors Gate

```typescript
describe("MMR hasVectors gate", () => {
  it("rejects Arrow Vectors where vec[0] is undefined", () => {
    const fakeArrowVec = { length: 4096, toArray: () => new Float32Array(4096) };
    const results: SearchResult[] = [
      makeResult("a", 0.9, fakeArrowVec as any),
      makeResult("b", 0.8, fakeArrowVec as any),
    ];
    // MMR should fall back to Jaccard, not cosine
    const reranked = mmrRerank(results, 2, 0.7);
    // With Jaccard, order depends on text similarity, not NaN cosine
    expect(reranked).toHaveLength(2);
    // Verify no NaN scores leaked through
    for (const r of reranked) {
      expect(Number.isNaN(r.score)).toBe(false);
    }
  });

  it("accepts proper number[] vectors for cosine path", () => {
    const vecA = [1, 0, 0, 0];
    const vecB = [0, 1, 0, 0]; // orthogonal to A
    const vecC = [0.99, 0.01, 0, 0]; // near-duplicate of A
    const results: SearchResult[] = [
      makeResult("a", 0.95, vecA),
      makeResult("b", 0.90, vecB),
      makeResult("c", 0.85, vecC),
    ];
    const reranked = mmrRerank(results, 2, 0.5);
    // With λ=0.5 (equal weight), MMR should prefer A (highest score)
    // then B (orthogonal = diverse) over C (near-duplicate of A)
    expect(reranked[0]!.chunk.id).toBe("a");
    expect(reranked[1]!.chunk.id).toBe("b"); // diverse, not the near-dup
  });
});
```

#### Test Group: Weighted RRF

```typescript
describe("Weighted RRF", () => {
  it("default weights (1.0, 1.0) match original behavior", () => {
    const vec = [makeResult("doc1", 0.9), makeResult("doc2", 0.8)];
    const fts = [makeResult("doc3", 0.7), makeResult("doc2", 0.6)];
    const original = hybridMerge(vec, fts, 10, 60);
    const weighted = hybridMerge(vec, fts, 10, 60, 1.0, 1.0);
    expect(weighted.map(r => r.chunk.id)).toEqual(original.map(r => r.chunk.id));
  });

  it("vectorWeight=2.0 boosts vector-first docs to top", () => {
    const vec = [makeResult("v1", 0.9), makeResult("v2", 0.8)];
    const fts = [makeResult("f1", 0.9), makeResult("f2", 0.8)];
    const merged = hybridMerge(vec, fts, 4, 60, 2.0, 0.5);
    // Vector docs should dominate top positions
    expect(merged[0]!.chunk.id).toBe("v1");
  });

  it("ftsWeight=0.0 makes FTS-only docs rank below all vector docs", () => {
    const vec = [makeResult("v1", 0.9)];
    const fts = [makeResult("f1", 0.9), makeResult("f2", 0.8)];
    const merged = hybridMerge(vec, fts, 3, 60, 1.0, 0.0);
    // v1 should be #1 (nonzero RRF), f1/f2 should have score=0
    expect(merged[0]!.chunk.id).toBe("v1");
    expect(merged[0]!.score).toBeGreaterThan(0);
  });

  it("dual-evidence docs get boosted by both weights", () => {
    const vec = [makeResult("shared", 0.9), makeResult("v-only", 0.85)];
    const fts = [makeResult("shared", 0.9), makeResult("f-only", 0.85)];
    const merged = hybridMerge(vec, fts, 3, 60, 1.0, 1.0);
    // "shared" should be #1 (dual evidence = summed RRF)
    expect(merged[0]!.chunk.id).toBe("shared");
  });
});
```

#### Test Group: Reranker Score Spread Guard

```typescript
describe("Reranker score spread guard", () => {
  it("returns input order when score spread < minSpread", () => {
    // Simulate saturated reranker: all scores ≈ 0.999
    const candidates = [
      makeResult("a", 0.9),
      makeResult("b", 0.8),
      makeResult("c", 0.7),
    ];
    // Mock reranker that returns compressed scores
    const reranked = applySpreadGuard(candidates, [
      { ...candidates[0]!, score: 0.9995 },
      { ...candidates[2]!, score: 0.9990 },
      { ...candidates[1]!, score: 0.9988 },
    ], 0.01);
    // spread = 0.0007 < 0.01 → should return input order
    expect(reranked.map(r => r.chunk.id)).toEqual(["a", "b", "c"]);
  });

  it("returns reranked order when spread is healthy", () => {
    const candidates = [
      makeResult("a", 0.9),
      makeResult("b", 0.8),
      makeResult("c", 0.7),
    ];
    const reranked = applySpreadGuard(candidates, [
      { ...candidates[1]!, score: 0.95 },  // b promoted
      { ...candidates[0]!, score: 0.80 },
      { ...candidates[2]!, score: 0.40 },
    ], 0.01);
    // spread = 0.55 > 0.01 → should return reranked order
    expect(reranked[0]!.chunk.id).toBe("b");
  });
});
```

#### Test Group: cosineSimilarity Defense

```typescript
describe("cosineSimilarity edge cases", () => {
  it("returns 0 for vectors with NaN values (defense)", () => {
    const a = [1, NaN, 0];
    const b = [0, 1, 0];
    const sim = cosineSimilarity(a, b);
    // Should return 0 or NaN — but never crash
    expect(typeof sim).toBe("number");
  });

  it("handles high-dimensional vectors (4096d) correctly", () => {
    // Simulate real embedding dimensionality
    const a = new Array(4096).fill(0).map(() => Math.random() - 0.5);
    const b = [...a]; // identical
    expect(cosineSimilarity(a, b)).toBeCloseTo(1.0, 5);
  });

  it("handles normalized unit vectors (dot product = cosine)", () => {
    const a = [0.6, 0.8]; // unit vector
    const b = [0.8, 0.6]; // unit vector
    const sim = cosineSimilarity(a, b);
    // dot product = 0.48 + 0.48 = 0.96
    expect(sim).toBeCloseTo(0.96, 2);
  });
});
```

---

## New Benchmarks

### Benchmark Config Additions for `run-beir-bench.ts`

Add these new configurations to the benchmark runner:

```
H: Vector + Reranker (no FTS)   — tests reranker value without FTS noise
I: Weighted RRF (2.0/0.5)      — tests vector-biased hybrid
J: Weighted RRF (1.5/0.5) + Reranker — tuned hybrid + reranker
K: Vector + MMR (fixed)         — tests MMR alone after Arrow fix
L: Full Pipeline (fixed)        — all stages with Arrow fix + weighted RRF
```

### New Diagnostic Script: `scripts/diag-stage-trace.ts`

Per-query stage-by-stage trace that logs:

```
Query: "Microbiota-derived short chain fatty acids..."
  [vector]   top-3: doc1(0.82), doc7(0.79), doc12(0.77)  — 2/3 relevant
  [fts]      top-3: doc45(BM25=12.3), doc1(8.7), doc99(7.1)  — 1/3 relevant
  [hybrid]   top-3: doc1(1.00), doc45(0.82), doc7(0.74)  — 1/3 relevant ← REGRESSION
  [reranker] top-3: doc45(0.999), doc1(0.998), doc12(0.997) — spread=0.002 ← SATURATED
  [mmr]      top-3: doc45(0.999), doc99(0.995), doc7(0.990) — ∆ from reranker: 2 swaps
  [ndcg@10]  vector=0.85, hybrid=0.72, final=0.68  ← 20% regression
```

This makes it trivial to identify exactly where each query degrades.

### Regression Guard: `scripts/regression-check.ts`

Runs after any pipeline change, compares against saved baseline:

```typescript
// Loads evaluation/results/beir-scifact-baseline.json
// Runs current pipeline on same queries
// Asserts: new NDCG@10 >= baseline - tolerance (default 0.02)
// Fails with per-config regression report if violated
```

---

## Debug Logging Framework

**Environment variable:** `MEMORY_SPARK_DEBUG=1`

All debug logs use `console.debug()` (suppressed by default) and follow the format:

```
[component] key=value key=value ...
```

### Log Points Added

| Component | Log | Data |
|-----------|-----|------|
| `lancedb.ts` | `[lancedb:rowToResult]` | `vectorType={constructor.name} length={n} bracketWorks={bool}` |
| `recall.ts` | `[mmrRerank]` | `strategy={cosine\|jaccard} vectorType={name} candidates={n} limit={n} lambda={n}` |
| `recall.ts` | `[hybridMerge]` | `total={n} dualEvidence={n} vectorOnly={n} ftsOnly={n} vectorWeight={n} ftsWeight={n}` |
| `reranker.ts` | `[reranker]` | `topN={n} spread={f} saturated={n}/{n} top3=[f, f, f] latencyMs={n}` |
| `hyde/generator.ts` | `[hyde]` | `activated={bool} latencyMs={n} docLength={n} rejected={reason\|null}` |

---

## Implementation Order

Each fix is independent and individually testable:

| Step | Fix | Depends On | Test Command | Estimated Time |
|------|-----|------------|--------------|----------------|
| 1 | Arrow Vector conversion | — | `npm test -- --grep "Arrow Vector"` | 15 min |
| 2 | hasVectors gate | Fix 1 | `npm test -- --grep "hasVectors"` | 10 min |
| 3 | Weighted RRF | — | `npm test -- --grep "Weighted RRF"` | 20 min |
| 4 | Reranker spread guard | — | `npm test -- --grep "spread guard"` | 15 min |
| 5 | HyDE timeout hardening | — | `npm test -- --grep "HyDE"` | 15 min |
| 6 | Debug logging framework | All fixes | Manual: `MEMORY_SPARK_DEBUG=1 npm run bench` | 10 min |
| 7 | New benchmark configs | All fixes | `npx tsx scripts/run-beir-bench.ts` | 30 min (runtime) |
| 8 | Regression guard script | Fix 7 | `npx tsx scripts/regression-check.ts` | 15 min |
| 9 | Full BEIR re-benchmark | All | `bash scripts/run-beir-pipeline.sh` | ~60 min (runtime) |

**Total implementation time:** ~3 hours (plus ~90 min benchmark runtime)

---

## Validation Criteria

### Must Pass (blocking)
- [ ] `npm run ci:quick` — typecheck + lint + all tests green
- [ ] All new test cases pass
- [ ] Arrow Vector: `cosineSimilarity()` returns valid numbers (not NaN) for LanceDB results
- [ ] MMR output changes when `lambda` changes (proves it's no longer a no-op)
- [ ] Config H (Vector + Reranker) NDCG@10 tracked separately
- [ ] Config I/J (Weighted RRF) NDCG@10 ≥ Config C (unweighted RRF)

### Should Pass (important)
- [ ] Full Pipeline NDCG@10 ≥ Vector-Only baseline (0.7709)
- [ ] Reranker spread guard fires on ≤ 20% of queries (down from 58%)
- [ ] HyDE activates on ≥ 80% of queries (up from 0%)

### Nice to Have
- [ ] Full Pipeline NDCG@10 ≥ 0.80 (exceeds vector-only by ~4%)
- [ ] Latency P95 ≤ 2000ms for Hybrid + Reranker + MMR

---

## Rollback Plan

Each fix is an individual git commit on `fix/phase7-pipeline-bugs` branch.

```bash
# Revert a single fix:
git revert <commit-sha>

# Revert everything:
git checkout main

# Nuclear option (reset branch):
git reset --hard origin/main
```

Baseline results are preserved in `evaluation/results/beir-scifact-summary-*.json`.

---

## Open Questions for Klein

1. **Weighted RRF defaults:** Should we ship with `vectorWeight=1.0, ftsWeight=1.0` (safe, no behavior change) or optimize aggressively based on benchmark results? I'd recommend shipping safe and tuning in a follow-up.
Find the middle ground; optimize for our end production goal and for benchmarks. Do not overfit and stay around safe values.

3. **Reranker spread guard:** The 0.01 default is conservative. We could make it 0.05 to catch more saturation. Preference?
I have no clue dude. You are to act as an AI expert and reserch best practices. you made most of this codebase.

5. **HyDE timeout:** Current behavior unclear — I need to read `generator.ts` to know the current timeout. Should I increase to 15s or 30s?
Same as aboe.

7. **Branch strategy:** Merge to `main` after all tests pass, or keep on feature branch for a soak period?
Soak period. production / profressional standards of producting code. Keep branches but merge to main when pushing. Keep them in history.

9. **Run all 3 datasets (SciFact + NFCorpus + FiQA)?** FiQA is resource-intensive and previously caused 502s. Recommend SciFact first, then NFCorpus if time permits.
All 3. SciFact -> NFCorpus -> FiQA sequentially.. This brings up a concern of mine -> You didn't address how drastically different the results were against the different dbs....
