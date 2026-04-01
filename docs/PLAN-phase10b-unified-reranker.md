# PLAN: Phase 10B — Unified Reranker Pipeline

**Date:** 2026-04-01  
**Author:** KleinClaw-Meta  
**Task:** `00889d7b` — Deep recon: BEIR benchmark underperformance  
**Status:** PROPOSED — awaiting Klein approval  
**Branch:** `fix/phase10b-unified-reranker` (to be created from current HEAD `3fff705`)

---

## Problem Statement

The BEIR benchmark script (`scripts/run-beir-bench.ts`) contains a **duplicate reranker code path** that bypasses critical normalization, error handling, and telemetry. This creates a class of bugs where blended configs (α > 0) behave fundamentally differently from pure reranker configs (α = 0), making benchmark results unreliable and non-comparable.

### Evidence

| Config | α | Code Path | Query Normalization | Error Handling | Telemetry Lines |
|--------|---|-----------|---------------------|----------------|-----------------|
| R | 0 | `reranker.rerank()` | ✅ `normalizeQueryForReranker()` | ✅ Returns passthrough | 300 |
| Q | 0.5 | Direct `fetch()` in bench | ❌ Uses raw `q.text` | ❌ Silent fallthrough | 0 |
| S | 0.3 | Direct `fetch()` in bench | ❌ Uses raw `q.text` | ❌ Silent fallthrough | 0 |
| T | 0.3 | Direct `fetch()` in bench | ❌ Uses raw `q.text` | ❌ Silent fallthrough | 0 |

### Impact

1. **Query normalization bypass** — SciFact queries are declarative claims. The reranker was fine-tuned on Q&A pairs. Without normalization, score discrimination collapses (spread ~0.02 vs ~18.0 in logit space). All blended configs are benchmarked with a fundamentally broken reranker signal.

2. **Silent error swallowing** — If the reranker returns a non-200 response, the `if (resp.ok)` guard silently falls through. No error logged. Candidates pass through unmodified. The benchmark reports results as if the reranker ran, but it didn't.

3. **Blind telemetry** — Config R produces 300 lines of per-query telemetry (logit spreads, blend metrics, normalization status). Configs Q/S/T produce zero. We literally cannot see what the reranker is doing.

4. **Non-comparable results** — R (α=0) = 0.7395 vs T (α=0.3) = 0.6608. This 14% gap was attributed to "more reranker weight = more damage." But R got normalized queries while T didn't. The gap is contaminated — we can't isolate the effect of α from the effect of normalization.

---

## Root Cause

Lines 569–596 of `run-beir-bench.ts` duplicate the reranker call logic for configs with `scoreBlendAlpha > 0`:

```typescript
// Phase 9A: Score blending — if alpha > 0, blend original + reranker scores
if (config.scoreBlendAlpha && config.scoreBlendAlpha > 0) {
  // Manual rerank call with blending (bypass the reranker's internal blending
  // to use the benchmark-specific alpha override)
  const normalizedQuery = candidates[0] ? q.text : "";  // ← BUG: not normalized!
  const pool = candidates.slice(0, 30);
  const resp = await fetch(`${rerankCfg!.baseUrl}/rerank`, { ... });
  if (resp.ok) {
    const allBlended = blendScores(pool, data.results, config.scoreBlendAlpha);
    candidates = allBlended.slice(0, k);
  }
  // On error, fall through with unmodified candidates  ← BUG: silent!
} else {
  candidates = await reranker.rerank(q.text, candidates, k);  // ← This path works correctly
}
```

This was introduced in Phase 9A (commit `8cec376`) because the `Reranker` interface reads `blendAlpha` from `RerankConfig` at construction time. The benchmark needed per-query alpha overrides, so it bypassed the class entirely. This "shortcut" skipped normalization, error handling, and telemetry.

---

## Fix Strategy

**One reranker path. Always.** Kill the direct `fetch()` in the benchmark and route ALL configs through `reranker.rerank()`.

### Fix 1: Add `alphaOverride` parameter to `reranker.rerank()`

**File:** `src/rerank/reranker.ts`

Extend the `Reranker` interface to accept an optional per-call alpha:

```typescript
export interface Reranker {
  rerank(
    query: string,
    candidates: SearchResult[],
    topN?: number,
    options?: { alphaOverride?: number },
  ): Promise<SearchResult[]>;
  probe(): Promise<boolean>;
}
```

Inside `sparkReranker.rerank()`, use `options?.alphaOverride ?? blendAlpha` (where `blendAlpha` is the config default). This preserves backward compatibility — production code that doesn't pass `options` behaves identically.

**Rationale:** The reranker already owns normalization, logit recovery, blending, spread guards, and telemetry. Adding an override parameter is a 3-line change that lets the benchmark control alpha without duplicating the entire pipeline.

### Fix 2: Delete duplicate code path in benchmark

**File:** `scripts/run-beir-bench.ts`

Replace lines 569–596 (the `if (config.scoreBlendAlpha > 0) { ... } else { ... }` branch) with:

```typescript
candidates = await reranker.rerank(
  q.text,
  candidates,
  k,
  config.scoreBlendAlpha != null ? { alphaOverride: config.scoreBlendAlpha } : undefined,
);
```

This is the entire reranker section. One line. All configs go through the same path.

**Rationale:** Eliminates the root cause — there's only one place where the reranker is called, so normalization/telemetry/error-handling can never be bypassed.

### Fix 3: Remove `rerankCfg` parameter from benchmark runner

**File:** `scripts/run-beir-bench.ts`

The `runBenchmark()` function currently accepts a separate `rerankCfg` parameter (baseUrl, apiKey, model) that was only used by the direct `fetch()` path. Remove it.

**Rationale:** Dead code after Fix 2. The `reranker` object already encapsulates connection details.

### Fix 4: Add error telemetry to `sparkReranker.rerank()` fallback path

**File:** `src/rerank/reranker.ts`

The existing fallback (`if (!resp.ok) { return pool.slice(0, topN); }`) is silent. Add:

```typescript
if (!resp.ok) {
  const body = await resp.text().catch(() => "");
  console.error(
    `[reranker] ERROR: ${resp.status} ${resp.statusText} — falling back to input order` +
    (body ? ` | body: ${body.slice(0, 200)}` : ""),
  );
  return pool.slice(0, topN);
}
```

**Rationale:** During Phase 10A benchmarking, if any fetch failed we'd have no signal. This is always-on (not gated behind VERBOSE) because reranker failures are operational anomalies, not debug noise.

---

## Validation Plan

### Step 1: Unit validation (pre-benchmark)

Run a single-query smoke test to confirm the unified path produces identical results to the old direct-fetch path:

```bash
VERBOSE=1 npx tsx scripts/run-beir-bench.ts --dataset scifact --configs Q --limit 5
```

Verify:
- `[reranker]` telemetry lines appear (they were absent before)
- `(normalized: "Is it true that...")` tag appears in logs
- Results are numerically reasonable (NDCG > 0)

### Step 2: Full benchmark — re-run Q, R, S, T

```bash
npx tsx scripts/run-beir-bench.ts --dataset scifact --configs Q,R,S,T 2>&1 | tee /tmp/beir-phase10b.log
```

**Expected outcome:** Q/S/T scores should improve significantly now that the reranker gets properly normalized queries. The "R beats Q" anomaly should disappear.

### Step 3: Regression check — re-run A, H

Run the baselines to confirm no regression:

```bash
npx tsx scripts/run-beir-bench.ts --dataset scifact --configs A,H 2>&1 | tee /tmp/beir-phase10b-baseline.log
```

- Config A (vector-only): should be unchanged (0.7709 ± 0.001)
- Config H (pure reranker, Phase 8): should be unchanged or improved

### Step 4: New config — clean comparison grid

After validation, introduce two clean configs that test the corrected pipeline:

| Config | Strategy | Purpose |
|--------|----------|---------|
| **U** | Vector → Normalized Reranker (α=0) | Clean pure-reranker with normalization (replaces R) |
| **V** | Vector → Normalized Logit Blend (α=0.5) | Clean blend (replaces Q) |
| **W** | Vector → Normalized Logit Blend (α=0.3) + Conditional | Clean conditional (replaces S) |

These provide an apples-to-apples comparison where the ONLY variable is alpha.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Behavioral change in R (α=0) | Low | Medium | R already goes through `reranker.rerank()`. Adding `alphaOverride` doesn't change the default path. |
| Regression in production recall | None | — | Production `recall.ts` already uses `reranker.rerank()` exclusively. This fix only changes the benchmark script. |
| `alphaOverride` conflicts with config alpha | Low | Low | Override takes precedence. Document clearly. |
| Telemetry volume increase | Certain | Low | Q/S/T will now produce 300 telemetry lines each (like R does). Acceptable — this is the correct behavior. |

---

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `src/rerank/reranker.ts` | Add `options?: { alphaOverride?: number }` to interface + impl. Add error logging to fallback. | ~15 |
| `scripts/run-beir-bench.ts` | Delete lines 569–596 (direct fetch). Replace with single `reranker.rerank()` call. Remove `rerankCfg` param. | ~-35 net |

**Total diff:** ~50 lines changed, ~35 lines deleted. Net reduction.

---

## Execution Order

1. Create branch `fix/phase10b-unified-reranker`
2. Fix 1: `reranker.ts` — add `alphaOverride` + error logging (commit: `fix(reranker): add per-call alphaOverride for benchmark flexibility`)
3. Fix 2+3: `run-beir-bench.ts` — delete duplicate path + remove `rerankCfg` (commit: `fix(bench): unify reranker path — delete duplicate fetch logic`)
4. Step 1 validation (smoke test)
5. Step 2 validation (full Q/R/S/T benchmark)
6. Step 3 validation (A/H regression check)
7. Commit results + write Phase 10B results summary
8. If reranker now provides positive lift: merge to main
9. If reranker still hurts: investigate further (reranker model may be fundamentally unsuited for SciFact-style claims even with normalization)

---

## Success Criteria

- [ ] ALL reranker calls go through `reranker.rerank()` — zero direct `fetch()` in benchmark
- [ ] ALL configs produce telemetry lines (logit spread, normalization status)
- [ ] Q and R produce comparable results (eliminating the normalization confound)
- [ ] Config A (vector-only) unchanged at 0.7709 ± 0.005
- [ ] At least one blended config achieves NDCG@10 ≥ 0.77 (matches or beats vector-only)

---

## Open Questions (Post-Implementation)

1. **Is the Nemotron reranker actually helping?** Once normalization is fixed, we get clean data. If the reranker STILL hurts with properly normalized queries, the model may simply not generalize to SciFact-style academic claims — even in interrogative form. That's a model-selection decision, not a code bug.

2. **Should `blendAlpha` be configurable per-pool?** Agent memory pools (mistakes, shared knowledge) have different retrieval characteristics. A 0.5 alpha might be optimal for Q&A but harmful for declarative knowledge entries.

3. **Multi-dataset validation?** SciFact is one benchmark. FiQA (financial Q&A) uses naturally interrogative queries — the normalization bug wouldn't affect it. Running FiQA would show the reranker's true capability on its native input format.
