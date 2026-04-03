# Phase 13 — Pipeline Hardening Plan

**Authored:** 2026-04-03  
**Status:** Ready (work deferred to next session per Klein)  
**Tracked:** oc-task `b770298b`

This plan documents real bugs and improvement opportunities identified via deep code audit of the production pipeline. All items are ordered by severity/impact. No speculative improvements — every issue has a root cause traced to source code.

---

## Priority 1 — Critical Bugs (affect production quality)

### P1-A: Score Clamping Poisons the Reranker Gate

**File:** `src/auto/recall.ts` line 1019  
**Code:** `r.score = Math.min(1.0, r.score * weight)`

**Root cause:**  
Source weighting applies a 1.6× multiplier to mistake chunks and 1.5× to captures. Any chunk with a base cosine similarity ≥ 0.625 gets clamped to `score = 1.0`. In a typical recall with 2+ relevant mistake chunks, the top-5 vector score spread becomes 0.0 (`1.0 - 1.0 = 0`), which triggers the gate's low-threshold branch: "tied set → skip reranker." But these ARE different chunks that the reranker could meaningfully rank. The gate fires incorrectly and the reranker never sees them.

**Impact:** Reranker is bypassed exactly when it would help most — on high-relevance queries where multiple mistake/capture chunks compete.

**Fix:** Move source weighting to a **post-rerank additive boost** rather than a pre-rerank multiplicative score modifier. The reranker sees raw cosine similarities (real signal), then after reranking we apply source weights to the final ranking. Replace `Math.min(1.0, score * weight)` with additive injection of a small rank bonus (`score + (weight - 1.0) * 0.1`), OR apply weighting only after the reranker returns.

```typescript
// Current (broken): inflates scores pre-gate, corrupts gate spread
r.score = Math.min(1.0, r.score * weight);

// Fix option A: additive post-rerank boost (best)
// Apply AFTER reranker.rerank() returns, not before

// Fix option B: unclamped pre-rerank (at least preserves spread)
r.score = r.score * weight; // no Math.min — let scores exceed 1.0
// Then normalize top score to 1.0 after weighting so gate spread is preserved
```

---

### P1-B: Position Preservation Guarantee (blending score-floor)

**File:** `src/rerank/reranker.ts` — `blendByRank()`  
**Root cause:** When a doc is weakly ranked by both vector (#3, score 0.23) and reranker (low logit), RRF blend places it outside the top-10 cutoff. Doc disappears. NDCG → 0.0.

**Impact:** Confirmed via BEIR audit — 34 SciFact queries hurt by Config U. The "helped" count (49) outweighs this, but the hurt cases are catastrophic zeros.

**Fix options (implement and benchmark all three):**
- **GUARD-A (Position clamp):** After blending, ensure any doc that ranked top-K_vec (K=5) in the original vector results has a guaranteed minimum blend score equal to the K+1th position. Simplest, safest.
- **GUARD-B (Additive rank bonus):** Blend score += `(K - vec_rank) * epsilon` for docs in top-K_vec. Soft guarantee that degrades gracefully.
- **GUARD-C (Two-pass):** Blend full pool, then inject top-K_vec results as "survivors" if they fell out of top-10, displacing the 10th-11th place blend results.

**Benchmark:** Add GUARD-A/B/C as new configs in `run-beir-bench.ts`, measure NDCG impact on SciFact.

---

### P1-C: HyDE Default ON with 15s Timeout

**File:** `src/config.ts` line ~555  
**Code:** `hyde: { enabled: true, timeoutMs: 15000 }`

**Root cause:** HyDE fires for every single recall query, blocking for up to 15 seconds if the Spark LLM is slow or cold. During benchmarks, HyDE failed 100% of the time (timeouts), adding 15s of wasted latency per query.

**Impact:** Adds 15s worst-case latency on every agent turn. In production, agent turns are expected to complete in <2s.

**Fix:** 
1. Default `hyde.enabled: false` in `buildDefaults()`.
2. Reduce timeout to 3–5s for when it IS enabled.
3. Add a config warning: "HyDE requires Nemotron-Super on port 18080. Disable if not available."

---

## Priority 2 — Architectural Improvements

### P2-A: MAX_RERANK_CANDIDATES=30 May Silently Drop Relevant Chunks

**File:** `src/rerank/reranker.ts` line ~57  
**Code:** `const MAX_RERANK_CANDIDATES = 30`

**Analysis:** With 5 pools × up to 8 results each after dedup = ~25-35 candidates typically. The 30-limit is occasionally binding (drops 5-10 candidates). The dropped candidates are ranked 31-35 by post-weighting score — they could be relevant.

**Fix:** Raise to 40 or make configurable via `rerank.topN` (already exists in config at 20 but isn't respected in this path). Cross-check: reranker latency scales linearly with candidate count. At p95 119ms for 30 docs, 40 docs ≈ 160ms. Acceptable.

```typescript
const MAX_RERANK_CANDIDATES = cfg.topN ?? 40;
```

---

### P2-B: Source Weighting Applied Pre-Dedup Biases Jaccard Dedup

**File:** `src/auto/recall.ts` — ordering of pipeline stages  
**Current order:** `applySourceWeighting → applyTemporalDecay → deduplicateSources → reranker.rerank`

**Problem:** `deduplicateSources` groups by path/parent and keeps the highest-scoring chunk. After source weighting, mistake chunks are inflated to 1.0 even if they are low-quality duplicates. The dedup keeps the "highest score" which is now 1.0 for everything boosted, so it arbitrarily keeps whichever was processed first.

**Fix:** Run dedup on raw cosine scores (before weighting), then apply weighting after dedup. New order: `deduplicateSources → applySourceWeighting → applyTemporalDecay → reranker.rerank`.

---

### P2-C: Gate Computes Spread on Post-Weighting Scores

**File:** `src/auto/recall.ts` — reranker call at line ~275  
**Problem:** The gate spread is computed inside `sparkReranker.rerank()` using the scores of incoming candidates — but those scores have already been inflated by source weighting. See P1-A. Once P1-A is fixed (weighting moved post-rerank), this resolves automatically.

**Dependency:** Fix P1-A first.

---

### P2-D: `recall.ts` Function Cognitive Complexity = 80

**File:** `src/auto/recall.ts` line 36  
**ESLint:** `sonarjs/cognitive-complexity` warning — 80 vs 25 allowed

The main `recallHandler` function is a monolithic 400+ line function. It's correct but unmaintainable. Break into:
- `buildQueryVectors()` — HyDE + multi-query expansion
- `searchAllPools()` — pool search orchestration
- `mergeAndWeight()` — source weighting, temporal decay, dedup
- `rankAndFilter()` — reranker + MMR + parent expansion
- `budgetAndFormat()` — dedup against context, token budget, security filter

Each is independently testable. This also unblocks proper unit tests for the reranker gate interaction.

---

## Priority 3 — Tuning / Polish

### P3-A: queryMessageCount=2 May Miss Recent Context

**Default:** `queryMessageCount: 2`  
**Issue:** Only the last 2 messages are used as the recall query. In a multi-turn conversation with 3 messages of relevant context, the 3rd back is ignored.  
**Fix:** Bump default to 3. Risk: slightly larger query vector, more noise. Easy to tune back.

### P3-B: npm audit: 220 Vulnerabilities (10 Critical)

Run `npm audit fix --force` and review the breaking changes. The critical ones are likely in transitive deps (LanceDB's arrow dependencies). Pin known-safe versions.

### P3-C: Integration Tests Always Fail in Dev (Spark ECONNREFUSED)

Integration tests ping live Spark at `10.99.1.1` — which doesn't exist in dev. They're skipped via `SKIP_INTEGRATION=1` in CI, but running `npm test` locally always shows 4 failures. Add a Spark health check at test start that skips the suite gracefully when Spark is unreachable instead of failing with ECONNREFUSED.

### P3-D: Documentation Coverage Badge Mismatch

README says "Coverage 91%" but vitest thresholds are at 35%. Update badge to reflect actual measured coverage (run `npm run test -- --coverage` to get real number).

---

## Summary Table

| ID | Severity | File | Issue | Fix Complexity |
|----|----------|------|-------|----------------|
| P1-A | 🔴 Critical | `recall.ts:1019` | Score clamping kills gate signal | Medium |
| P1-B | 🔴 Critical | `reranker.ts` | No position preservation (Phase 13) | Medium |
| P1-C | 🔴 Critical | `config.ts` | HyDE on by default, 15s timeout | Trivial |
| P2-A | 🟠 High | `reranker.ts:57` | MAX_RERANK_CANDIDATES=30 too low | Trivial |
| P2-B | 🟠 High | `recall.ts` | Dedup runs after weighting (wrong order) | Simple |
| P2-C | 🟠 High | `recall.ts` | Gate spread polluted by pre-weighting | Blocked by P1-A |
| P2-D | 🟡 Medium | `recall.ts:36` | Cognitive complexity 80 — unmaintainable | Large refactor |
| P3-A | 🟢 Low | `config.ts` | queryMessageCount=2, could be 3 | Trivial |
| P3-B | 🟢 Low | `package.json` | 220 npm vulnerabilities | Run audit fix |
| P3-C | 🟢 Low | `integration.test.ts` | ECONNREFUSED failures in dev | Simple |
| P3-D | 🟢 Low | `README.md` | Coverage badge mismatch | Trivial |

## Recommended Session Order

1. P1-C first (trivial, immediate latency win in production)
2. P1-A + P2-B together (fixing score clamping requires reordering pipeline stages anyway)
3. P1-B (Phase 13 position preservation — benchmark GUARD-A/B/C)
4. P2-A (raise reranker candidate limit — 2 line change)
5. P2-D (refactor — valuable for long-term maintenance, not a correctness fix)
6. P3-x cleanup in parallel (low risk, do at end)
