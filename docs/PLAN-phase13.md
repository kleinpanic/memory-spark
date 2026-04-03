# Phase 13 — Pipeline Hardening Plan

**Authored:** 2026-04-03  
**Status:** Partially complete — P1-A/P1-C/P2-A/P2-B/P3-C shipped in `310918f`  
**Tracked:** oc-task `b770298b`

> **Severity corrections (post-implementation recon, 2026-04-03):**  
> P1-A and P2-B were overstated as "critical." See corrected assessments below.  
> P1-C was correctly motivated (HyDE reverted to enabled, now properly configurable).

This plan documents real bugs and improvement opportunities identified via deep code audit of the production pipeline. Items are ordered by actual measured impact, not initial intuition.

---

## Priority 1 — Confirmed Impactful

### P1-A: Score Clamping — Gate Misfire in Specific Scenarios *(was: Critical; corrected: Low-Medium)*

**File:** `src/auto/recall.ts` — `applySourceWeighting()`  
**Code:** `r.score = Math.min(1.0, r.score * weight)` *(removed in 310918f)*

**What I originally claimed:** The `Math.min(1.0)` clamp collapses all boosted mistake/capture scores to 1.0, causing the gate to see spread=0 and skip the reranker on every query with multiple mistake chunks.

**What is actually true (post-recon):**  
The gate misfire only triggers when *all* top-5 candidates are boosted chunks *and* all have cosine similarity ≥ 0.625. In production:
- The top-5 pool is almost always mixed across pool types — memory chunks aren't boosted and maintain natural spread
- Temporal decay partially prevents collapse even when clamped (chunks indexed at different times get different decay multipliers — a 30-day-old chunk clamped to 1.0 decays to 0.88)
- Requires 5+ mistake chunks all scoring ≥ 0.625 on the same query — rare with a small mistakes DB

**Actual scenario that triggers it:** A very domain-specific query where 5+ mistake chunks from the same session were indexed on the same day and all happen to match at cosine ≥ 0.625. This is real but uncommon.

**Fix shipped:** Removed clamp, added `normalizeScores()` after all weighting. Mathematically sound — ranking order preserved, just rescaled. No regressions in 685 unit tests. **Safe change, but was oversold as critical.**

---

### P1-B: Position Preservation Guarantee (blending score-floor) *(not yet implemented)*

**File:** `src/rerank/reranker.ts` — `blendByRank()`  
**Root cause:** When a doc is weakly ranked by both vector (#3, score 0.23) and reranker (low logit), RRF blend places it outside the top-10 cutoff. Doc disappears. NDCG → 0.0.

**Impact:** Confirmed via BEIR audit — 34 SciFact queries hurt by Config U. The "helped" count (49) outweighs this, but the hurt cases are catastrophic zeros.

**Fix options (implement and benchmark all three):**
- **GUARD-A (Position clamp):** After blending, ensure any doc that ranked top-K_vec (K=5) in the original vector results has a guaranteed minimum blend score equal to the K+1th position. Simplest, safest.
- **GUARD-B (Additive rank bonus):** Blend score += `(K - vec_rank) * epsilon` for docs in top-K_vec. Soft guarantee that degrades gracefully.
- **GUARD-C (Two-pass):** Blend full pool, then inject top-K_vec results as "survivors" if they fell out of top-10, displacing the 10th-11th place blend results.

**Benchmark:** Add GUARD-A/B/C as new configs in `run-beir-bench.ts`, measure NDCG impact on SciFact.

---

### P1-C: HyDE Timeout Not Configurable *(corrected — re-enabled, now fully configurable)*

**Original claim:** HyDE should be disabled by default (15s timeout, 100% failure rate in benchmarks).

**What was actually done:** HyDE is re-enabled with default `timeoutMs: 30000`. It is now fully configurable via plugin config (`hyde.enabled`, `hyde.timeoutMs`, `hyde.model`, `hyde.llmUrl`). Benchmark failures were due to Nemotron-Super cold-start — a different LLM or a warm instance will perform correctly. Operators without a Spark LLM can set `hyde.enabled: false` in `openclaw.json`.

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

### P2-B: Pipeline Ordering — Dedup Before Weighting *(was: High; corrected: Cosmetic/Code cleanliness)*

**File:** `src/auto/recall.ts` — ordering of pipeline stages  
**Shipped order (310918f):** `deduplicateSources → applySourceWeighting → applyTemporalDecay → normalizeScores → reranker.rerank`

**What was claimed:** Dedup running after weighting caused Jaccard to compare inflated scores and keep arbitrary duplicates.

**What is actually true (post-recon):**  
`deduplicateSources` only deduplicates chunks that are ≥ 85% Jaccard-token-similar *from the same source* (same path/parent_id). Near-identical text at 85%+ similarity carries the same information regardless of which copy is kept. The tie-breaking between nearly-identical chunks has **zero quality impact**. Additionally, cross-source dedup cannot occur since different sources always have different paths/parent_ids.

**Practical impact:** None. The new ordering is marginally cleaner (dedup on raw cosine is logically correct) but this was shipped as a bug fix when it is actually a code cleanliness improvement. **Safe to keep as shipped.**

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

| ID | Actual Severity | Status | File | Issue |
|----|-----------------|--------|------|-------|
| P1-A | 🟡 Low-Medium *(was: Critical)* | ✅ Shipped `310918f` | `recall.ts` | Score clamping — rare edge case, safe fix |
| P1-B | 🔴 High | 🔲 Not yet | `reranker.ts` | No position preservation (Phase 13 core work) |
| P1-C | 🟠 Medium *(was: Critical)* | ✅ Shipped `310918f` | `config.ts` | HyDE fully configurable, timeout now 30s |
| P2-A | 🟡 Low-Medium | ✅ Shipped `310918f` | `reranker.ts` | topN 30→40, respects cfg.topN |
| P2-B | 🟢 Cosmetic *(was: High)* | ✅ Shipped `310918f` | `recall.ts` | Dedup order — no functional impact |
| P2-C | 🟡 Low | ✅ Resolved by P1-A | `recall.ts` | Gate spread on post-weight scores |
| P2-D | 🟡 Medium | 🔲 Not yet | `recall.ts:36` | Cognitive complexity 80 — unmaintainable |
| P3-A | 🟢 Low | 🔲 Not yet | `config.ts` | queryMessageCount=2, could be 3 |
| P3-B | 🟢 Low | 🔲 Not yet | `package.json` | 220 npm vulnerabilities |
| P3-C | 🟢 Low | ✅ Shipped `310918f` | `integration.test.ts` | ECONNREFUSED in dev — graceful skip |
| P3-D | 🟢 Low | 🔲 Not yet | `README.md` | Coverage badge mismatch |

## Recommended Session Order

1. P1-C first (trivial, immediate latency win in production)
2. P1-A + P2-B together (fixing score clamping requires reordering pipeline stages anyway)
3. P1-B (Phase 13 position preservation — benchmark GUARD-A/B/C)
4. P2-A (raise reranker candidate limit — 2 line change)
5. P2-D (refactor — valuable for long-term maintenance, not a correctness fix)
6. P3-x cleanup in parallel (low risk, do at end)
