# Phase 12: Pipeline Overhaul — Fix the Plumbing, Not the Models

**Status:** Planning → Ready for approval  
**Date:** 2026-04-01 (revised)  
**Author:** meta  
**Context:** 28-config BEIR SciFact benchmark complete. Forensic telemetry analysis on 300 queries.

## Core Principle

> The models are solid. The plumbing loses their signal.

- `llama-embed-nemotron-8b` produces excellent vectors (0.7709 NDCG@10 standalone)
- `llama-nemotron-rerank-1b-v2` is a strong cross-encoder — but our code mangles its output
- Every "advanced" pipeline stage (reranker, MMR, HyDE, MQ) either hurts or is neutral because of code bugs, not model quality

## Evidence Base (from 300-query telemetry)

| Finding | Data |
|---------|------|
| `recoverLogit` is a no-op | M=T, N=Q — identical metrics. Monotonic transform + min-max = same ranking. |
| Reranker is net-negative | 13 queries lose relevant docs from top-10, 9 gain. Net: -4 queries. |
| 299/300 top-1 scores = 1.0 | Min-max normalization destroys all discrimination — best always → 1.0, worst → 0.0 |
| Reranker reorders 297/300 | The model IS working. Our normalization + blending code throws away its signal. |
| Tight vector clusters | On loser queries, vector scores span only ~0.05 (e.g., 0.226–0.277). Reranker gambles on tied sets. |
| Blending at α=0.5 limits damage | Config N anchors to vector order, preventing catastrophic reranker reorders. |

## Current Standings

| Config | Strategy | NDCG@10 | vs A | Notes |
|--------|----------|---------|------|-------|
| **U** | Logit Blend α=0.4 | **0.7889** | +2.3% | Current best (but logit recovery is no-op — same as standard blend) |
| A | Vector-Only | 0.7709 | — | Strong baseline |
| N | Blended Reranker α=0.5 | 0.7863 | +2.0% | Identical to Q (logit version) |
| K | Vector → Adaptive MMR | 0.7622 | -1.1% | MMR mostly neutral, occasionally harmful |
| R | Pure Reranker α=0 | 0.7395 | -4.1% | Reranker active but net-negative due to normalization |
| MQ-A | Multi-Query Vector | 0.7609 | -1.3% | Dilution from off-target reformulations |

---

## Fix 1: Kill Min-Max Normalization (CRITICAL)

### The Actual Bug

Min-max normalization is the root cause of the reranker underperformance. Here's why:

1. Reranker outputs sigmoid scores in range [0.83, 1.0] — compressed but **ordered correctly**
2. `recoverLogit()` maps these to logits [~1.6, ~16.0] — still **ordered correctly** (monotonic)
3. `minMaxNormalize()` maps to [0.0, 1.0] — still **ordered correctly** but now **all information about score magnitude is gone**

The problem: min-max normalization after ANY monotonic transform preserves only **rank order**, not **confidence gaps**. A reranker that assigns 0.999 to doc A and 0.835 to doc B (massive confidence gap) gets the same treatment as 0.999 to doc A and 0.998 to doc B (basically a tie).

When we then blend with vector scores using α-interpolation:
```
blended = α × normVector + (1-α) × normReranker
```
The reranker's signal has been flattened to uniform spacing. Its strong "this doc is much better" signals are indistinguishable from its weak "these are about the same" signals.

### The Fix: Rank-Based Fusion (RRF) for Blending

Replace score-based α-blending with Reciprocal Rank Fusion between vector and reranker orderings:

```typescript
function blendByRank(
  vectorPool: SearchResult[],         // sorted by vector score
  rerankResults: RerankResult[],      // sorted by reranker score
  k: number = 60,                     // RRF constant
  vectorWeight: number = 1.0,
  rerankerWeight: number = 1.0,
): SearchResult[] {
  const rrfScores = new Map<string, number>();

  // Vector rank contribution
  vectorPool.forEach((r, i) => {
    const id = r.chunk.id;
    rrfScores.set(id, (rrfScores.get(id) ?? 0) + vectorWeight / (k + i + 1));
  });

  // Reranker rank contribution
  rerankResults.forEach((r, i) => {
    const id = vectorPool[r.index]!.chunk.id;
    rrfScores.set(id, (rrfScores.get(id) ?? 0) + rerankerWeight / (k + i + 1));
  });

  // Sort by RRF score
  return [...rrfScores.entries()]
    .sort(([, a], [, b]) => b - a)
    .map(([id]) => vectorPool.find(r => r.chunk.id === id)!)
    .filter(Boolean);
}
```

**Why RRF instead of score-based blending:**
- Scale-invariant: doesn't matter if reranker scores are 0.83–1.0 or 0.0–100.0
- No normalization needed: works on rank positions directly
- Proven: standard in IR literature for fusing heterogeneous rankers
- The `k` constant controls how much top ranks dominate (k=60 is standard)
- Vector/reranker weights replace the α parameter

**Why NOT percentile/z-score normalization:**
- Percentile normalization is equivalent to rank-based fusion (same information content)
- Z-score requires assumptions about score distributions that don't hold across queries
- RRF is simpler, better understood, and has well-studied hyperparameters

### Cleanup
- Remove `recoverLogit()` — correct math, but useless in practice
- Remove `minMaxNormalize()` from the reranker blend path
- Keep `minMaxNormalize()` for other uses (FTS sigmoid normalization, etc.)

### New Configs
| Config | Strategy |
|--------|----------|
| RRF-A | Vector ⊕ Reranker via RRF (k=60, equal weight) |
| RRF-B | Vector ⊕ Reranker via RRF (k=60, vector=1.5, reranker=1.0) |
| RRF-C | Vector ⊕ Reranker via RRF (k=60, vector=1.0, reranker=1.5) |
| RRF-D | Vector ⊕ Reranker via RRF (k=20, equal weight) — sharper top-rank focus |

### Success Criteria
- RRF-A should beat Config U (0.7889) without any logit tricks
- Pure reranker (still Config R) stays the same — this fix is about the BLEND, not the reranker itself

---

## Fix 2: Dynamic Reranker Gate (HIGH)

### The Bug

The reranker reorders the top-5 in 297/300 queries. On 13 of those, it pushes relevant docs OUT of the top-10. The pattern:

- Vector scores are tightly clustered (spread < 0.05)
- Reranker reshuffles what is essentially a tied set
- When it picks wrong, relevant docs fall off the cliff

The reranker has no way to say "I'm not confident here, keep the original order." It ALWAYS reorders.

### The Fix: Spread-Aware Reranker Gate

```typescript
function shouldRerank(candidates: SearchResult[], threshold: number = 0.08): boolean {
  if (candidates.length < 2) return false;

  const top5 = candidates.slice(0, 5).map(c => c.score);
  const spread = Math.max(...top5) - Math.min(...top5);

  // If vector scores are tightly clustered, the embedding model couldn't 
  // distinguish these docs. The reranker MIGHT help — but it's a gamble.
  // If there's a clear vector winner (high spread), trust the embedding.
  if (spread > threshold) {
    // Vector is confident — reranker is more likely to mess this up
    return false;
  }

  // Tight cluster — reranker's cross-attention might break the tie
  return true;
}
```

**Key insight:** This is the OPPOSITE of the current `conditionalRerank` logic (Config O/S), which skips reranking when spread IS high. That was correct intuition — when vector is confident, don't let the reranker override it. But the threshold (0.15) was set wrong and the implementation was checking the wrong direction on some code paths.

### Refinement: Soft Gate
Instead of hard skip, use the spread to set the blend weight dynamically:
```typescript
const spread = computeSpread(candidates);
const dynamicVectorWeight = Math.min(1.0, spread / 0.15);  // 0→0, 0.15→1.0
// High spread → high vector weight → trust vector
// Low spread → low vector weight → trust reranker
```

### New Configs
| Config | Strategy |
|--------|----------|
| GATE-A | RRF blend + hard gate (skip reranker if spread > 0.08) |
| GATE-B | RRF blend + soft gate (dynamic vector weight from spread) |
| GATE-C | RRF blend + soft gate (threshold=0.12) |

### Success Criteria
- The 13 "loser" queries should no longer lose relevant docs
- The 9 "winner" queries should still gain their relevant docs (soft gate preserves this)
- Net improvement: Config GATE-B > RRF-A > Config U

---

## Fix 3: Dynamic MMR Gate (MEDIUM)

### The Bug
MMR is either ON or OFF per config. On SciFact:
- Average pairwise cosine similarity of top-5: 0.497
- MMR at λ=0.9 displaces relevant docs that happen to be topically similar
- Scientific papers about the same claim ARE supposed to be similar — diversity hurts here

### The Fix: Query-Adaptive MMR

```typescript
function shouldApplyMMR(candidates: SearchResult[]): { apply: boolean; lambda: number } {
  const top5vecs = candidates.slice(0, 5).map(c => c.chunk.vector);
  const avgPairwiseSim = computeAvgPairwiseCosine(top5vecs);

  if (avgPairwiseSim > 0.85) {
    // Near-duplicate results — diversity IS needed
    return { apply: true, lambda: 0.7 };
  }

  if (avgPairwiseSim > 0.70) {
    // Moderately similar — light diversity
    return { apply: true, lambda: 0.9 };
  }

  // Results are already diverse enough — skip MMR
  return { apply: false, lambda: 1.0 };
}
```

**Requirements:**
- Must use actual vectors (Arrow `.toArray()` fix from Phase 7 is prerequisite)
- Gate function is cheap: 10 pairwise cosine calculations on 5 vectors
- Log the decision + avgPairwiseSim for tuning

### New Configs
| Config | Strategy |
|--------|----------|
| DMMR-A | RRF blend + dynamic MMR (thresholds: 0.85/0.70) |
| DMMR-B | RRF blend + soft reranker gate + dynamic MMR |

### Success Criteria
- On SciFact: MMR should trigger on <10% of queries (most are already diverse)
- On NFCorpus (if we test it): MMR should trigger more often (broader topic queries)
- DMMR never hurts NDCG vs the non-MMR equivalent

---

## Fix 4: Selective Multi-Query Expansion (MEDIUM)

### The Bug
MQ-A unions 3 reformulations with the original query, flooding the candidate pool with off-target results. NDCG drops from 0.7709 → 0.7609.

### The Fix: Confidence-Gated Expansion

```typescript
async function expandIfNeeded(
  query: string,
  vectorResults: SearchResult[],
  confidenceThreshold: number = 0.35,
): Promise<SearchResult[]> {
  // If vector already has a clear winner, don't expand
  if (vectorResults.length > 0 && vectorResults[0]!.score >= confidenceThreshold) {
    return vectorResults;
  }

  // Low confidence — try reformulations to rescue recall
  const reformulations = await expandQuery(query, config);
  const extraVectors = await Promise.all(
    reformulations.slice(0, 2).map(r => embed.embedQuery(r))
  );

  for (const vec of extraVectors) {
    const extraResults = await backend.vectorSearch(vec, { maxResults: 10, ... });
    // Only add docs NOT already in results AND above minimum threshold
    for (const r of extraResults) {
      if (!vectorResults.some(v => v.chunk.id === r.chunk.id) && r.score > 0.15) {
        vectorResults.push(r);
      }
    }
  }

  return vectorResults.sort((a, b) => b.score - a.score);
}
```

**Key change:** Original query results are NEVER displaced. Reformulations only ADD candidates to the pool for the reranker to evaluate.

### New Configs
| Config | Strategy |
|--------|----------|
| SMQ-A | Selective MQ (threshold=0.35, max 2 reformulations) |
| SMQ-B | Selective MQ (threshold=0.25, max 1 reformulation) |

### Success Criteria
- Recall@40 improves over baseline (more candidates for reranker)
- NDCG@10 does NOT decrease vs baseline (no dilution)
- MQ triggers on <30% of SciFact queries (most have confident vectors)

---

## Fix 5: Conditional HyDE (LOW)

### The Bug
HyDE worsens top-1 in 72% of cases where it changes the result. SciFact queries are already document-like claims — the LLM just paraphrases them with hallucinated details.

### The Fix
Same confidence gate as Fix 4, but with additive merging:
```typescript
if (vectorResults[0]!.score < 0.30) {
  // Very low confidence — vocabulary mismatch likely
  const hydeDoc = await generateHypothetical(query);
  const hydeVec = await embed.embedDocument(hydeDoc); // no prefix → document space
  const hydeResults = await backend.vectorSearch(hydeVec, { maxResults: 10 });
  // Merge additively — don't replace
  vectorResults = dedup_merge(vectorResults, hydeResults);
}
```

**Also:** Swap Nemotron-Super (10.5s) → Nemotron-Nano or Qwen3-4B (<1s) for the generation model. HyDE docs don't need to be perfect — just close enough to the right vector neighborhood.

### Success Criteria
- HyDE triggers on <15% of queries
- When it triggers, it improves Recall@40
- Latency overhead < 2s (fast model)

---

## Implementation Order

| Priority | Fix | Estimated Impact | Effort |
|----------|-----|-----------------|--------|
| 1 | **RRF Blending** (Fix 1) | +3-5% NDCG over current | ~2h code, ~1h bench |
| 2 | **Dynamic Reranker Gate** (Fix 2) | Eliminates 13 loser queries | ~1h code, ~1h bench |
| 3 | **Dynamic MMR Gate** (Fix 3) | Prevents MMR harm | ~1h code, ~30m bench |
| 4 | **Selective MQ** (Fix 4) | Recall improvement without dilution | ~1h code, ~1h bench |
| 5 | **Conditional HyDE** (Fix 5) | Edge case rescue | ~30m code, ~30m bench |

Total: ~10h work, sequential (each fix builds on the previous).

## Benchmark Plan

Run in phases, not all 28+ configs at once:

**Phase 12A:** Fixes 1+2 (RRF + Gate)
- Configs: RRF-A through RRF-D, GATE-A through GATE-C
- Dataset: SciFact 300
- Expected runtime: ~45 min

**Phase 12B:** Fix 3 (Dynamic MMR) on top of best from 12A
- Configs: DMMR-A, DMMR-B
- Dataset: SciFact 300
- Expected runtime: ~15 min

**Phase 12C:** Fixes 4+5 (Selective MQ + HyDE) on top of best from 12B
- Configs: SMQ-A, SMQ-B + HyDE variants
- Dataset: SciFact 300
- Expected runtime: ~30 min

**Phase 12D:** Cross-dataset validation
- Best config from 12C tested on NFCorpus + FiQA
- Validates generalization beyond SciFact

## Task Cleanup

| Task | Action |
|------|--------|
| `recoverLogit` code | Remove from blend path (keep in codebase as reference, add `@deprecated` JSDoc) |
| `minMaxNormalize` in reranker | Remove from blend path only |
| `f50a05e8` (14 audit findings) | Close — most fixes landed in Phase 7-11 |
| `11cbd574` (stuck session) | Close — obsolete |
| `0696904e` (audit results) | Close — superseded by this analysis |
| `6b36833e` (MQ expansion) | Close — code done, folded into Fix 4 |
| `8999599f` (HyDE fast model) | Close — folded into Fix 5 |

## Success Criteria (Overall)

| Metric | Current Best | Target |
|--------|-------------|--------|
| NDCG@10 (SciFact) | 0.7889 (U) | > 0.82 |
| Pure Reranker usefulness | Net -4 queries | Net positive |
| Recall@10 | 0.9143 (N) | > 0.93 |
| p95 Latency | ~1500ms | < 1600ms (with MQ/HyDE gating, should be lower) |
| Harmful stage count | MMR + HyDE + MQ all hurt | No stage hurts; each neutral or positive |
