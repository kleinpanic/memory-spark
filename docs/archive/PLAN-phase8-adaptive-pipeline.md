# Phase 8: Adaptive Pipeline Architecture

## Problem Statement

Vector-only retrieval achieves 0.77 NDCG@10 on SciFact. Every pipeline stage we add
(RRF hybrid, reranker, MMR) makes it *worse*. This is not because those stages are
unnecessary — it's because their implementations are static when they need to be dynamic.

The diagnostic (50 SciFact queries) revealed:

| Stage | NDCG@10 | Δ vs Vector |
|-------|---------|-------------|
| Vector-Only | 0.8882 | baseline |
| Hybrid (RRF) | 0.8456 | -4.3% |
| Hybrid + Reranker | 0.8268 | -6.1% |
| Hybrid + MMR | 0.8382 | -5.0% |

## Root Causes

### 1. RRF Rank-Washout (the poison pill)
RRF assumes both retrieval systems return comparable quality results with significant
overlap. When overlap is low (mean 5.7/10), RRF promotes irrelevant FTS-only docs
because `1/(k+rank_fts)` competes directly with `1/(k+rank_vec)` regardless of
absolute quality.

**The bug is not RRF itself — it's that RRF is overlap-unaware.**

### 2. Reranker on Polluted Pool
The reranker receives RRF output (already contaminated). A cross-encoder CAN fix
ranking mistakes, but when 40%+ of its input is junk, it wastes discrimination
capacity sorting irrelevant docs instead of fine-tuning relevant ones.

**The bug is not the reranker — it's the input pipeline order.**

### 3. Static MMR Lambda
λ=0.9 is hardcoded. For queries where vector already found the perfect answer
(tight score cluster, one dominant result), ANY diversity penalty hurts. For
exploratory queries with flat score distributions, diversity genuinely helps.

**The bug is not MMR — it's that lambda doesn't adapt to the query.**

### 4. Cascade Amplification
Each broken stage amplifies the previous one's errors. This is a pipeline ordering
and gating problem, not a component problem.

---

## The Fix: Adaptive Pipeline with Dynamic Gating

### Fix 1: Overlap-Aware Hybrid Fusion (replace static RRF)

**Concept:** Measure vector↔FTS agreement at query time and adapt fusion accordingly.

```
overlap_ratio = |vector_top_K ∩ fts_top_K| / K

if overlap_ratio > 0.6:
  # High agreement — both systems see the same thing
  # Standard RRF works well here, dual-evidence docs get proper boost
  → RRF with equal weights (current behavior)

elif overlap_ratio > 0.3:
  # Medium agreement — vector is primary, FTS contributes selectively
  # Only promote FTS docs that ALSO appear in vector results (dual evidence)
  # FTS-only docs get demoted weight
  → Weighted RRF: vectorWeight=2.0, ftsWeight=0.5

else:
  # Low agreement — systems are seeing different things
  # Vector is almost certainly more reliable for semantic queries
  # FTS-only results are likely keyword noise
  → Vector-primary: only include FTS docs with dual evidence
  → FTS-only docs only used as fallback if vector returns < K results
```

**Implementation:**
- Compute overlap before calling hybridMerge()
- Pass overlap_ratio into hybridMerge() 
- hybridMerge() adjusts vectorWeight/ftsWeight dynamically
- Log the overlap ratio for telemetry

**Why this works:** When both systems agree (high overlap), RRF's rank aggregation
is statistically meaningful. When they disagree, trusting the higher-quality signal
(vector with instruction-tuned 4096d Nemotron) is the right call.

### Fix 2: Reranker-as-Fusioner (dual-path architecture)

**Concept:** Instead of Hybrid→Reranker (reranker cleans up RRF mess), give the
reranker the RAW candidate pool and let it do the fusion.

```
Path A (current, broken):
  Vector → FTS → RRF → Reranker
  Problem: Reranker inherits RRF's ranking mistakes

Path B (proposed):
  Vector top-2K ∪ FTS top-2K → Dedupe → Reranker(query, candidate) → Sort
  The cross-encoder scores each (query, doc) pair INDEPENDENTLY
  It doesn't care whether the doc came from vector or FTS
  The reranker IS the fusion step
```

**Implementation:**
- New function: `rerankerFusion(query, vectorResults, ftsResults, reranker, limit)`
- Deduplicates by doc ID (keep higher-scored version)
- Passes full union to reranker.rerank()
- Reranker produces final ranking from scratch
- No RRF involved at all in this path

**Why this works:** Cross-encoders process (query, document) pairs with full
attention — they're fundamentally better at relevance scoring than rank-based
heuristics like RRF. We're using a 1B parameter model; let it do its job.

This is what the NVIDIA RAG Blueprint recommends: retrieve broadly, rerank precisely.

### Fix 3: Adaptive MMR Lambda

**Concept:** Lambda should respond to the score distribution of the current result set.

```
scores = results.map(r => r.score)
spread = max(scores) - min(scores)
entropy = -Σ(p_i * log(p_i))  where p_i = score_i / Σscores

if spread > 0.3:
  # Wide score spread — ranking is confident, one clear winner
  # Diversity would demote the winner — bad
  λ = 0.95 (almost pure relevance)

elif spread > 0.1:
  # Medium spread — ranking has some confidence but results are similar
  # Light diversity helps surface different perspectives
  λ = 0.85

else:
  # Tight cluster — ranker can't distinguish, all results look equivalent
  # Diversity is genuinely useful here — break ties with novelty
  λ = 0.7
```

**Implementation:**
- Compute score spread and/or coefficient of variation before MMR
- Map to lambda via configurable thresholds
- Log the computed lambda for telemetry
- Configurable: `mmrLambda: "adaptive"` vs `mmrLambda: 0.9` (fixed, backward compat)

**Why this works:** When the ranker is confident (wide spread), we trust it.
When it's uncertain (tight cluster), diversity is a tiebreaker. This is exactly
what Klein suggested and I should have listened.

### Fix 4: Pipeline Routing (the dynamic approach)

**Concept:** Not every query needs every stage. Route queries to the optimal
pipeline configuration based on measurable signals.

```
Signals available at query time:
- Query length (short factual vs. long exploratory)
- Vector score confidence (top-1 score, score spread)
- Vector↔FTS overlap ratio
- Corpus metadata (scientific vs. conversational vs. code)

Routing logic:
  if vector_top1_score > 0.7:
    # Vector is very confident — don't mess with it
    → Vector-only (maybe reranker for fine-tuning)
  
  elif overlap_ratio > 0.5:
    # Both systems agree — hybrid adds value
    → Adaptive RRF → Reranker → Adaptive MMR
  
  elif query_is_keyword_heavy:
    # FTS might genuinely help (exact names, IDs, acronyms)
    → Reranker-as-Fusioner (union → cross-encoder)
  
  else:
    # Default: vector-primary with reranker refinement
    → Vector → Reranker → Adaptive MMR
```

**Implementation:**
- New `PipelineRouter` class with configurable thresholds
- Selects pipeline variant per-query
- Logs routing decision for telemetry
- Benchmark with routing telemetry to measure which paths get chosen

---

## New Benchmark Configs

| Config | Pipeline | Purpose |
|--------|----------|---------|
| H | Vector → Reranker (no RRF) | Isolate reranker value on clean input |
| I | Overlap-Aware Hybrid → Reranker | Test adaptive RRF fix |
| J | Reranker-as-Fusioner (union → rerank) | Test cross-encoder fusion |
| K | Vector → Adaptive MMR | Test dynamic lambda in isolation |
| L | Full Adaptive Pipeline (router) | All fixes combined |

## Testing Plan

### Unit Tests
- [ ] Overlap ratio computation (0%, 50%, 100% overlap cases)
- [ ] Adaptive RRF weight selection by overlap tier
- [ ] Adaptive MMR lambda by score spread
- [ ] Reranker-as-Fusioner dedup logic
- [ ] Pipeline router decision logic

### Integration Tests
- [ ] End-to-end SciFact 50-query diagnostic with each new config
- [ ] Regression: Vector-only must still hit 0.77+ NDCG@10
- [ ] Telemetry logging captures all adaptive decisions

### Benchmark Suite
- [ ] Full SciFact run (300 queries × configs A-L)
- [ ] NFCorpus and FiQA runs for generalization
- [ ] Per-query comparison: adaptive vs. static

---

## Implementation Order

1. Fix 1: Overlap-Aware Hybrid → modify `hybridMerge()` in `recall.ts`
2. Fix 3: Adaptive MMR Lambda → modify `mmrRerank()` in `recall.ts`  
3. Fix 2: Reranker-as-Fusioner → new function in `recall.ts`
4. Unit tests for all three
5. New benchmark configs H-L
6. Fix 4: Pipeline Router → new `PipelineRouter` class
7. Full benchmark run
8. CI integration
