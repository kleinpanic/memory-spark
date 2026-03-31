# Phase 4: Fix MMR, Pipeline Ordering, Instruction Prefix, and BM25 Saturation

**Status:** Planning  
**Parent task:** `00889d7b` (BEIR benchmark underperformance recon)  
**Date:** 2026-03-30 (revised after cross-reference research)  
**Research:** `evaluation/research-phase4-crossref.md`  

---

## Problem Statement

The retrieval pipeline has four compounding issues that degrade BEIR benchmark scores:

1. **MMR uses Jaccard similarity** (bag-of-words overlap) instead of cosine on embedding vectors. Penalizes domain vocabulary overlap, not semantic redundancy. Drops ~25% of relevant results on SciFact.
2. **Pipeline ordering is wrong** — MMR runs before the reranker, throwing away candidates the reranker would have rescued.
3. **Nemotron instruction prefix may be missing** — `llama-embed-nemotron-8b` requires `Instruct: ...\nQuery: ` prefix for queries. If `queryInstruction` config is empty, all queries embed without it. Estimated 10-20% NDCG impact.
4. **BM25 sigmoid saturation** — hardcoded midpoint of 3.0 maps all FTS scores to >0.98 for scientific text (raw BM25 scores 5-20+). The `minScore` filter passes everything. RRF ignores these scores (rank-only), but minScore doesn't.

---

## Execution Plan (priority order)

### 4A: Verify and Fix Nemotron Instruction Prefix ⚡ HIGHEST ROI

**Problem:** The `queryInstruction` config field exists in `src/config.ts` but may have been empty or undefined during BEIR benchmarks. If undefined, `embedQuery()` silently skips the instruction prefix, and all queries embed as raw text — wrong subspace for an instruction-aware model.

**Fix:**
1. Verify current config: check if `queryInstruction` is populated in production config
2. Set a sensible default in `src/config.ts`: `"Given a question, retrieve relevant passages that answer it"`
3. Add a startup warning if `queryInstruction` is empty when using an instruction-aware model (Nemotron, E5, INSTRUCTOR)
4. Verify with a test: embed the same query with and without prefix, confirm vectors differ

**Files:**
- `src/config.ts` — default value for `queryInstruction`
- `src/embed/provider.ts` — startup warning if prefix missing

**Risk:** Very low. If the prefix was already working, this is a no-op. If it wasn't, instant 10-20% improvement.

### 4B: Fix BM25 Sigmoid Saturation

**Problem:** `rowToSearchResult()` in the LanceDB backend applies a sigmoid with midpoint 3.0 to BM25 scores. For scientific corpora where raw BM25 scores are 5-20+, this maps everything to >0.98. The `minScore` filter then passes all FTS results regardless of quality.

Note: `hybridMerge()` (RRF) uses **rank position only**, not the sigmoid-normalized scores. So the sigmoid doesn't affect hybrid merging — it only damages the `minScore` filter on FTS results.

**Fix:**
- Remove `minScore` filtering from FTS results before passing to `hybridMerge()`. FTS quality is handled by RRF rank position, not score magnitude.
- Keep the sigmoid for display/debugging purposes if needed, but don't use it for filtering.
- Alternatively: make `sigmoidMidpoint` configurable and corpus-adaptive (compute from actual BM25 score distribution at index time).

**Files:**
- `src/backends/lancedb.ts` — remove or bypass minScore filter for FTS results
- `src/config.ts` — optionally make sigmoidMidpoint configurable

**Risk:** Low. RRF already handles FTS ranking correctly. Removing the minScore filter just stops it from accidentally passing garbage.

### 4C: Replace Jaccard with Cosine Similarity in MMR

**Problem:** `mmrRerank()` uses `jaccardSimilarity(tokenSets[i], tokenSets[sIdx])` — word overlap. Two chunks about different aspects of the same system ("agent config" vs "agent memory") share vocabulary and get penalized even though they contain distinct information.

**Fix:**
1. Add `vector?: number[]` to `SearchResult` (if not already present)
2. Ensure LanceDB search returns embedding vectors alongside results (check if `.select()` includes the vector column)
3. Implement `cosineSimilarity(a: number[], b: number[])` utility
4. Replace Jaccard computation in `mmrRerank()` with cosine on vectors
5. Keep Jaccard as fallback if vectors unavailable (graceful degradation)

**Files:**
- `src/auto/recall.ts` — rewrite MMR similarity computation
- `src/types.ts` — add `vector` field to `SearchResult` if needed
- `src/backends/lancedb.ts` — return vectors from search results

**Risk:** Low. Need to verify vector availability from LanceDB without extra DB roundtrip.

### 4D: Fix Pipeline Ordering (MMR After Reranker)

**Problem:** Current order: `retrieve → merge → weight → decay → MMR → reranker → expand`. MMR runs before the reranker, removing candidates the reranker would have approved.

**Evidence:** Config D (Hybrid + Reranker) and Config G (Full Pipeline = D + MMR) produce identical NDCG (0.5308). This proves MMR after reranker is harmless — the reranker already selects a tight set. The damage happens when MMR runs first.

**Fix:** Reorder to:
```
retrieve → merge → weight → decay → source dedup → reranker → MMR → expand
```

Specifically in `recall.ts`:
```typescript
// Old: mmrRerank → reranker
// New: reranker gets full candidate set, MMR trims reranked output
const reranked = await reranker.rerank(queryText, candidates, cfg.maxResults * 2);
const final = mmrRerank(reranked, cfg.maxResults, lambda);
```

**Files:**
- `src/auto/recall.ts` — reorder the pipeline stages

**Risk:** Low. NVIDIA confirms this ordering. Reranker sees more candidates = better relevance. MMR on reranked output = safe diversity.

**Note:** Reranker latency (41s per query — Issue 7) should be addressed separately. The ordering fix is correct regardless of latency.

### 4E: Make Lambda Configurable (default 0.9)

**Problem:** Lambda hardcoded at 0.7. Too aggressive for factual retrieval.

**Fix:**
- Add `mmrLambda?: number` to `AutoRecallConfig` in config schema
- Default: `0.9` (research confirms: factual retrieval needs 0.85-0.95)
- Valid range: [0.0, 1.0], clamped at boundaries
- Wire through `recall()` → `mmrRerank()`

**Files:**
- `src/config.ts` — add schema field with default
- `src/auto/recall.ts` — read from config

**Risk:** Trivial.

### 4F: Source-Level Deduplication (Pre-Reranker)

**Problem:** Multiple chunks from the same source with overlapping text (different chunk boundaries) waste reranker and MMR slots.

**Fix:**
- New `deduplicateSources(results)` function
- Group by `parentId` or source path
- Within each group: if Jaccard > 0.85 (here token overlap IS the right metric — checking for near-identical text), keep the higher scorer
- Run before reranker so it sees only distinct chunks

**Files:**
- `src/auto/recall.ts` — new function, called before reranker in the pipeline

**Risk:** Low. Only drops genuinely redundant chunks from the same source.

### 4G: Tests

**New tests:**
- Cosine-based MMR keeps relevant-but-different chunks (same domain, different content)
- Cosine-based MMR drops near-duplicate chunks (high cosine similarity)
- Lambda=1.0 = pure relevance ordering
- Lambda=0.5 = strong diversity bias
- Source dedup: collapses overlapping chunks from same parent
- Source dedup: preserves chunks from different sources
- Pipeline ordering: verify MMR runs after reranker in the call chain
- Instruction prefix: verify embedQuery applies prefix, embedDocument does not
- BM25 minScore: FTS results pass to RRF regardless of score

**Files:**
- `tests/unit.test.ts` — MMR, dedup, and config tests
- `tests/hyde.test.ts` — if HyDE-related tests needed

### 4H: Comprehensive Test Suite + Local CI/CD

**Goal:** Add a thorough test suite covering all Phase 4 changes, then run the full CI/CD pipeline locally to validate everything before any benchmarking.

**New test coverage:**
- **Instruction prefix integration test:** Full round-trip — query with prefix embeds differently than without
- **BM25 filtering test:** FTS results pass through to RRF regardless of sigmoid score
- **Cosine MMR correctness:** Given known vectors, verify MMR selects diverse-but-relevant over redundant
- **Pipeline ordering test:** Assert call order in recall pipeline (reranker before MMR)
- **Source dedup edge cases:** Same-source overlap, cross-source vocabulary overlap (should NOT dedup), single-chunk groups
- **Lambda boundary tests:** 0.0, 0.5, 0.9, 1.0 produce expected rank orderings
- **Regression tests:** Existing pipeline behavior preserved where not explicitly changed

**Local CI/CD run:**
- `npx vitest run` — all unit + integration tests pass
- `npx tsc --noEmit` — no type errors
- Lint pass if configured

**Note:** Full BEIR re-benchmarking happens AFTER all 8 phases are complete, not here. This step validates code correctness only.

---

## Execution Order

```
4A (instruction prefix) → 4B (BM25 sigmoid) → 4C (cosine MMR) → 4D (pipeline ordering) → 4E (lambda config) → 4F (source dedup) → 4G (tests) → 4H (benchmarks)
```

4A and 4B are the highest-ROI fixes and independent of MMR changes.  
4C and 4D are the core MMR fixes.  
4E and 4F are configuration and polish.  
4G validates everything. 4H runs the full test suite + local CI to confirm correctness before any benchmarking.

---

## Success Criteria

- [ ] Instruction prefix verified active for Nemotron embeddings
- [ ] BM25 minScore filter no longer blocks FTS results before RRF
- [ ] MMR uses cosine similarity on embedding vectors
- [ ] Pipeline order: retrieve → merge → weight → decay → dedup → rerank → MMR → expand
- [ ] Lambda configurable via `mmrLambda`, default 0.9
- [ ] Source-level dedup collapses near-identical chunks before reranker
- [ ] All existing tests pass + new Phase 4 tests pass
- [ ] Full test suite passes locally (`vitest run` + `tsc --noEmit`)
- [ ] No hardcoded lambda values in production code
- [ ] BEIR re-benchmark deferred to post-Phase-8 (all phases complete first)

---

## Deferred to Future

- **Dynamic MMR lambda** — moved to `future-improvements.md`. No production systems use this. Need empirical ablation data (lambda sweep across BEIR datasets) before implementing. Proposed range if implemented: 0.85–0.95 (±0.05 from baseline 0.9).
- **Query decomposition** — Phase 5 candidate
- **Query rewriting** — Phase 5 candidate
- **Self-reflection** — Phase 5 candidate
- **GPU-accelerated ANN (cuVS)** — Phase 5+ candidate

---

## References

- Carbonell & Goldstein (1998). "The Use of MMR, Diversity-Based Reranking." — Original MMR paper
- Gao et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels." — HyDE
- Su et al. (2023). "One Embedder, Any Task: Instruction-Finetuned Text Embeddings." — INSTRUCTOR
- NVIDIA RAG Blueprint (2025): https://github.com/NVIDIA-AI-Blueprints/rag
- Full cross-reference report: `evaluation/research-phase4-crossref.md`
