# ISSUES.md — memory-spark BEIR Benchmark Root Cause Analysis

> **Date:** 2026-03-30 (last updated 2026-04-12)
> **Benchmark:** BEIR (SciFact, NFCorpus, FiQA partial)
> **Embedding Model:** nvidia/llama-embed-nemotron-8b (4096-dim, instruction-aware)
> **Reranker:** nvidia/llama-nemotron-rerank-1b-v2
> **Vector DB:** LanceDB 0.27+
> **FTS:** LanceDB built-in Tantivy BM25

## Fix Status

| Issue | Status | Notes |
|-------|--------|-------|
| 1. BM25 Sigmoid Saturation | ✅ Fixed | `sigmoidMidpoint: 10.0` in config.ts |
| 2. Missing Instruction Prefix | ✅ Fixed | `queryInstruction` added to config; query format corrected |
| 3. Hybrid Merge Bug in Runner | ✅ Fixed | Vector and FTS results now kept separate until fusion |
| 4. HyDE Never Activated | ✅ Fixed | HyDE fires when `cfg.hyde.enabled: true`; `--hyde` CLI flag available |
| 5. HyDE Vector Averaging | ✅ Fixed | Replaced with Gao et al. (2022) replacement approach |
| 6. MMR Destroys Recall | ✅ Fixed | `useMmr: false` for all benchmark configs |
| 7. Reranker Latency | ⚠️ Spark GPU | 41s p50 — needs GPU-enabled vLLM on Spark |

---

## Results Summary

| Config | Label | SciFact NDCG@10 | NFCorpus NDCG@10 | Notes |
|--------|-------|-----------------|-------------------|-------|
| A | Vector-Only | **0.7451** | 0.1862 | Best overall on SciFact |
| B | FTS-Only | 0.6232 | 0.2729 | Best on NFCorpus |
| C | Hybrid | 0.4737 | 0.2281 | **Worse than either A or B alone** |
| D | Hybrid + Reranker | 0.5308 | 0.2796 | Partially rescues hybrid |
| E | Hybrid + MMR | 0.3634 | 0.1544 | **Worst config** |
| F | Hybrid + HyDE | 0.4737 | 0.2281 | Identical to C (HyDE never ran) |
| G | Full Pipeline | 0.5308 | 0.2796 | Identical to D (MMR no-op) |

**Key observation:** Hybrid (C) is *worse* than vector-only (A) by 36% on SciFact. This is the opposite of what should happen — hybrid should improve or at minimum match the best single signal.

---

## Issue 1: BM25 Sigmoid Score Saturation

**Severity:** 🔴 Critical
**Component:** `src/storage/lancedb.ts` → `rowToSearchResult()` (line ~340)
**Impact:** Completely breaks hybrid merge. FTS signal loses all discriminative power.

### The Problem

BM25 raw scores are normalized to [0,1] via sigmoid:
```
score = 1 / (1 + exp(-(bm25_score - midpoint)))
```
With `midpoint = 3.0` (the default).

BM25 scores for BEIR scientific text corpora range **5–25+**. With midpoint 3.0:

| Raw BM25 | Sigmoid Output | Information Lost |
|----------|---------------|------------------|
| 3.0 | 0.500 | — (midpoint) |
| 5.0 | 0.881 | Low |
| 7.0 | 0.982 | Moderate |
| 10.0 | 0.9991 | Severe |
| 15.0 | 0.999999 | Total |
| 20.0 | ~1.0 | Total |

### Evidence

```
SciFact FTS top-1 score distribution (300 queries):
  min:  0.9834
  max:  1.0000
  >0.99: 299/300 queries (99.7%)
  <0.50: 0/300 queries
```

All FTS results are compressed into the range [0.983, 1.000]. A document with BM25=7 (marginal match) scores 0.982 — nearly identical to BM25=20 (perfect match). **The sigmoid destroys the relative ranking within FTS results.**

### Why This Kills Hybrid

In `hybridMerge()`, vector scores (cosine similarity, range 0.2–0.6 typically) compete against FTS scores (range 0.983–1.000). FTS results always win rank priority, regardless of actual relevance. The hybrid merge becomes "FTS ranking with vector results pushed to the bottom."

### Root Cause

The sigmoid midpoint was calibrated for small personal knowledge bases (short markdown files, agent memory captures) where BM25 scores cluster around 2-4. It was never recalibrated for full document corpora.

### Literature

- BM25 score distributions are corpus-dependent and unbounded. Normalizing them requires corpus-specific calibration or rank-based fusion.
- Robertson & Zaragoza (2009), "The Probabilistic Relevance Framework: BM25 and Beyond" — BM25 is a *ranking* function, not a *scoring* function. Converting raw BM25 to probabilities is non-trivial and corpus-specific.
- Cormack, Clarke & Büttcher (2009) — score normalization across different retrieval models is an open problem; rank-based fusion avoids it entirely.

---

## Issue 2: Missing Instruction Prefix for Nemotron Embeddings

**Severity:** 🔴 Critical
**Component:** `src/embed/provider.ts` → `makeOpenAiCompat()` → `embedQuery()`
**Impact:** Query vectors land in wrong region of embedding space. Degrades all vector retrieval.

### The Problem

`llama-embed-nemotron-8b` is an **instruction-aware** bi-encoder. Per the [model card](https://huggingface.co/nvidia/llama-embed-nemotron-8b) and [paper](https://arxiv.org/abs/2511.07025):

- **Queries** must be formatted: `Instruct: {task_instruction}\nQuery: {query_text}`
- **Documents** are embedded as raw text (no prefix)

The asymmetric formatting is fundamental to the model's architecture. During training, the model learned to project queries and documents into aligned but distinct subspaces. Without the instruction prefix, query vectors fall into the document subspace, reducing the query-document alignment the model was optimized for.

### Evidence

Empirical test on Spark endpoint (port 18091):

```
Input: "what is retrieval augmented generation"
Vector: [0.0133, -0.0002, 0.0051, -0.0171, 0.0090]

Input: "Instruct: Given a web search query, retrieve relevant passages.\nQuery: what is retrieval augmented generation"
Vector: [-0.0103, 0.0022, 0.0177, -0.0183, 0.0148]
```

The vectors are **completely different** — signs flipped, magnitudes changed. This is not a minor perturbation; the model's internal routing is fundamentally different with the instruction prefix.

Additionally: the OpenAI-compatible `input_type: "query"` parameter has **no effect** on the Spark TEI endpoint. It returned the identical vector as raw text. The instruction must be inline.

### Current Code (broken)

```typescript
// src/embed/provider.ts
async embedQuery(text) {
  const results = await embed(text);  // ← raw text, no instruction
  return results[0]!;
}
```

### Literature

- Babakhin et al. (2025), "Llama-Embed-Nemotron-8B: A Universal Text Embedding Model" — The model uses instruction-based task specialization. The SentenceTransformers integration automatically applies `encode_query()` vs `encode_document()` with different prefixes.
- Su et al. (2023), "One Embedder, Any Task: Instruction-Finetuned Text Embeddings" (INSTRUCTOR) — Established that instruction-aware embeddings outperform non-instructed variants by 5-10% NDCG on retrieval tasks.

---

## Issue 3: Benchmark Runner Hybrid Merge Bug

**Severity:** 🔴 Critical (invalidates benchmark results for configs C-G)
**Component:** `scripts/run-beir-bench.ts` → `runRetrieval()` (lines 285-289)
**Impact:** Hybrid merge receives corrupted inputs. Benchmark results for C/D/E/F/G are unreliable.

### The Problem

```typescript
// Current code (buggy):
let candidates: SearchResult[] = [];
if (config.useVector) {
  const vResults = await backend.vectorSearch(...);
  candidates.push(...vResults);   // ← concatenated into candidates
}
if (config.useFts) {
  const fResults = await backend.ftsSearch(...);
  candidates.push(...fResults);   // ← concatenated into candidates
}

// Hybrid merge:
const vectorResults = candidates.filter((r) => r.score > 0 && r.chunk.id.startsWith("beir-"));
const ftsResults = candidates.filter((r) => r.score > 0);
candidates = hybridMerge(vectorResults, ftsResults, k * 2);
```

Both `vectorResults` and `ftsResults` filter from the combined `candidates` array. Since all BEIR chunk IDs start with `beir-`, `vectorResults` matches everything. `ftsResults` also matches everything. **Both variables contain the same mixed set of results.**

`hybridMerge()` receives identical inputs for its vector and FTS parameters. It cannot distinguish which results came from which source, so the "dual evidence boost" logic fires for all duplicates regardless of origin.

### Note

This bug is in the **benchmark runner only**, not in the production `recall.ts` which correctly calls `vectorSearch()` and `ftsSearch()` separately and passes them independently to `hybridMerge()`.

---

## Issue 4: HyDE Never Activated in Benchmark

**Severity:** 🟡 Important
**Component:** `scripts/run-beir-bench.ts` (line 353) + `scripts/run-beir-pipeline.sh`
**Impact:** Config F results are meaningless (identical to Config C).

### The Problem

Config F is defined with `useHyde: true`, but HyDE activation is gated on a CLI flag:

```typescript
const enableHyde = args.includes("--hyde");
const hydeConfig = enableHyde ? cfg.hyde : undefined;
```

The pipeline script never passes `--hyde`:
```bash
npx tsx scripts/run-beir-bench.ts --dataset "$ds"
```

### Evidence

```
SciFact C NDCG: 0.4736870964979056
SciFact F NDCG: 0.4736870964979056  ← identical to 16 decimal places
```

---

## Issue 5: HyDE Vector Averaging (Wrong Approach)

**Severity:** 🟡 Important
**Component:** `scripts/run-beir-bench.ts` (line 246) + `src/auto/recall.ts` (line ~50)
**Impact:** When HyDE runs, it dilutes both query and hypothetical document signals.

### The Problem

```typescript
// Current: average query and HyDE vectors
queryVector = queryVector.map((v, idx) => (v + hydeVector[idx]) / 2);
```

### Literature

Gao et al. (2022), "Precise Zero-Shot Dense Retrieval without Relevance Labels" — the original HyDE paper specifies:

> "We use the language model to generate a hypothetical document, embed it with an unsupervised contrastive encoder, and use this embedding to search against the corpus."

The hypothetical document vector **replaces** the query vector entirely. The insight is that the hypothetical document, despite being generated (possibly hallucinated), lives in **document space** — the same semantic region as the actual corpus documents. The raw query lives in **query space** and must cross the query-document gap.

Averaging is a common implementation mistake that creates a vector in neither query space nor document space — a "no man's land" that reduces retrieval quality. Multiple studies (Gao et al. 2022, Yu et al. 2023) confirm that direct replacement outperforms averaging.

### Additional Concern

With instruction-aware embeddings (Issue 2), HyDE becomes more nuanced:
- The raw query should use `Instruct: ...\nQuery: {query}` formatting
- The hypothetical document should be embedded as a **document** (no instruction prefix)
- This naturally bridges the query→document gap, which is exactly HyDE's purpose

---

## Issue 6: MMR Destroys Recall

**Severity:** 🟡 Important
**Component:** `src/auto/recall.ts` → `mmrRerank()` + benchmark runner
**Impact:** MMR removes relevant results that happen to share terminology.

### The Problem

MMR with `lambda=0.7` uses Jaccard similarity on token sets to penalize results similar to already-selected ones. For scientific corpora where relevant documents share domain terminology (e.g., multiple papers about the same biological mechanism), MMR aggressively removes relevant results.

### Evidence

```
SciFact Recall@10:
  A (Vector):     0.884
  C (Hybrid):     0.846
  E (Hybrid+MMR): 0.637  ← 28% recall loss vs vector-only
```

MMR removed ~25% of relevant documents. In a scientific domain, this is expected — papers about "gene expression in cancer cells" will share many tokens even when discussing different aspects.

### Also: D ≡ G (MMR is a no-op in Full Pipeline)

```
SciFact D NDCG: 0.5308
SciFact G NDCG: 0.5308  ← identical
```

The reranker returns `topN` results. MMR then runs on this already-limited set and finds nothing to diversify — the reranker already selected the best candidates.

### Literature

- Carbonell & Goldstein (1998), "The Use of MMR, Diversity-Based Reranking" — MMR is designed for **document summarization** where diversity is the goal. For precision-oriented retrieval (like BEIR), high lambda (>0.9) or no MMR is standard.
- Relevant in production RAG (avoiding repetitive context injection), but harmful for pure retrieval benchmarks where recall matters.

---

## Issue 7: Reranker Latency and Candidate Quality

**Severity:** 🟡 Important
**Component:** `src/rerank/reranker.ts` + Spark reranker endpoint (port 18096)
**Impact:** 41-second p50 latency per query. Reranker can't fix garbage-in from hybrid.

### The Problem

**Latency:**
```
SciFact Config D (Hybrid + Reranker):
  p50 latency: 41,909ms
  p95 latency: 43,009ms
  mean:        40,995ms
```

41 seconds per query is ~60× slower than vector-only (703ms). For production RAG where user-facing latency matters, this is unacceptable. Even for batch benchmarks, it makes iteration painful.

**Candidate Quality:**
The reranker receives candidates from the broken hybrid merge. It partially rescues the ranking (0.531 vs 0.474) but can't overcome the fact that the candidate set itself is corrupted by FTS score saturation (Issue 1) and the merge bug (Issue 3).

### Possible Causes for Latency

1. Reranker model may be running on CPU (no GPU offload)
2. No request batching — each query-document pair scored individually
3. Network round-trip overhead to Spark node
4. Model may be cold-starting or swapping

### Literature

- Cross-encoder rerankers are inherently slower than bi-encoders (O(n) forward passes for n candidates vs O(1) for bi-encoder).
- Standard practice: limit reranker input to top-20 to top-50 candidates from the first-stage retriever.
- NVIDIA NIM reranker benchmarks show <100ms for 25 documents — our 41s suggests a configuration/deployment issue, not a model limitation.

---

## Cross-Cutting Observations

### NFCorpus vs SciFact Divergence

NFCorpus is fundamentally different from SciFact:
- **NFCorpus:** Medical documents with multi-label relevance grades (0/1/2), many relevant docs per query
- **SciFact:** Claim verification with typically 1-2 relevant docs per query

Vector-only performs much worse on NFCorpus (0.186 vs 0.745 on SciFact) while FTS performs better (0.273 vs 0.623). This suggests the embedding model handles well-formed scientific claims better than broad medical queries — consistent with the instruction prefix issue (Issue 2), where medical queries would benefit more from proper task instruction.

### Production vs Benchmark Code Paths

The production code in `src/auto/recall.ts` and `src/manager.ts` has the same fundamental Issues 1, 2, 5, and 6, but NOT Issue 3 (which is benchmark-runner-only). The production pipeline correctly separates vector and FTS results before merging.

However, the production pipeline also applies:
- Source weighting (captures 1.5×, sessions 0.5×) — not tested in BEIR
- Temporal decay — not applicable to static BEIR corpus
- Context dedup against recent messages — not applicable
- Parent-child chunk expansion — not tested

These production-only features may introduce additional quality variations not captured by BEIR.
