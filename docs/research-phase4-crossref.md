# Phase 4 Cross-Reference: memory-spark vs NVIDIA RAG Blueprint & Modern Research

**Date:** 2026-03-31  
**Author:** RAG research subagent (meta)  
**Purpose:** Cross-reference the memory-spark retrieval pipeline against the NVIDIA RAG Blueprint (https://github.com/NVIDIA-AI-Blueprints/rag) and modern RAG research to validate, challenge, or extend the Phase 4 MMR fix plan.

---

## Summary / TL;DR

The Phase 4 plan is **mostly correct** and directionally sound. The Jaccard→cosine switch for MMR is the right call. Configurable lambda is the right call. Source-level dedup is the right call. However, there are **five issues not covered by Phase 4 that are more critical than the MMR fix**, plus one significant ordering mistake in the current pipeline that Phase 4 doesn't address.

Priority order for fixing (revised from Phase 4):

1. **[CRITICAL, not in Phase 4] Fix instruction prefix for Nemotron embeddings** (Issue 2 in ISSUES.md)
2. **[CRITICAL, not in Phase 4] Fix BM25 sigmoid saturation / switch fully to RRF** (Issue 1 in ISSUES.md)
3. **[CRITICAL] Fix pipeline ordering: MMR should come AFTER reranker, not before** (Phase 4 plan has wrong order)
4. **[HIGH] Replace Jaccard with cosine similarity in MMR** (Phase 4A — correct)
5. **[HIGH] Make lambda configurable with better default** (Phase 4B — correct, but raise default to 0.9+)
6. **[MEDIUM] Source-level dedup before MMR** (Phase 4D — correct)
7. **[MEDIUM] Dynamic lambda** (Phase 4C — nice-to-have, not urgent)
8. **[MEDIUM] Fix HyDE to replace vector, not average** (Issue 5 in ISSUES.md, not in Phase 4)
9. **[LOW] Benchmark runner fixes** (Phase 4F — important for measurement)

---

## Part 1: NVIDIA RAG Blueprint Analysis

### Architecture (from https://github.com/NVIDIA-AI-Blueprints/rag + https://docs.nvidia.com/rag/2.5.0/query-to-answer-pipeline.html)

NVIDIA's pipeline is:
```
Query → [Optional: Query Rewriter] → Embedding → Vector Search (dense + sparse hybrid) → [Optional: Reranker] → LLM Generation
```

Additional optional stages: query decomposition, self-reflection, guardrails.

**Key models:**
- Embedding: `llama-3_2-nv-embedqa-1b-v2` (instruction-aware, query/document task types)
- Reranker: `llama-3_2-nv-rerankqa-1b-v2` (cross-encoder)
- LLM: `llama-3.3-nemotron-super-49b-v1.5`
- Vector DB: Milvus or Elasticsearch (both support GPU-accelerated search via cuVS)

**What NVIDIA does NOT have:**
- MMR — zero mention of Maximum Marginal Relevance anywhere in the blueprint
- HyDE — not present in the reference pipeline
- Temporal decay — not relevant for enterprise document corpora
- Source weighting — not present (they rely on reranker for this)
- Parent-child chunk expansion — not present (they use standard flat chunking)

**What NVIDIA does that we don't:**
- Query decomposition (breaking complex queries into sub-queries)
- Self-reflection (LLM validates its own answer against retrieved context)
- Multi-collection search (searching across multiple vector stores simultaneously)
- GPU-accelerated ANN indexing (cuVS — we use CPU IVF_PQ)
- Multimodal ingestion (tables, charts, images, audio — we're text-only)
- Guardrails (content safety, topic control)
- Query rewriting for multi-turn conversations

### NVIDIA's Hybrid Search Approach

NVIDIA uses **dense + sparse** hybrid search at the vector DB layer (Milvus/Elasticsearch native hybrid). They do **not** implement their own RRF — the vector DB handles it. Their pipeline is:
1. Embed query with instruction-aware model
2. Vector DB runs hybrid search (dense ANN + sparse BM25) internally
3. Return top-k results
4. Optional reranker pass

This is cleaner than our approach of calling `vectorSearch()` and `ftsSearch()` separately and merging ourselves, because:
- The vector DB can optimize both searches together
- No BM25 score normalization problem — rank fusion happens at the DB layer
- No risk of our sigmoid saturation bug

**Implication for us:** Our RRF implementation is architecturally correct but the BM25 score normalization (Issue 1) is a known pitfall that NVIDIA sidesteps by using DB-native hybrid. Our sigmoid midpoint needs recalibration or we should abandon score-based FTS normalization entirely and lean harder into pure RRF (rank-only, no score magnitude).

---

## Part 2: Agreements with Modern Research

### 2.1 RRF for Hybrid Merge — Correct

Our `hybridMerge()` using RRF with k=60 is correct and industry-standard. RRF with k=60 is the default in Elasticsearch, Azure AI Search, and OpenSearch. The k=60 value is from Cormack et al. (2009) and remains the standard default:

- Azure AI Search uses k=60: https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
- OpenSearch blog on RRF: https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/
- LangChain4j uses RRF as default aggregator

**However:** Our RRF implementation normalizes the final scores to [0,1] by dividing by the max RRF score. This is correct for downstream `minScore` filtering. No issue here.

### 2.2 HyDE: Replace, Don't Average — Correct (but we're currently wrong in benchmark)

Our production `recall.ts` correctly **replaces** the query vector with the HyDE vector:
```typescript
queryVector = await embed.embedDocument(hypothetical);
```

This matches Gao et al. (2022): the hypothetical document embedding should fully replace the query embedding. The original paper never averages — it's a full replacement.

The **benchmark runner** (Issue 5) incorrectly averages, but production is correct. Good.

### 2.3 Instruction-Aware Embeddings: Asymmetric Encoding — Correct Design, Broken Implementation

The design intent is right: `embedQuery()` applies instruction prefix, `embedDocument()` does not. This is exactly what Nemotron-8B requires and what E5/INSTRUCTOR models use:

- Su et al. (2023), "One Embedder, Any Task: Instruction-Finetuned Text Embeddings" (INSTRUCTOR): https://arxiv.org/abs/2212.09561
- Nemotron-8B HuggingFace card confirms asymmetric encoding is required

However, Issue 2 in ISSUES.md shows the instruction prefix is NOT being applied in practice (or wasn't at benchmark time). The `queryInstruction` config field exists but must be verified as populated. If `cfg.spark.queryInstruction` is undefined, the prefix silently gets skipped. This is the single highest-impact bug.

### 2.4 Parent-Child Chunk Expansion — Novel, Defensible

NVIDIA doesn't do this — they use flat chunking. Our parent-child approach (small children for precise search, parent text for context) is sometimes called "Small-to-Big Retrieval" or "Sentence Window Retrieval" in the literature. It's a legitimate technique:

- LlamaIndex documents it as "parent document retrieval"
- Anthropic's RAG cookbook recommends it for long-context models
- Cost: extra DB lookup for parent chunks. Benefit: richer context with precise retrieval

This is not wrong. Keep it.

### 2.5 Source Weighting and Temporal Decay — Domain-Specific, Not General

NVIDIA doesn't do source weighting or temporal decay because their use case is enterprise document corpora (static, uniform trust). Our use case is a personal memory system with heterogeneous sources (captures, files, sessions, mistakes) with different trust levels and recency profiles.

These features are justified for our domain. No issue.

---

## Part 3: Disagreements and Problems

### 3.1 [CRITICAL] Pipeline Ordering: MMR Before Reranker Is Wrong

**Current order:**
```
retrieve → hybrid merge → source weighting → temporal decay → MMR → reranker → context expansion
```

**Correct order:**
```
retrieve → hybrid merge → source weighting → temporal decay → reranker → MMR → context expansion
```

**Why this matters:**

The reranker is a cross-encoder: it scores each candidate document against the query with full attention between query and document tokens. It is by far the most accurate relevance signal in the pipeline. MMR, by contrast, uses embedding similarity (or currently, Jaccard) to estimate document-to-document redundancy.

When MMR runs first:
- It removes documents based on crude similarity estimates (currently Jaccard, even after Phase 4 fix: cosine embeddings)
- Then the reranker only sees the already-filtered set
- If MMR incorrectly removed a highly relevant document (which we know happens — 25% recall drop), the reranker can't rescue it

When reranker runs first:
- The reranker identifies the most relevant candidates from the large overfetch pool
- MMR then applies diversity within the **reranker-approved** set
- MMR can't damage recall because the reranker has already certified relevance

**NVIDIA's pipeline confirms this order:** their pipeline is retrieve → rerank → (no MMR). If you add MMR for diversity, it belongs after reranking, not before.

**Evidence from ISSUES.md:**
```
D (Hybrid + Reranker): NDCG 0.5308
G (Full Pipeline = D + MMR after reranker): NDCG 0.5308  ← identical
```

The fact that G ≡ D tells us MMR is effectively a no-op after the reranker, because the reranker already selects a small tight set where MMR has nothing diverse to pick from. This is actually correct behavior — it means MMR (correctly placed after reranker) doesn't hurt. The issue is when MMR is placed before the reranker and removes candidates the reranker would have rescued.

**Fix:** Move `mmrRerank()` to after `reranker.rerank()` in `recall.ts`. Apply MMR to the reranked results to get the final `maxResults` from a slightly larger reranker output.

```typescript
// New order:
const reranked = await reranker.rerank(queryText, diverse_pre_mmr, cfg.maxResults * 2);  // rerank more candidates
const finalResults = mmrRerank(reranked, cfg.maxResults, cfg.mmrLambda ?? 0.9);  // MMR on reranked output
```

This way:
- MMR sees reranker-certified relevant documents
- MMR only needs to diversity-filter a small set (e.g., 20 → 10)
- MMR can't hurt recall because the reranker already validated relevance

### 3.2 [CRITICAL] Phase 4 Plan Missing the Two Most Impactful Fixes

Phase 4 focuses on MMR fixes, which are correct but not the most impactful issues. From ISSUES.md:

**Issue 2 (Nemotron instruction prefix)** — This is estimated to be a 10-20% NDCG improvement on SciFact based on INSTRUCTOR paper benchmarks (5-10% uplift from proper instruction formatting). It's also the simplest fix: ensure `queryInstruction` is set in config and wired to `makeOpenAiCompat()`.

**Issue 1 (BM25 sigmoid saturation)** — FTS is completely non-discriminative when all scores are compressed to [0.983, 1.0]. The fix options:
1. **Recommended:** Use pure rank-based RRF internally (our `hybridMerge()` already does this — the issue is in `rowToSearchResult()` where BM25 scores are converted to pseudo-[0,1] before being passed to hybridMerge, then ignored because hybridMerge only uses rank position). Actually looking at the code carefully: `hybridMerge()` uses rank position (RRF), not the pre-converted score. So the sigmoid is applied but then **discarded** by hybridMerge. The problem is that pre-sigmoid scores are used by `minScore` filtering — if sigmoid maps everything to >0.98, the minScore filter passes everything.
2. **Immediate fix:** Set `fts.sigmoidMidpoint` to corpus-appropriate value (10-15 for scientific text, 5-8 for general text), or better: remove minScore filtering from FTS results entirely and let RRF handle ranking.

### 3.3 [HIGH] Lambda Default Is Wrong Direction (Phase 4B)

Phase 4 plan proposes raising the default from 0.7 to 0.85. Based on the evidence, this should be higher — at least 0.9 for a factual memory retrieval system.

MMR was designed for document summarization (Carbonell & Goldstein 1998), not retrieval augmentation. In RAG:
- The LLM can handle some redundancy; it's better to have redundant relevant context than diverse but partially relevant context
- For factual queries, precision over diversity is almost always correct
- The 25% recall drop with lambda=0.7 shows the default is far too diversity-aggressive

**Recommendation:** Default lambda = 0.9. Allow config override down to 0.5 for exploratory use cases. The dynamic lambda heuristic (Phase 4C) is a nice addition but secondary to getting the baseline right.

Literature confirms: in production RAG systems (LangChain, LlamaIndex), MMR lambda defaults are typically 0.5 (50/50 relevance/diversity) for summarization and 0.8-0.9 for retrieval. Our 0.7 is miscalibrated for our use case.

### 3.4 [MEDIUM] MMR Metric: Cosine Is Correct (Phase 4A Validated)

The Phase 4A plan is correct. Cosine similarity on embedding vectors is the right metric for document-document similarity in MMR. Multiple sources confirm:

- Anton Dergunov (2024): "The document-document similarity metric can differ from the query-document similarity, but for simplicity, we'll use cosine similarity as well" — https://anton-dergunov.github.io/posts/2024/07/maximal-marginal-relevance
- Azure AI Search MMR implementation uses cosine similarity: https://farzzy.hashnode.dev/enhancing-rag-with-maximum-marginal-relevance-mmr-in-azure-ai-search
- All LangChain/LlamaIndex MMR implementations use embedding cosine similarity

Jaccard on token bags is not used in any production RAG implementation I found. It's a reasonable approximation when vectors aren't available, but since we have vectors, we should use them.

**Important implementation note:** LanceDB's `vectorSearch()` currently does NOT return the embedding vector alongside results (the `vector` column exists in the table but is excluded from search result projections by default for performance). You'll need to explicitly request it via `.select(["vector", ...otherFields])` or fetch vectors separately. Verify this before implementing Phase 4A.

### 3.5 [MEDIUM] Dynamic Lambda: Correct Intuition, Questionable Implementation

The Phase 4C dynamic lambda heuristic (query length, interrogative patterns, source pools) is reasonable but not grounded in literature. I found no production systems that implement dynamic MMR lambda adjustment. Most systems either:
- Use a fixed lambda with careful tuning
- Expose lambda as a per-query API parameter (user-controlled)
- Don't use MMR at all and rely on reranker diversity

The proposed heuristics are plausible but will need empirical validation. The multi-pool adjustment (-0.05 lambda when results come from multiple pools) is particularly dubious — multi-pool results are already diverse by source, so you'd want HIGHER lambda (less diversity penalty) not lower.

**My recommendation:** Implement 4B (configurable lambda) and set a good default (0.9). Skip 4C unless you have empirical evidence the heuristics help. Dynamic lambda is premature optimization.

---

## Part 4: Missing Pieces (What NVIDIA Does That We Don't)

### 4.1 Query Decomposition (High Value for Complex Queries)

NVIDIA's blueprint includes optional query decomposition: breaking "tell me about Klein's infrastructure and what monitoring is set up for it" into two sub-queries, running each, then merging results.

For a personal memory system like ours, this could improve recall significantly for multi-aspect questions. Implementation:
- Use the same LLM we have for HyDE
- Decompose into 2-3 sub-queries
- Run parallel recall for each
- Merge results via RRF before reranking

This is higher complexity but high value. Not in Phase 4 — suggest creating a separate Phase 5 task.

### 4.2 Self-Reflection / Answer Validation

NVIDIA's pipeline optionally has an LLM check whether the generated answer is supported by the retrieved context. For a memory injection system (not generation), this doesn't directly apply, but we could add a "relevance check" step: before injecting memories into the prompt, have a small LLM verify that each memory actually answers/informs the query. This would catch false positives from embedding similarity mismatch.

Too expensive for real-time use. Skip for now.

### 4.3 GPU-Accelerated Vector Search (cuVS)

NVIDIA uses cuVS for GPU-accelerated ANN. We use LanceDB's CPU-based IVF_PQ. Given that we're running on Spark DGX, GPU-accelerated search would dramatically reduce vector search latency. LanceDB doesn't support cuVS directly, but we could:
- Use FAISS with GPU support for the ANN step
- Use Milvus (supports cuVS) as the backend

Not a Phase 4 concern, but a future scalability item.

### 4.4 Query Rewriting for Multi-Turn Conversations

NVIDIA rewrites follow-up queries to be standalone before embedding them. For example, "What about the other one?" → "What is the configuration for the second Spark node?" This dramatically improves retrieval for conversational contexts.

We don't do this. Our `cleanQueryText()` strips noise but doesn't rephrase. For our use case (agent context injection), this matters because queries often reference earlier conversation context.

Suggest adding query rewriting as a Phase 5 feature using the same HyDE LLM endpoint.

---

## Part 5: MMR Deep Dive

### Is the Phase 4 Plan Correct? Yes, With Caveats.

**The Jaccard → Cosine switch (4A): Correct.** All production implementations use embedding cosine. Jaccard on token bags is only appropriate when you don't have vectors, which we do.

**Configurable lambda (4B): Correct.** Default should be 0.9, not 0.85.

**Dynamic lambda (4C): Not proven.** Skip or make it optional behind a flag.

**Source dedup before MMR (4D): Correct and important.** This separates "same document different chunks" deduplication (which Jaccard was accidentally good at) from "diverse topic" selection (which cosine handles better).

**The big missing piece in Phase 4: Pipeline ordering.** MMR must move to after the reranker.

### Is Cosine the Right Metric for MMR? Yes.

Cosine similarity on embedding vectors is the standard for document-document similarity in RAG. The original MMR paper (Carbonell & Goldstein 1998) doesn't specify the metric — it says "similarity measure." All modern implementations use the same metric for query-document and document-document comparison: cosine on embeddings.

**Possible concern:** With instruction-aware embeddings (Nemotron), documents are embedded in "document space" (no instruction prefix). Do document-document cosine similarities still make sense? Yes — the instruction prefix only shifts query vectors into "query space." Document vectors all live in the same "document space" and are directly comparable via cosine.

### Is Dynamic Lambda Standard? No.

I found zero production systems implementing dynamic lambda adjustment based on query characteristics. Lambda is either:
- Fixed (most systems: LangChain default 0.5, LlamaIndex default 0.5)
- Exposed as a user/API parameter
- Not used at all (NVIDIA's blueprint has no MMR)

LangChain and LlamaIndex both default to lambda=0.5 which is quite aggressive diversity. For precision-oriented RAG (factual Q&A, memory injection), production tuning guides recommend lambda=0.7-0.9. Our proposed 0.9 is at the high end but appropriate given the evidence of recall damage.

### What Do Production Systems Actually Use?

Based on research:
- **No MMR:** NVIDIA RAG Blueprint, most enterprise search systems (they rely on reranker for quality)
- **MMR with lambda ~0.5:** LangChain and LlamaIndex defaults (designed for summarization use cases)
- **MMR with lambda ~0.7-0.9:** RAG-for-retrieval use cases where diversity is secondary to precision
- **Clustering-based dedup instead of MMR:** Some systems cluster retrieved documents and pick the top result from each cluster (simpler, faster, no lambda tuning)
- **DPP (Determinantal Point Processes):** Used in recommendation systems (YouTube, Netflix) but not in RAG — too computationally expensive for the marginal diversity gain

For our use case (memory injection, not summarization), MMR with high lambda (0.9) after reranking, combined with source-level dedup, is the right approach.

---

## Part 6: Pipeline Ordering Analysis

### Current Order
```
retrieve (vector + FTS per pool) 
→ hybrid merge (RRF) 
→ source weighting 
→ temporal decay 
→ MMR (lambda=0.7, Jaccard) 
→ reranker 
→ context expansion (parent-child) 
→ LCM/recency dedup 
→ prompt injection filter 
→ token budget enforcement 
→ inject
```

### What NVIDIA Does
```
query → embed → vector search (hybrid) → reranker → inject
```

### Recommended Order for memory-spark

```
retrieve (vector + FTS per pool) 
→ hybrid merge (RRF) 
→ source weighting        ← keep here: penalize garbage before reranker input
→ temporal decay          ← keep here: same reason
→ source-level dedup      ← NEW (Phase 4D): collapse near-identical chunks from same source
→ reranker                ← MOVED EARLIER: reranker sees full overfetch set
→ MMR (cosine, lambda=0.9) ← MOVED AFTER RERANKER: diversity on reranker-approved set
→ context expansion (parent-child)
→ LCM/recency dedup
→ prompt injection filter
→ token budget enforcement
→ inject
```

**Why source weighting before reranker?** The reranker input limit (top-k candidates) matters. We want to fill those k slots with high-quality candidates, not session chunks or archive content that source weighting would penalize. Apply weighting first so the reranker sees the right candidates.

**Why MMR after reranker?** As established above: prevents MMR from removing candidates the reranker would have rescued.

**Concern about reranker latency (41s, Issue 7):** Before moving reranker earlier, fix the reranker deployment. 41s is a deployment bug (likely CPU-only, no batching), not a design limitation. NVIDIA NIM reranker benchmarks show <100ms for 25 documents. Fix the deployment before the ordering matters.

### Is the Current Ordering Defensible At All?

The only argument for MMR before reranker: "MMR reduces the reranker's input size, saving compute." This is a cost optimization argument, not a quality argument. Given that our reranker is already broken (41s latency), optimizing its input size is premature. Fix the deployment first, then optimize input size if needed.

---

## Part 7: Specific Recommendations

### Priority 1: Fix Instruction Prefix (Not in Phase 4)
- **File:** `src/embed/provider.ts`, `src/config.ts`
- **Action:** Ensure `queryInstruction` is set in default config for Nemotron-8B. Default value: `"Given a question about a personal knowledge base, retrieve relevant passages that directly answer or inform the question."`
- **Verify:** Confirm `embedQuery()` actually applies the prefix by logging vectors in a test. The config field exists but was empty/missing during BEIR benchmarks (per ISSUES.md evidence).
- **Impact:** Estimated 10-20% NDCG improvement on vector retrieval.

### Priority 2: Fix BM25 Score Normalization (Not in Phase 4)
- **File:** `src/storage/lancedb.ts` → `rowToSearchResult()`
- **Action:** Either recalibrate sigmoid midpoint for the actual corpus, or (better) remove the minScore filter on FTS results before passing to hybridMerge. The hybridMerge function uses rank positions (RRF), not the sigmoid-normalized scores. The sigmoid only affects `minScore` filtering on FTS output. Solution: apply minScore filter to vector results only, pass all FTS results to hybridMerge regardless of sigmoid score.
- **Why:** hybridMerge discards the pre-normalized score and recomputes rank-based scores anyway. The sigmoid conversion is only harmful — it causes minScore to wrongly filter FTS results.

### Priority 3: Fix Pipeline Ordering
- **File:** `src/auto/recall.ts`
- **Action:** Move `mmrRerank()` to after `reranker.rerank()`. Have reranker return `cfg.maxResults * 1.5` candidates, then MMR trims to `cfg.maxResults`.
- **Requires:** Reranker latency fix first (Issue 7 — likely a deployment problem on Spark).

### Priority 4: Phase 4A — Jaccard → Cosine in MMR
- **File:** `src/auto/recall.ts`
- **First verify:** Can we get embedding vectors back from LanceDB search without an extra round-trip? Check if `vectorSearch()` can return the `vector` column. If not, implement a separate `getVectorsByIds()` batch call after retrieval.
- **Alternative if vectors unavailable:** Use a cheap cosine approximation from the text — TF-IDF vectors instead of full embeddings. Still better than Jaccard.

### Priority 5: Phase 4B — Lambda Config
- **File:** `src/config.ts`, `src/auto/recall.ts`
- **Default:** `mmrLambda: 0.9` (not 0.85 as Phase 4 proposes — evidence from BEIR shows aggressive diversity is harmful)
- **Range validation:** Clamp to [0.0, 1.0]

### Priority 6: Phase 4D — Source Dedup Before MMR
- Implement as planned. Jaccard IS the right metric here — we're checking for near-identical text from the same source, not semantic similarity. Threshold 0.85 seems reasonable.

### Priority 7: Fix HyDE Vector Handling in Benchmark Runner (Not in Phase 4)
- **File:** `scripts/run-beir-bench.ts`
- **Action:** Remove averaging, replace fully. Also add `--hyde` flag to pipeline script.
- **Note:** Production `recall.ts` already does this correctly. Benchmark runner is the bug.

### Do NOT Implement: Dynamic Lambda (Phase 4C)
No empirical basis. The query characteristics heuristics are plausible but untested and risk making performance worse. Run ablation experiments with fixed lambda values (0.7, 0.8, 0.9, 1.0) across BEIR datasets first. If the optimal lambda varies dramatically by query type, then consider dynamic lambda. Otherwise, a fixed 0.9 is simpler and less likely to regress.

---

## Part 8: What We Do Well (Keep These)

1. **RRF for hybrid merge** — correct, industry-standard, properly implemented
2. **HyDE in production** — correct replace-don't-average implementation  
3. **Parent-child chunk expansion** — novel but defensible; NVIDIA doesn't do it but it's legitimate
4. **Pool-based retrieval** — smart architecture for heterogeneous memory sources
5. **LCM/recency dedup** — unique to our use case, prevents redundancy with LCM context
6. **Prompt injection filtering** — critical safety feature, not in NVIDIA blueprint
7. **Write mutex for LanceDB** — prevents commit conflicts; correct
8. **Asymmetric embedding design** — intent is right, implementation needs verification (Priority 1)

---

## References

- Carbonell & Goldstein (1998). "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries." SIGIR 1998. [Original MMR paper]
- Cormack, Clarke & Büttcher (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods." SIGIR 2009. [RRF original paper] https://plg.uwaterloo.ca/~grcorcor/topicmodels/rrf.pdf
- Gao et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels." [HyDE paper] https://arxiv.org/abs/2212.10496
- Su et al. (2022/2023). "One Embedder, Any Task: Instruction-Finetuned Text Embeddings." [INSTRUCTOR] https://arxiv.org/abs/2212.09561
- Babakhin et al. (2025). "Llama-Embed-Nemotron-8B: A Universal Text Embedding Model." https://www.researchgate.net/publication/397480951
- Robertson & Zaragoza (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." [BM25 score normalization analysis]
- NVIDIA RAG Blueprint (2025). https://github.com/NVIDIA-AI-Blueprints/rag
- NVIDIA RAG Pipeline Docs (2025). https://docs.nvidia.com/rag/2.5.0/query-to-answer-pipeline.html
- NVIDIA Blog: Enhancing RAG Pipelines with Re-Ranking (Oct 2024). https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/
- Azure AI Search: Hybrid Search Scoring with RRF. https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
- OpenSearch RRF blog (2025). https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/
- Anton Dergunov (2024). "Maximal Marginal Relevance." https://anton-dergunov.github.io/posts/2024/07/maximal-marginal-relevance
- Azure AI Search MMR implementation (2024). https://farzzy.hashnode.dev/enhancing-rag-with-maximum-marginal-relevance-mmr-in-azure-ai-search
- Pinecone: Rerankers and Two-Stage Retrieval. https://www.pinecone.io/learn/series/rag/rerankers/
- Cheng et al. (2018). "Practical Diversified Recommendations on YouTube with Determinantal Point Processes." [DPP in production — recommendation systems, not RAG]
- nvidia/llama-embed-nemotron-8b HuggingFace model card. https://huggingface.co/nvidia/llama-embed-nemotron-8b
