# Phase 5: Reranker Performance + Vision Pipeline Architecture

**Status:** Planning  
**Parent task:** `00889d7b` (BEIR benchmark underperformance recon)  
**Date:** 2026-03-31  
**Depends on:** Phase 4 complete ✅  

---

## Problem Statement

Two interrelated problems:

### 1. Reranker Latency (Original Phase 5)
The current text reranker (`llama-nemotron-rerank-1b-v2`) exhibits 41-second latency per query — orders of magnitude slower than NVIDIA's benchmarks (<100ms for 25 docs). This makes the full pipeline unusable for interactive RAG.

### 2. OCR Pipeline Fragility (New Direction)
The current document ingestion pipeline is:
```
PDF/Image → GLM-4V OCR → extracted text → chunk → embed text → store
```

This is fundamentally lossy:
- **Layout destruction:** Tables, columns, headers lose spatial relationships
- **Chart/figure blindness:** OCR can describe charts in text, but loses the visual signal entirely
- **Formula mangling:** Math notation gets garbled or lost
- **Handwriting/scan quality:** OCR accuracy degrades on non-pristine documents
- **Pipeline complexity:** OCR is a separate service (GLM-4V on Spark), adds latency, adds failure modes (502s), adds cost
- **No visual context:** A human reading a PDF sees layout, emphasis, diagrams — OCR strips all of that

### The Vision Alternative

NVIDIA has released multimodal variants of the exact models we already use:

| Current (Text-Only) | New (Vision-Language) | Size |
|---|---|---|
| `llama-nemotron-embed-8b` | **`llama-nemotron-embed-vl-1b-v2`** | 1.7B (Llama 3.2 1B + SigLip2 400M) |
| `llama-nemotron-rerank-1b-v2` | **`llama-nemotron-rerank-vl-1b-v2`** | ~1B |

The VL models accept **both text and images** as input and project them into the **same vector space**. This enables:

```
PDF → render pages as images → embed images directly → store
Query (text) → embed query → search same vector space → rerank with vision
```

**No OCR. No text extraction. No layout parsing. The model sees the actual page.**

---

## Architecture Comparison

### Current Pipeline (Text-Only + OCR)
```
Ingest:  PDF → GLM-4V OCR → text chunks → embed-8b → LanceDB
Query:   text → embed-8b(query) → vector search → FTS search → RRF merge
         → rerank-1b(text) → MMR → expand → LLM
```

- 4 services: OCR (GLM-4V), embedder, reranker, LLM
- OCR is the bottleneck and single point of failure
- Text-only: no visual understanding

### Proposed Pipeline (Multimodal Vision)
```
Ingest:  PDF → render page images → embed-vl-1b(image) → LanceDB
         Markdown/text → chunk → embed-vl-1b(text) → LanceDB
Query:   text → embed-vl-1b(query) → vector search → FTS search → RRF merge
         → rerank-vl-1b(query, images) → MMR → expand → VLM
```

- 3 services: embedder-vl, reranker-vl, VLM (one fewer!)
- No OCR dependency
- Images and text live in unified vector space
- Reranker sees the actual page image when scoring relevance
- FTS still works for text documents (markdown, notes) — hybrid is preserved

### Hybrid Approach (Recommended — Phase 5)
```
Ingest:  PDF → render page images → embed-vl-1b(image) → LanceDB [visual index]
         PDF → basic text extract (pdftotext) → chunk → embed-vl-1b(text) → LanceDB [text index]
         Markdown/text → chunk → embed-vl-1b(text) → LanceDB [text index]
Query:   text → embed-vl-1b(query) → search both indexes → RRF merge
         → rerank-vl-1b(query, page_image OR text) → MMR → VLM
```

Why hybrid: text documents (markdown, agent memory, notes) don't have visual representations. The unified VL embedder handles both — text-to-text AND text-to-image retrieval in the same space.

---

## Execution Plan

### 5A: Diagnose Current Reranker Latency (~1h)

Before replacing anything, understand why the current reranker is 400x slower than benchmarks.

**Steps:**
1. SSH to Spark, check `nvidia-smi` — is the reranker on GPU or CPU?
2. Check the reranker service config — is it running via vLLM, TGI, or raw transformers?
3. Test with controlled payloads:
   - 1 document: baseline latency
   - 10 documents: scaling behavior  
   - 30 documents: target operating point
4. Check if batching is enabled or if we're sending serial requests
5. Profile: is latency in model inference, data transfer, or preprocessing?

**Expected outcome:** Root cause identified. Likely one of:
- CPU inference (not on GPU) → fix: move to GPU
- No batching (serial pair scoring) → fix: batch API
- vLLM overhead for small models → fix: use transformers directly or NIM container
- Network round-trip to Spark → fix: measure, optimize

**Deliverable:** Latency diagnostic report with numbers.

### 5B: Limit Reranker Candidate Pool (~30min)

Regardless of vision migration, the reranker should never receive unbounded input.

**File:** `src/rerank/reranker.ts`

```typescript
const MAX_RERANK_CANDIDATES = 30;
const toRerank = candidates.slice(0, MAX_RERANK_CANDIDATES);
return await reranker.rerank(query, toRerank, topN);
```

**Scientific basis:**
- Cross-encoders are O(n) — each document is scored independently against the query
- Top-30 from first-stage retrieval captures >99% of relevant documents (Nogueira et al., 2020)
- NVIDIA Blueprint uses top-25 by default
- Reduces reranker wall time by ~60% for large result sets

**Tests:** Unit test asserting candidate truncation. Benchmark comparison showing minimal NDCG impact.

### 5C: Score Calibration Telemetry (~30min)

Add logging to understand reranker score distributions.

**File:** `src/rerank/reranker.ts`

```typescript
// After reranking, log score distribution
const scores = results.map(r => r.score);
logger.debug(`Reranker scores: min=${Math.min(...scores).toFixed(3)}, max=${Math.max(...scores).toFixed(3)}, mean=${(scores.reduce((a,b)=>a+b,0)/scores.length).toFixed(3)}, spread=${(Math.max(...scores)-Math.min(...scores)).toFixed(3)}`);
```

**Questions to answer:**
- Are scores in [0,1] or raw logits?
- Is the distribution well-separated (good discrimination) or clustered (poor discrimination)?
- Does the text-only reranker struggle on certain document types (tables, code, structured data)?

### 5D: Evaluate Vision Models on Spark (~2h)

**Goal:** Determine if we can run the VL models on the DGX Spark.

**Models to evaluate:**
1. **`nvidia/llama-nemotron-embed-vl-1b-v2`** (1.7B params)
   - Llama 3.2 1B language encoder + SigLip2 400M image encoder
   - Produces 4096-dim embeddings (same as current text-only model!)
   - Supports: text-only, image-only, and interleaved text+image input
   - OpenAI-compatible API when served via vLLM

2. **`nvidia/llama-nemotron-rerank-vl-1b-v2`** (~1B params)
   - Multimodal cross-encoder: scores (query_text, document_image) pairs
   - Also scores (query_text, document_text) — backward compatible
   - Improves retrieval accuracy by 6-7% over text-only reranker on visual documents
   - vLLM support merged (commit b428fd7)

**Steps:**
1. Check Spark GPU memory budget: `nvidia-smi` (what's currently loaded?)
2. Estimate VRAM for VL models:
   - embed-vl-1b: ~3.5GB FP16, ~2GB INT8
   - rerank-vl-1b: ~2.5GB FP16, ~1.5GB INT8
   - Current embed-8b: ~16GB FP16 (would FREE this!)
3. Test serving embed-vl via vLLM on Spark:
   ```bash
   vllm serve nvidia/llama-nemotron-embed-vl-1b-v2 \
     --task embed --dtype float16 --max-model-len 4096
   ```
4. Test serving rerank-vl via vLLM:
   ```bash
   vllm serve nvidia/llama-nemotron-rerank-vl-1b-v2 \
     --task score --dtype float16
   ```
5. Benchmark latency and throughput for:
   - Text embedding (must match or beat current 8B model speed)
   - Image embedding (new capability — establish baseline)
   - Text reranking (must beat current 41s catastrophe)
   - Image reranking (new capability — establish baseline)

**Key question:** The VL embedder is 1.7B vs current 8B. Is the quality tradeoff acceptable?
- embed-vl-1b: MTEB Retrieval score ~62 (text-only)
- embed-8b: MTEB Retrieval score ~72 (text-only)
- But embed-vl-1b gains back quality via multimodal understanding on visual docs

**Deliverable:** Benchmark report comparing:
- VL-1B text retrieval quality vs current 8B text retrieval
- VL-1B image retrieval quality (new capability)
- Latency comparison across all operations
- VRAM footprint analysis

### 5E: Design Multimodal Ingestion Pipeline (~2h)

If 5D shows the VL models are viable, design the new ingestion pipeline.

**New document types and their processing:**

| Document Type | Current Pipeline | New Pipeline |
|---|---|---|
| **Markdown/text** | chunk → embed-8b(text) | chunk → embed-vl-1b(text) |
| **PDF (text-heavy)** | GLM-4V OCR → text → chunk → embed | `pdftotext` → chunk → embed-vl-1b(text) + render pages → embed-vl-1b(image) |
| **PDF (visual/charts)** | GLM-4V OCR → lossy text → embed | render pages → embed-vl-1b(image) ← **massive quality gain** |
| **Images** | GLM-4V describe → text → embed | embed-vl-1b(image) directly |
| **Screenshots** | Not supported well | embed-vl-1b(image) directly |

**Key design decisions:**

1. **Dual indexing for PDFs:** Store both text chunks AND page images. Text for FTS (BM25 still works on extracted text). Images for visual retrieval. RRF merges both signals.

2. **Page-level vs chunk-level for images:** Each PDF page = one image embedding. This is coarser than text chunking (which might have 3-5 chunks per page), but it's what ColPali/VL models are optimized for. The reranker-VL then does fine-grained relevance scoring on the page image.

3. **Backward compatibility:** Text-only documents (markdown, notes, agent memory) continue through the text pipeline. The VL embedder handles text just as well (it's a superset). No regression for existing content.

4. **OCR removal:** GLM-4V OCR is completely eliminated from the pipeline. `pdftotext` (a simple, fast, no-GPU utility) handles basic text extraction for FTS indexing. Visual understanding comes from the VL models.

5. **Storage impact:** Page images add storage. A typical PDF page renders to ~200KB PNG at 150 DPI. For 1000 pages, that's ~200MB of images. LanceDB stores the 4096-dim vectors (~16KB each), not the images. Images are stored on disk with path references.

**Files to modify:**
- `src/ingest/` — new image ingestion path
- `src/embed/provider.ts` — add image embedding support
- `src/rerank/reranker.ts` — add image reranking support  
- `src/storage/lancedb.ts` — store image metadata alongside vectors
- `src/config.ts` — multimodal configuration options

### 5F: Implement VL Embedding Provider (~3h)

**File:** `src/embed/provider.ts`

Extend the `EmbedProvider` interface:

```typescript
interface EmbedProvider {
  embedQuery(text: string): Promise<number[]>;
  embedDocument(text: string): Promise<number[]>;
  embedImage(imagePath: string): Promise<number[]>;  // NEW
  embedBatch(texts: string[]): Promise<number[][]>;
  embedImageBatch(imagePaths: string[]): Promise<number[][]>;  // NEW
}
```

The VL embedder API (vLLM OpenAI-compatible):
```typescript
// Text embedding (same as current)
POST /v1/embeddings
{ "model": "nvidia/llama-nemotron-embed-vl-1b-v2", "input": "query text" }

// Image embedding (new)
POST /v1/embeddings  
{ "model": "nvidia/llama-nemotron-embed-vl-1b-v2",
  "input": [{ "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } }] }
```

**Important:** The VL embedder uses the same 4096-dim output as our current 8B model. This means we could potentially run both in parallel during migration — VL for new content, 8B for existing — and they'd (roughly) share the same vector space. However, for optimal results, a full reindex is recommended.

### 5G: Implement VL Reranker (~2h)

**File:** `src/rerank/reranker.ts`

The VL reranker accepts (query, document) pairs where document can be text OR image:

```typescript
// Text reranking (backward compatible)
POST /v1/score
{ "model": "nvidia/llama-nemotron-rerank-vl-1b-v2",
  "query": "What is the agent configuration?",
  "documents": ["The agent config handles model routing..."] }

// Image reranking (new)
POST /v1/score
{ "model": "nvidia/llama-nemotron-rerank-vl-1b-v2",
  "query": "What is the agent configuration?",
  "documents": [{ "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } }] }
```

The reranker sees the actual page image and determines if it's relevant to the query. This is far superior to reranking OCR'd text, especially for:
- Tables (layout preserved)
- Charts/graphs (visual data preserved)
- Code with syntax highlighting
- Forms, receipts, structured documents

### 5H: PDF Page Renderer (~1h)

Simple utility to convert PDF pages to images for embedding:

```typescript
// Using pdf-img-convert or pdf2pic (Node.js)
async function renderPdfPages(pdfPath: string, dpi = 150): Promise<string[]> {
  // Returns array of image file paths, one per page
}
```

**DPI choice:** 150 DPI balances quality vs embedding speed. The SigLip2 vision encoder resizes to 384×384 anyway, so ultra-high DPI is wasted. 150 DPI gives clear text at typical page sizes.

**Alternative:** `pdftoppm` (poppler-utils) — fast, no dependencies, battle-tested.

### 5I: Integration Testing + BEIR Re-benchmark (~4h)

1. **Unit tests:** Verify VL embedder produces 4096-dim vectors for both text and images
2. **Integration test:** End-to-end: ingest a PDF (image path), query with text, get relevant page back
3. **BEIR re-benchmark:** Run full BEIR suite with:
   - Config A: VL vector-only (text queries, text docs) — compare to current 8B baseline
   - Config C: VL hybrid (text + FTS) — should exceed both baselines
   - Config D: VL hybrid + VL reranker — target: NDCG@10 ≥ 0.85 on SciFact
4. **Visual document test:** Create a small test set of PDFs with tables/charts. Compare:
   - Old pipeline: OCR → text → embed → search
   - New pipeline: render → image embed → search
   - Measure: retrieval accuracy on visual content queries

---

## Migration Strategy

### Phase 5.1 (Minimal — Current Sprint)
- 5A-5C: Fix current text reranker (latency diagnosis, candidate limiting, telemetry)
- No architecture changes. Immediate production improvement.

### Phase 5.2 (Evaluation — Next Sprint)  
- 5D: Benchmark VL models on Spark
- Decision point: if VL quality is acceptable, proceed to 5.3. If not, stay with optimized text-only pipeline.

### Phase 5.3 (Migration — Following Sprint)
- 5E-5H: Implement multimodal pipeline
- 5I: Full testing and benchmarking
- Decommission GLM-4V OCR service
- Reindex all documents with VL embedder

### Rollback Plan
- Keep text-only embed-8b service running during migration
- Both pipelines can coexist (separate LanceDB tables)
- If VL quality is insufficient, revert to text-only with Phase 4 fixes
- GLM-4V OCR service is not deleted until VL pipeline is proven in production

---

## Resource Impact

### VRAM Budget (Spark — 128GB total)

| Model | Current VRAM | New VRAM | Delta |
|---|---|---|---|
| embed-8b (text) | ~16GB FP16 | — | **Freed** |
| embed-vl-1b (multimodal) | — | ~3.5GB FP16 | **New** |
| rerank-1b (text) | ~2GB | — | **Replaced** |
| rerank-vl-1b (multimodal) | — | ~2.5GB FP16 | **New** |
| Nemotron-Super-120B (LLM) | ~80GB NVFP4 | ~80GB | No change |

**Net result:** ~12GB VRAM freed. The VL models are SMALLER than what they replace.

### Storage Impact
- Page images: ~200KB per page × estimated 5000 pages = ~1GB
- Additional vectors: 4096-dim × 4 bytes × 5000 = ~80MB
- Minimal compared to existing LanceDB size (~1.1GB for BEIR alone)

### Latency Impact (Expected)
- Embedding: 1.7B model should be **faster** than 8B model (4.7x fewer params)
- Reranking: 1B model on GPU should be **<100ms** vs current 41s catastrophe
- Image processing: ~50ms per page render + ~20ms per image embed
- Net: significant improvement across the board

---

## What Dies

| Component | Status | Reason |
|---|---|---|
| GLM-4V OCR service | **Decommissioned** | Replaced by vision embedder |
| OCR text extraction pipeline | **Removed** | No longer needed |
| OCR error handling / retry logic | **Removed** | Complexity reduction |
| `src/ocr/` directory | **Archived** | Keep for reference |
| 502 errors from OCR service | **Gone** | One fewer service to fail |

**What we gain:** One fewer GPU service, simpler pipeline, visual understanding, smaller models, lower latency, no more OCR-induced text corruption.

---

## Alternative Approaches Considered

### ColPali / ColQwen2 (Late Interaction)
- **Pro:** State-of-the-art visual document retrieval (62.7 NDCG@5 on ViDoRe-v2)
- **Pro:** Token-level matching — finds specific regions within a page
- **Con:** Late interaction requires storing per-token embeddings (~1000 vectors per page vs 1)
- **Con:** Not OpenAI-compatible API — requires custom serving
- **Con:** LanceDB doesn't natively support late interaction / MaxSim scoring
- **Verdict:** More complex, higher quality ceiling. Consider for Phase 7 if VL models aren't sufficient.

### Nomic Embed Multimodal 7B
- **Pro:** 62.7 NDCG@5 on ViDoRe-v2 (SOTA)
- **Con:** 7B params — larger than current 8B text model
- **Con:** Not NVIDIA ecosystem — less optimized for our Spark hardware
- **Verdict:** Worth evaluating if NVIDIA VL models underperform.

### Keep OCR + Add Vision Reranker Only
- **Pro:** Minimal code changes — just swap reranker
- **Con:** Still dependent on OCR quality for initial retrieval
- **Con:** Reranker can't rescue documents that OCR completely mangled
- **Verdict:** Half-measure. If we're touching the reranker, go all the way.

---

## References

1. NVIDIA (2026). "Llama Nemotron Embed VL 1B v2." HuggingFace model card.
2. NVIDIA (2026). "Llama Nemotron Rerank VL 1B v2." HuggingFace model card.
3. NVIDIA (2026). "Small Yet Mighty: Improve Accuracy in Multimodal Search with Llama Nemotron RAG Models." HuggingFace blog.
4. Faysse et al. (2024). "ColPali: Efficient Document Retrieval with Vision Language Models." arXiv:2407.01449
5. Nogueira et al. (2020). "Document Ranking with a Pretrained Sequence-to-Sequence Model." EMNLP.
6. Nomic (2025). "Nomic Embed Multimodal: Open Source Multimodal Embedding Models." nomic.ai blog.
7. NVIDIA (2026). "Best-in-Class Multimodal RAG: How the Llama 3.2 NeMo Retriever Embedding Model Boosts Pipeline Accuracy." NVIDIA Technical Blog.
