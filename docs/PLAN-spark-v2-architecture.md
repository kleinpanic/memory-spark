# Spark Infrastructure v2 — Architecture Migration Plan

**Authored:** 2026-04-03  
**Status:** Planned — not yet started  
**Context:** memory-spark v0.4.0 is stable. This plan describes the next major infrastructure evolution for the Spark node service stack.

---

## Current State (v1)

The current Spark service stack runs on the NVIDIA DGX Spark node:

| Port | Service | Model | Role |
|------|---------|-------|------|
| 18080 | vLLM chat completions | `NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | HyDE generation, GLM-OCR (dual-use) |
| 18091 | Embedding service | `nvidia/llama-embed-nemotron-8b` | Query + document embedding |
| 18094 | STT (Parakeet) | `nvidia/Parakeet-CTC-1.1B` | Voice-to-text transcription |
| 18096 | Reranker | `nvidia/llama-nemotron-rerank-1b-v2` | Cross-encoder reranking |
| 18097 | EasyOCR | EasyOCR (legacy Python service) | Fallback OCR for scanned PDFs |
| 18110 | Summarizer | — | Text summarization |
| 18112 | NER | — | Named entity recognition |
| 18113 | Zero-shot classifier | — | Capture classification |

### Known Problems with v1

1. **Nemotron-Super-120B is overkill for HyDE.** Generating a 150-token hypothetical document does not require a 120B parameter reasoner. Cold-start latency is high; VRAM pressure is severe. During BEIR benchmarks, HyDE failed 100% of the time due to timeouts on this model.

2. **EasyOCR (port 18097) is a legacy service** — Python-based, slow, unreliable on complex layouts. It was the original OCR fallback and has been superseded by GLM-OCR running on port 18080. Port 18097 should be retired.

3. **GLM-OCR shares port 18080 with the reasoning LLM** — this is a scheduling conflict. GLM-OCR (zai-org/GLM-OCR, 0.9B) and Nemotron-Super-120B share a single vLLM endpoint, meaning OCR jobs compete with HyDE/inference jobs for VRAM. They should be separate services.

4. **No multimodal (VL) embedding.** The current embed model (`llama-embed-nemotron-8b`) is text-only. Documents with diagrams, tables, and figures lose their visual information entirely during chunking. A vision-language (VL) embedding model would capture both.

5. **Reranker discrimination is weak at 1B scale.** The `llama-nemotron-rerank-1b-v2` shows high score saturation (58% of top results ≥ 0.999) and poor discrimination on scientific claims. A VL-capable reranker or a larger cross-encoder would help.

---

## Proposed v2 Architecture

### Core Principle
Replace OCR → text extraction with VL embedding. Instead of converting images to text then embedding the text, embed the visual representation directly. This preserves layout, tables, equations, and figures that OCR routinely loses.

### Service Stack Changes

| Port | v1 | v2 | Notes |
|------|----|----|-------|
| 18080 | Nemotron-Super-120B (HyDE + OCR dual-use) | **Small fast LLM** for HyDE only | Nemotron-Mini-4B-Instruct or similar |
| 18081 | *(free)* | **GLM-OCR** (dedicated port) | Decoupled from HyDE LLM |
| 18091 | `llama-embed-nemotron-8b` (text-only) | **VL Embedding model** | Multimodal: text + image |
| 18096 | `llama-nemotron-rerank-1b-v2` | **VL Reranker or larger cross-encoder** | Better discrimination |
| 18097 | EasyOCR (legacy) | **RETIRED** | Replaced by VL embedding |
| 18094 | Parakeet STT | Parakeet STT | No change |
| 18110–18113 | NER, zero-shot, summarizer | NER, zero-shot, summarizer | No change |

---

## Migration Details

### M1: Replace Nemotron-Super with Small LLM for HyDE

**Problem:** 120B parameter reasoner is absurd for generating a 150-token paragraph. VRAM: ~60GB. Cold start: 30–90s.

**Solution:** Use a small instruction-tuned model — `nvidia/Nemotron-Mini-4B-Instruct` (4B params, ~8GB VRAM) or similar. These models generate coherent HyDE documents at <1s latency. Quality difference for HyDE: negligible (HyDE just needs a fluent passage in the right domain, not deep reasoning).

**Config change:**
```json
{
  "hyde": {
    "model": "nvidia/Nemotron-Mini-4B-Instruct",
    "timeoutMs": 5000
  }
}
```

**VRAM freed:** ~52GB — enough to run the VL embedder and VL reranker concurrently.

**Port:** Keep 18080 for backward compatibility.

---

### M2: Dedicate GLM-OCR to Port 18081

**Problem:** GLM-OCR (0.9B VL model) and Nemotron-Super-120B share port 18080 — OCR jobs during ingestion block HyDE during recall.

**Solution:** Run GLM-OCR on a dedicated vLLM instance at port 18081.

```bash
# New GLM-OCR service (systemd or podman)
vllm serve zai-org/GLM-OCR \
  --port 18081 \
  --gpu-memory-utilization 0.15 \
  --max-model-len 8192
```

**Config change:**
```json
{
  "spark": {
    "glmOcr": "http://<sparkHost>:18081/v1"
  }
}
```

**Code change:** Update `src/config.ts` `SparkServices.glmOcr` default port to 18081.

---

### M3: VL Embedding Model (Replaces Text-Only Embed + EasyOCR for images)

**Problem:** Current embedding is text-only. PDF pages with figures, charts, equations embed as blank/garbled text after OCR. VL models embed the visual token sequence directly alongside text.

**Candidate models (as of 2026-04):**

| Model | Dims | VRAM | Notes |
|-------|------|------|-------|
| `nvidia/llama-embed-nemotron-8b-vl` | 4096 | ~16GB | Drop-in replacement if released |
| `BAAI/bge-visualized` | 1024 | ~8GB | BGE family, text+image |
| `Qwen/Qwen2-VL-7B-Instruct` | — | ~14GB | Full VL model, dual-encoder mode |
| `nomic-ai/nomic-embed-vision` | 768 | ~4GB | Lightweight, text+image |

**Preferred:** Whatever NVIDIA releases as the VL successor to `llama-embed-nemotron-8b`. Track https://build.nvidia.com/explore/retrieval.

**What changes in code:**
- `src/embed/provider.ts`: Add `vl` provider type alongside `spark`/`openai`/`gemini`
- `src/ingest/parsers.ts`: For image-heavy PDFs, pass raw page image to VL embedder instead of (or alongside) OCR text
- `src/storage/backend.ts`: Dimension field needs to be flexible (4096 → variable based on model)
- Pool schema: `content_type: "image"` chunks store the image embedding, retrieved alongside text chunks

**Important:** Text-only queries still work — VL embedding models handle text queries natively. Queries with attached images would also work. This is additive.

---

### M4: VL Reranker (Replaces llama-nemotron-rerank-1b-v2)

**Problem:** The 1B reranker shows high score saturation (58% ≥ 0.999) and poor discrimination, especially on scientific claims vs QA-format text.

**Options:**
1. **Larger cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-12-v2` (12 layers) or `BAAI/bge-reranker-v2-m3` (566M, multilingual) — better discrimination at text-only
2. **VL cross-encoder:** If the embed model is VL, the reranker should also handle (query_text, image_chunk) pairs — rare but correct
3. **NVIDIA Nemotron reranker larger variant** — if/when released

**Near-term recommendation:** Swap to `BAAI/bge-reranker-v2-m3`. Serves via the same Torchserve/vLLM reranker API. Better discrimination, ~twice the latency (240ms vs 119ms) — acceptable given GATE-A skips 78% of queries.

**Port:** Keep 18096.

---

### M5: Retire EasyOCR (Port 18097)

**Problem:** EasyOCR is slow, Python-dependent, fails on complex layouts, and is already superseded by GLM-OCR (port 18081 in v2). The fallback chain `pdfjs → GLM-OCR → EasyOCR` becomes `pdfjs → GLM-OCR` in v2.

**Code change in `src/ingest/parsers.ts`:**
- Remove `EASY_OCR_TIMEOUT_MS` and `easyOcrRequest` function
- Remove the EasyOCR fallback branch in `parsePdf()`
- Remove `spark.ocr` from `SparkServices` config type
- Update integration test to not probe port 18097

**Config change:** Remove `spark.ocr` field. Add deprecation warning if present.

---

## Migration Sequence

Phase the migration to avoid breaking production:

```
Phase A (immediate, no hardware change):
  - M1: Swap HyDE model to Nemotron-Mini on port 18080
  - M2: Move GLM-OCR to dedicated port 18081
  - Benefit: frees 52GB VRAM, removes scheduling conflict

Phase B (requires VL model selection):
  - M3: Deploy VL embedder, add vl provider in embed/provider.ts
  - Requires re-indexing all existing chunks (vector dim change)
  - Plan: dual-write period, then cutover

Phase C (after Phase B is stable):
  - M4: Replace reranker with better cross-encoder
  - M5: Retire EasyOCR service + remove from codebase

Phase D (future, hardware permitting):
  - Run VL embed + rerank on dedicated partition
  - Enable image-native RAG for multimodal workspaces
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| VL embed dims differ from 4096 | High | Medium | Schema migration script, config-driven dims |
| Re-indexing breaks existing recalls | Medium | High | Keep old index as fallback during cutover |
| Small HyDE LLM generates lower-quality hypotheticals | Low | Low | HyDE is best-effort; NDCG impact small vs direct embed |
| GLM-OCR on 18081 VRAM conflict | Low | Medium | Set `--gpu-memory-utilization 0.15` to cap headroom |
| BGE reranker API incompatible | Low | Low | Same rerank API format; test against integration suite |

---

## Open Questions

1. **Which VL embedding model?** Need to benchmark `bge-visualized` vs `nomic-embed-vision` vs NVIDIA VL embed on our specific workload (mostly markdown + code, occasionally PDFs with figures).
2. **Re-indexing strategy:** Live cutover vs parallel index build vs scheduled downtime?
3. **Image storage:** VL-embedded image chunks need the raw image stored somewhere (not just the vector). Do we store image hashes in LanceDB and images on disk, or encode them?
4. **Query-side:** When an agent sends a text query, does the VL embedder handle it identically to the text-only embedder? (Answer: yes for all models listed, but verify instruction prefix format.)

---

## Related Files

- `src/config.ts` — `SparkServices` type, `buildDefaults()` port assignments
- `src/embed/provider.ts` — provider selection (`spark`, `openai`, `gemini` → add `vl`)
- `src/ingest/parsers.ts` — OCR chain (`glmOcrRequest`, `easyOcrRequest`)
- `src/storage/lancedb.ts` — vector dimension handling
- `docs/ARCHITECTURE.md` — update when v2 ships
- `paper/memory-spark.tex` — §8 Future Work documents this plan
