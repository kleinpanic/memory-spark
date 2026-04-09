# Stack Research — memory-spark v1.0 (2026-current validation)

**Domain:** Scientifically-rigorous RAG memory plugin for LLM agents
**Researched:** 2026-04-09
**Researcher:** gsd-new-milestone research phase
**Overall confidence:** HIGH (all recommendations sourced from HuggingFace model cards, official docs, or peer-reviewed papers ≤ 2026-02)

---

## TL;DR Verdicts on Existing Stack

| Component | Current | Verdict | Reason |
|-----------|---------|---------|--------|
| LanceDB (vector+FTS) | `@lancedb/lancedb ^0.27.1` | **KEEP** | Native FTS + pre-filter pushdown (2025 release) is 2026-current; only local-first hybrid DB with disk-native scale |
| Nemotron text embedder | `nvidia/llama-embed-nemotron-8b` (4096d) | **REPLACE** | Released 2025-10-21. Top MMTEB multilingual Borda rank, but **license is non-commercial (`customized-nscl-v1` + Llama 3.1 community)**. Incompatible with v1.0 public release. |
| Reranker | `nvidia/llama-nemotron-rerank-1b-v2` | **REPLACE** | Same NVIDIA non-commercial license class; additionally shows score saturation (58% ≥ 0.999) per existing audit. |
| HyDE LLM | `Nemotron-Super-3-122B` | **DOWNGRADE** (already planned) | 120B for 150-token hypothetical is absurd; 100% timeout in BEIR runs. |
| HyDE itself | enabled, configurable | **KEEP + GATE** | HyDE is *not* replaced in 2026. It's refined: gated on low-confidence queries, combined with cross-encoder validation. HyPE is additive, not a replacement. |
| Hybrid vector+FTS + RRF | present | **KEEP** | Still SOTA pattern; Anthropic Contextual Retrieval (2024) validates hybrid BM25 + vector + reranker as the gold standard (-49% retrieval errors). |
| MMR diversity | present | **KEEP** | No 2026 replacement with better cost/benefit for small top-k. |
| EasyOCR fallback | port 18097 | **RETIRE** (already planned) | Superseded by GLM-OCR + VL embedding directly. |
| Flat 500-char chunking | present | **AUGMENT** | Keep for short agent memories (already atomic). Add *contextual retrieval* for workspace doc ingestion (documented +49% recall improvement, Anthropic 2024). |
| Tool injection | TOOLS.md static prompt | **REPLACE WITH SEMANTIC RETRIEVAL** (already planned) | `content_type="tool"` per langgraph-bigtool pattern. Cited paper (arXiv:2603.20313) shows 97.1% hit@3, 99.6% token reduction. |
| Eval: BEIR + mock IR metrics | partial | **EXPAND** | Add golden dataset with gold answers + RAGAS-lite (context precision/recall) OR ARES if time permits. |

---

## Recommended Stack (2026-current)

### 1. Embedding Model (the biggest change)

| Attribute | Current | Recommended |
|-----------|---------|-------------|
| Model | `nvidia/llama-embed-nemotron-8b` | **`Qwen/Qwen3-VL-Embedding-8B`** (primary) OR **`Qwen/Qwen3-VL-Embedding-2B`** (dev/low-VRAM) |
| HF ID | `nvidia/llama-embed-nemotron-8b` | `Qwen/Qwen3-VL-Embedding-8B` |
| Parameters | 8B | 8B (2B variant for dev) |
| Dims | 4096 fixed | Up to **4096 with MRL down to 64** (configurable at inference) |
| Context | 32k | 32k |
| Modalities | text only | **text + images + screenshots + video + mixed** |
| Languages | multilingual (English focus) | 30+ |
| License | `customized-nscl-v1` + Llama 3.1 community (NON-COMMERCIAL) | **Apache 2.0 (commercial OK)** |
| MMEB-V2 overall | n/a (text-only) | **77.9** (8B), 73.2 (2B) — SOTA open source |
| ViDoRe v1/v2 | n/a | 88.2 / 69.9 (8B) |
| Release | 2025-10-21 | 2026-01 (arXiv 2601.04720) |
| Serving | vLLM `runner=pooling` (built-in) | vLLM `runner=pooling` (built-in, example in model card) |

**Why Qwen3-VL-Embedding-8B over alternatives:**

1. **License clears public release.** Apache 2.0. This is the *only* 2026-current VL embedder at this quality tier that clears commercial/public-release usage without negotiation.
2. **MRL support** means you can store 4096-dim for max quality, then experiment with 1024d or 512d for storage-efficient tables without retraining — solves the "4096 = 4x storage overhead" concern from `docs/RESEARCH-SOTA-2026.md`.
3. **Built on Qwen3-VL foundation**, the same model Jina-v4 is derived from — but Qwen licensed it cleanly rather than Jina-style non-commercial.
4. **Native vLLM embedding runner** — drop-in replacement for the existing `llama-embed-nemotron-8b` vLLM deployment on port 18091.
5. Matches the v2 architecture plan goal in `docs/PLAN-spark-v2-architecture.md` M3 exactly (multimodal VL embedder), but with a better license answer than the `llama-embed-nemotron-8b-vl` "if released" placeholder.

**Alternatives explicitly rejected (with reasoning):**

| Model | HF ID | Reject Reason |
|-------|-------|---------------|
| `nvidia/nemotron-colembed-vl-8b-v2` | ✓ | **CC-BY-NC-4.0 — non-commercial.** Ranks #1 on ViDoRe v3 (63.54 NDCG@10), but cannot ship in a public v1.0. Strong candidate *if* you negotiate commercial license with NVIDIA. |
| `nvidia/nemotron-colembed-vl-4b-v2` | ✓ | Same CC-BY-NC-4.0 restriction. |
| `nvidia/llama-nemotron-embed-vl-1b-v2` (NIM) | ✓ | NVIDIA NIM endpoint, same license family. |
| `jinaai/jina-embeddings-v4` | ✓ | **CC-BY-NC-4.0 + Qwen Research License.** Jina themselves confirm: "cannot offer as commercial product." Need Jina managed API for production. 2048-dim single-vector + 128-dim multi-vector, ViDoRe SOTA among open — but license blocks us. |
| `nomic-ai/nomic-embed-vision` | ✓ | 768 dims, Apache 2.0, but one generation behind on MMEB/ViDoRe. Keep as fallback reference only. |
| `BAAI/bge-visualized` | ✓ | Older generation (2024), BAAI family. Not competitive on ViDoRe v2/v3. |
| `Qwen/Qwen3-Embedding-8B` (text-only) | ✓ | Would work (Apache 2.0, MRL 32-4096, MTEB multilingual #1 70.58 as of 2025-06), but **abandons the VL upgrade path.** Use only as text-only fallback if VL inference blows latency budget. |
| `nvidia/llama-embed-nemotron-8b` (status quo) | ✓ | Non-commercial license (blocks public release). Tops MMTEB but text-only. |

**Confidence: HIGH.** Direct HuggingFace model card verification of dims, license, benchmark, release date, vLLM support.

---

### 2. Reranker

| Attribute | Current | Recommended |
|-----------|---------|-------------|
| Model | `nvidia/llama-nemotron-rerank-1b-v2` | **`Qwen/Qwen3-VL-Reranker-2B`** (primary) OR **`BAAI/bge-reranker-v2-m3`** (text-only fallback) |
| HF ID | n/a non-commercial | `Qwen/Qwen3-VL-Reranker-2B` |
| Parameters | 1B | 2B |
| Context | — | 32k |
| Modalities | text | **text + image (cross-modal reranking)** |
| License | NVIDIA non-commercial | **Apache 2.0** |
| MMEB-v2 retrieval avg | n/a | 75.1 (image: 73.8, VisDoc: 83.4) |
| ViDoRe v3 | n/a | 60.8 |
| JinaVDR | n/a | 80.9 |

**Why Qwen3-VL-Reranker-2B:**

1. **License.** Apache 2.0 — same licensing reason as the embedder.
2. **Paired with Qwen3-VL-Embedding-8B** in the same arXiv paper (2601.04720) as a designed two-stage pipeline. Cross-encoder scores on same modality as the retriever = consistent discrimination.
3. **Fixes the score saturation** observed with Nemotron 1B (58% ≥ 0.999 per existing audit). 2B+ discriminator models trained on multi-task retrieval datasets don't exhibit this collapse.
4. **Cross-modal reranking** — when the retriever returns an image chunk and the query is text, a text-only reranker has nothing to say. The Qwen3-VL-Reranker scores (query_text, image_page) pairs natively.

**bge-reranker-v2-m3 as fallback:**

- 566M params, multilingual, Apache 2.0, same hosted-API shape.
- Placed 4th behind Nemotron, gte_modernbert, and Jina in 2026 reranker evals — **but is text-only**.
- Recommended *only* if Qwen3-VL-Reranker-2B exceeds VRAM budget or latency requirement at your batch sizes.
- ~240ms latency vs llama-nemotron 119ms per existing `docs/PLAN-spark-v2-architecture.md` M4 — acceptable since GATE-A skips 78% of queries.

**Reranker latency frame of reference (2026 benchmarks, agentset.ai):**

| Model | Hit@1 | Latency |
|-------|-------|---------|
| `jina-reranker-v3` | 81.33% | 188ms |
| `nvidia/llama-nemotron-rerank-1b-v2` | 83.00% | 243ms |
| `BAAI/bge-reranker-v2-m3` | ~78% | ~240ms (batch 16) |
| `Qwen/Qwen3-Reranker-4B` (text) | higher | >1s (autoregressive) ← **avoid** |
| `Qwen/Qwen3-Reranker-0.6B` (text) | ~78% | ~85ms GPU |

**Critical warning:** `Qwen/Qwen3-Reranker-4B` (text) uses causal LM yes/no-logit decoding → **>1s/query**. This is documented: SequenceClassification rerankers (one forward pass) are fundamentally faster. Qwen3-VL-Reranker-2B is the *VL* variant with a different inference path — confirm latency on your hardware before committing.

**Explicitly rejected:**

| Model | Reject Reason |
|-------|---------------|
| `nvidia/llama-nemotron-rerank-1b-v2` | Non-commercial license + documented saturation bug |
| `Qwen/Qwen3-Reranker-4B` (text) | >1s latency — autoregressive yes/no decoding blows budget |
| `jinaai/jina-reranker-v3` | Jina production = managed API; self-hosted checkpoint licensing uncertain |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Apache 2.0 and fast, but 2022-era, BEIR-domain only, no VL |

**Confidence: HIGH** for model selection; **MEDIUM** for latency projections (need on-hardware benchmark before commitment).

---

### 3. Vector Database + Hybrid Search

| Attribute | Current | Recommended |
|-----------|---------|-------------|
| Database | `@lancedb/lancedb ^0.27.1` | **KEEP LanceDB. Pin ≥ 0.27.1.** |
| Hybrid search | vector + Tantivy FTS + RRF | **Migrate to native FTS** (pass `use_tantivy=False` in new index builds) |
| Filter pushdown | post-filter on FTS | **Native FTS supports pre-filter pushdown** (2025 release) |
| Dimension | 4096 fixed | Use config-driven dim to support MRL truncation |

**Why LanceDB stays:**

1. **Embedded, disk-native, no separate service.** This is the single biggest architectural advantage for a plugin that ships inside another user's workspace. Qdrant/Weaviate/Milvus all require a running server process; Pinecone is SaaS.
2. **Native FTS has shipped** (LanceDB team's 2025 "No more Tantivy!" blog + WikiSearch demo on 41M docs). This directly addresses the current post-filter limitation in your code's Tantivy path. Pass `use_tantivy=False` or use the native FTS API.
3. **Hybrid search API exists** with built-in BM25 + vector + RRF merge — you already use this via `hybridMerge.ts`. No migration cost.
4. **2026 roadmap** includes Lance-native SQL retrieval via DuckDB and uber-scale multi-bucket storage. Active development.
5. **Multi-modal ready** — LanceDB markets itself as "AI-Native Multimodal Lakehouse" and is used by Continue.dev's local-first AI stack. Supports multi-vector (ColBERT-style) storage if you ever migrate to the NVIDIA ColEmbed family.

**Alternatives explicitly considered and rejected for this plugin:**

| Alternative | Why Not |
|-------------|---------|
| Qdrant | Requires server process. Best filter pushdown of the three — but its strength (complex payload filters) isn't your bottleneck. Migration cost > benefit. |
| Weaviate | Best-in-class hybrid search with `relativeScoreFusion` — but also requires a server. Overkill for per-user memory. |
| Milvus | Distributed-scale vector DB. Wrong shape for single-agent memory. |
| Pinecone | SaaS, not embeddable, plugin ships user data to third-party = privacy violation. Hard NO. |
| pgvector | Requires Postgres. Simplicity lost, hybrid search weaker. |

**LanceDB migration notes (LOW-risk):**

- Your code path `hybridMerge.ts` already consumes `(vectorHits, ftsHits)` tuples from LanceDB. Switching to native FTS is one flag.
- **Re-index required** when switching native FTS (index format differs). Coordinate with VL embedder cutover (both require rebuild) to do a single migration.
- Verify `apache-arrow ^18.0.0` still aligns with `@lancedb/lancedb ^0.27.1` after pin (LanceDB bumps Arrow versions aggressively — check their changelog on pin).

**Confidence: HIGH** on keep-LanceDB decision; **MEDIUM** on the native FTS + pre-filter claim (verified via LanceDB blog and docs index, but didn't retrieve the exact API page). Flag for phase-1 recon: confirm the TypeScript SDK exposes native FTS (Python SDK does).

---

### 4. HyDE and Query Expansion

**Verdict: KEEP HyDE. Refine, don't replace.**

The assertion in `docs/RESEARCH-SOTA-2026.md` (2026-03-26 internal notes) that "HyDE — NOT for us" is **partially stale**. That conclusion was driven by two things:

1. Nemotron-Super-120B's 100% timeout rate during BEIR runs — which is the **LLM choice**, not HyDE the technique.
2. A latency budget framing that assumed every query must use HyDE.

**2026 literature view:**

- HyDE has *not* been replaced by any 2026 technique. It's refined.
- **HyPE** (Hypothetical Prompt Embeddings, Vake et al. 2025) is **additive, not substitutive**: index-time query generation per document. Paper claims +42pts precision, +45pts recall on specific datasets. But: it requires an LLM pass *per document at ingest*, which is a totally different cost profile.
- **Step-back prompting** (Google Brain 2024) is a different technique — rewriting to a more abstract query. Not a HyDE replacement either.
- **Current best practice** (Haystack, Milvus, Zilliz 2026 docs): gate HyDE on low-confidence queries, validate with cross-encoder, apply domain-specific prompt guardrails.

**Recommended approach:**

1. **Phase A (existing plan is correct):** swap HyDE LLM from Nemotron-Super-120B to `nvidia/Nemotron-Mini-4B-Instruct` or equivalent 4B-8B instruction model. VRAM: ~8GB, latency <1s. HyDE doesn't need reasoning depth; it needs fluent domain language.
2. **Gate HyDE on confidence:** if top-1 vector sim > threshold, skip HyDE (current GATE-A pattern already does this — validate the thresholds).
3. **Consider HyPE for workspace doc ingestion path only** — at ingest time, generate 3-5 hypothetical queries per chunk and store them as additional searchable text. Cheap because it's offline. This is where the +42pt recall claim comes from. **Flag for post-v1.0** — too much scope for this milestone.

**Alternative HyDE LLMs to consider (must be fast + fluent + license-clear):**

| Model | Size | License | Notes |
|-------|------|---------|-------|
| `nvidia/Nemotron-Mini-4B-Instruct` | 4B | NVIDIA Open Model (check version) | Referenced in existing PLAN-spark-v2-architecture M1 |
| `Qwen/Qwen3-4B-Instruct` | 4B | Apache 2.0 | Clean license, 2026-current |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Llama community license | Smaller, faster |
| `microsoft/Phi-4-mini-instruct` | 3.8B | MIT | Fastest small model, great for HyDE's "just needs fluency" requirement |

**Recommended HyDE LLM:** `Qwen/Qwen3-4B-Instruct` or `microsoft/Phi-4-mini-instruct`. Both Apache/MIT, both fast, both fluent. Avoids NVIDIA license uncertainty entirely.

**Confidence: MEDIUM-HIGH.** HyDE's "not replaced" status is well-documented. The specific LLM recommendation is flexible; benchmark the top-3 candidates on your HyDE corpus before committing.

---

### 5. Semantic Tool Retrieval

**Verdict: IMPLEMENT as planned.** `docs/RESEARCH-TOOLS-INJECTION-2026.md` cites arXiv 2603.20313 (121 MCP tools, 99.6% token reduction, 97.1% hit@3, MRR=0.91, sub-100ms). This is **validated 2026 SOTA**.

**Recommended implementation:**

- Store tool schemas as `content_type="tool"` chunks in the existing LanceDB pool.
- Embed tool descriptions with the **same embedder** as memories (Qwen3-VL-Embedding-8B). Text-only for tools is fine; VL is overkill.
- Expose `memory_tools_retrieve(query)` plugin tool following the **langgraph-bigtool pattern** (arXiv 2603.20313 + LangChain repo).
- At recall time, detect tool-related queries and boost `content_type="tool"` chunks via metadata filter + score boost.

**Production pattern validated:**

The langgraph-bigtool pattern is the **most mature production pattern** as of 2026 (per LangChain official docs). Two-phase: agent starts with `retrieve_tools` as only tool → calls it → retrieved top-K tools become available. Supports hundreds/thousands of tools.

**Confidence: HIGH.** Source paper is peer-reviewable and has concrete numbers. LangChain ref implementation exists.

---

### 6. RAG Evaluation Framework

**Verdict: Build golden dataset + RAGAS-lite. Defer ARES to future.**

**2026 landscape:**

| Framework | Strength | Weakness | Use |
|-----------|----------|----------|-----|
| **RAGAS** | Reference-free, 4 core metrics (context precision/recall, faithfulness, answer relevance), synthetic test gen, LLM-as-judge, widely adopted | Heuristic prompts, no confidence scores, depends on judge LLM quality | **Recommended** for v1.0 — fastest path to credible eval |
| **ARES** (Stanford) | Fine-tuned judges per metric, confidence scoring, PPI statistical rigor, **+59.3pts context relevance, +14.4pts answer relevance vs RAGAS** | Requires ≥50 annotated triples per metric, training pipeline per deployment, heavier infrastructure | Defer to post-v1.0 |
| **BEIR** | Standard IR eval, 15+ datasets, NDCG@k, accepted in literature | Doesn't eval answer quality, only retrieval | **KEEP** — you have it, use for SciFact/FiQA/NFCorpus as planned |
| **Custom golden dataset** | Domain-specific (agent memory), gold answers, multi-phrasing | Generation cost, annotation quality concerns | **Build this** — Nemotron-Super-3-122B on DGX Spark |

**Recommended evaluation stack for v1.0:**

1. **BEIR benchmarks** (already planned): SciFact, FiQA, NFCorpus against `testDbBEIR`. Report NDCG@10 in paper.
2. **Golden dataset** (already planned): 50-200 query-answer-source triples, generated via Nemotron-Super-3-122B on DGX Spark, on **scrubbed/synthetic OpenClaw data only** (critical per privacy constraint).
3. **RAGAS-lite** (MVP): implement just `context_precision` and `context_recall` against the golden dataset. Use Nemotron-Super-3-122B as the judge LLM. This is the cheapest route to credible metrics.
4. **Full RAGAS** (stretch per PROJECT.md "Out of Scope" note): all 4 metrics + synthetic test gen if time allows. PROJECT.md already flags this as stretch — respect that.
5. **ARES**: **do not attempt in v1.0**. Training pipeline overhead + annotation burden is incompatible with timeline.

**Critical methodology requirement (scientific rigor):**

- **Never** run BEIR queries against `testDbOCMemory`. BEIR and agent-memory are different corpora; mixing produces garbage scores (already in PROJECT.md Key Decisions).
- **Report with confidence intervals** where possible. Even naive bootstrap on query-level NDCG scores gives you something citable.
- **Release the golden dataset generation script + scrubbing pipeline** as part of the paper's supplementary material (not the data itself — the pipeline). Reproducibility without privacy leak.

**Confidence: HIGH.** RAGAS vs ARES comparison is peer-reviewed (ARES paper arXiv 2311.09476 reports the +59.3/+14.4 deltas directly).

---

### 7. Chunking Strategy

**Verdict: Keep flat chunking for agent memories. Add contextual retrieval for workspace docs.**

**Key insight from 2026 literature + your existing notes:**

- Agent memories are **already atomic short facts (~50-200 tokens)**. They don't need chunking at all — they're propositions by construction. `docs/RESEARCH-SOTA-2026-VALIDATED.md` correctly identifies this.
- Workspace document ingestion (PDFs, markdown, wikis) is a **different problem**. For that, 2026 SOTA is:

**Recommended chunking by content type:**

| Content Type | Strategy | Rationale |
|--------------|----------|-----------|
| Agent memory (captured facts/prefs) | **No chunking** — embed whole | Already ≤200 tokens, already atomic |
| Session JSONL transcripts | **500-800 tokens, no overlap** | Matches existing code; 2025 Vectara NAACL finding: overlap provides no benefit with hybrid retrieval |
| Markdown docs | **Header-aware recursive + contextual retrieval** | Preserves structure; contextual retrieval gives +49% recall reduction per Anthropic 2024 paper |
| PDFs with figures/tables | **Page-level VL embedding** (via Qwen3-VL-Embedding) | Don't OCR then chunk — embed the visual page directly. This is the core architectural win from the v2 migration. |
| Code files | **Function/class boundary chunking** | Standard; existing parsers OK |
| Workspace wiki / long docs | **Parent-child hierarchical** | Retrieve by child (500 tok) for precision, return parent (2000 tok) for context |

**Contextual retrieval recipe (Anthropic 2024, validated in 2026):**

For each workspace doc chunk, prepend a short (50-100 token) LLM-generated context summary before embedding:

```
<context>
This chunk is from the "Configuration" section of the memory-spark README,
which describes how to set up the LanceDB backend and embedder endpoints.
</context>

[original chunk text]
```

Numbers: Anthropic reported **-49% retrieval errors** with contextual embeddings + contextual BM25. This is the single highest-leverage change you can make to workspace ingestion.

**Cost:** One LLM call per chunk at ingest. Use the same small HyDE LLM (`Qwen3-4B-Instruct` or `Phi-4-mini`). Offline, cacheable.

**Late chunking** (Jina 2024): defers chunking until after token-level embedding, then mean-pools per chunk. +10-12% on anaphoric references. Worth testing for docs with pronouns ("it", "this system"). Lower priority than contextual retrieval.

**Proposition chunking** (from existing notes): atomic facts per chunk. Highest precision on factoid queries (+20-30%), but expensive to generate. **Already implicit** in your `extract-facts`-style capture pipeline. Not worth re-implementing separately.

**Explicitly reject:**

- **Chunk overlap 10-20%** — 2026 LanceDB blog finding: with SPLADE/hybrid retrieval, overlap provides no benefit and wastes storage. Your existing notes correctly identify this. Keep 0 overlap.
- **Agentic chunking** (LLM decides per-doc): overkill, fragile, non-reproducible. Avoid for scientific rigor (the output chunking changes depending on LLM mood).

**Confidence: HIGH.** Anthropic's contextual retrieval numbers are from their published cookbook + AWS Bedrock case study.

---

## Full Installation Snapshot

### Model endpoints (Spark node, vLLM)

```bash
# Embedder (primary replacement)
vllm serve Qwen/Qwen3-VL-Embedding-8B \
  --port 18091 \
  --runner pooling \
  --dtype bfloat16 \
  --trust-remote-code \
  --gpu-memory-utilization 0.30

# Reranker
vllm serve Qwen/Qwen3-VL-Reranker-2B \
  --port 18096 \
  --dtype bfloat16 \
  --trust-remote-code \
  --gpu-memory-utilization 0.15

# HyDE LLM (small + fast)
vllm serve Qwen/Qwen3-4B-Instruct \
  --port 18080 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.20 \
  --max-model-len 8192

# GLM-OCR (dedicated port — per existing v2 plan M2)
vllm serve zai-org/GLM-OCR \
  --port 18081 \
  --gpu-memory-utilization 0.15 \
  --max-model-len 8192

# Golden dataset generator (keep on DGX Spark, invoke offline, not a runtime dep)
# Nemotron-Super-3-122B — offline only, not exposed as a runtime service
```

### Node dependencies (no change to package.json for the stack decisions, only config)

```json
{
  "dependencies": {
    "@lancedb/lancedb": "^0.27.1",
    "apache-arrow": "^18.0.0"
  }
}
```

All stack decisions above are **service-side**, not npm-dependency-side. Code changes are limited to:

- `src/config.ts` — `SparkServices` port assignments + new embedder dimension field
- `src/embed/provider.ts` — new `qwen3-vl` provider (builds on existing `spark` provider HTTP client)
- `src/rerank/client.ts` — new reranker endpoint format (Qwen VL reranker API differs slightly from llama-nemotron-rerank API — verify before cutover)
- `src/storage/lancedb.ts` — dimension field becomes config-driven (not hardcoded 4096)
- `src/hyde/client.ts` — model name config
- `src/ingest/parsers.ts` — remove EasyOCR branch

---

## Alternatives Considered (Consolidated Table)

| Dimension | Our Choice | Strong Alternative | When to Pick Alternative |
|-----------|------------|-------------------|--------------------------|
| Text embedder (if VL too heavy) | Qwen3-VL-Embedding-8B | `Qwen/Qwen3-Embedding-8B` (text-only, Apache 2.0, MTEB #1 multilingual 70.58) | VRAM-constrained deployment or latency budget blows |
| VL embedder (if license negotiable) | Qwen3-VL-Embedding-8B | `nvidia/nemotron-colembed-vl-8b-v2` (ViDoRe v3 #1, 63.54 NDCG@10) | You negotiate commercial license with NVIDIA — strictly better on ViDoRe |
| Reranker (VL) | Qwen3-VL-Reranker-2B | `BAAI/bge-reranker-v2-m3` | Text-only workload or 2B VL too heavy |
| Vector DB | LanceDB | Qdrant | You're building a service, not a plugin |
| HyDE LLM | Qwen3-4B-Instruct or Phi-4-mini | Nemotron-Mini-4B-Instruct | You've already validated the NVIDIA license chain |
| Eval framework | RAGAS-lite + golden dataset + BEIR | ARES | You have 50+ annotated triples per metric and a training pipeline |
| Chunking (docs) | Header-aware + contextual retrieval | Late chunking (Jina) | Heavy anaphoric content (legal, multi-turn transcripts) |

---

## What NOT to Use (and why)

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `nvidia/llama-embed-nemotron-8b` | Non-commercial license (`customized-nscl-v1` + Llama 3.1 community). Blocks v1.0 public release. | `Qwen/Qwen3-VL-Embedding-8B` |
| `nvidia/llama-nemotron-rerank-1b-v2` | Same license class + documented score saturation (58% ≥ 0.999) | `Qwen/Qwen3-VL-Reranker-2B` |
| `nvidia/nemotron-colembed-vl-*-v2` | CC-BY-NC-4.0 non-commercial despite SOTA ViDoRe scores | `Qwen/Qwen3-VL-Embedding-8B` (unless you license from NVIDIA) |
| `jinaai/jina-embeddings-v4` | CC-BY-NC-4.0 + Qwen Research License. Jina's own statement: not commercial. | `Qwen/Qwen3-VL-Embedding-8B` |
| `Qwen/Qwen3-Reranker-4B` (text) | >1s/query — autoregressive yes/no decoding blows latency budget. The *text* reranker is not viable. | `Qwen/Qwen3-VL-Reranker-2B` (different inference path) or `BAAI/bge-reranker-v2-m3` |
| `Nemotron-Super-3-122B` as a runtime HyDE model | 120B for 150-token generation. 100% timeout rate per your existing BEIR runs. | `Qwen/Qwen3-4B-Instruct` or `microsoft/Phi-4-mini-instruct` (runtime); keep Nemotron-Super for offline golden dataset generation only |
| EasyOCR | Python-dependent, slow, bad on complex layouts, superseded by VL embedding (no OCR step at all) | Retire. VL embeds the image page directly. |
| Pinecone | SaaS; plugin ships user data to third party = privacy violation per PROJECT.md constraint | LanceDB (embedded, local) |
| Chunk overlap 10-20% | 2026 LanceDB study: no benefit with hybrid retrieval, wastes storage | Zero overlap with hybrid retrieval (your current setting) |
| Agentic chunking | Non-reproducible, fails scientific-rigor requirement | Header-aware recursive + contextual retrieval |
| ARES for v1.0 | Requires ≥50 annotated triples per metric + training pipeline + fine-tuned judges — scope risk | RAGAS-lite + golden dataset |
| Tantivy FTS in LanceDB | Post-filter only, no pre-filter pushdown, no incremental indexing | Native LanceDB FTS (`use_tantivy=False`) — shipped 2025 |

---

## Stale Claims in Existing Research Docs (Flag for Doc Overhaul Phase)

| Doc | Claim | Status | Correction |
|-----|-------|--------|------------|
| `docs/RESEARCH-SOTA-2026.md` | "HyDE — +5-15% recall" | Still accurate (2024 paper numbers) | OK, but clarify the scope: this is for ambiguous short queries, not agent memories |
| `docs/RESEARCH-SOTA-2026.md` | "Voyage-3-large 1024 dims leads MTEB by 9.74% over OpenAI" | **STALE** (2024 framing) | 2026 SOTA: Qwen3-Embedding-8B #1 multilingual MTEB (70.58), llama-embed-nemotron-8b #1 Borda rank. Voyage-3 is no longer the leader. |
| `docs/RESEARCH-SOTA-2026.md` | "PQ compression 16x reduction typical" | Still valid as baseline | But: Matryoshka (built into Qwen3-VL, Jina-v4, Qwen3-Embedding) gives dimension reduction without PQ. Prefer MRL over PQ for new embedders. |
| `docs/RESEARCH-SOTA-2026-VALIDATED.md` | "HyDE NOT for us — latency budget" | **Partially stale** | Conclusion was correct *given* Nemotron-Super-120B as HyDE LLM. With Qwen3-4B-Instruct or Phi-4-mini HyDE at <1s, the latency objection disappears. Keep HyDE, change the LLM. |
| `docs/RESEARCH-SOTA-2026-VALIDATED.md` | "4096 dims is fine — not overkill for a local model" | **Refine** | True for storage cost, but MRL lets you have 4096 for max quality AND 512 for efficient tables. Not either/or in 2026. |
| `docs/RESEARCH-SOTA-2026-VALIDATED.md` | "nvidia/llama-embed-nemotron-8b: 72.31 on MTEB English (v2)" | **Verify** | The 72.31 number is from `NV-Embed-v2` (different model). `llama-embed-nemotron-8b` is 69.46 Mean Task on MMTEB multilingual v2. The "72.31 on English v2" attribution in the doc is a citation error. Fix in doc overhaul. |
| `docs/PLAN-spark-v2-architecture.md` M3 | "nvidia/llama-embed-nemotron-8b-vl (if released)" | **STALE** — use Qwen3-VL-Embedding-8B instead | NVIDIA did release VL embedders (`nemotron-colembed-vl-*-v2`) but under CC-BY-NC-4.0. Non-commercial. Plan should pivot to Qwen3-VL-Embedding-8B (Apache 2.0). |
| `docs/PLAN-spark-v2-architecture.md` M4 | "BAAI/bge-reranker-v2-m3 or equivalent" | **OK as fallback; upgrade preferred to Qwen3-VL-Reranker-2B** | bge-reranker-v2-m3 is text-only; if you migrate to VL embeddings, you want a VL reranker to score (query, image) pairs correctly. |
| `docs/PLAN-spark-v2-architecture.md` M1 | "nvidia/Nemotron-Mini-4B-Instruct" | **License-check** | Verify NVIDIA Open Model license terms for this specific variant. If non-commercial, swap to `Qwen/Qwen3-4B-Instruct` (Apache 2.0) or `microsoft/Phi-4-mini-instruct` (MIT). |
| `docs/PLAN-spark-v2-architecture.md` Open Question 1 | "Which VL embedding model?" | **Resolved:** Qwen3-VL-Embedding-8B (license + MMEB + MRL + Apache 2.0) |
| `docs/RESEARCH-TOOLS-INJECTION-2026.md` | arXiv 2603.20313, 97.1% hit@3, langgraph-bigtool | **Validated** — still current 2026 SOTA | No change needed |

---

## Version Compatibility Notes

| Package | Version | Note |
|---------|---------|------|
| `@lancedb/lancedb` | `^0.27.1` | Keep. Verify native FTS support in the TypeScript SDK (Python SDK confirmed). Flag for recon phase. |
| `apache-arrow` | `^18.0.0` | Verify against LanceDB 0.27.1 compatibility matrix on each bump. LanceDB is aggressive about Arrow version bumps. |
| vLLM (Spark node) | verify ≥ supporting Qwen3-VL pooling runner | Qwen3-VL-Embedding-8B model card shows `runner="pooling"` — verify vLLM version supports this on your DGX Spark install. |
| `transformers` | `≥ 4.57.0` | Required by Qwen3-VL-Embedding-2B. Server-side only; not an npm dep. |
| `qwen-vl-utils` | `≥ 0.0.14` | Server-side only. |

---

## Key Sources

### Primary (model cards, verified directly)

- [Qwen/Qwen3-VL-Embedding-2B — HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) — dims 64-2048, 2B params, Apache 2.0, MMEB-V2 73.2, 2026-01 release
- [Qwen/Qwen3-VL-Embedding-8B — HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) — dims 64-4096, 8B params, Apache 2.0, MMEB-V2 77.9, ViDoRe v1 88.2, vLLM example
- [Qwen/Qwen3-VL-Reranker-2B — HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) — 2B params, Apache 2.0, MMEB-v2 75.1, ViDoRe v3 60.8, JinaVDR 80.9
- [nvidia/llama-embed-nemotron-8b — HuggingFace](https://huggingface.co/nvidia/llama-embed-nemotron-8b) — 4096 dims, customized-nscl-v1 + Llama 3.1 community license (NON-COMMERCIAL), 2025-10-21
- [nvidia/nemotron-colembed-vl-8b-v2 — HuggingFace](https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2) — 8.8B params, CC-BY-NC-4.0, ViDoRe v3 63.54 NDCG@10, 2026-01-26
- [jinaai/jina-embeddings-v4 — HuggingFace](https://huggingface.co/jinaai/jina-embeddings-v4) — 3.8B params, 2048 dims MRL, CC-BY-NC-4.0 + Qwen Research License
- [jina-embeddings-v4 commercial license discussion](https://huggingface.co/jinaai/jina-embeddings-v4/discussions/44) — Jina's own confirmation of non-commercial status

### Papers (peer-reviewed or preprint)

- arXiv:2601.04720 — "Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking" (2026)
- arXiv:2602.03992 — "Nemotron ColEmbed V2: Top-Performing Late Interaction Embedding Models for Visual Document Retrieval" (2026)
- arXiv:2511.07025 — "Llama-Embed-Nemotron-8B: A Universal Text Embedding Model for Multilingual and Cross-Lingual Tasks" (2025-11)
- arXiv:2506.18902 — "jina-embeddings-v4" (2025)
- arXiv:2603.20313 — Semantic tool retrieval (121 MCP tools, 99.6% token reduction, 97.1% hit@3)
- arXiv:2311.09476 — "ARES: An Automated Evaluation Framework for RAG Systems"
- arXiv:2309.15217 — "Ragas: Automated Evaluation of Retrieval Augmented Generation"
- arXiv:2504.19754 — "Reconstructing Context: Evaluating Advanced Chunking Strategies for RAG" (2025)

### Official blogs / docs

- [NVIDIA blog — Llama-Embed-Nemotron-8B Multilingual MTEB #1](https://huggingface.co/blog/nvidia/llama-embed-nemotron-8b)
- [NVIDIA blog — Nemotron ColEmbed V2 ViDoRe V3 #1](https://huggingface.co/blog/nvidia/nemotron-colembed-v2)
- [Qwen3-VL-Embedding blog post](https://qwen.ai/blog?id=qwen3-vl-embedding)
- [LanceDB WikiSearch: Native Full-Text Search on 41M docs](https://lancedb.com/blog/feature-full-text-search/)
- [LanceDB Hybrid Search blog](https://lancedb.com/blog/hybrid-search-rag-for-real-life-production-grade-applications-e1e727b3965a/)
- [Anthropic Contextual Retrieval — platform.claude.com cookbook](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)
- [Anthropic Contextual Retrieval — AWS Bedrock case study](https://aws.amazon.com/blogs/machine-learning/contextual-retrieval-in-anthropic-using-amazon-bedrock-knowledge-bases/)
- [langchain-ai/langgraph-bigtool — GitHub](https://github.com/langchain-ai/langgraph-bigtool)

### Benchmark comparisons (2026)

- [Agentset reranker leaderboard](https://agentset.ai/rerankers) — latency/Hit@1 for Jina, Nemotron, BGE, Qwen3
- [aimultiple reranker benchmark 2026](https://aimultiple.com/rerankers)
- [MMEB Leaderboard — TIGER-Lab HuggingFace Space](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard)
- [MTEB March 2026 leaderboard snapshot](https://awesomeagents.ai/leaderboards/embedding-model-leaderboard-mteb-march-2026/)

### Secondary (validation support)

- BentoML — Best Open-Source Embedding Models 2026
- SiliconFlow — Most accurate reranker for real-time search 2026
- Firecrawl — Best Chunking Strategies for RAG 2026
- Vectorize.io — Best AI Agent Memory Systems 2026

---

## Confidence Summary

| Recommendation | Confidence | Basis |
|----------------|------------|-------|
| Replace embedder with `Qwen/Qwen3-VL-Embedding-8B` | **HIGH** | Direct HF model card verification of license, dims, benchmarks, vLLM support. Decisive factor is license (Apache 2.0 vs competitors' non-commercial). |
| Replace reranker with `Qwen/Qwen3-VL-Reranker-2B` | **HIGH** (model choice), **MEDIUM** (latency) | Model card verified; latency needs on-hardware measurement before cutover |
| Keep LanceDB, use native FTS | **HIGH** (keep), **MEDIUM** (TS SDK native FTS availability) | Native FTS confirmed in Python SDK via LanceDB blog. Verify TS SDK parity in phase 1 recon. |
| Keep HyDE, swap LLM to Qwen3-4B or Phi-4-mini | **HIGH** (keep HyDE), **MEDIUM** (specific LLM) | HyDE's continued relevance is documented. LLM choice is flexible among the small-instruct class. |
| Implement semantic tool retrieval per langgraph-bigtool | **HIGH** | Source paper has concrete numbers; reference implementation exists |
| RAGAS-lite + golden dataset + BEIR, defer ARES | **HIGH** | RAGAS/ARES tradeoff is peer-reviewed; ARES training burden is documented |
| Contextual retrieval for workspace docs, flat for memories | **HIGH** | Anthropic -49% retrieval error number is from their published cookbook |
| Retire EasyOCR | **HIGH** | Already planned; VL embedding removes the OCR dependency entirely |

**Overall research confidence: HIGH.** Every recommendation traces to a HuggingFace model card, published paper, or official documentation dated 2025-10 or later. Latency projections and TypeScript SDK capability gaps are the two MEDIUM-confidence items flagged for phase-1 verification.

---

## Roadmap Implications for This Milestone

1. **License audit is blocking.** The biggest finding is that the current NVIDIA embedder + reranker stack is non-commercial-licensed and **cannot ship in v1.0 public release as-is**. This reorders priorities: license-clean replacement is a P0 alongside the 11 critical bug fixes, not a Phase B "Spark v2" nice-to-have.

2. **VL embedder migration = single biggest engineering win.** Preserves diagrams/tables/figures in PDF ingestion, solves the license issue, gets you on 2026-current MMEB/ViDoRe SOTA with a clean license — all with one model swap. Recommend elevating this from PLAN-spark-v2-architecture Phase B to v1.0 required.

3. **HyDE re-validation.** The "kill HyDE" verdict in prior research was framed incorrectly. With a small-instruct LLM, HyDE's latency objection disappears. Keep the HyDE code path, change the LLM, re-run BEIR to get the +5-15% recall number honestly.

4. **Contextual retrieval for docs is the highest-ROI chunking change.** -49% retrieval errors (Anthropic's number) dwarfs any proposition/late-chunking marginal gain. Single-LLM-call-per-chunk at ingest, cacheable. Consider adding to v1.0 if timeline permits.

5. **Golden dataset methodology must be reproducible.** For paper credibility, publish the generation pipeline + scrubbing steps. Never publish the data itself (privacy). This is the centerpiece of the "scientific rigor" core value.

6. **Tool retrieval is validated SOTA.** Ship it per langgraph-bigtool pattern without further research. Numbers are concrete, pattern is mature.

7. **RAGAS-lite over full RAGAS over ARES.** Respect the scope guard in PROJECT.md. ARES is a research project in itself; RAGAS with context precision/recall + the golden dataset is enough for a credible v1.0 paper.

---

*Stack research for: scientifically-rigorous RAG memory plugin (memory-spark v1.0)*
*Researched: 2026-04-09*
*Overall confidence: HIGH*
