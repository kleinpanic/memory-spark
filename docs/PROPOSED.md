# PROPOSED.md — Proposed But Not Yet Implemented

**Purpose:** This document is an index of every scoped but unimplemented feature, improvement, and architectural change in memory-spark. It exists so nothing stays lost in plan docs — everything is findable here with status, ownership, and links to source documents.

**How to use:** If you're scoping a new feature, check here first. If it already exists as a proposal, link to that instead of re-inventing it.

**Legend:**
- 🔲 Not started
- 🔶 In progress / partially started
- ✅ Shipped (moved out of this doc)

---

## Tool Capabilities

### `tool_registry` Pool + `memory_tools_search` Tool
**Source:** `docs/RESEARCH-TOOLS-INJECTION-2026.md` (research, not implemented)

A cross-agent shared pool containing all tool schemas indexed from:
- All `SKILL.md` files in `~/.openclaw/skills/`
- All agent `TOOLS.md` files
- OpenClaw tool manifests
- MCP tool schemas

Plus a new `memory_tools_search(query, tags?)` tool that lets agents search the registry by *task description* (e.g. "send an iMessage with image attachment") and retrieve relevant tools. Currently `agent_tools` is per-agent only and auto-injected, not explicitly searchable.

**Status:** 🔲 Not started
**Files to touch:** `src/storage/pool.ts`, `src/auto/recall.ts`, `index.ts`, `docs/PLUGIN-API.md`, `docs-site/index.html`

**Research backing:** arxiv 2603.20313 — 121 MCP tools indexed with dense embeddings → 99.6% token reduction, 97.1% hit rate at K=3

---

## Pipeline Improvements

### P1-B: Position Preservation in Reranker Blend
**Source:** `docs/PLAN-phase13.md` § Priority 1

When a doc is weakly ranked by both vector AND reranker, the RRF blend drops it out of top-10 — causing NDCG → 0.0. Confirmed on 34 SciFact queries.

**Options:**
- **GUARD-A:** Ensure top-K_vec docs get guaranteed minimum blend score
- **GUARD-B:** Additive rank bonus `(K - vec_rank) * epsilon`
- **GUARD-C:** Two-pass — blend full pool, then inject survivors

**Status:** 🔲 Not started
**File:** `src/rerank/reranker.ts`

---

### P2-D: `recall.ts` Cognitive Complexity Refactor
**Source:** `docs/PLAN-phase13.md` § Priority 2

`recallHandler` is a 400+ line monolithic function with ESLint cognitive complexity of 80 (limit: 25). Break into independently testable sub-functions:

- `buildQueryVectors()` — HyDE + multi-query expansion
- `searchAllPools()` — pool search orchestration
- `mergeAndWeight()` — source weighting, temporal decay, dedup
- `rankAndFilter()` — reranker + MMR + parent expansion
- `budgetAndFormat()` — LCM dedup, token budget, security filter

**Status:** 🔲 Not started
**File:** `src/auto/recall.ts`
**Prerequisite:** P1-B (position preservation)

---

### Parent-Child Hierarchical Chunking
**Source:** `docs/RESEARCH-SOTA-2026.md` § Item 3

Store small child chunks (200 tokens) for precise retrieval, with large parent chunks (2000 tokens) for context. Retrieve by child similarity, return parent for generation context.

**Status:** 🔲 Not started
**Impact:** Best of both worlds — precision retrieval + contextual generation
**See also:** Phase 2 in `docs/ARCHITECTURE.md`

---

### Proposition Chunking
**Source:** `docs/RESEARCH-SOTA-2026.md` § Item 4

Break content into atomic factual statements. Each chunk = one fact. Highest retrieval precision for factoid queries.

**Status:** 🔲 Not started
**Impact:** +20-30% precision on factoid queries (but expensive to generate)

---

### Query Rewriting / Expansion
**Source:** `docs/RESEARCH-SOTA-2026.md` § Item 2 + `docs/ARCHITECTURE.md` Phase 3

Rewrite user queries before embedding: expand acronyms, add synonyms, step-back prompting. We have multi-query expansion (Phase 11B) but not query rewriting.

**Status:** 🔲 Not started
**Impact:** +3-8% NDCG on production queries

---

## Reference Library (Phase 2)

**Source:** `docs/ARCHITECTURE.md` § Phase 2

Full Phase 2 implementation:

- [ ] `memory_reference_search` tool (documented in PLUGIN-API.md, but underlying indexing pipeline not complete)
- [ ] PDF text extraction pipeline (improved — GLM-OCR + vLLM, see Spark v2 below)
- [ ] Version tracking for reference documents (detect when indexed docs change)
- [ ] FTS coverage for reference pool content

**Status:** 🔲 Not started (reference_search tool exists; reference ingestion pipeline does not)
**File:** `src/ingest/pipeline.ts`

---

## Classification Pipeline (Phase 3)

**Source:** `docs/ARCHITECTURE.md` § Phase 3

Replace heuristic-only classification with LLM+heuristic hybrid:
- [ ] Integrate zero-shot classifier (Nemotron) for content routing
- [ ] Implement pool routing based on classification output
- [ ] Quality scoring refinement with LLM feedback

**Status:** 🔲 Not started
**File:** `src/classify/zero-shot.ts`, `src/classify/quality.ts`

---

## Evaluation Overhaul (Phase 4)

**Source:** `docs/ARCHITECTURE.md` § Phase 4

- [ ] Fix all metric bugs (Codex findings — see `docs/AUDIT-2026-04-02.md`)
- [ ] Calibrate BM25 sigmoid from corpus score distribution
- [ ] Build golden dataset per pool (not just global)
- [ ] A/B framework for pipeline ablation studies
- [ ] Full BEIR benchmark (SciFact + FiQA + NFCorpus) run with all 36 configs

**Also needed (from `docs/RESEARCH-SOTA-2026.md`):**
- [ ] RAGAS evaluation framework (Context Precision, Context Recall, Faithfulness, Answer Relevance)
- [ ] ARES framework (synthetic data generation, trained judge models)
- [ ] Golden dataset with approved answers and source document links
- [ ] Adversarial query set

**Status:** 🔲 Not started
**Files:** `evaluation/`, `docs/EVALUATION.md`, `docs/BENCHMARKS.md`

---

## Phase 13 Remaining Items

**Source:** `docs/PLAN-phase13.md`

| ID | Item | Status |
|----|------|--------|
| P1-B | Position preservation (reranker blend) | 🔲 Not started |
| P2-D | `recall.ts` cognitive complexity refactor | 🔲 Not started |
| P3-A | `queryMessageCount` default 2 → 3 | 🔲 Not started |
| P3-B | `npm audit fix --force` (220 vulnerabilities) | 🔲 Not started |
| P3-D | Coverage badge mismatch (README says 91%, actual 35%) | 🔲 Not started |

---

## Spark v2 Infrastructure Migration

**Source:** `docs/PLAN-spark-v2-architecture.md`

Full v2 architecture for the Spark node service stack. Not started.

### M1: Small LLM for HyDE (Port 18080)
Replace Nemotron-Super-120B (overkill for 150-token HyDE) with `nvidia/Nemotron-Mini-4B-Instruct` or similar. Frees ~52GB VRAM.

**Status:** 🔲 Not started

### M2: GLM-OCR to Dedicated Port (18081)
Move GLM-OCR off shared port 18080 to its own vLLM instance at port 18081.

**Status:** 🔲 Not started

### M3: VL Embedding Model (Port 18091)
Replace text-only `llama-embed-nemotron-8b` with a vision-language embedder. Multimodal: text + image embedded directly, preserving layout/figures/tables that OCR loses.

**Status:** 🔲 Not started (depends on model availability from NVIDIA)
**Candidates:** `nvidia/llama-embed-nemotron-8b-vl`, `BAAI/bge-visualized`, `Qwen/Qwen2-VL-7B-Instruct`
**Impact:** Native image/figure/chart embedding, no OCR text loss
**Warning:** Requires re-indexing all existing chunks (dimension change)

### M4: VL or Better Reranker (Port 18096)
Replace `llama-nemotron-rerank-1b-v2` (58% score saturation, poor discrimination) with `BAAI/bge-reranker-v2-m3` or NVIDIA VL reranker.

**Status:** 🔲 Not started

### M5: Retire EasyOCR (Port 18097)
Legacy Python OCR superseded by GLM-OCR. Remove from codebase entirely.

**Status:** 🔲 Not started
**File:** `src/ingest/parsers.ts`

---

## Production RAG Checklist (from SOTA Research)

**Source:** `docs/RESEARCH-SOTA-2026.md` § Production RAG Checklist

These are individual items tracked in the research doc. Some overlap with other sections above.

| Item | Status | Notes |
|------|--------|-------|
| Query rewriting / HyDE | 🔲 Not started | Multi-query exists; query rewriting doesn't |
| Parent-child hierarchical chunks | 🔲 Not started | See Pipeline Improvements |
| Chunk overlap | 🔲 Not started | Research says minimal impact for hybrid; test before implementing |
| Rerank pool expansion (10→50+) | 🔲 Not started | SOTA uses 100; we use 30 |
| RAGAS/ARES evaluation | 🔲 Not started | See Evaluation Overhaul |
| Golden dataset with gold answers | 🔲 Not started | See Evaluation Overhaul |
| Adversarial query set | 🔲 Not started | |
| Embedding cache | ✅ Shipped | `src/embed/cached-provider.ts` |
| Change detection (hash-based) | 🔲 Not started | Detect when indexed docs change without full re-scan |
| Incremental re-indexing | 🔲 Not started | |
| Observability / monitoring dashboard | 🔲 Not started | |

---

## Phase 5: Production Deployment

**Source:** `docs/ARCHITECTURE.md` § Phase 5

- [ ] Full reindex with pool assignment on existing chunks
- [ ] Config migration tool for existing installations
- [ ] Performance profiling and optimization
- [ ] Documentation complete
- [ ] v1.0 release

**Status:** 🔲 Not started

---

## Known Code Risk

**Source:** `docs/STATE.md` § Known Code-Level Risk

### Soft-Gate Boundary Discontinuity
`computeRerankerGate()` soft mode has piecewise branch behavior around `lowThreshold=0.02` that jumps unexpectedly. Hard gate logic is correct.

**Status:** 🔲 Not started
**File:** `src/rerank/reranker.ts`
**Impact:** Non-blocking for current production (hard gate is used), but should be fixed before benchmark policy lock.

---

*Last updated: 2026-04-08. To add a proposed item, create a plan doc in `docs/` or `docs/archive/` and add an entry here with source link and status.*
