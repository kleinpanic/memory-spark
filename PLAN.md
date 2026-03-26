# memory-spark Upgrade Plan (v0.2.0)

> Written: 2026-03-26 | Author: KleinClaw-Meta
> Status: **APPROVED FOR EXECUTION**
> Repo: `~/codeWS/TypeScript/memory-spark`
> LanceDB: `~/.openclaw/data/memory-spark/lancedb`

---

## Current State (Honest Assessment)

### What Works
- Hybrid search: Vector (IVF_PQ, 4096-dim Nemotron-Embed-8B) + FTS (BM25) merged via RRF (k=60)
- Reranking: Nemotron-Rerank-1B via Spark node
- Quality gating: Noise patterns score 0 → never indexed (session dumps, casual chat, Discord metadata)
- Language filter: Configurable, defaults to English
- Auto-recall hook: Injects `<relevant-memories>` into agent context via `before_prompt_build`
- Auto-capture hook: Stores agent decisions/facts via `agent_end`
- Temporal decay: `0.8 + 0.2 * exp(-0.03 * ageDays)` with 0.8 floor
- Mistake amplification: 1.6x relevance boost on `MISTAKES/` directory files
- Reference library: Indexed separately with `content_type: reference`

### What's Broken or Missing
1. **Timestamps were reset on every restart** — FIXED (now uses file mtime), but all 18,605 existing chunks have wrong timestamps. Need full table rebuild.
2. **2,964 garbage chunks** still in index (218 zh-CN + 2,746 old session dumps) — quality filter only blocks NEW ingestion, doesn't retroactively purge.
3. **46.4% token waste** — recalled memories include session headers, Discord metadata, "lmfao" — the hard-cut fix prevents future indexing but existing data needs purging.
4. **99.6 KB per chunk** — 4096-dim Float32 vectors = 16 KB raw, but LanceDB stores ~99.6 KB with metadata overhead. 1.81 GB for 18K chunks.
5. **Flat 500-char chunks** — no hierarchy, no parent-child, no overlap analysis.
6. **No query intelligence** — raw agent prompts go straight to embedding. No rewriting, no HyDE, no expansion.
7. **Mock-only evaluation** — 60 hand-picked queries with simulated scores. No live eval, no RAGAS, no adversarial testing.
8. **Only 83 capture chunks** vs 18,522 memory chunks — auto-capture is barely firing.
9. **No context-aware tool suggestions** — recalled memories don't influence which tools agents are prompted to use.
10. **No hot-add directories** — can't add new directories to watch at runtime.
11. **No health monitoring** — no visibility into embed queue depth, failures, Spark connectivity, staleness.
12. **No memory linking** — chunks are isolated, no graph/relationship between related memories.
13. **Bootstrap bloat** — AGENTS.md files are massive (20KB+) with content that should be in memory-spark instead.

### Index Audit (Live Numbers)
```
Total chunks:           18,605
Source breakdown:       memory 99.6%, capture 0.4%
zh-CN noise:            218 chunks
Old session dumps:      2,746 chunks
LanceDB disk:           1.81 GB
Storage per chunk:      99.6 KB
Vector dims:            4096 (Float32)
Indices:                IVF_PQ (vector), FTS (text)
```

---

## SOTA References

| Technique | Source | Key Finding |
|-----------|--------|-------------|
| HyDE | [Gao et al. 2022](https://arxiv.org/abs/2212.10496) | Hypothetical document embeddings improve recall +5-15% on ambiguous queries |
| Parent-Child Chunking | [PremAI Production RAG Guide 2026](https://blog.premai.io/building-production-rag-architecture-chunking-evaluation-monitoring-2026-guide/) | Child retrieval + parent context = precision + completeness |
| Reranking Pool | [ColBERTv2](https://arxiv.org/abs/2112.01488); [Qdrant Hybrid Reranking](https://qdrant.tech/articles/hybrid-search/) | Rerank top-50-100 instead of top-10 for +3-5% NDCG |
| RAGAS Framework | [Shahul et al. 2023](https://arxiv.org/abs/2309.15217) | Context Precision/Recall, Faithfulness, Answer Relevance — LLM-as-judge |
| ARES Framework | [Saad-Falcon et al. 2023](https://arxiv.org/abs/2311.09476) | PPI-based confidence intervals, synthetic training data for judge models |
| BEIR 2.0 | [Thakur et al. 2025](https://github.com/beir-cellar/beir) | 18-dataset zero-shot benchmark. Top: Voyage-Large-2 at 54.8% NDCG@10 |
| Query Rewriting | [Microsoft Advanced RAG](https://learn.microsoft.com/en-us/azure/developer/ai/advanced-retrieval-augmented-generation) | Step-back prompting, acronym expansion, intent classification |
| Chunk Overlap | [PremAI 2026](https://blog.premai.io/building-production-rag-architecture-chunking-evaluation-monitoring-2026-guide/) | No benefit with hybrid search (SPLADE study). Only for dense-only. |
| Embedding Dims | [Voyage-3-large](https://docs.voyageai.com/) | 1024 dims beats 3072-dim OpenAI on MTEB. Less dims ≠ less quality |
| Proposition Chunking | [Chen et al. 2023](https://arxiv.org/abs/2312.06648) | Atomic factual statements = highest precision for factoid queries |
| Adaptive Chunking | PremAI 2026 citing clinical study | 87% accuracy vs 13% for fixed-size on same corpus |
| RAG Survey 2025 | [arxiv 2506.00054](https://arxiv.org/abs/2506.00054) | SimRAG self-training, LQR layered retrieval, Re2G reranking layers |

---

## Phase 1: Fix Fundamentals
**Goal:** Clean data, correct timestamps, remove garbage.
**Estimate:** 1-2 hours

### 1.1 Full Table Rebuild
- Drop existing LanceDB table
- Re-index all workspace files with correct mtime timestamps
- Verify age distribution: old files should show days/months, not "1h ago"
- **Test:** Query for a Feb 2026 file → should show "~30d ago", not "1h ago"

### 1.2 Garbage Purge
- Delete all chunks where `path` contains `/zh-CN/`, `/zh-TW/`, `/ja/`, `/ko/`
- Delete all chunks matching session dump patterns (already blocked by quality scorer)
- **Test:** `scripts/index-audit.ts` should show 0 zh-CN, 0 session dumps
- **Metric:** Total chunk count should drop from ~18,605 to ~15,000

### 1.3 Rerank Pool Expansion
- Change rerank candidate pool from 10 → 50
- File: `src/auto/recall.ts` — the `vectorSearch().limit()` and `ftsSearch().limit()` calls
- **Test:** Recall@5 should improve on queries where relevant doc was ranked 11-50
- **Metric:** Measure before/after on live eval

### 1.4 Storage Audit
- Run `scripts/vector-audit.ts` after rebuild
- Measure actual per-chunk size
- If >50 KB/chunk, investigate LanceDB compaction (`table.compact_files()`)
- **Target:** <30 KB/chunk

---

## Phase 2: Query Intelligence
**Goal:** Don't just embed raw queries — make them smarter before search.
**Estimate:** 3-4 hours
**References:** Microsoft Advanced RAG, HyDE paper

### 2.1 Query Rewriting
- Before embedding the agent's prompt for recall, run a lightweight rewrite:
  - Expand acronyms (e.g., "DGX" → "DGX Spark NVIDIA Grace Hopper")
  - Extract the core question from conversational noise
  - Add domain context (e.g., "in the context of OpenClaw agent configuration")
- Implementation: New `src/query/rewriter.ts` module
- Use cheapest available model (spark-ollama Qwen3-4B or Nemotron-Terminal-32B)
- **Test:** Side-by-side: raw query vs rewritten query → compare Recall@5
- **Metric:** +5% Recall@5 minimum

### 2.2 HyDE (Hypothetical Document Embeddings)
- For complex queries, have the LLM generate a hypothetical answer paragraph
- Embed THAT paragraph instead of the query
- Search with the hypothetical document embedding
- Implementation: New `src/query/hyde.ts` module
- Gate: Only trigger for queries >20 tokens (short queries don't benefit)
- **Test:** Query "how do I configure vLLM GPU memory?" → HyDE generates a paragraph about GPU_MEMORY_UTILIZATION → embedding matches the actual doc better
- **Metric:** +10% Recall@5 on ambiguous queries
- **Cost:** 1 extra LLM call per recall. Use cheapest model.

### 2.3 Embedding Cache
- Cache query embeddings with TTL (1 hour default)
- Same query from same agent within the hour → skip embed call
- Implementation: In-memory LRU cache in `src/embed/cache.ts`
- **Metric:** Reduce embed API calls by ~30-50%

---

## Phase 3: Chunking Overhaul
**Goal:** Replace flat chunks with hierarchical parent-child architecture.
**Estimate:** 4-6 hours
**References:** PremAI 2026 guide, LlamaIndex ParentDocumentRetriever

### 3.1 Parent-Child Chunks
- **Child chunks:** 200 tokens, used for vector search (precise matching)
- **Parent chunks:** 2000 tokens, returned as context to the agent
- Store both in LanceDB with a `parent_id` field linking child → parent
- On retrieval: search children, deduplicate by parent_id, return parent text
- **Schema change:**
  ```
  + chunk_type: "child" | "parent"
  + parent_id: string | null
  + child_ids: string[] (JSON)
  ```
- **Test:** Query matches a specific sentence → agent gets the full surrounding section
- **Metric:** Answer quality improvement measured via RAGAS Faithfulness score

### 3.2 Recursive Chunking
- Replace current line-based splitter with recursive strategy:
  - Split on `\n\n` (paragraphs) first
  - Then `\n` (lines)
  - Then `. ` (sentences)
  - Then ` ` (words)
- Respect markdown headers as hard boundaries
- **Reference:** PremAI 2026 — recursive 512-token = 69% accuracy vs semantic chunking's 54%

### 3.3 Contextual Prefix Preservation
- Already implemented: prepend source/path/heading to text before embedding
- Verify this still works with parent-child architecture
- Parent chunks get the prefix; child chunks inherit parent's prefix

---

## Phase 4: New Plugin Tools
**Goal:** Give agents better control over memory.
**Estimate:** 3-4 hours

### 4.1 `memory_watch` — Hot-Add Directories
- Tool: `memory_watch(path, recursive?, fileTypes?)`
- Adds a directory to the file watcher at runtime without config change
- Persists watched paths in `data/watched-paths.json`
- Survives gateway restarts
- **Use case:** Agent discovers a new project dir and wants it indexed

### 4.2 `memory_health` — System Diagnostics
- Tool: `memory_health()`
- Returns:
  - Embed queue depth + pending count
  - Recent embed failures + error types
  - Spark endpoint connectivity (embed, rerank, classify)
  - Index stats (total chunks, age distribution, content types)
  - Staleness report (oldest chunks, chunks with wrong timestamps)
  - Token waste estimate (garbage chunks still in index)
- **Use case:** Heartbeat health checks, debugging recall quality

### 4.3 `memory_link` — Relationship Graph
- Tool: `memory_link(sourceId, targetId, relation)`
- Creates a directed edge between two memory chunks
- Relations: `related`, `supersedes`, `contradicts`, `depends_on`
- Stored in separate LanceDB table `memory_links`
- On recall: if a retrieved chunk has links, include link metadata in output
- **Use case:** Agent marks an old decision as superseded by a new one
- **Schema:**
  ```
  memory_links table:
    source_id: string
    target_id: string
    relation: string
    created_at: string
    created_by: string (agent_id)
  ```

### 4.4 Context-Aware Tool Suggestions
- Extend `before_prompt_build` hook
- After recall, analyze retrieved memory categories
- If memories mention infrastructure → suggest `exec`, `nodes` tools
- If memories mention configuration → suggest `gateway`, `config` tools
- If memories mention code → suggest `read`, `edit`, `coding-agent` tools
- Inject 1-2 sentence tool suggestion into `appendSystemContext`
- **Implementation:** New `src/auto/tool-suggest.ts`
- **Metric:** Measure if agents use suggested tools more often (log analysis)

---

## Phase 5: Bootstrap Bloat Reduction
**Goal:** Move static knowledge from AGENTS.md into memory-spark.
**Estimate:** 2-3 hours

### 5.1 Audit AGENTS.md Files
- For each agent: measure AGENTS.md token count
- Identify sections that are:
  - **Static facts** (infra inventory, host names, paths) → move to memory-spark reference docs
  - **Behavioral rules** (safety, policy) → keep in AGENTS.md
  - **Historical context** (past incidents) → move to memory-spark
- **Target:** Reduce AGENTS.md from ~5000 tokens to ~2000 tokens per agent

### 5.2 Reference Library Expansion
- Move extracted static facts to `~/Documents/OpenClaw/ReferenceLibrary/`
- Tag as `content_type: reference` for high-priority recall
- Verify agents can retrieve this info via `memory_reference_search`

### 5.3 Dynamic Context Injection
- Use the `before_prompt_build` hook to inject relevant reference docs
- Agent asks about Spark → auto-inject Spark infrastructure facts
- This replaces having the facts hardcoded in AGENTS.md

---

## Phase 6: Real Evaluation (CRITICAL)
**Goal:** Kill mock mode. Run real benchmarks. Get credible numbers.
**Estimate:** 4-6 hours
**References:** RAGAS, ARES, BEIR 2.0

### 6.1 Golden Dataset
- Expand `evaluation/ground-truth.json` from 60 → 100+ queries
- Add **gold answers** (the correct response for each query)
- Add **source document paths** (which chunk should be retrieved)
- Add **multiple phrasings** per query (for robustness testing)
- Add **difficulty ratings** (easy/medium/hard)
- Add **query types** (factoid, comparison, procedural, troubleshooting)

### 6.2 RAGAS Integration
- Install `ragas` or implement the 4 core metrics manually:
  - **Context Precision:** % of retrieved chunks that are relevant
  - **Context Recall:** % of relevant chunks that were retrieved
  - **Faithfulness:** Is the LLM's answer grounded in retrieved context?
  - **Answer Relevance:** Does the answer address the query?
- Use LLM-as-judge (cheapest model: spark-ollama or or-qwen3-4b)
- **Test:** Run full eval suite, get real numbers, compare to SOTA

### 6.3 Live LanceDB Evaluation
- Replace mock mode with actual vector search + reranking
- Measure real latency (embed time + search time + rerank time)
- Measure real Recall@5, NDCG@10, MRR on live data

### 6.4 Adversarial Queries
- Generate adversarial variants of golden dataset:
  - **Paraphrase:** Same meaning, different words
  - **Negation:** "What is NOT the correct GPU memory setting?"
  - **Entity swap:** "What model does the nicholas node run?" (when data is about user)
- Measure retrieval robustness gap (BEIR 2.0 shows 13-30% degradation on adversarial)

### 6.5 Weak Model Testing
- Run the full pipeline feeding recalled context to the weakest available model
- Candidate: `spark-ollama/qwen3-4b` or `or-qwen3-4b:free`
- If the weakest model gives correct answers with our context → retrieval is doing its job
- If it fails → retrieval quality isn't sufficient, need more context or better chunks

---

## Phase 7: Performance & Storage Optimization
**Goal:** Faster, smaller, cheaper.
**Estimate:** 2-3 hours

### 7.1 Dimension Reduction Analysis
- Current: 4096-dim Float32 = 16 KB per vector
- Option A: PCA to 1024 dims (investigate quality loss)
- Option B: Matryoshka embeddings (if Nemotron supports truncation)
- Option C: Switch to Voyage-3-large (1024 dims, better MTEB scores, but API cost)
- **Test:** Compare Recall@5 at 4096 vs 1024 dims on golden dataset

### 7.2 LanceDB Compaction
- Run `table.compact_files()` after rebuild
- Run `table.cleanup_old_versions()`
- Measure storage reduction
- **Target:** <30 KB/chunk (from 99.6 KB)

### 7.3 FP16 Vectors
- Store vectors as Float16 instead of Float32
- 2x storage reduction, typically <1% quality loss
- Requires LanceDB schema change

---

## Success Criteria

| Metric | Current | Target | SOTA |
|--------|---------|--------|------|
| NDCG@10 (live) | Unknown (mock: 0.889) | 0.75+ | 0.548 (BEIR cross-domain) |
| Recall@5 (live) | Unknown (mock: 0.903) | 0.80+ | — |
| Context Precision (RAGAS) | Not measured | 0.85+ | — |
| Faithfulness (RAGAS) | Not measured | 0.90+ | — |
| p95 Latency | ~118ms | <80ms | <50ms |
| Token waste per recall | 46.4% | <5% | 0% |
| Storage per chunk | 99.6 KB | <30 KB | ~4 KB (1024d) |
| Total index size | 1.81 GB | <500 MB | — |
| Adversarial robustness gap | Not measured | <15% | 13.8% (hybrid, BEIR 2.0) |

---

## Execution Order

```
Phase 1 (fundamentals)  ──→  Phase 6.1-6.3 (eval baseline)
         ↓                              ↓
Phase 2 (query intelligence) ──→  Phase 6 re-eval (measure improvement)
         ↓                              ↓
Phase 3 (chunking overhaul)  ──→  Phase 6 re-eval
         ↓                              ↓
Phase 4 (new tools)          ──→  Phase 6 re-eval
         ↓                              ↓
Phase 5 (bootstrap bloat)   ──→  Phase 6.4-6.5 (adversarial + weak model)
         ↓
Phase 7 (performance)       ──→  Final benchmarks + README update
```

**Key principle:** Eval FIRST, then build. Measure baseline before any change. Re-measure after every phase. No more mock numbers.
