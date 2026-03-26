# memory-spark Architecture

## Overview

memory-spark is an OpenClaw plugin that provides autonomous, Spark-powered long-term memory for AI agents. It replaces the built-in flat-file memory with a full RAG (Retrieval-Augmented Generation) pipeline backed by LanceDB, NVIDIA embedding/reranking models, and NER/classification services on the DGX Spark node.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        OpenClaw Gateway                         │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │ Agent Turn   │──>│before_prompt │──>│  Auto-Recall        │  │
│  │ (any agent)  │   │   _build     │   │  (recall.ts)        │  │
│  └─────────────┘   └──────────────┘   │  1. Build query      │  │
│        │                               │  2. Embed query      │  │
│        │                               │  3. Vector search    │  │
│        │                               │  4. FTS search       │  │
│        │                               │  5. Hybrid merge     │  │
│        │                               │  6. Mistakes inject  │  │
│        │                               │  7. Source weighting  │  │
│        │                               │  8. Temporal decay   │  │
│        │                               │  9. MMR diversity    │  │
│        │                               │  10. Cross-encoder   │  │
│        │                               │  11. LCM dedup       │  │
│        │                               │  12. Security filter │  │
│        │                               │  13. Token budget    │  │
│        │                               └──────────┬──────────┘  │
│        │                                          │              │
│        │                              <relevant-memories> XML    │
│        │                                          │              │
│        v                                          v              │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │ Agent End    │──>│  agent_end   │──>│  Auto-Capture       │  │
│  │ (response)   │   │   hook       │   │  (capture.ts)       │  │
│  └─────────────┘   └──────────────┘   │  1. Garbage filter   │  │
│                                       │  2. Quality gate     │  │
│                                       │  3. Length check      │  │
│                                       │  4. Zero-shot class  │  │
│                                       │  5. NER extraction   │  │
│                                       │  6. Embed + store    │  │
│                                       └─────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     9 Plugin Tools                        │   │
│  │  memory_search | memory_get | memory_store | memory_forget│   │
│  │  memory_reference_search | memory_index_status            │   │
│  │  memory_forget_by_path | memory_inspect | memory_reindex  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/gRPC
                              v
┌─────────────────────────────────────────────────────────────────┐
│                     DGX Spark Node (127.0.0.1)                   │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Embed Server  │  │ Rerank Server│  │ vLLM (Nemotron)    │    │
│  │ :18081        │  │ :18098       │  │ :18080 (chat)      │    │
│  │ llama-embed-  │  │ llama-nemo-  │  │ Nemotron-Super-    │    │
│  │ nemotron-8b   │  │ tron-rerank  │  │ 120B-A12B          │    │
│  │ (4096d)       │  │ -1b-v2       │  │                    │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ NER Server    │  │ Zero-shot    │  │ GLM-OCR (vLLM)     │    │
│  │ :18112        │  │ :8013        │  │ :18090              │    │
│  │ bert-large-   │  │ bart-large-  │  │ GLM-Edge-V-5B      │    │
│  │ NER           │  │ mnli         │  │ (PDF/image OCR)    │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Local Storage                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              LanceDB (on-disk, ~/.openclaw/...)            │   │
│  │  Table: memory_chunks                                     │   │
│  │  Indexes: IVF_PQ (vector, 64 sub-vectors)                │   │
│  │           FTS (text, tantivy-based)                       │   │
│  │  Fields: id, path, source, agent_id, text, vector,       │   │
│  │          updated_at, category, entities, confidence,      │   │
│  │          content_type, quality_score, token_count,        │   │
│  │          parent_heading                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Ingestion Pipeline
```
File/Content → Parse (markdown/PDF/image via OCR) → Chunk (500 tokens, 50 overlap)
  → Quality Gate (30+ noise patterns, min score 0.3) → NER (entity extraction)
  → Embed (4096d via Spark) → Store (LanceDB with mtime preservation)
```

### Recall Pipeline (13 stages)
```
Recent Messages → Clean Query → Embed Query
  → Vector Search (IVF_PQ, cosine) + FTS Search (tantivy, sequential)
  → Hybrid Merge (preserve cosine + rank boost)
  → Dynamic Mistakes Injection (separate filtered search)
  → Source Weighting (configurable: captures 1.5x, sessions 0.5x, etc.)
  → Temporal Decay (0.8 + 0.2 * exp(-0.03 * ageDays), 80% floor)
  → MMR Diversity (λ=0.7, Jaccard similarity)
  → Cross-Encoder Rerank (Nemotron-Rerank-1B via Spark)
  → LCM Dedup (40% overlap with messages + LCM summaries)
  → Prompt Injection Filter
  → Token Budget Enforcement (default: 2000 tokens)
  → Format as <relevant-memories> XML
```

### Capture Pipeline
```
Agent Response → Garbage Filter (30+ patterns) → Quality Gate (score ≥ 0.5)
  → Length Check (≥ 30 chars) → Zero-shot Classification (if enabled)
  → NER Extraction → Embed → Store with content_type="capture"
```

## Content Types

| Type | Source | Description |
|------|--------|-------------|
| `knowledge` | Agent workspace files | AGENTS.md, MEMORY.md, daily notes, etc. |
| `reference` | Reference library | OpenClaw docs, API docs, source code |
| `capture` | Auto-capture hook | Facts, preferences, decisions from conversations |
| `sessions` | Session transcripts | (Disabled by default — LCM handles sessions) |

## Key Design Decisions

1. **Sequential vector → FTS**: LanceDB has a bug where FTS + `.where()` causes Arrow panics. We run searches sequentially so vector results are safe even if FTS fails.

2. **Source weighting before reranking**: Penalize low-quality sources (sessions 0.5x, archives 0.4x) BEFORE the cross-encoder reranker, so they don't waste the limited reranker slots.

3. **Dynamic mistakes injection**: Mistakes get a separate vector search with a lower threshold (0.7x normal), ensuring they're always considered by the reranker regardless of the main query.

4. **Configurable weights**: All source and path weights are in the plugin config, not hardcoded. Operators can tune MISTAKES.md boost from 1.6x to whatever they want.

5. **LCM coordination**: We extract `<content>` blocks from LCM summaries in the conversation and check 40% token overlap to avoid injecting what LCM already provides.

6. **mtime preservation**: File indexing uses filesystem `mtime` for `updated_at`, not indexing time. This prevents temporal decay from being reset on reindex.
