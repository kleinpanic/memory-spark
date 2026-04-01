# memory-spark ⚡

**GPU-Accelerated Persistent Memory for OpenClaw AI Agents**

Hybrid search · RRF fusion · Dynamic reranker gate · Cross-encoder reranking · 18 plugin tools

[![Tests](https://img.shields.io/badge/tests-483%2F483-brightgreen)]() [![Version](https://img.shields.io/badge/version-0.4.0-blue)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

## What is this?

memory-spark is an OpenClaw plugin that gives AI agents long-term memory backed by a full RAG pipeline. Instead of relying on flat files or hoping the context window remembers, agents get:

- **Auto-recall**: Relevant memories injected before every turn (13-stage pipeline)
- **Auto-capture**: Facts, preferences, and decisions extracted and stored automatically
- **18 tools**: Search, store, forget, inspect, debug, bulk ingest — full agent self-service
- **Cross-agent memory**: All agents share the same knowledge base
- **Dynamic reranker gate**: Intelligently skips reranking when vector search is confident, cutting latency by 50%+ while improving recall

## Architecture

```
Agent Turn → Embed Query → Multi-Query Expansion → Vector Search + FTS
  → RRF Hybrid Merge → Source Weighting → Temporal Decay
  → Source Dedup → Dynamic Reranker Gate → Cross-Encoder Rerank
  → RRF Blend → MMR Diversity → Parent-Child Expansion
  → LCM Dedup → Security Filter → Token Budget
  → <relevant-memories> XML injection
```

All ML inference runs on a local NVIDIA DGX Spark — no cloud API calls for embeddings or reranking.

**Stack:**

| Component | Model | Port |
|-----------|-------|------|
| Embeddings | nvidia/llama-embed-nemotron-8b (4096d) | 18081 |
| Reranker | nvidia/llama-nemotron-rerank-1b-v2 | 18098 |
| LLM (HyDE) | Nemotron-Super-120B-A12B (NVFP4) | 18080 |
| NER | bert-large-NER | 18112 |
| Zero-shot | bart-large-mnli | 8013 |
| Storage | LanceDB (IVF_PQ + FTS) | local |

## Benchmark Results (BEIR SciFact, 300 queries)

| Config | NDCG@10 | Δ Baseline | Recall@10 | Latency |
|--------|---------|------------|-----------|---------|
| Vector-Only (baseline) | 0.7709 | — | 0.9037 | 528ms |
| RRF-D (k=20) | 0.7798 | +0.90% | 0.8924 | 1452ms |
| **GATE-A (hard) ★** | **0.7802** | **+0.94%** | **0.9137** | **732ms** |
| GATE-D (soft+RRF) | 0.7803 | +0.94% | 0.8924 | 1413ms |

**GATE-A** is the production default — best recall, best latency, near-best NDCG. The hard gate skipped reranking for 78% of queries (vector was confident), only firing the cross-encoder on 64 ambiguous queries where it could actually help.

## Quick Start

```bash
# Clone and install
git clone https://github.com/kleinpanic/memory-spark.git
cd memory-spark
npm install

# Build
npm run build

# Run tests (483 unit + integration tests)
npm test

# Run BEIR benchmark (requires Spark services)
npx tsx scripts/run-beir-bench.ts --dataset scifact --config A,GATE-A

# Run full benchmark suite (all configs × all datasets, ~7-8h)
bash scripts/run-full-benchmark.sh
```

### OpenClaw Plugin Installation

Add to your `openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "memory-spark": {
        "enabled": true,
        "path": "/path/to/memory-spark",
        "config": {
          "embed": {
            "provider": "spark",
            "model": "nvidia/llama-embed-nemotron-8b",
            "spark": { "baseUrl": "http://localhost:18081" }
          },
          "rerank": {
            "enabled": true,
            "rerankerGate": "hard",
            "blendMode": "rrf",
            "spark": { "baseUrl": "http://localhost:18098" }
          },
          "autoRecall": {
            "enabled": true,
            "agents": ["main", "dev", "meta"]
          },
          "autoCapture": {
            "enabled": true,
            "agents": ["main", "dev"]
          }
        }
      }
    }
  }
}
```

## Plugin Tools (18)

### Core Memory

| Tool | Purpose |
|------|---------|
| `memory_search` | Vector + FTS hybrid search across all knowledge |
| `memory_get` | Read a file by path and line range |
| `memory_store` | Store a fact, preference, or decision |
| `memory_forget` | Remove memories matching a query |
| `memory_forget_by_path` | Remove all chunks for a file path |
| `memory_bulk_ingest` | Batch store multiple memories in one call |

### Search & Discovery

| Tool | Purpose |
|------|---------|
| `memory_reference_search` | Search indexed reference docs (read-only pools) |
| `memory_temporal` | Time-windowed search ("what did I learn last week?") |
| `memory_related` | Find semantically similar memories by chunk ID |
| `memory_mistakes_search` | Search agent mistake patterns |
| `memory_rules_search` | Search shared rules across agents |

### Storage & Admin

| Tool | Purpose |
|------|---------|
| `memory_mistakes_store` | Store a mistake pattern for future recall |
| `memory_rules_store` | Store a shared rule for all agents |
| `memory_inspect` | Simulate recall — see what would be injected |
| `memory_reindex` | Trigger re-index (single file or full scan) |
| `memory_index_status` | Health dashboard with service probes, pool/agent breakdown |
| `memory_recall_debug` | Full pipeline trace — see every stage's scores and decisions |
| `memory_gate_status` | Show reranker gate configuration and mode |

## Key Features

### Dynamic Reranker Gate (Phase 12)

The gate analyzes vector score distribution *before* calling the expensive cross-encoder:

- **Spread > 0.08** (vector confident): Skip reranker, trust vector ranking
- **Spread < 0.02** (tied set): Skip reranker, it's gambling on noise
- **Spread 0.02–0.08** (ambiguous): Fire reranker — this is where it helps

Result: 78% of queries skip reranking with **no loss in NDCG** and a **+1.1% improvement in recall**.

### Reciprocal Rank Fusion (RRF)

Replaces score-based hybrid merging. BM25 scores (5–20+) and cosine similarities (0.2–0.6) are on incompatible scales. RRF fuses by rank position only — scale-invariant, no normalization needed.

### Multi-Query Expansion (Phase 11B)

Generates query reformulations via LLM, embeds each, unions results by chunk ID (keeping highest score). Improves recall for ambiguous or under-specified queries.

### HyDE (Hypothetical Document Embeddings)

Generates a hypothetical answer document and embeds *that* instead of the raw question. Bridges the question↔answer semantic gap. Uses asymmetric embedding — hypothetical docs go through `embedDocument` (no instruction prefix) to land in document space.

### Parent-Child Chunking

Small child chunks embed precisely for search. After retrieval, parent text is used for context injection — giving the LLM much more surrounding context while keeping search precision high.

## Configuration

See [docs/CONFIGURATION.md](./docs/CONFIGURATION.md) for the full reference.

Key configuration areas:
- **Source weights**: Captures 1.5×, mistakes 1.6×, sessions 0.5×, archive 0.4×
- **Reranker gate**: Mode (hard/soft/off), thresholds (0.08/0.02), blend mode (rrf/score)
- **Temporal decay**: Floor (0.8), rate (0.03) — gentle decay, recent memories boosted
- **Auto-recall/capture**: Per-agent enable/disable, token budgets, query message count
- **HyDE**: Enable/disable, model config, timeout, quality filters

## Documentation

| Doc | Content |
|-----|---------|
| [Architecture](./docs/ARCHITECTURE.md) | System design, pipeline stages, data flow |
| [Configuration](./docs/CONFIGURATION.md) | Full config reference with defaults |
| [Benchmarks](./docs/BENCHMARKS.md) | BEIR results, methodology, all datasets |
| [Plugin API](./docs/PLUGIN-API.md) | All 18 tools with input/output examples |
| [Tuning Guide](./docs/TUNING.md) | Threshold tuning, RRF weights, HyDE, MMR |
| [Technical Report](./docs/TECHNICAL-REPORT.md) | Deep-dive into pipeline engineering |
| [Changelog](./docs/CHANGELOG.md) | Version history |

## Project Structure

```
src/
  auto/          # Auto-recall (13-stage pipeline) and auto-capture hooks
    recall.ts    # Core recall pipeline with RRF, gate, MMR, parent expansion
    capture.ts   # Fact extraction + dedup (0.92 threshold) + quality gating
    mistakes.ts  # MISTAKES.md enforcement
  classify/      # NER, zero-shot classification, quality scoring
  embed/         # Embed provider, queue with circuit breaker, cache
  hyde/          # Hypothetical Document Embeddings generation
  ingest/        # File parsing, chunking, workspace discovery
  query/         # Multi-query expansion
  rerank/        # Cross-encoder reranker with RRF blend + dynamic gate
  storage/       # LanceDB backend (IVF_PQ + FTS hybrid)
  security.ts    # Prompt injection detection, memory formatting
  config.ts      # Full config schema
index.ts         # OpenClaw plugin entry (18 tools + 3 hooks)
tools/           # Standalone tool implementations
scripts/         # BEIR benchmarks, diagnostics, migration
tests/           # 483 unit + integration tests
evaluation/      # BEIR datasets and results
docs/            # Architecture, config, benchmarks, tuning
```

## Development

```bash
# Watch mode
npm run dev

# Type check
npx tsc --noEmit

# Run specific test file
npx vitest run tests/reranker.test.ts

# Run BEIR benchmark (specific configs)
npx tsx scripts/run-beir-bench.ts --dataset scifact --config A,GATE-A,RRF-D

# Pipeline diagnostic (trace a single query)
VERBOSE=1 npx tsx scripts/diag-pipeline-stages.ts "What is memory-spark?"
```

## License

MIT
