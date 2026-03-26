# memory-spark ⚡

**GPU-Accelerated Persistent Memory for OpenClaw AI Agents**

Hybrid search · Cross-encoder reranking · Temporal decay · Configurable weights · 9 plugin tools

## What is this?

memory-spark is an OpenClaw plugin that gives AI agents long-term memory backed by a full RAG pipeline. Instead of relying on flat files or hoping the context window remembers, agents get:

- **Auto-recall**: Relevant memories injected before every turn (13-stage pipeline)
- **Auto-capture**: Facts, preferences, and decisions extracted and stored automatically
- **9 tools**: Search, store, forget, inspect, reindex — agents can manage their own memory
- **Cross-agent memory**: All agents share the same knowledge base

## Architecture

```
Agent Turn → Embed Query → Vector Search + FTS → Hybrid Merge
  → Mistakes Injection → Source Weighting → Temporal Decay
  → MMR Diversity → Cross-Encoder Rerank → LCM Dedup
  → Security Filter → Token Budget → <relevant-memories> XML
```

All ML inference runs on a local DGX Spark node — no cloud API calls for embeddings or reranking.

**Stack:**
| Component | Model | Port |
|-----------|-------|------|
| Embeddings | nvidia/llama-embed-nemotron-8b (4096d) | 18081 |
| Reranker | nvidia/llama-nemotron-rerank-1b-v2 | 18098 |
| LLM | Nemotron-Super-120B-A12B (NVFP4) | 18080 |
| NER | bert-large-NER | 18112 |
| Zero-shot | bart-large-mnli | 8013 |
| OCR | GLM-Edge-V-5B (vLLM) | 18090 |
| Storage | LanceDB (IVF_PQ + FTS) | local |

## Current Metrics (v0.2.1)

| Metric | Value |
|--------|-------|
| Unit tests | 144/144 |
| E2E tests (dev) | 7/7 |
| E2E tests (prod) | 6/7 (1 stale data) |
| Content relevance | 90% (vector), 90% (hybrid) |
| ESLint errors | 0 |
| Plugin tools | 9 |
| Index size | ~15K chunks (dev), ~22K (prod) |

## Quick Start

```bash
# Install dependencies
npm install

# Build
npm run build

# Run unit tests
npm test

# Run standalone indexer (requires Spark)
MEMORY_SPARK_DATA_DIR=./test-data npx tsx scripts/standalone-index.ts

# Run E2E benchmark
MEMORY_SPARK_DATA_DIR=./test-data npx tsx scripts/e2e-benchmark.ts

# Run search quality eval
MEMORY_SPARK_DATA_DIR=./test-data npx tsx scripts/quick-eval-v2.ts
```

## Plugin Tools

| Tool | Purpose |
|------|---------|
| `memory_search` | Vector + FTS search across all knowledge |
| `memory_get` | Read a file by path and line range |
| `memory_store` | Store a fact, preference, or decision |
| `memory_forget` | Remove memories matching a query |
| `memory_reference_search` | Search indexed reference docs |
| `memory_index_status` | Health dashboard with service probes |
| `memory_forget_by_path` | Remove all chunks for a file |
| `memory_inspect` | Simulate recall — see what would be injected |
| `memory_reindex` | Trigger re-index (single file or full scan) |

## Configuration

All weights, thresholds, and agents are configurable in `openclaw.json`. See [docs/CONFIGURATION.md](./docs/CONFIGURATION.md).

Key config: source weights (captures 1.5x, sessions 0.5x), path weights (MISTAKES.md 1.6x), temporal decay parameters, auto-recall/capture agent lists, token budgets.

## Documentation

| Doc | Content |
|-----|---------|
| [Architecture](./docs/ARCHITECTURE.md) | System design, pipeline flow, diagrams |
| [Configuration](./docs/CONFIGURATION.md) | Full config reference with defaults |
| [Tools](./docs/TOOLS.md) | All 9 tools: params, usage, examples |
| [Evaluation](./docs/EVALUATION.md) | Benchmarks, metrics, test coverage |
| [Known Issues](./docs/KNOWN-ISSUES.md) | Bugs, workarounds, planned fixes |
| [SOTA Research](./docs/RESEARCH-SOTA-2026.md) | Literature review and gap analysis |

## Project Structure

```
src/
  auto/          # Auto-recall and auto-capture hooks
    recall.ts    # 13-stage recall pipeline
    capture.ts   # Fact extraction + quality gating
    mistakes.ts  # MISTAKES.md enforcement
  classify/      # NER, zero-shot, quality scoring
  embed/         # Embed provider, queue, cache, dims lock
  ingest/        # File parsing, chunking, workspace discovery
  rerank/        # Cross-encoder reranker
  storage/       # LanceDB backend
  security.ts    # Prompt injection detection, memory formatting
  config.ts      # Full config schema with defaults
  manager.ts     # High-level search manager
index.ts         # OpenClaw plugin entry point (hooks + tools)
scripts/         # Standalone indexer, benchmarks, eval
tests/           # Unit tests, integration tests, fixtures
docs/            # Architecture, config, tools, evaluation
```

## License

MIT
