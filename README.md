# memory-spark

An OpenClaw memory plugin with LanceDB vector storage and NVIDIA Spark GPU-accelerated embeddings. Drop-in replacement for `memory-core` with autonomous recall, capture, file ingestion, hybrid search, and a reference library system.

## Does it actually work?

Yes. We test this rigorously — not with academic metrics, but with **real agent scenarios**.

### Practical Agent Utility Eval (16 scenarios)
Tests whether agents can answer real questions using retrieved context:

```
✅ Safety:          4/4 (100%) — restart rules, config safety, model policy
✅ Infrastructure:  4/4 (100%) — IPs, tunnels, machine topology
✅ Workflow:        3/3 (100%) — task tracking, sudo patterns, iMessage
✅ History:         2/3  (67%) — past incidents, post-mortem learnings
✅ Reference:       2/2 (100%) — token limits, GPU memory settings

Overall: 15/16 (94%) with 92% bonus context coverage
```

### A/B Performance Lift (12 hard questions)
Questions about private infrastructure that no model could answer without context:

```
Without memory-spark:   0% correct (model is blind to your setup)
With memory-spark:    100% correct (all ground truth facts retrieved)
Performance lift:    +100%
```

Every question — Spark node IPs, safety rules, past incidents, exact config values — is answered correctly only because memory-spark surfaces the right institutional knowledge.

Run the evals yourself:
```bash
npx tsx scripts/practical-eval.ts   # Retrieval quality
npx tsx scripts/ab-eval.ts          # A/B performance lift
npx tsx scripts/benchmark.ts        # Academic metrics (MRR, Recall@5)
```

## Features

- **Auto-recall** — injects relevant memories before every agent turn (`before_prompt_build` hook)
- **Auto-capture** — extracts and stores facts/preferences from conversations (`agent_end` hook)
- **File watcher** — auto-indexes workspace memory dirs, sessions, and watched paths on boot + change
- **Reference library** — indexes structured documentation with tag-based filtering (`memory_reference_search`)
- **PDF / DOCX / audio ingest** — parse and embed documents via `pdftotext`, `mammoth`, and STT
- **Hybrid search + rerank** — vector (IVF_PQ) + BM25 full-text, reranked by Cohere-compatible cross-encoder
- **Contextual retrieval** — prepends source/path/heading context before embedding for better matches
- **Temporal decay** — exponential freshness scoring (`0.8 + 0.2 * exp(-0.03 * ageDays)`) with 0.8 floor
- **NVIDIA Spark GPU embeddings** — `nvidia/llama-embed-nemotron-8b` (4096d), 0.25s/call on Blackwell GPU
- **MISTAKES.md enforcement** — auto-creates and boosts mistake files (1.6x weight) across agent workspaces
- **Quality filtering** — strips noise (Discord metadata, JSON envelopes, bootstrap spam) before indexing
- **Serialized embed queue** — retry with exponential backoff, health monitoring, auto-cooldown
- **Dimension lock** — refuses to start if provider/dims mismatch, preventing silent vector corruption
- **Zero-touch** — configure once, everything else is automatic

## Architecture

```
memory-spark/
├── index.ts                 # Plugin entry: register(), tool handlers, hooks
├── src/
│   ├── config.ts            # Config schema + defaults + .env loader
│   ├── manager.ts           # MemorySearchManager: search, store, forget
│   ├── auto/
│   │   ├── recall.ts        # before_prompt_build hook — injects memories
│   │   ├── capture.ts       # agent_end hook — classifies + stores new facts
│   │   └── mistakes.ts      # MISTAKES.md auto-creation + weight boost
│   ├── embed/
│   │   ├── provider.ts      # EmbedProvider: Spark → OpenAI → Gemini fallback
│   │   ├── queue.ts         # EmbedQueue: serial requests, retry, health tracking
│   │   ├── chunker.ts       # Markdown-aware token-bounded chunking with overlap
│   │   └── dims-lock.ts     # Dimension lock: validates provider/model/dims on startup
│   ├── ingest/
│   │   ├── pipeline.ts      # extract → chunk → quality filter → embed → upsert
│   │   ├── quality.ts       # Quality gate: strips noise, validates content
│   │   ├── watcher.ts       # Chokidar watcher + boot pass + reference library indexing
│   │   ├── workspace.ts     # Agent workspace discovery, session scanning
│   │   └── parsers.ts       # PDF (pdftotext+OCR), DOCX (mammoth), audio (STT), text
│   ├── classify/
│   │   ├── ner.ts           # NER via Spark 18112 (dslim/bert-base-NER)
│   │   └── zero-shot.ts     # Zero-shot classifier via Spark 18113 (bart-large-mnli)
│   ├── rerank/
│   │   └── reranker.ts      # Cohere-compatible reranker via Spark 18096
│   └── storage/
│       ├── backend.ts       # StorageBackend interface
│       └── lancedb.ts       # LanceDB: IVF_PQ + FTS indexes, mergeInsert, vectorSearch
├── scripts/
│   ├── practical-eval.ts    # Real-world agent scenario eval (94% pass rate)
│   ├── ab-eval.ts           # A/B performance lift measurement (+100%)
│   └── benchmark.ts         # Academic retrieval metrics (MRR, Recall@5, Precision@5)
└── test-unit.ts             # 106 unit tests
```

## Requirements

- [OpenClaw](https://openclaw.ai) — the plugin host
- [LanceDB](https://lancedb.github.io/lancedb/) — `npm install @lancedb/lancedb`
- `pdftotext` (poppler-utils) — for PDF parsing
- NVIDIA Spark microservices (optional but recommended):
  - **Embed** — port 18091 (`nvidia/llama-embed-nemotron-8b`, 4096d)
  - **Reranker** — port 18096 (`nvidia/llama-nemotron-rerank-1b-v2`)
  - **NER** — port 18112 (`dslim/bert-base-NER`)
  - **Zero-shot** — port 18113 (`facebook/bart-large-mnli`)
  - **OCR** — port 18097
  - **STT** — port 18094

Falls back to OpenAI or Gemini embeddings if Spark is unavailable.

## Quick Start

### 1. Install

```bash
cd ~/.openclaw/extensions
git clone https://github.com/kleinpanic/memory-spark
cd memory-spark && npm install && npm run build
```

### 2. Configure OpenClaw

In `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "slots": { "memory": "memory-spark" },
    "allow": ["memory-spark"],
    "entries": {
      "memory-spark": {
        "enabled": true,
        "config": {
          "embed": {
            "provider": "spark",
            "model": "nvidia/llama-embed-nemotron-8b",
            "spark": {
              "baseUrl": "http://YOUR_SPARK_HOST:18091/v1",
              "apiKey": "${SPARK_BEARER_TOKEN}"
            }
          },
          "rerank": {
            "enabled": true,
            "spark": {
              "baseUrl": "http://YOUR_SPARK_HOST:18096/v1",
              "apiKey": "${SPARK_BEARER_TOKEN}"
            }
          },
          "autoRecall": {
            "enabled": true,
            "minScore": 0.2,
            "maxTokens": 1500
          },
          "autoCapture": {
            "enabled": true
          },
          "reference": {
            "enabled": true,
            "paths": ["~/Documents/OpenClaw/ReferenceLibrary/openclaw"],
            "tags": { "openclaw": "openclaw" }
          }
        }
      }
    },
    "load": { "paths": ["~/.openclaw/extensions/memory-spark"] }
  }
}
```

### 3. Environment

Set your Spark bearer token in `~/.openclaw/.env`:

```
SPARK_BEARER_TOKEN=your_token_here
```

## Tools

Exposes six tools to agents:

| Tool | Description |
|------|-------------|
| `memory_search` | Hybrid vector + FTS search with reranking |
| `memory_reference_search` | Search reference docs only, with optional tag filter |
| `memory_index_status` | Show chunk counts, index health, age distribution |
| `memory_get` | Read a section of an indexed file by path and line range |
| `memory_store` | Store a fact/preference/decision manually |
| `memory_forget` | Delete memories by query match |
| `memory_forget_by_path` | Delete all chunks from a specific file path |

## Reference Library

Index structured documentation for targeted agent retrieval:

```json
{
  "reference": {
    "enabled": true,
    "paths": [
      "~/Documents/OpenClaw/ReferenceLibrary/openclaw",
      "~/Documents/OpenClaw/ReferenceLibrary/vllm",
      "~/Documents/OpenClaw/ReferenceLibrary/nvidia-dgx-spark"
    ],
    "tags": {
      "openclaw": "openclaw",
      "vllm": "vllm",
      "dgx-spark": "nvidia-dgx-spark"
    }
  }
}
```

Query with tag filtering:
```
memory_reference_search(query="engine arguments", tag="vllm")
```

Reference docs are indexed with `content_type: "reference"` and kept separate from daily agent captures.

## Testing

```bash
npm run build                         # Compile TypeScript
npx tsx test-unit.ts                  # 106 unit tests
npx tsx scripts/practical-eval.ts     # Agent scenario eval
npx tsx scripts/ab-eval.ts            # A/B performance lift
npx tsx scripts/benchmark.ts          # Retrieval metrics (needs live Spark)
```

## License

MIT

## Acknowledgments

Built on [LanceDB](https://lancedb.github.io/lancedb/), [NVIDIA Nemotron](https://developer.nvidia.com/nemotron), and [OpenClaw](https://openclaw.ai).
