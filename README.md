# memory-spark

An OpenClaw memory plugin with LanceDB vector storage and NVIDIA Spark GPU-accelerated embeddings. Drop-in replacement for `memory-core` with autonomous recall, capture, file ingestion, and hybrid search.

## Features

- **Auto-recall** — injects relevant memories before every agent turn (`before_prompt_build` hook)
- **Auto-capture** — extracts and stores facts/preferences from conversations (`agent_end` hook)
- **File watcher** — auto-indexes your workspace memory dirs, sessions, and watched paths on boot and on change
- **PDF / DOCX / audio ingest** — parse and embed documents via `pdftotext`, `mammoth`, and STT
- **Hybrid search + rerank** — vector search + BM25 full-text, reranked by Cohere-compatible cross-encoder
- **NVIDIA Spark GPU embeddings** — `nvidia/llama-embed-nemotron-8b` (4096d), 0.25s/call on Blackwell GPU
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
│   │   └── capture.ts       # agent_end hook — classifies + stores new facts
│   ├── embed/
│   │   ├── provider.ts      # EmbedProvider: Spark → OpenAI → Gemini fallback
│   │   ├── queue.ts         # EmbedQueue: serial requests, retry, health tracking
│   │   ├── chunker.ts       # Markdown-aware token-bounded chunking with overlap
│   │   └── dims-lock.ts     # Dimension lock: validates provider/model/dims on startup
│   ├── ingest/
│   │   ├── pipeline.ts      # extract → chunk → NER → embed → upsert
│   │   ├── watcher.ts       # Chokidar watcher + boot pass across all agent workspaces
│   │   ├── workspace.ts     # Agent workspace discovery, session scanning
│   │   └── parsers.ts       # PDF (pdftotext+OCR), DOCX (mammoth), audio (STT), text
│   ├── classify/
│   │   ├── ner.ts           # NER via Spark 18112 (dslim/bert-base-NER)
│   │   └── zero-shot.ts     # Zero-shot classifier via Spark 18113 (bart-large-mnli)
│   ├── rerank/
│   │   └── reranker.ts      # Cohere-compatible reranker via Spark 18096
│   └── storage/
│       ├── backend.ts       # StorageBackend interface
│       └── lancedb.ts       # LanceDB: connect, upsert (mergeInsert), vectorSearch, FTS, delete
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

## Setup

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
    "entries": { "memory-spark": { "enabled": true } },
    "load": { "paths": ["~/.openclaw/extensions/memory-spark"] }
  }
}
```

### 3. Environment

Set your Spark bearer token in `~/.openclaw/.env`:

```
SPARK_BEARER_TOKEN=your_token_here
```

### 4. Config (openclaw.json)

```json
{
  "memory": {
    "backend": "lancedb",
    "lancedb": { "uri": "~/.openclaw/data/memory-spark/lancedb" },
    "embed": {
      "provider": "spark",
      "model": "nvidia/llama-embed-nemotron-8b",
      "baseUrl": "http://localhost:18091/v1"
    },
    "recall": {
      "enabled": true,
      "topK": 5,
      "minScore": 0.3,
      "maxTokens": 1500
    },
    "capture": {
      "enabled": true,
      "categories": ["fact", "preference", "decision"]
    },
    "watch": {
      "enabled": true,
      "indexOnBoot": true,
      "debounceMs": 2000,
      "paths": []
    }
  }
}
```

## Tools

Exposes four tools to agents:

| Tool | Description |
|------|-------------|
| `memory_search` | Hybrid vector + FTS search with reranking |
| `memory_get` | Retrieve a specific chunk by ID |
| `memory_store` | Store a fact/note manually |
| `memory_forget` | Delete memories by path or query |

## License

MIT

## Acknowledgments

Built on [LanceDB](https://lancedb.github.io/lancedb/), [NVIDIA Nemotron](https://developer.nvidia.com/nemotron), and [OpenClaw](https://openclaw.ai).
