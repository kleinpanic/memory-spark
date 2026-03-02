# memory-spark

Autonomous Spark-powered memory plugin for OpenClaw.
Drop-in replacement for `memory-core` — same tools, massively upgraded pipeline.

## What it does

- **Auto-indexes** files dropped into configured paths. No commands. No cron.
- **Auto-recalls** relevant memories before every agent turn via `before_prompt_build` hook injection.
- **Auto-captures** facts, preferences, and decisions from conversations via `agent_end` hook.
- **LanceDB** primary storage with native vector ANN + Tantivy FTS + hybrid search.
- **Spark-backed** embed (18091), reranker (18096), OCR (18097), NER (18112), zero-shot (18113), STT (18094).
- **Portable**: provider falls back Spark → OpenAI → Gemini. Works on setups without a DGX node.
- **Auto-migrates** existing memory-core SQLite-vec data on first boot (re-embeds with new provider).

## Architecture

```
~/.openclaw/extensions/memory-spark/
├── index.ts                   ← plugin entry, registers everything with OC
├── src/
│   ├── config.ts              ← config schema + defaults
│   ├── manager.ts             ← MemorySearchManager (OC tool interface)
│   ├── embed/
│   │   ├── provider.ts        ← Spark/OpenAI/Gemini embedding with fallback chain
│   │   └── chunker.ts         ← token-aware, markdown-aware chunking
│   ├── rerank/
│   │   └── reranker.ts        ← Spark cross-encoder reranking (Cohere-compatible API)
│   ├── storage/
│   │   ├── backend.ts         ← StorageBackend interface
│   │   ├── lancedb.ts         ← LanceDB primary backend
│   │   └── sqlite-vec.ts      ← SQLite-vec migration source + fallback
│   ├── ingest/
│   │   ├── watcher.ts         ← chokidar file watcher (gateway service)
│   │   ├── pipeline.ts        ← file → chunk → NER → embed → store pipeline
│   │   └── parsers.ts         ← .md/.txt, .pdf (pdftotext+OCR), .docx, audio/STT
│   ├── auto/
│   │   ├── recall.ts          ← before_prompt_build: inject top-K memories
│   │   └── capture.ts         ← agent_end: classify + store turn facts
│   └── classify/
│       ├── ner.ts             ← Spark NER entity tagging per chunk
│       └── zero-shot.ts       ← Spark zero-shot: fact/pref/decision/snippet/none
└── scripts/
    └── migrate.ts             ← one-time migration from memory-core
```

## Spark microservices used

| Port  | Model                                  | Role                          |
|-------|----------------------------------------|-------------------------------|
| 18091 | Qwen/Qwen3-Embedding-4B               | Embedding (2560d)             |
| 18096 | nvidia/llama-nemotron-rerank-1b-v2    | Post-search reranking         |
| 18097 | TrOCR / EasyOCR                       | Scanned PDF text extraction   |
| 18112 | dslim/bert-base-NER                   | Entity tagging per chunk      |
| 18113 | bart-large-mnli                       | Zero-shot capture classify    |
| 18110 | bart-large-cnn                        | Large doc pre-summarization   |
| 18094 | nvidia/parakeet-ctc-1.1b              | Audio → transcript (STT)      |

## Activation (openclaw.json)

```json
"plugins": {
  "allow": ["memory-spark", ...],
  "entries": {
    "memory-core": { "enabled": false },
    "memory-spark": {
      "enabled": true,
      "config": {
        "backend": "lancedb",
        "embed": { "provider": "spark" },
        "rerank": { "enabled": true },
        "autoRecall": {
          "enabled": true,
          "agents": ["main", "school", "research", "dev"]
        },
        "autoCapture": {
          "enabled": true,
          "agents": ["main", "school", "research"]
        },
        "watch": {
          "enabled": true,
          "paths": [
            { "path": "~/Documents/school", "agents": ["school"] },
            { "path": "~/Documents/OpenClaw", "agents": ["meta", "main"] }
          ]
        }
      }
    }
  }
}
```

## Build

```bash
cd ~/.openclaw/extensions/memory-spark
npm install
npm run build
```

## CLI

```bash
openclaw memory status          # indexing stats per agent
openclaw memory sync            # force re-index watched paths
openclaw memory migrate         # run memory-core → LanceDB migration
```

## Portability (no Spark node)

Set `embed.provider` to `"openai"` or `"gemini"`. Reranker auto-disables.
Watcher, auto-recall, and auto-capture all still work — just without local inference.

## Implementation status

All subsystems are scaffolded with full type definitions and inline implementation notes.
Each file has a clear TODO contract. Implementation order:

1. `storage/lancedb.ts` — core data layer
2. `embed/provider.ts` — OpenAI-compat HTTP client for Spark
3. `embed/chunker.ts` — markdown-aware token-bounded chunking
4. `manager.ts` — wire backend + embed + reranker into OC tool interface
5. `rerank/reranker.ts` — Cohere-compat rerank HTTP call
6. `ingest/parsers.ts` — PDF + docx text extraction
7. `ingest/pipeline.ts` — full ingest pipeline
8. `ingest/watcher.ts` — chokidar file watcher
9. `auto/recall.ts` — before_prompt_build injection
10. `auto/capture.ts` — agent_end classification + storage
11. `classify/ner.ts` + `classify/zero-shot.ts` — Spark inference calls
12. `scripts/migrate.ts` — one-time migration runner
