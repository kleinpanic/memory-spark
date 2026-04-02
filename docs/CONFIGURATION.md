# memory-spark Configuration Reference

## Plugin Config Location

In `openclaw.json` under `plugins.entries.memory-spark.config`:

```json
{
  "plugins": {
    "entries": {
      "memory-spark": {
        "enabled": true,
        "config": {
          // All options below
        }
      }
    }
  }
}
```

## Full Configuration

### Embedding (`embed`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | string | `"spark"` | Embed provider: `spark`, `openai`, `gemini` |
| `model` | string | `"nvidia/llama-embed-nemotron-8b"` | Model name |
| `dims` | number | `4096` | Embedding dimensions |
| `spark.host` | string | env `SPARK_HOST` | Spark node hostname/IP |
| `spark.port` | number | `18091` | Spark embed server port |
| `spark.bearerToken` | string | env `SPARK_BEARER_TOKEN` | Auth token |

### Reranking (`rerank`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `true` | Enable cross-encoder reranking |
| `provider` | string | `"spark"` | Rerank provider |
| `model` | string | `"nvidia/llama-nemotron-rerank-1b-v2"` | Model name |
| `spark.host` | string | env `SPARK_HOST` | Spark node hostname/IP |
| `spark.port` | number | `18096` | Rerank server port |

### Auto-Recall (`autoRecall`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `true` | Enable automatic memory injection |
| `agents` | string[] | `["*"]` | Agents to enable recall for (`*` = all) |
| `ignoreAgents` | string[] | `[]` | Agents to exclude |
| `maxResults` | number | `5` | Max memories to inject |
| `minScore` | number | `0.1` | Minimum similarity threshold |
| `queryMessageCount` | number | `2` | Messages to use for query building |
| `maxInjectionTokens` | number | `2000` | Token budget for injection |
| `weights` | object | See below | Source and path weighting config |

#### Recall Weights (`autoRecall.weights`)

```json
{
  "weights": {
    "sources": {
      "capture": 1.5,    // Agent-captured facts/preferences
      "memory": 1.0,     // Workspace files (baseline)
      "sessions": 0.5,   // Session transcripts (penalized)
      "reference": 1.0   // Reference library docs
    },
    "paths": {
      "MEMORY.md": 1.4,
      "TOOLS.md": 1.3,
      "AGENTS.md": 1.2,
      "SOUL.md": 1.2,
      "USER.md": 1.3,
      "memory/learnings.md": 0.1
    },
    "pathPatterns": {
      "mistakes": 1.6,
      "memory/archive/": 0.4
    }
  }
}
```

- `paths`: Exact filename match (after stripping workspace prefix)
- `pathPatterns`: Substring match (case-insensitive). If exact match exists, pattern is NOT applied.

### Auto-Capture (`autoCapture`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `true` | Enable automatic fact extraction |
| `agents` | string[] | `["*"]` | Agents to capture from |
| `ignoreAgents` | string[] | `[]` | Agents to exclude |
| `categories` | string[] | `["fact","preference","decision"]` | Categories to capture |
| `minConfidence` | number | `0.6` | Minimum classification confidence |
| `minMessageLength` | number | `30` | Minimum message length (chars) |
| `useClassifier` | boolean | `true` | Use zero-shot classifier (vs heuristic) |

### Watcher (`watch`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `true` | Enable filesystem watcher |
| `paths` | object[] | auto-discovered | Watch paths and patterns |

### Reference Library (`reference`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `true` | Enable reference library indexing |
| `paths` | object[] | `[]` | Reference paths with tags |

Example:
```json
{
  "reference": {
    "enabled": true,
    "paths": [
      { "path": "~/.local/share/npm/lib/node_modules/openclaw/docs", "tag": "openclaw", "recursive": true },
      { "path": "~/codeWS/TypeScript/memory-spark/src", "tag": "internal", "recursive": true }
    ]
  }
}
```

### Ingestion (`ingest`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `chunkSize` | number | `500` | Target tokens per chunk |
| `chunkOverlap` | number | `50` | Overlap tokens between chunks |
| `sessionIndexing` | boolean | `false` | Index session JSONL files (off by default, LCM handles sessions) |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `SPARK_HOST` | Spark node hostname (e.g. `10.99.1.1`) |
| `SPARK_BEARER_TOKEN` | Spark auth token |
| `MEMORY_SPARK_DATA_DIR` | Override data directory (for testing) |
| `MEMORY_SPARK_LANCEDB_PATH` | Override LanceDB path directly |
