# memory-spark Configuration Reference

> **Auto-generated from `src/config.ts` — v0.4.0 (2026-04-02)**

memory-spark is configured via OpenClaw's plugin config system. All fields have sensible defaults — you only need to override what you want to change.

```jsonc
// In openclaw.json → plugins.entries.memory-spark.config
{
  "sparkHost": "192.168.1.99",        // Override Spark node IP
  "sparkBearerToken": "${SPARK_BEARER_TOKEN}", // Use env template
  // ... any overrides below
}
```

---

## Quick Start (Minimal Config)

If Spark is running on the same machine (default ports), you need zero config. The defaults work.

For a remote Spark node:
```json
{
  "sparkHost": "YOUR_SPARK_IP",
  "sparkBearerToken": "${SPARK_BEARER_TOKEN}"
}
```

---

## Top-Level Fields

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `backend` | `"lancedb"` | `"lancedb"` | Storage backend (only LanceDB supported) |
| `lancedbDir` | string | `~/.openclaw/memory-spark/lancedb` | LanceDB data directory |
| `sparkHost` | string | `"localhost"` | Override all Spark endpoint hostnames |
| `sparkBearerToken` | string | from env | Override `SPARK_BEARER_TOKEN` env var |

---

## Embedding (`embed`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `embed.provider` | `"spark"` \| `"openai"` \| `"gemini"` | `"spark"` | Embedding provider |
| `embed.spark.baseUrl` | string | `http://localhost:18091/v1` | Spark embed endpoint |
| `embed.spark.apiKey` | string | from env | Bearer token |
| `embed.spark.model` | string | `nvidia/llama-embed-nemotron-8b` | Embedding model |
| `embed.spark.dimensions` | number | `4096` | Vector dimensions |
| `embed.spark.queryInstruction` | string | `"Given a query, retrieve relevant passages..."` | Instruction prefix for asymmetric query embeddings. Documents are embedded raw (no prefix). Required for instruction-tuned models like llama-embed-nemotron-8b. |
| `embed.openai.apiKey` | string | — | OpenAI API key |
| `embed.openai.model` | string | `text-embedding-3-small` | OpenAI model |
| `embed.gemini.model` | string | `gemini-embedding-001` | Gemini model |

---

## Reranking (`rerank`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rerank.enabled` | boolean | `true` | Enable cross-encoder reranking |
| `rerank.spark.baseUrl` | string | `http://localhost:18096/v1` | Spark reranker endpoint |
| `rerank.spark.apiKey` | string | from env | Bearer token |
| `rerank.spark.model` | string | `nvidia/llama-nemotron-rerank-1b-v2` | Reranker model |
| `rerank.spark.minScoreSpread` | number | `0.01` | Min score spread to trust reranker results |
| `rerank.topN` | number | `5` | Number of candidates to rerank |
| `rerank.scoreBlendAlpha` | number | `0.0` | Score interpolation: 0=pure reranker, 1=ignore reranker |
| `rerank.blendMode` | `"score"` \| `"rrf"` | `"score"` | Blend strategy. `"rrf"` recommended (scale-invariant) |
| `rerank.rrfK` | number | `60` | RRF smoothing constant (lower = top ranks dominate more) |
| `rerank.rrfVectorWeight` | number | `1.0` | RRF vector rank weight |
| `rerank.rrfRerankerWeight` | number | `1.0` | RRF reranker rank weight |

### Dynamic Reranker Gate

Skip the cross-encoder when vector retrieval is already confident or tied (reranker adds noise in both cases).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rerank.rerankerGate` | `"off"` \| `"hard"` \| `"soft"` | `"off"` | Gate mode. `"hard"` = production recommended (GATE-A) |
| `rerank.rerankerGateThreshold` | number | `0.08` | High spread: skip reranker (vector confident) |
| `rerank.rerankerGateLowThreshold` | number | `0.02` | Low spread: skip reranker (tied set, gambling) |

**GATE-A (Hard Gate)** skips reranking for ~78% of queries with +0.94% NDCG lift and 50% latency reduction on SciFact.

---

## Auto-Recall (`autoRecall`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `autoRecall.enabled` | boolean | `true` | Enable automatic memory injection before each turn |
| `autoRecall.agents` | string[] | `["*"]` | Agents to enable recall for (`"*"` = all) |
| `autoRecall.ignoreAgents` | string[] | `[]` | Agents to exclude (overrides `agents`) |
| `autoRecall.maxResults` | number | `5` | Maximum memories to inject |
| `autoRecall.minScore` | number | `0.65` | Minimum relevance score to inject |
| `autoRecall.queryMessageCount` | number | `4` | Recent messages to build query from |
| `autoRecall.maxInjectionTokens` | number | `2000` | Token budget for injected memories |
| `autoRecall.ftsEnabled` | boolean | `true` | Enable hybrid search (vector + BM25 FTS) |
| `autoRecall.overfetchMultiplier` | number | `4` | Overfetch factor for initial search |
| `autoRecall.dedupOverlapThreshold` | number | `0.4` | Context dedup overlap threshold |

### Source & Path Weighting

```jsonc
"autoRecall": {
  "weights": {
    "sources": {
      "capture": 1.5,    // Auto-captured facts/decisions
      "memory": 1.0,     // MEMORY.md content
      "sessions": 0.5,   // Session transcripts
      "reference": 1.0   // Reference library
    },
    "pathPatterns": {
      "mistakes": 1.6,   // MISTAKES.md gets priority
      "TOOLS": 1.2       // Tool docs boosted
    }
  }
}
```

### Temporal Decay

```jsonc
"autoRecall": {
  "temporalDecay": {
    "floor": 0.8,   // Old content keeps ≥80% of score
    "rate": 0.03    // Decay rate per day
  }
}
```

Formula: `score *= floor + (1 - floor) * exp(-rate * ageDays)`

### MMR Diversity

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `autoRecall.mmrLambda` | number | `0.9` | MMR diversity. 1.0=pure relevance, 0.0=max diversity |
| `autoRecall.hybridVectorWeight` | number | `1.0` | RRF weight for vector results |
| `autoRecall.hybridFtsWeight` | number | `1.0` | RRF weight for FTS results |

### Multi-Query Expansion

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `autoRecall.queryExpansion.enabled` | boolean | `false` | Enable LLM query reformulation |
| `autoRecall.queryExpansion.llmUrl` | string | Spark LLM URL | vLLM endpoint |
| `autoRecall.queryExpansion.model` | string | Spark model | LLM model name |
| `autoRecall.queryExpansion.numReformulations` | number | `3` | Number of query variants |
| `autoRecall.queryExpansion.maxTokens` | number | `100` | Max tokens per reformulation |
| `autoRecall.queryExpansion.temperature` | number | `0.7` | LLM temperature |

---

## Auto-Capture (`autoCapture`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `autoCapture.enabled` | boolean | `true` | Enable automatic fact extraction from turns |
| `autoCapture.agents` | string[] | `["*"]` | Agents to capture from |
| `autoCapture.ignoreAgents` | string[] | `[]` | Agents to exclude |
| `autoCapture.categories` | string[] | `["fact", "decision", "preference", ...]` | Categories to capture |
| `autoCapture.minConfidence` | number | `0.75` | Minimum classifier confidence |
| `autoCapture.minMessageLength` | number | `30` | Min chars to consider for capture |
| `autoCapture.useClassifier` | boolean | `true` | Use Spark zero-shot classifier (falls back to heuristic) |

---

## File Watching (`watch`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `watch.enabled` | boolean | `true` | Enable filesystem watching for auto-indexing |
| `watch.paths` | WatchPath[] | agent workspace | Paths to watch |
| `watch.fileTypes` | string[] | `[".md", ".txt", ...]` | File extensions to index |
| `watch.debounceMs` | number | `2000` | Debounce time for file changes |
| `watch.indexOnBoot` | boolean | `true` | Index all watched files on startup |
| `watch.excludePatterns` | string[] | `["archive", ...]` | Glob patterns to skip |
| `watch.excludePathsExact` | string[] | `["memory/learnings.md"]` | Exact relative paths to skip |
| `watch.indexSessions` | boolean | `false` | Index session JSONL transcripts |

---

## HyDE (Hypothetical Document Embeddings) (`hyde`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `hyde.enabled` | boolean | `true` | Enable HyDE for improved retrieval |
| `hyde.llmUrl` | string | `http://localhost:18080/v1` | vLLM chat completions URL |
| `hyde.model` | string | Nemotron-Super | LLM model for hypothetical docs |
| `hyde.maxTokens` | number | `150` | Max tokens for generated doc |
| `hyde.temperature` | number | `0.7` | Generation temperature |
| `hyde.timeoutMs` | number | `10000` | LLM call timeout |
| `hyde.apiKey` | string | from env | Bearer token |

---

## Chunking (`chunk`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `chunk.maxTokens` | number | `400` | Max tokens per chunk (flat mode) |
| `chunk.overlapTokens` | number | `50` | Overlap between chunks |
| `chunk.minTokens` | number | `20` | Minimum tokens to index |
| `chunk.hierarchical` | boolean | `true` | Enable parent-child chunking |
| `chunk.parentMaxTokens` | number | `2000` | Parent chunk size |
| `chunk.childMaxTokens` | number | `200` | Child chunk size |
| `chunk.childOverlapTokens` | number | `25` | Child chunk overlap |

---

## Full-Text Search (`fts`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fts.enabled` | boolean | `true` | Enable BM25 alongside vector search |
| `fts.sigmoidMidpoint` | number | `3.0` | BM25 sigmoid midpoint for score normalization |

---

## Embed Cache (`embedCache`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `embedCache.enabled` | boolean | `true` | Cache query embeddings (recall only) |
| `embedCache.maxSize` | number | `256` | Max cached embeddings (LRU) |
| `embedCache.ttlMs` | number | `1800000` | Cache TTL (30 minutes) |

---

## Search Tuning (`search`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `search.refineFactor` | number | `20` | ANN refinement (higher = more precise) |
| `search.maxWriteRetries` | number | `3` | Write conflict retries |
| `search.ivfPartitions` | number | `10` | IVF_PQ partitions |
| `search.ivfSubVectors` | number | `64` | IVF_PQ sub-vectors |

---

## Ingestion (`ingest`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ingest.minQuality` | number | `0.3` | Minimum quality score to index |
| `ingest.language` | string | `"en"` | Primary language (`"all"` to disable filtering) |
| `ingest.languageThreshold` | number | `0.3` | Non-Latin character ratio threshold |

---

## Reference Library (`reference`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `reference.enabled` | boolean | `true` | Enable reference doc indexing |
| `reference.paths` | string[] | `[]` | Additional paths to index |
| `reference.chunkSize` | number | `800` | Chunk size for reference docs |
| `reference.tags` | Record | `{}` | Path prefix → tag mapping |

---

## Spark Endpoints (`spark`)

These are auto-configured from `sparkHost`. Only override individually if services are on different hosts/ports.

| Key | Default | Service |
|-----|---------|---------|
| `spark.embed` | `http://localhost:18091/v1` | Embedding (llama-embed-nemotron-8b) |
| `spark.rerank` | `http://localhost:18096/v1` | Reranking (llama-nemotron-rerank-1b-v2) |
| `spark.ner` | `http://localhost:18091/v1` | Named Entity Recognition |
| `spark.zeroShot` | `http://localhost:8013/v1` | Zero-shot classifier |
| `spark.stt` | `http://localhost:8014/v1` | Speech-to-text |
| `spark.ocr` | `http://localhost:18080/ocr` | Legacy OCR |
| `spark.glmOcr` | `http://localhost:18099/v1` | GLM-OCR (vision) |
| `spark.summarizer` | `http://localhost:18080/v1` | Summarization LLM |

---

## Production Config Example (GATE-A)

```json
{
  "sparkHost": "192.168.1.99",
  "sparkBearerToken": "${SPARK_BEARER_TOKEN}",
  "rerank": {
    "enabled": true,
    "blendMode": "rrf",
    "rrfK": 60,
    "rerankerGate": "hard",
    "rerankerGateThreshold": 0.08,
    "rerankerGateLowThreshold": 0.02
  },
  "autoRecall": {
    "maxResults": 5,
    "minScore": 0.65,
    "maxInjectionTokens": 2000,
    "mmrLambda": 0.9
  },
  "autoCapture": {
    "minConfidence": 0.75,
    "useClassifier": true
  }
}
```

This configuration achieves NDCG@10 of 0.7802 on BEIR SciFact with 50% latency reduction via the hard gate.
