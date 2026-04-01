# Plugin API — 18 Tools

memory-spark exposes 18 tools to OpenClaw agents via the plugin tool registration API.

## Core Memory Tools

### `memory_search`

Search the knowledge base using hybrid vector + FTS search.

**Parameters:**
- `query` (string, required) — Search query
- `maxResults` (number, optional, default 10) — Maximum results to return

**Returns:** Ranked results with scores, paths, and snippets.

**Example:**
```
memory_search("what model does the Spark embed service use?")
→ 1. [config/spark-services.md] (score: 0.89) nvidia/llama-embed-nemotron-8b...
```

---

### `memory_get`

Read a section of an indexed file by path and line range.

**Parameters:**
- `path` (string, required) — File path (as shown in search results)
- `startLine` (number, optional) — Start line
- `endLine` (number, optional) — End line

---

### `memory_store`

Store a fact, preference, or decision into agent memory.

**Parameters:**
- `text` (string, required) — Content to store
- `path` (string, optional) — Virtual path for categorization
- `contentType` (string, optional) — One of: knowledge, decision, preference, mistake

**Example:**
```
memory_store("Klein prefers Opus for complex coding tasks", path: "preferences/models")
```

---

### `memory_forget`

Remove memories matching a query (vector similarity based).

**Parameters:**
- `query` (string, required) — Query to match memories against
- `maxRemove` (number, optional, default 5) — Maximum memories to remove

---

### `memory_forget_by_path`

Remove all indexed chunks from a specific file path.

**Parameters:**
- `path` (string, required) — File path to remove

---

### `memory_bulk_ingest`

Batch store multiple memories in one call. More efficient than repeated `memory_store`.

**Parameters:**
- `items` (array, required, 1-100 items) — Each item:
  - `text` (string, required) — Content
  - `path` (string, optional) — Virtual path
  - `source` (string, optional) — Source identifier
  - `tags` (string[], optional) — Tags

**Example:**
```json
{
  "items": [
    {"text": "Python 3.12 drops distutils", "path": "facts/python"},
    {"text": "Use pathlib over os.path", "path": "preferences/coding"}
  ]
}
```

---

## Search & Discovery Tools

### `memory_reference_search`

Search indexed reference docs (read-only pools — ingested from files, not captured).

**Parameters:**
- `query` (string, required)
- `maxResults` (number, optional)

---

### `memory_temporal`

Search memories within a specific time window.

**Parameters:**
- `query` (string, required)
- `after` (string, optional) — ISO date. Only return memories after this date.
- `before` (string, optional) — ISO date. Only return memories before this date.
- `maxResults` (number, optional)

**Example:**
```
memory_temporal("deployment changes", after: "2026-03-25", before: "2026-04-01")
```

---

### `memory_related`

Find semantically similar memories given a chunk ID.

**Parameters:**
- `chunkId` (string, required) — ID of the memory to find neighbors for
- `maxResults` (number, optional, default 5)

---

### `memory_mistakes_search`

Search agent mistake patterns.

**Parameters:**
- `query` (string, required)
- `maxResults` (number, optional)

---

### `memory_rules_search`

Search shared rules across agents.

**Parameters:**
- `query` (string, required)
- `maxResults` (number, optional)

---

## Storage & Admin Tools

### `memory_mistakes_store`

Store a mistake pattern for future recall. Mistake memories get a 1.6× weight boost during auto-recall.

**Parameters:**
- `text` (string, required) — The mistake pattern to remember
- `agentId` (string, optional) — Which agent made the mistake

---

### `memory_rules_store`

Store a shared rule for all agents.

**Parameters:**
- `text` (string, required) — The rule text

---

### `memory_inspect`

Simulate an auto-recall — see exactly what would be injected for a given query without actually injecting it.

**Parameters:**
- `query` (string, required) — Query to simulate

---

### `memory_reindex`

Trigger a re-index of a single file or a full workspace scan.

**Parameters:**
- `path` (string, optional) — Specific file to re-index. Omit for full scan.

---

### `memory_index_status`

Health dashboard showing: chunk counts, pool breakdown, agent breakdown, service health probes, cache stats, reranker gate config.

**Parameters:**
- `agentId` (string, optional) — Filter stats to a specific agent

**Output includes:**
- Total chunks
- Backend status
- Index list (IVF_PQ, FTS)
- Top paths by chunk count
- Chunks by pool (agent_memory, shared_knowledge, etc.)
- Chunks by agent
- Embed/Reranker/Cache health
- Reranker gate mode and thresholds
- Auto-recall/capture config

---

### `memory_recall_debug`

Full pipeline trace for a query — shows what happens at every stage.

**Parameters:**
- `query` (string, required)
- `maxResults` (number, optional)

**Output includes:**
- Vector search: count, top scores + paths
- FTS search: count, top scores + paths
- Hybrid merge: count, top scores
- Reranked: count, top scores + text preview
- Gate config: mode, thresholds, blend mode

---

### `memory_gate_status`

Show the current reranker gate configuration.

**Parameters:** None.

**Output:**
- Gate mode (hard/soft/off)
- Thresholds (high, low)
- Blend mode (rrf/score)
- RRF k value
- Human-readable explanation of current gate behavior
