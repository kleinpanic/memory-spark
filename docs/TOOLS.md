# memory-spark Plugin Tools

## Overview

memory-spark exposes 9 tools to agents via the OpenClaw plugin system. Tools fall into three categories:

1. **Search**: Find information in the knowledge base
2. **Manage**: Store, forget, and maintain memories
3. **Diagnostic**: Inspect, debug, and manage the pipeline

## Tool Reference

### memory_search
**Category:** Search | **Auto-injected:** No (agent must call explicitly)

Search the knowledge base and memory for relevant information. Use when auto-recall didn't surface what you need, or for specific lookups.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | ✅ | — | What to search for |
| maxResults | number | | 10 | Max results to return |

**When to use:** When auto-recall missed something, when you need a specific fact, config detail, past decision, or mistake. Searches across all agent workspaces, reference docs, and captured knowledge.

---

### memory_get
**Category:** Search | **Auto-injected:** No

Read a section of an indexed file by path and line range.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| path | string | ✅ | — | Relative path to the file |
| from | number | | 1 | Start line (1-indexed) |
| lines | number | | 50 | Lines to read |

**When to use:** When you found a file via `memory_search` or `memory_reference_search` and need to read a specific section.

---

### memory_store
**Category:** Manage | **Auto-injected:** No

Explicitly store a piece of information in long-term memory. Use for facts, preferences, or decisions the user wants remembered.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| text | string | ✅ | — | The information to remember |
| category | string | | — | Category: fact, preference, decision, code-snippet |

**When to use:** When the user says "remember this", or when you learn an important fact that should persist across sessions.

---

### memory_forget
**Category:** Manage | **Auto-injected:** No

Remove memories matching a query. Use when the user wants to correct or delete stored information.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | ✅ | — | What to forget — matches and removes similar memories |

---

### memory_reference_search
**Category:** Search | **Auto-injected:** No

Search reference documentation (textbooks, API docs, source code docs). Use instead of web search when relevant reference material has been indexed.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | ✅ | — | What to search for in reference documentation |
| tag | string | | — | Filter by tag (e.g. 'internal', 'openclaw') |
| maxResults | number | | 10 | Max results |

**When to use:** When you need documentation about OpenClaw, APIs, or other indexed reference material. More reliable than web search for known-good docs.

---

### memory_index_status
**Category:** Diagnostic | **Auto-injected:** No

Show memory index statistics: chunk counts by type, index health, Spark service probes, embed cache stats, and configuration summary.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| agentId | string | | current | Agent ID to scope stats to |

**Returns:** Chunk counts, source breakdown, service health (embed/reranker probes), embed queue stats (queued/failed/healthy), cache performance (hit rate), and config summary (autoRecall/autoCapture agents).

---

### memory_forget_by_path
**Category:** Manage | **Auto-injected:** No

Remove all indexed chunks from a specific file path. Use when reference docs are outdated or a file has been deleted.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| path | string | ✅ | — | Relative file path whose chunks should be removed |

---

### memory_inspect
**Category:** Diagnostic | **Auto-injected:** No

Simulate auto-recall for a query. Shows exactly what would be injected into context, with scores, sources, and weights applied. Use to debug recall quality or verify that important memories (MISTAKES, TOOLS) are being retrieved.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | string | ✅ | — | Simulate a query to see what would be recalled |
| maxResults | number | | 5 | Max results to show |

**When to use:** When you suspect recall is missing important information, or to verify that weight changes are working as expected.

---

### memory_reindex
**Category:** Manage | **Auto-injected:** No

Trigger a re-index of memory files. With a path, re-indexes just that file. Without a path, triggers a full boot-pass re-scan.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| path | string | | — | Specific file path to re-index (omit for full re-scan) |

**When to use:** After editing a workspace file that should be reflected in search immediately, or when the index seems stale.

## Auto-Injection vs Tool Use

### Auto-Injection (before_prompt_build)
- Runs automatically on every agent turn
- Agent does NOT choose or control this
- Injects `<relevant-memories>` XML into context
- Uses the last N messages as query (configurable via `queryMessageCount`)
- Budget: default 2000 tokens (configurable via `maxInjectionTokens`)

### Tool-Based Search
- Agent must actively decide to call `memory_search` or `memory_reference_search`
- Results are returned to the agent, not injected into context
- No token budget limit (agent controls how much to read)
- More precise: agent crafts the query intentionally

**Best practice:** Auto-injection handles 80% of cases. Use tools when auto-injection misses something specific, or when you need to search reference docs.
