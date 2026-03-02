# Implementation Instructions for memory-spark

## What this is
An OpenClaw plugin that replaces the built-in `memory-core` plugin. It provides autonomous memory management: auto-indexing files, auto-recall before agent turns, auto-capture of facts from conversations, backed by LanceDB + Spark microservices.

## Critical Type Information

### AgentTool (from @mariozechner/pi-agent-core)
Tools must use TypeBox schemas. The exact interface:

```typescript
import { Type, type TSchema, type Static } from "@sinclair/typebox";

interface Tool<TParameters extends TSchema = TSchema> {
  name: string;
  description: string;
  parameters: TParameters;
}

interface AgentTool<TParameters extends TSchema = TSchema, TDetails = any> extends Tool<TParameters> {
  label: string;
  execute: (toolCallId: string, params: Static<TParameters>, signal?: AbortSignal, onUpdate?: AgentToolUpdateCallback<TDetails>) => Promise<AgentToolResult<TDetails>>;
}

interface AgentToolResult<T> {
  content: (TextContent | ImageContent)[];
  details: T;
}

type TextContent = { type: "text"; text: string };
type ImageContent = { type: "image"; data: string; mimeType: string };
```

Example tool construction:
```typescript
import { Type } from "@sinclair/typebox";

const params = Type.Object({
  query: Type.String({ description: "What to search for" }),
  maxResults: Type.Optional(Type.Number({ description: "Max results" })),
});

const tool: AnyAgentTool = {
  name: "memory_search",
  description: "Search the knowledge base",
  label: "Memory Search",
  parameters: params,
  execute: async (toolCallId, params) => ({
    content: [{ type: "text", text: "results here" }],
    details: {},
  }),
};
```

### Plugin API (from openclaw/plugin-sdk)
The real exports from `"openclaw/plugin-sdk"`:
- `emptyPluginConfigSchema` function
- Types: `AnyAgentTool`, `OpenClawPluginApi`, `OpenClawPluginService`, `OpenClawPluginServiceContext`, `PluginRuntime`

Hook types are NOT exported from the barrel. Use structural typing for hook events:
```typescript
// Don't import PluginHookBeforePromptBuildEvent - it's not exported
// Instead use the structure directly:
type BeforePromptBuildEvent = { prompt: string; messages: unknown[] };
type BeforePromptBuildResult = { systemPrompt?: string; prependContext?: string };
type AgentEndEvent = { messages: unknown[]; success: boolean; error?: string; durationMs?: number };
type AgentContext = { agentId?: string; sessionKey?: string; sessionId?: string; workspaceDir?: string };
```

Hook registration:
```typescript
api.on("before_prompt_build", async (event, ctx) => {
  return { prependContext: "injected text" };
});

api.on("agent_end", async (event, ctx) => {
  // no return value needed
});
```

Plugin tool factory:
```typescript
api.registerTool(
  (ctx) => {
    // ctx has: config, sessionKey, agentId, etc.
    return [tool1, tool2]; // AnyAgentTool[]
  },
  { names: ["memory_search", "memory_get"] }
);
```

Service registration:
```typescript
api.registerService({
  id: "my-service",
  start: async (ctx) => { /* ctx.logger, ctx.config, ctx.stateDir */ },
  stop: async (ctx) => { /* cleanup */ },
});
```

### LanceDB API (from @lancedb/lancedb v0.14)
```typescript
import * as lancedb from "@lancedb/lancedb";

const db = await lancedb.connect("/path/to/db");

// Create table with data (schema inferred from first record)
const table = await db.createTable("my_table", [
  { id: "1", text: "hello", vector: [0.1, 0.2, ...], updated_at: "2026-01-01" }
]);

// Or open existing
const table = await db.openTable("my_table");

// Add data
await table.add([{ id: "2", text: "world", vector: [...] }]);

// Vector search
const results = await table.vectorSearch([0.1, 0.2, ...])
  .distanceType("cosine")
  .limit(10)
  .where("agent_id = 'main'")
  .toArray();
// results: [{ id, text, vector, _distance, ... }]

// FTS (needs createIndex first)
await table.createIndex("text", { config: lancedb.Index.fts() });
const ftsResults = await table.search("query text", "fts", "text")
  .limit(10)
  .toArray();

// Delete
await table.delete("path = 'some/path'");

// Merge insert (upsert)
await table.mergeInsert("id")
  .whenMatchedUpdateAll()
  .whenNotMatchedInsertAll()
  .execute([{ id: "1", text: "updated", vector: [...] }]);

// Count
const count = await table.countRows();

// List tables
const names = await db.tableNames();
```

### Spark Microservice Endpoints
All are standard HTTP REST:

**Embed (18091)** — OpenAI-compatible:
```
POST http://dgx-spark.local:18091/v1/embeddings
{ "model": "Qwen/Qwen3-Embedding-4B", "input": ["text1", "text2"] }
→ { "data": [{ "embedding": [0.1, ...], "index": 0 }, ...] }
```

**Rerank (18096)** — Cohere-compatible:
```
POST http://dgx-spark.local:18096/v1/rerank
{ "model": "nvidia/llama-nemotron-rerank-1b-v2", "query": "...", "documents": ["doc1", "doc2"], "top_n": 5 }
→ { "results": [{ "index": 0, "relevance_score": 0.95 }, ...] }
```

**NER (18112)** — HuggingFace pipeline:
```
POST http://dgx-spark.local:18112
{ "inputs": "text here" }
→ [{ "entity_group": "PER", "score": 0.99, "word": "Klein" }, ...]
```

**Zero-shot (18113)** — HuggingFace pipeline:
```
POST http://dgx-spark.local:18113
{ "inputs": "text", "parameters": { "candidate_labels": ["fact","preference","decision","code-snippet","none"] } }
→ { "labels": ["fact", ...], "scores": [0.85, ...] }
```

## What to implement

Every file with `throw new Error("... not yet implemented")` needs real code. The architecture and interfaces in config.ts and storage/backend.ts are correct — implement against them.

### Key implementation notes:
1. `storage/lancedb.ts` — Use one table called "memory_chunks" (not per-agent tables). Filter by agent_id column. Use mergeInsert for upserts.
2. `embed/provider.ts` — Use native fetch(). The Spark endpoint is OpenAI-compatible. Handle batch size limits (max ~100 texts per call).
3. `embed/chunker.ts` — Use simple character-based estimation (4 chars ≈ 1 token). Split markdown on headers first, then paragraphs, then hard-split.
4. `rerank/reranker.ts` — Simple fetch to Spark 18096. Passthrough fallback on failure.
5. `ingest/parsers.ts` — Use fs.readFile for text. Use child_process.execFile for pdftotext. Use mammoth for docx.
6. `ingest/watcher.ts` — Use chokidar. Debounce changes. Run boot pass on start.
7. `auto/recall.ts` — Return { prependContext } from before_prompt_build hook.
8. `auto/capture.ts` — Fire-and-forget in agent_end hook. Never throw.
9. `classify/ner.ts` and `classify/zero-shot.ts` — Simple fetch calls. Return [] or "none" on failure.
10. `index.ts` — Use TypeBox for tool parameters. Proper AgentTool construction. api.on() for hooks.
11. `scripts/migrate.ts` — Read existing SQLite-vec DBs, extract text, re-embed, store in LanceDB.
12. `config.ts` — Implement proper deep merge in resolveConfig().

### Import rules:
- `"openclaw/plugin-sdk"` for: `emptyPluginConfigSchema`, types `OpenClawPluginApi`, `AnyAgentTool`, `OpenClawPluginService`
- `"@sinclair/typebox"` for: `Type`, `Static`, `TSchema`
- `"@lancedb/lancedb"` for LanceDB
- `"chokidar"` for file watching
- `"pdf-parse"` for PDF extraction (fallback if pdftotext unavailable)
- `"mammoth"` for docx extraction
- Node built-ins: `node:fs/promises`, `node:path`, `node:crypto`, `node:child_process`, `node:os`

### TypeScript config:
- Target: ES2022, Module: ESNext, Strict: true
- TypeBox v0.34.x (installed via openclaw peer dep)
- All files are ESM (.ts with type: module in package.json)

## Build and test
```bash
cd ~/.openclaw/extensions/memory-spark
npx tsc --noEmit  # type check only
npm run build      # full build to dist/
```
