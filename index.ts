/**
 * memory-spark — OpenClaw Memory Plugin
 *
 * Drop-in replacement for memory-core.
 * kind: "memory" → occupies the memory slot exclusively.
 *
 * Replicates ALL memory-core functionality:
 *   - memory_search + memory_get tools
 *   - Auto-indexes workspace memory dirs per agent
 *   - Auto-indexes workspace root files (MEMORY.md, SOUL.md, USER.md, etc.)
 *   - Auto-indexes session JSONL transcripts
 *   - Auto-recall: injects memories before each agent turn
 *   - Auto-capture: stores facts/preferences after each turn
 *   - File watcher on external configured paths
 *   - Migration from existing memory-core on first boot
 */

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { Type } from "@sinclair/typebox";
import type { OpenClawPluginApi, OpenClawPluginConfigSchema } from "openclaw/plugin-sdk";

import { createAutoCaptureHandler } from "./src/auto/capture.js";
import { createAutoRecallHandler } from "./src/auto/recall.js";
import { resolveConfig, type MemorySparkConfig } from "./src/config.js";
import { withCache, type CachedEmbedProvider } from "./src/embed/cached-provider.js";
import { validateDimsLock } from "./src/embed/dims-lock.js";
import { createEmbedProvider, type EmbedProvider } from "./src/embed/provider.js";
import { EmbedQueue } from "./src/embed/queue.js";
import { createWatcher, type Watcher } from "./src/ingest/watcher.js";
import { MemorySparkManager } from "./src/manager.js";
import { createReranker, type Reranker } from "./src/rerank/reranker.js";
import type { StorageBackend } from "./src/storage/backend.js";
import { LanceDBBackend } from "./src/storage/lancedb.js";

// ---------------------------------------------------------------------------
// Singleton plugin state
// ---------------------------------------------------------------------------
interface PluginState {
  cfg: MemorySparkConfig;
  backend: StorageBackend;
  embed: EmbedProvider;
  queue: EmbedQueue;
  cachedEmbed: CachedEmbedProvider;
  reranker: Reranker;
  watcher: Watcher | null;
}

let state: PluginState | null = null;
let initPromise: Promise<PluginState> | null = null;

async function getState(
  cfg: MemorySparkConfig,
  logger: { info: (m: string) => void; warn: (m: string) => void; error: (m: string) => void },
): Promise<PluginState> {
  if (state) return state;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    logger.info("memory-spark: initializing...");

    // Single-table backend with pool-based logical isolation.
    // LanceDB 0.27+ supports FTS+WHERE natively, so no multi-table workaround needed.
    // Pool column provides logical separation: agent_memory, shared_mistakes, shared_rules, etc.
    const backend: StorageBackend = new LanceDBBackend(cfg);
    await backend.open();
    logger.info("memory-spark: LanceDB backend initialized (single-table + pool architecture)");

    const embed = await createEmbedProvider(cfg.embed);
    logger.info(`memory-spark: embed → ${embed.id}/${embed.model} (${embed.dims}d)`);

    // Dimension safety: lock dims on first boot, refuse mismatch on subsequent boots
    const dimsCheck = await validateDimsLock(cfg.lancedbDir, embed.id, embed.model, embed.dims);
    if (!dimsCheck.ok) {
      logger.error(`memory-spark: FATAL — ${dimsCheck.error}`);
      throw new Error(dimsCheck.error);
    }

    // Queue: serialized embed requests with retry/backoff
    const queue = new EmbedQueue(
      embed,
      {
        concurrency: 1,
        maxRetries: 3,
        baseDelayMs: 2000,
        maxDelayMs: 30000,
        timeoutMs: 30000,
        unhealthyThreshold: 5,
        unhealthyCooldownMs: 60000,
      },
      logger,
    );

    const reranker = await createReranker(cfg.rerank);

    // Cache wraps the queue for query embeddings only (recall).
    // Document embeddings (indexing) bypass the cache via queue directly.
    const cachedEmbed = withCache(queue, {
      enabled: cfg.embedCache?.enabled ?? true,
      maxSize: cfg.embedCache?.maxSize ?? 256,
      ttlMs: cfg.embedCache?.ttlMs ?? 30 * 60 * 1000,
    });

    state = { cfg, backend, embed, queue, cachedEmbed, reranker, watcher: null };
    logger.info("memory-spark: ready");
    return state;
  })();

  return initPromise;
}

// ---------------------------------------------------------------------------
// Tool schemas (TypeBox)
// ---------------------------------------------------------------------------
const SearchParams = Type.Object({
  query: Type.String({ description: "What to search for" }),
  maxResults: Type.Optional(Type.Number({ description: "Max results (default 10)" })),
});

const GetParams = Type.Object({
  path: Type.String({ description: "Relative path to the file" }),
  from: Type.Optional(Type.Number({ description: "Start line (1-indexed)" })),
  lines: Type.Optional(Type.Number({ description: "Lines to read (default 50)" })),
});

const StoreParams = Type.Object({
  text: Type.String({ description: "The information to remember" }),
  category: Type.Optional(
    Type.String({ description: "Category: fact, preference, decision, code-snippet" }),
  ),
});

const ForgetParams = Type.Object({
  query: Type.String({ description: "What to forget — matches and removes similar memories" }),
});

const ReferenceSearchParams = Type.Object({
  query: Type.String({ description: "What to search for in reference documentation" }),
  tag: Type.Optional(Type.String({ description: "Filter by tag (e.g. 'internal', 'openclaw')" })),
  maxResults: Type.Optional(Type.Number({ description: "Max results (default 10)" })),
});

const MistakesSearchParams = Type.Object({
  query: Type.String({ description: "Search for relevant mistakes, errors, and lessons learned" }),
  maxResults: Type.Optional(Type.Number({ description: "Max results (default 5)" })),
});

const MistakesStoreParams = Type.Object({
  description: Type.String({ description: "What went wrong — the mistake or error" }),
  rootCause: Type.Optional(Type.String({ description: "Root cause analysis" })),
  fix: Type.Optional(Type.String({ description: "How it was fixed" })),
  lessons: Type.Optional(Type.String({ description: "Key takeaways to avoid repeating this" })),
  severity: Type.Optional(
    Type.String({ description: "Severity: critical, high, medium, low (default: medium)" }),
  ),
  shared: Type.Optional(
    Type.Boolean({ description: "Share with all agents (default: false — per-agent only)" }),
  ),
});

const RulesStoreParams = Type.Object({
  rule: Type.String({
    description:
      "The rule, preference, or guideline to store. E.g. 'Never use Gemini Flash for coding tasks'",
  }),
  scope: Type.Optional(
    Type.String({
      description: "Scope: 'global' (all agents, default) or an agent ID for agent-specific rules",
    }),
  ),
  category: Type.Optional(
    Type.String({
      description: "Category: preference, constraint, workflow, safety (default: preference)",
    }),
  ),
});

const RulesSearchParams = Type.Object({
  query: Type.String({ description: "Search for rules, preferences, or guidelines" }),
  maxResults: Type.Optional(Type.Number({ description: "Max results (default 5)" })),
});

const IndexStatusParams = Type.Object({
  agentId: Type.Optional(
    Type.String({ description: "Agent ID to scope stats to (default: current agent)" }),
  ),
});

const ForgetByPathParams = Type.Object({
  path: Type.String({ description: "Relative file path whose indexed chunks should be removed" }),
});

const InspectParams = Type.Object({
  query: Type.String({ description: "Simulate a query to see what would be recalled" }),
  maxResults: Type.Optional(Type.Number({ description: "Max results to show (default: 5)" })),
});

const ReindexParams = Type.Object({
  path: Type.Optional(
    Type.String({ description: "Specific file path to re-index (omit for full re-scan)" }),
  ),
});

// ---------------------------------------------------------------------------
// Plugin definition
// ---------------------------------------------------------------------------
const memorySpark = {
  id: "memory-spark",
  name: "Memory Spark",
  version: "0.1.0",
  description:
    "Autonomous Spark-powered memory: LanceDB + local embed/rerank + auto-recall/capture.",
  kind: "memory" as const,
  configSchema: {
    safeParse(value: unknown) {
      if (value === undefined || value === null) return { success: true, data: undefined };
      if (typeof value !== "object" || Array.isArray(value)) {
        return {
          success: false,
          error: { issues: [{ path: [], message: "expected config object" }] },
        };
      }
      // Passthrough — resolveConfig() handles deep merging and type coercion
      return { success: true, data: value };
    },
    jsonSchema: {
      type: "object",
      additionalProperties: true,
      properties: {
        sparkHost: {
          type: "string",
          description: "Spark node IP/hostname (overrides SPARK_HOST env)",
        },
        sparkBearerToken: {
          type: "string",
          description: "Spark bearer token (overrides SPARK_BEARER_TOKEN env)",
        },
        backend: { type: "string", enum: ["lancedb"] },
        lancedbDir: { type: "string", description: "Path to LanceDB data directory" },
        autoRecall: {
          type: "object",
          properties: {
            enabled: { type: "boolean" },
            agents: {
              type: "array",
              items: { type: "string" },
              description: 'Agent IDs or ["*"] for all',
            },
            ignoreAgents: {
              type: "array",
              items: { type: "string" },
              description: "Agent IDs to exclude from recall",
            },
            maxResults: {
              type: "number",
              description: "Max memories to inject per turn (default: 5)",
            },
            minScore: { type: "number", description: "Minimum similarity score (default: 0.75)" },
            queryMessageCount: {
              type: "number",
              description: "Recent messages used as recall query (default: 2)",
            },
            maxInjectionTokens: {
              type: "number",
              description: "Token budget for injected memories (default: 2000)",
            },
            mmrLambda: {
              type: "number",
              description: "MMR diversity lambda 0-1 (default: 0.7, higher=more relevant)",
            },
            dedupOverlapThreshold: {
              type: "number",
              description: "Context dedup overlap threshold 0-1 (default: 0.4)",
            },
            overfetchMultiplier: {
              type: "number",
              description: "Overfetch multiplier for search (default: 4)",
            },
            ftsEnabled: {
              type: "boolean",
              description: "Use FTS alongside vector search (default: true)",
            },
            temporalDecay: {
              type: "object",
              properties: {
                floor: { type: "number", description: "Minimum decay multiplier (default: 0.8)" },
                rate: { type: "number", description: "Decay rate per day (default: 0.03)" },
              },
            },
          },
        },
        autoCapture: {
          type: "object",
          properties: {
            enabled: { type: "boolean" },
            agents: {
              type: "array",
              items: { type: "string" },
              description: 'Agent IDs or ["*"] for all',
            },
            ignoreAgents: {
              type: "array",
              items: { type: "string" },
              description: "Agent IDs to exclude from capture",
            },
            categories: { type: "array", items: { type: "string" } },
            minConfidence: { type: "number" },
            minMessageLength: {
              type: "number",
              description: "Minimum chars to consider for capture (default: 30)",
            },
            useClassifier: {
              type: "boolean",
              description: "Use Spark zero-shot classifier (default: true)",
            },
          },
        },
        embed: {
          type: "object",
          properties: {
            provider: { type: "string", enum: ["spark", "openai", "gemini"] },
          },
        },
        rerank: {
          type: "object",
          properties: {
            enabled: { type: "boolean" },
            topN: { type: "number" },
          },
        },
        fts: {
          type: "object",
          description: "Full-text search (BM25) tuning",
          properties: {
            enabled: {
              type: "boolean",
              description: "Enable FTS alongside vector search (default: true)",
            },
            sigmoidMidpoint: {
              type: "number",
              description: "BM25 sigmoid normalization center (default: 3.0)",
            },
          },
        },
        chunk: {
          type: "object",
          description: "Document chunking configuration",
          properties: {
            maxTokens: { type: "number", description: "Max tokens per chunk (default: 400)" },
            overlapTokens: {
              type: "number",
              description: "Token overlap between chunks (default: 50)",
            },
            minTokens: {
              type: "number",
              description: "Min tokens for a chunk to be indexed (default: 20)",
            },
          },
        },
        embedCache: {
          type: "object",
          description: "Query embedding cache settings",
          properties: {
            enabled: { type: "boolean", description: "Enable embed cache (default: true)" },
            maxSize: { type: "number", description: "Max cached embeddings (default: 256)" },
            ttlMs: { type: "number", description: "Cache TTL in milliseconds (default: 1800000)" },
          },
        },
        search: {
          type: "object",
          description: "Vector search and index tuning",
          properties: {
            refineFactor: { type: "number", description: "ANN refinement factor (default: 20)" },
            maxWriteRetries: {
              type: "number",
              description: "Write conflict retry count (default: 3)",
            },
            ivfPartitions: { type: "number", description: "IVF_PQ partitions (default: 10)" },
            ivfSubVectors: { type: "number", description: "IVF_PQ sub-vectors (default: 64)" },
          },
        },
        watch: {
          type: "object",
          properties: {
            enabled: { type: "boolean" },
            indexOnBoot: { type: "boolean" },
            indexSessions: {
              type: "boolean",
              description: "Index session JSONL transcripts (default: false)",
            },
          },
        },
        ingest: {
          type: "object",
          properties: {
            minQuality: { type: "number", description: "Minimum quality score 0-1 (default: 0.3)" },
            language: {
              type: "string",
              description: "Primary language (default: 'en', use 'all' to disable filtering)",
            },
            languageThreshold: {
              type: "number",
              description: "Non-Latin char ratio threshold (default: 0.3)",
            },
          },
        },
        reference: {
          type: "object",
          properties: {
            enabled: { type: "boolean" },
            paths: {
              type: "array",
              items: { type: "string" },
              description: "Additional paths to index as reference",
            },
            chunkSize: {
              type: "number",
              description: "Chunk size for reference docs (default: 800)",
            },
          },
        },
      },
    },
  } as OpenClawPluginConfigSchema,

  register(api: OpenClawPluginApi) {
    const cfg = resolveConfig(api.pluginConfig as Partial<MemorySparkConfig> | undefined);

    // -------------------------------------------------------------------
    // 1. Tools: memory_search + memory_get
    //    Tool context gives us workspaceDir + agentId per agent session
    // -------------------------------------------------------------------
    api.registerTool(
      (ctx) => {
        const agentId = ctx.agentId ?? "default";
        const workspaceDir =
          ctx.workspaceDir ?? path.join(os.homedir(), ".openclaw", `workspace-${agentId}`);

        const searchTool = {
          name: "memory_search",
          description:
            "Search the knowledge base and memory for relevant information. Use when auto-recall didn't surface what you need, or when you need to search for something specific (a fact, config detail, past decision, or mistake). Searches across all agent workspaces, reference docs, and captured knowledge.",
          label: "Memory Search",
          parameters: SearchParams,
          execute: async (_toolCallId: string, params: { query: string; maxResults?: number }) => {
            const s = await getState(cfg, api.logger);
            const manager = new MemorySparkManager({
              cfg,
              agentId,
              workspaceDir,
              backend: s.backend,
              embed: s.embed,
              reranker: s.reranker,
              queue: s.queue,
            });
            const results = await manager.search(params.query, { maxResults: params.maxResults });
            if (results.length === 0) {
              return {
                content: [{ type: "text" as const, text: "No relevant memories found." }],
                details: {},
              };
            }
            const text = results
              .map(
                (r, i) =>
                  `${i + 1}. [${r.citation ?? r.path}] (score: ${r.score.toFixed(2)}) ${r.snippet}`,
              )
              .join("\n\n");
            return {
              content: [{ type: "text" as const, text }],
              details: { resultCount: results.length },
            };
          },
        };

        const getTool = {
          name: "memory_get",
          description: "Read a section of an indexed file by path and line range.",
          label: "Memory Get",
          parameters: GetParams,
          execute: async (
            _toolCallId: string,
            params: { path: string; from?: number; lines?: number },
          ) => {
            const s = await getState(cfg, api.logger);
            const manager = new MemorySparkManager({
              cfg,
              agentId,
              workspaceDir,
              backend: s.backend,
              embed: s.embed,
              reranker: s.reranker,
              queue: s.queue,
            });
            const result = await manager.readFile({
              relPath: params.path,
              from: params.from,
              lines: params.lines,
            });
            return {
              content: [{ type: "text" as const, text: result.text || "(empty)" }],
              details: {},
            };
          },
        };

        const storeTool = {
          name: "memory_store",
          description:
            "Explicitly store a piece of information in long-term memory. Use for facts, preferences, or decisions the user wants remembered.",
          label: "Memory Store",
          parameters: StoreParams,
          execute: async (_toolCallId: string, params: { text: string; category?: string }) => {
            const s = await getState(cfg, api.logger);
            const { looksLikePromptInjection } = await import("./src/security.js");
            if (looksLikePromptInjection(params.text)) {
              return {
                content: [
                  { type: "text" as const, text: "Refused: text contains suspicious patterns." },
                ],
                details: {},
              };
            }
            const vector = await s.queue.embedQuery(params.text);
            // Duplicate check
            const existing = await s.backend
              .vectorSearch(vector, {
                query: params.text,
                maxResults: 1,
                minScore: 0.92,
                agentId,
                source: "capture",
              })
              .catch(() => []);
            if (existing.length > 0 && existing[0]!.score >= 0.92) {
              return {
                content: [
                  { type: "text" as const, text: "Already stored (similar memory exists)." },
                ],
                details: {},
              };
            }
            const { tagEntities } = await import("./src/classify/ner.js");
            const entities = await tagEntities(params.text, cfg).catch(() => [] as string[]);
            const crypto = await import("node:crypto");
            const now = new Date();
            await s.backend.upsert([
              {
                id: crypto.randomUUID().slice(0, 16),
                path: `capture/${agentId}/${now.toISOString().slice(0, 10)}`,
                source: "capture",
                agent_id: agentId,
                start_line: 0,
                end_line: 0,
                text: params.text,
                vector,
                updated_at: now.toISOString(),
                category: params.category ?? "fact",
                entities: JSON.stringify(entities),
                confidence: 1.0,
              },
            ]);
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Stored in memory: "${params.text.slice(0, 80)}..."`,
                },
              ],
              details: {},
            };
          },
        };

        const forgetTool = {
          name: "memory_forget",
          description:
            "Remove memories matching a query. Use when the user wants to correct or delete stored information.",
          label: "Memory Forget",
          parameters: ForgetParams,
          execute: async (_toolCallId: string, params: { query: string }) => {
            const s = await getState(cfg, api.logger);
            const vector = await s.queue.embedQuery(params.query);
            const matches = await s.backend
              .vectorSearch(vector, {
                query: params.query,
                maxResults: 5,
                minScore: 0.7,
                agentId,
                source: "capture",
              })
              .catch(() => []);
            if (matches.length === 0) {
              return {
                content: [{ type: "text" as const, text: "No matching memories found to forget." }],
                details: {},
              };
            }
            const ids = matches.map((m) => m.chunk.id);
            await s.backend.deleteById(ids);
            const previews = matches.map((m) => `• "${m.chunk.text.slice(0, 60)}..."`).join("\n");
            return {
              content: [
                { type: "text" as const, text: `Forgot ${ids.length} memories:\n${previews}` },
              ],
              details: { deleted: ids.length },
            };
          },
        };

        const referenceSearchTool = {
          name: "memory_reference_search",
          description:
            "Search reference documentation (textbooks, API docs, source code docs). Use instead of web search when relevant reference material has been indexed.",
          label: "Reference Search",
          parameters: ReferenceSearchParams,
          execute: async (
            _toolCallId: string,
            params: { query: string; tag?: string; maxResults?: number },
          ) => {
            const s = await getState(cfg, api.logger);
            const limit = params.maxResults ?? 10;
            let queryVector: number[];
            try {
              queryVector = await s.queue.embedQuery(params.query);
            } catch {
              return {
                content: [
                  {
                    type: "text" as const,
                    text: "Embedding unavailable — cannot search reference docs.",
                  },
                ],
                details: {},
              };
            }

            const fetchN = limit * 3;
            // Use pool filtering (canonical) — searches both reference_library and reference_code
            const referencePools = ["reference_library", "reference_code"];
            const [vectorResults, ftsResults] = await Promise.all([
              s.backend
                .vectorSearch(queryVector, {
                  query: params.query,
                  maxResults: fetchN,
                  pools: referencePools,
                })
                .catch(() => []),
              s.backend
                .ftsSearch(params.query, {
                  query: params.query,
                  maxResults: fetchN,
                  pools: referencePools,
                })
                .catch(() => []),
            ]);

            // Merge and deduplicate by id
            const seen = new Set<string>();
            const merged = [...vectorResults, ...ftsResults]
              .filter((r) => {
                if (seen.has(r.chunk.id)) return false;
                seen.add(r.chunk.id);
                return true;
              })
              .sort((a, b) => b.score - a.score);

            // Tag filter — match path prefix against cfg.reference.tags
            let results = merged;
            if (params.tag) {
              const tagMap = cfg.reference.tags;
              const matchingPrefixes = Object.entries(tagMap)
                .filter(([, t]) => t === params.tag)
                .map(([prefix]) => prefix);
              if (matchingPrefixes.length > 0) {
                results = merged.filter((r) =>
                  matchingPrefixes.some((prefix) => r.chunk.path.startsWith(prefix)),
                );
              }
            }

            const final = results.slice(0, limit);
            if (final.length === 0) {
              return {
                content: [
                  {
                    type: "text" as const,
                    text: "No reference documentation found for that query.",
                  },
                ],
                details: {},
              };
            }
            const text = final
              .map(
                (r, i) =>
                  `${i + 1}. [${r.chunk.path}:${r.chunk.start_line}] (score: ${r.score.toFixed(2)})\n${r.chunk.text.slice(0, 400)}`,
              )
              .join("\n\n");
            return {
              content: [{ type: "text" as const, text }],
              details: { resultCount: final.length },
            };
          },
        };

        const indexStatusTool = {
          name: "memory_index_status",
          description:
            "Show memory index statistics: chunk counts by type, index health, and top indexed paths.",
          label: "Index Status",
          parameters: IndexStatusParams,
          execute: async (_toolCallId: string, params: { agentId?: string }) => {
            const s = await getState(cfg, api.logger);
            const targetAgentId = params.agentId ?? agentId;

            // Use LanceDBBackend.getStats() if available
            const lanceStats = (
              s.backend as import("./src/storage/lancedb.js").LanceDBBackend
            ).getStats?.(targetAgentId);
            const baseStatus = await s.backend.status();

            let statsText = `Memory Index Status\n`;
            statsText += `===================\n`;
            statsText += `Backend: ${baseStatus.backend}\n`;
            statsText += `Total chunks: ${baseStatus.chunkCount}\n`;
            statsText += `Ready: ${baseStatus.ready}\n`;

            if (lanceStats) {
              const stats = await lanceStats;
              if (stats.indices.length > 0) {
                statsText += `\nIndexes:\n`;
                for (const idx of stats.indices) {
                  statsText += `  - ${idx.name} (${idx.indexType}) on [${idx.columns.join(", ")}]\n`;
                }
              } else {
                statsText += `\nIndexes: none (ANN search will use brute force)\n`;
              }

              if (stats.topPaths.length > 0) {
                statsText += `\nTop paths by chunk count:\n`;
                for (const p of stats.topPaths.slice(0, 10)) {
                  statsText += `  ${p.chunkCount.toString().padStart(4)} chunks — ${p.path}\n`;
                }
              }
            }

            // ── Services Health ──
            statsText += `\nServices:\n`;
            statsText += `  Embed: ${s.embed.id}/${s.embed.model} (${s.embed.dims}d)\n`;

            // Spark embed probe
            try {
              const probeOk = await s.embed.probe();
              statsText += `  Embed health: ${probeOk ? "✅ healthy" : "❌ unreachable"}\n`;
            } catch {
              statsText += `  Embed health: ❌ probe failed\n`;
            }

            // Embed queue stats
            const queueStats = s.queue.stats;
            statsText += `  EmbedQueue: ${queueStats.queued} queued, ${queueStats.failed} failed, ${queueStats.healthy ? "✅ healthy" : "❌ unhealthy"}\n`;

            // Cache stats
            const cs = s.cachedEmbed.cacheStats();
            statsText += `  EmbedCache: ${cs.size}/${cs.maxSize} entries, hit rate ${cs.hitRate} (${cs.hits} hits, ${cs.misses} misses)\n`;

            // Reranker probe
            statsText += `  Reranker: ${cfg.rerank.enabled ? "enabled" : "off"}\n`;
            if (cfg.rerank.enabled) {
              try {
                const rerankOk = await s.reranker.probe();
                statsText += `  Reranker health: ${rerankOk ? "✅ healthy" : "❌ unreachable"}\n`;
              } catch {
                statsText += `  Reranker health: ❌ probe failed\n`;
              }
            }

            // ── Pool & Agent Breakdown ──
            if (lanceStats) {
              const stats = await lanceStats;
              if (stats.poolBreakdown && Object.keys(stats.poolBreakdown).length > 0) {
                statsText += `\nChunks by pool:\n`;
                for (const [pool, count] of Object.entries(stats.poolBreakdown).sort((a, b) => b[1] - a[1])) {
                  statsText += `  ${count.toString().padStart(6)} — ${pool}\n`;
                }
              }
              if (stats.agentBreakdown && Object.keys(stats.agentBreakdown).length > 0) {
                statsText += `\nChunks by agent:\n`;
                for (const [agent, count] of Object.entries(stats.agentBreakdown).sort((a, b) => b[1] - a[1])) {
                  statsText += `  ${count.toString().padStart(6)} — ${agent}\n`;
                }
              }
            }

            // ── Reranker Gate Config ──
            statsText += `\nReranker Gate:\n`;
            statsText += `  Mode: ${cfg.rerank.rerankerGate ?? "hard"}\n`;
            statsText += `  Blend: ${cfg.rerank.blendMode ?? "rrf"}\n`;
            statsText += `  Thresholds: [${cfg.rerank.rerankerGateLowThreshold ?? 0.02}, ${cfg.rerank.rerankerGateThreshold ?? 0.08}]\n`;

            // ── Config Summary ──
            statsText += `\nConfig:\n`;
            statsText += `  AutoRecall: ${cfg.autoRecall.enabled ? cfg.autoRecall.agents.join(",") : "off"}\n`;
            statsText += `  AutoCapture: ${cfg.autoCapture.enabled ? cfg.autoCapture.agents.join(",") : "off"}\n`;

            return { content: [{ type: "text" as const, text: statsText }], details: {} };
          },
        };

        const forgetByPathTool = {
          name: "memory_forget_by_path",
          description:
            "Remove all indexed chunks from a specific file path. Use when reference docs are outdated or a file has been deleted.",
          label: "Forget by Path",
          parameters: ForgetByPathParams,
          execute: async (_toolCallId: string, params: { path: string }) => {
            const s = await getState(cfg, api.logger);
            const removed = await s.backend.deleteByPath(params.path, agentId);
            if (removed === 0) {
              return {
                content: [
                  { type: "text" as const, text: `No chunks found for path: ${params.path}` },
                ],
                details: { removed: 0 },
              };
            }
            return {
              content: [
                { type: "text" as const, text: `Removed ${removed} chunks from: ${params.path}` },
              ],
              details: { removed },
            };
          },
        };

        const inspectTool = {
          name: "memory_inspect",
          description:
            "Simulate auto-recall for a query. Shows exactly what would be injected into context, with scores, sources, and weights applied. Use to debug recall quality or verify that important memories (MISTAKES, TOOLS) are being retrieved.",
          label: "Inspect Recall",
          parameters: InspectParams,
          execute: async (_toolCallId: string, params: { query: string; maxResults?: number }) => {
            const s = await getState(cfg, api.logger);
            const maxR = params.maxResults ?? 5;

            // Embed the query
            let vector: number[];
            try {
              vector = await s.cachedEmbed.embedQuery(params.query);
            } catch {
              return {
                content: [{ type: "text" as const, text: "Embedding failed — Spark may be down." }],
                details: {},
              };
            }

            // Vector search
            const vectorResults = await s.backend
              .vectorSearch(vector, {
                query: params.query,
                maxResults: maxR * 4,
                minScore: cfg.autoRecall.minScore,
                agentId,
              })
              .catch(() => []);

            // FTS search
            const ftsResults = await s.backend
              .ftsSearch(params.query, {
                query: params.query,
                maxResults: maxR * 2,
              })
              .catch(() => []);

            // Import pipeline functions
            const { applySourceWeighting, applyTemporalDecay } =
              await import("./src/auto/recall.js");

            // Merge
            const merged = [...vectorResults];
            const seenIds = new Set(vectorResults.map((r) => r.chunk.id));
            for (const r of ftsResults) {
              if (!seenIds.has(r.chunk.id)) {
                r.score *= 0.7;
                merged.push(r);
              }
            }

            // Apply weights
            applySourceWeighting(merged, cfg.autoRecall.weights);
            applyTemporalDecay(merged);
            merged.sort((a, b) => b.score - a.score);

            const top = merged.slice(0, maxR);

            let text = `Recall Inspection for: "${params.query}"\n`;
            text += `${"=".repeat(50)}\n`;
            text += `Vector results: ${vectorResults.length} | FTS results: ${ftsResults.length} | Merged: ${merged.length}\n\n`;

            for (let i = 0; i < top.length; i++) {
              const r = top[i]!;
              text += `[${i + 1}] Score: ${r.score.toFixed(3)} | Source: ${r.chunk.source} | Path: ${r.chunk.path}\n`;
              text += `    Agent: ${r.chunk.agent_id ?? "?"} | Updated: ${r.chunk.updated_at?.slice(0, 10) ?? "?"}\n`;
              text += `    Preview: ${r.chunk.text.slice(0, 200).replace(/\n/g, " ")}\n\n`;
            }

            if (top.length === 0) {
              text += "(No results found)\n";
            }

            return { content: [{ type: "text" as const, text }], details: { count: top.length } };
          },
        };

        const reindexTool = {
          name: "memory_reindex",
          description:
            "Trigger a re-index of memory files. With a path, re-indexes just that file. Without a path, triggers a full boot-pass re-scan of all workspace files.",
          label: "Re-index",
          parameters: ReindexParams,
          execute: async (_toolCallId: string, params: { path?: string }) => {
            const s = await getState(cfg, api.logger);
            if (params.path) {
              // Single file re-index
              try {
                const { ingestFile } = await import("./src/ingest/pipeline.js");
                // Delete existing chunks first to force a fresh re-index
                await s.backend.deleteByPath(params.path, agentId);
                const result = await ingestFile({
                  filePath: params.path,
                  backend: s.backend,
                  embed: s.queue,
                  cfg,
                  agentId,
                  workspaceDir,
                  source: "memory",
                });
                return {
                  content: [
                    {
                      type: "text" as const,
                      text: `Re-indexed: ${params.path} → ${result.chunksAdded} chunks`,
                    },
                  ],
                  details: { chunks: result.chunksAdded },
                };
              } catch (err) {
                return {
                  content: [
                    {
                      type: "text" as const,
                      text: `Failed to re-index ${params.path}: ${err instanceof Error ? err.message : String(err)}`,
                    },
                  ],
                  details: {},
                };
              }
            } else {
              // Full re-scan
              if (s.watcher) {
                s.watcher.triggerBootPass();
                return {
                  content: [
                    {
                      type: "text" as const,
                      text: "Full re-scan triggered. New and modified files will be indexed in the background.",
                    },
                  ],
                  details: {},
                };
              }
              return {
                content: [
                  {
                    type: "text" as const,
                    text: "Watcher not active — cannot trigger re-scan. Try restarting the plugin.",
                  },
                ],
                details: {},
              };
            }
          },
        };

        // ── Mistakes tools ──────────────────────────────────────────────────

        const mistakesSearchTool = {
          name: "memory_mistakes_search",
          description:
            "Search for relevant mistakes, errors, and lessons learned across all agents. Use when you need to check if a similar mistake has been made before, or to recall how a past error was fixed.",
          label: "Search Mistakes",
          parameters: MistakesSearchParams,
          execute: async (_toolCallId: string, params: { query: string; maxResults?: number }) => {
            const s = await getState(cfg, api.logger);
            const maxR = params.maxResults ?? 5;
            let vector: number[];
            try {
              vector = await s.cachedEmbed.embedQuery(params.query);
            } catch {
              return {
                content: [
                  {
                    type: "text" as const,
                    text: "Embedding unavailable — cannot search mistakes.",
                  },
                ],
                details: {},
              };
            }

            // Search both agent-specific and shared mistake pools
            const [agentMistakes, sharedMistakes] = await Promise.all([
              s.backend
                .vectorSearch(vector, {
                  query: params.query,
                  maxResults: maxR,
                  minScore: 0.1,
                  agentId,
                  pools: ["agent_mistakes"],
                })
                .catch(() => [] as import("./src/storage/backend.js").SearchResult[]),
              s.backend
                .vectorSearch(vector, {
                  query: params.query,
                  maxResults: maxR,
                  minScore: 0.1,
                  pools: ["shared_mistakes"],
                })
                .catch(() => [] as import("./src/storage/backend.js").SearchResult[]),
            ]);

            // Merge and deduplicate by ID, keep highest score
            const seen = new Set<string>();
            const results = [...agentMistakes, ...sharedMistakes]
              .sort((a, b) => b.score - a.score)
              .filter((r) => {
                if (seen.has(r.chunk.id)) return false;
                seen.add(r.chunk.id);
                return true;
              })
              .slice(0, maxR);

            if (results.length === 0) {
              return {
                content: [{ type: "text" as const, text: "No relevant mistakes found." }],
                details: {},
              };
            }
            const text = results
              .map(
                (r, i) =>
                  `${i + 1}. [${r.chunk.path}] (score: ${r.score.toFixed(2)})\n${r.chunk.text.slice(0, 400)}`,
              )
              .join("\n\n");
            return {
              content: [{ type: "text" as const, text }],
              details: { resultCount: results.length },
            };
          },
        };

        const mistakesStoreTool = {
          name: "memory_mistakes_store",
          description:
            "Log a mistake, error, or incident for future reference. Stored in the shared mistakes pool visible to all agents. Use when something goes wrong and you want to ensure it's remembered and not repeated.",
          label: "Log Mistake",
          parameters: MistakesStoreParams,
          execute: async (
            _toolCallId: string,
            params: {
              description: string;
              rootCause?: string;
              fix?: string;
              lessons?: string;
              severity?: string;
              shared?: boolean;
            },
          ) => {
            const s = await getState(cfg, api.logger);
            const { looksLikePromptInjection } = await import("./src/security.js");
            const fullText = [
              `Mistake: ${params.description}`,
              params.rootCause ? `Root Cause: ${params.rootCause}` : "",
              params.fix ? `Fix: ${params.fix}` : "",
              params.lessons ? `Lessons: ${params.lessons}` : "",
              `Severity: ${params.severity ?? "medium"}`,
              `Agent: ${agentId}`,
              `Date: ${new Date().toISOString().slice(0, 10)}`,
            ]
              .filter(Boolean)
              .join("\n");

            if (looksLikePromptInjection(fullText)) {
              return {
                content: [
                  { type: "text" as const, text: "Refused: text contains suspicious patterns." },
                ],
                details: {},
              };
            }

            const vector = await s.queue.embedQuery(fullText);
            const crypto = await import("node:crypto");
            const now = new Date();
            const pool = params.shared ? "shared_mistakes" : "agent_mistakes";
            const chunk: import("./src/storage/backend.js").MemoryChunk = {
              id: crypto.randomUUID().slice(0, 16),
              path: `mistakes/${agentId}/${now.toISOString().slice(0, 10)}`,
              source: "capture",
              agent_id: agentId,
              start_line: 0,
              end_line: 0,
              text: fullText,
              vector,
              updated_at: now.toISOString(),
              category: "mistake",
              content_type: "mistake",
              entities: "[]",
              confidence: 1.0,
              pool,
            };

            await s.backend.upsert([chunk]);
            const shareLabel = params.shared ? "shared with all agents" : "per-agent";
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Logged mistake (${shareLabel}, ${params.severity ?? "medium"}): "${params.description.slice(0, 80)}..."`,
                },
              ],
              details: { severity: params.severity ?? "medium", pool },
            };
          },
        };

        // ── Rules tools ────────────────────────────────────────────────────

        const rulesStoreTool = {
          name: "memory_rules_store",
          description:
            "Store a global rule, preference, or guideline that should be recalled when relevant. Rules are shared across all agents by default. Use for persistent behavioral guidelines like 'Never use Flash for coding' or 'Klein prefers concise responses'.",
          label: "Store Rule",
          parameters: RulesStoreParams,
          execute: async (
            _toolCallId: string,
            params: { rule: string; scope?: string; category?: string },
          ) => {
            const s = await getState(cfg, api.logger);
            const { looksLikePromptInjection } = await import("./src/security.js");
            const category = params.category ?? "preference";
            const scope = params.scope ?? "global";

            const fullText = `[${category.toUpperCase()}] ${params.rule}`;
            if (looksLikePromptInjection(fullText)) {
              return {
                content: [
                  { type: "text" as const, text: "Refused: text contains suspicious patterns." },
                ],
                details: {},
              };
            }

            const vector = await s.queue.embedQuery(fullText);
            const crypto = await import("node:crypto");
            const now = new Date();
            const chunk: import("./src/storage/backend.js").MemoryChunk = {
              id: crypto.randomUUID().slice(0, 16),
              path: `rules/${scope}/${category}`,
              source: "capture",
              agent_id: scope === "global" ? "shared" : scope,
              start_line: 0,
              end_line: 0,
              text: fullText,
              vector,
              updated_at: now.toISOString(),
              category,
              content_type: "rule",
              entities: "[]",
              confidence: 1.0,
              pool: "shared_rules",
            };

            await s.backend.upsert([chunk]);
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Stored rule (${category}, ${scope}): "${params.rule.slice(0, 80)}..."`,
                },
              ],
              details: { scope, category },
            };
          },
        };

        const rulesSearchTool = {
          name: "memory_rules_search",
          description:
            "Search for stored rules, preferences, and guidelines. Use to check existing rules before adding new ones, or to find relevant guidelines for a task.",
          label: "Search Rules",
          parameters: RulesSearchParams,
          execute: async (_toolCallId: string, params: { query: string; maxResults?: number }) => {
            const s = await getState(cfg, api.logger);
            const maxR = params.maxResults ?? 5;
            let vector: number[];
            try {
              vector = await s.cachedEmbed.embedQuery(params.query);
            } catch {
              return {
                content: [{ type: "text" as const, text: "Embedding unavailable." }],
                details: {},
              };
            }
            const results = await s.backend
              .vectorSearch(vector, {
                query: params.query,
                maxResults: maxR,
                minScore: 0.1,
                pool: "shared_rules",
              })
              .catch(() => [] as import("./src/storage/backend.js").SearchResult[]);
            if (results.length === 0) {
              return {
                content: [{ type: "text" as const, text: "No matching rules found." }],
                details: {},
              };
            }
            const text = results
              .map((r, i) => `${i + 1}. (score: ${r.score.toFixed(2)}) ${r.chunk.text}`)
              .join("\n\n");
            return {
              content: [{ type: "text" as const, text }],
              details: { resultCount: results.length },
            };
          },
        };

        // ── Phase C: New tools (v0.4.0) ──────────────────────────────────

        const recallDebugTool = {
          name: "memory_recall_debug",
          description:
            "Show the full recall pipeline trace for a query — what the agent sees behind the scenes. " +
            "Shows vector scores, hybrid merge, reranker gate decision, MMR diversity, and final token budget. " +
            "Use to understand why certain memories were or weren't recalled.",
          label: "Recall Debug",
          parameters: Type.Object({
            query: Type.String({ description: "Query to trace through the recall pipeline" }),
            maxResults: Type.Optional(
              Type.Number({ description: "Max results to trace", minimum: 1, maximum: 50 }),
            ),
          }),
          execute: async (
            _toolCallId: string,
            params: { query: string; maxResults?: number },
          ) => {
            const s = await getState(cfg, api.logger);
            const maxR = params.maxResults ?? 10;

            // Run vector search
            const queryVector = await s.embed.embedQuery(params.query);
            const vectorResults = await s.backend
              .vectorSearch(queryVector, {
                query: params.query,
                maxResults: maxR * 4,
                minScore: 0.05,
                agentId,
              })
              .catch(() => [] as import("./src/storage/backend.js").SearchResult[]);

            // Run FTS search
            const ftsResults = await s.backend
              .ftsSearch(params.query, {
                query: params.query,
                maxResults: maxR * 4,
                minScore: 0.05,
                agentId,
              })
              .catch(() => [] as import("./src/storage/backend.js").SearchResult[]);

            // Hybrid merge
            const { hybridMerge } = await import("./src/auto/recall.js");
            const merged = hybridMerge(vectorResults, ftsResults, maxR * 2);

            // Reranker (includes gate)
            const reranked = await s.reranker.rerank(params.query, merged, maxR);

            const trace = {
              query: params.query,
              stages: {
                vector: {
                  count: vectorResults.length,
                  topScores: vectorResults.slice(0, 5).map((r) => ({
                    score: Number(r.score.toFixed(4)),
                    path: r.chunk.path,
                    pool: r.chunk.pool,
                  })),
                },
                fts: {
                  count: ftsResults.length,
                  topScores: ftsResults.slice(0, 5).map((r) => ({
                    score: Number(r.score.toFixed(4)),
                    path: r.chunk.path,
                  })),
                },
                hybrid: {
                  count: merged.length,
                  topScores: merged.slice(0, 5).map((r) => ({
                    score: Number(r.score.toFixed(4)),
                    path: r.chunk.path,
                  })),
                },
                reranked: {
                  count: reranked.length,
                  topScores: reranked.slice(0, 5).map((r) => ({
                    score: Number(r.score.toFixed(4)),
                    path: r.chunk.path,
                    text: r.chunk.text.slice(0, 100),
                  })),
                },
              },
              config: {
                gateMode: cfg.rerank.rerankerGate ?? "hard",
                gateThreshold: cfg.rerank.rerankerGateThreshold ?? 0.08,
                gateLowThreshold: cfg.rerank.rerankerGateLowThreshold ?? 0.02,
                blendMode: cfg.rerank.blendMode ?? "rrf",
              },
            };

            return {
              content: [
                {
                  type: "text" as const,
                  text: JSON.stringify(trace, null, 2),
                },
              ],
              details: { stages: 4 },
            };
          },
        };

        const bulkIngestTool = {
          name: "memory_bulk_ingest",
          description:
            "Ingest multiple documents or notes into memory in a single batch. " +
            "More efficient than calling memory_store repeatedly. " +
            "Each item needs text content; path, source, and tags are optional.",
          label: "Bulk Ingest",
          parameters: Type.Object({
            items: Type.Array(
              Type.Object({
                text: Type.String({ description: "Content to store" }),
                path: Type.Optional(
                  Type.String({ description: "Virtual path for the memory" }),
                ),
                source: Type.Optional(
                  Type.String({ description: "Source identifier (default: capture)" }),
                ),
                tags: Type.Optional(
                  Type.Array(Type.String(), { description: "Tags for categorization" }),
                ),
              }),
              { minItems: 1, maxItems: 100 },
            ),
          }),
          execute: async (
            _toolCallId: string,
            params: {
              items: Array<{
                text: string;
                path?: string;
                source?: string;
                tags?: string[];
              }>;
            },
          ) => {
            const s = await getState(cfg, api.logger);
            const results: Array<{ index: number; status: string; id?: string; error?: string }> =
              [];

            for (let i = 0; i < params.items.length; i++) {
              const item = params.items[i]!;
              try {
                const vector = await s.queue.embedDocument(item.text);
                const chunk = {
                  id: `capture-${Date.now()}-${i}-${Math.random().toString(36).slice(2, 8)}`,
                  text: item.text,
                  path: item.path ?? `capture/bulk-${Date.now()}`,
                  source: (item.source ?? "capture") as "memory" | "ingest" | "capture" | "sessions",
                  agent_id: agentId,
                  pool: "agent_memory",
                  updated_at: new Date().toISOString(),
                  content_type: "knowledge" as const,
                  tags: item.tags,
                  start_line: 0,
                  end_line: 0,
                };
                // Attach the vector to the chunk for LanceDB storage
                const chunkWithVector = { ...chunk, vector };
                await s.backend.upsert([chunkWithVector]);
                results.push({ index: i, status: "ok", id: chunk.id });
              } catch (err) {
                results.push({
                  index: i,
                  status: "error",
                  error: err instanceof Error ? err.message : String(err),
                });
              }
            }

            const ok = results.filter((r) => r.status === "ok").length;
            const failed = results.filter((r) => r.status === "error").length;
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Bulk ingest complete: ${ok} stored, ${failed} failed.\n${
                    failed > 0
                      ? results
                          .filter((r) => r.status === "error")
                          .map((r) => `  Item ${r.index}: ${r.error}`)
                          .join("\n")
                      : ""
                  }`,
                },
              ],
              details: { stored: ok, failed },
            };
          },
        };

        const temporalTool = {
          name: "memory_temporal",
          description:
            "Search memories within a specific time window. " +
            'Use when you need to find what was learned recently or during a specific period. Example: "what did I learn last week?"',
          label: "Temporal Search",
          parameters: Type.Object({
            query: Type.String({ description: "Search query" }),
            after: Type.Optional(
              Type.String({
                description: "ISO date — only return memories after this date",
              }),
            ),
            before: Type.Optional(
              Type.String({
                description: "ISO date — only return memories before this date",
              }),
            ),
            maxResults: Type.Optional(
              Type.Number({ description: "Max results", minimum: 1, maximum: 50 }),
            ),
          }),
          execute: async (
            _toolCallId: string,
            params: { query: string; after?: string; before?: string; maxResults?: number },
          ) => {
            const s = await getState(cfg, api.logger);
            const maxR = params.maxResults ?? 10;
            const queryVector = await s.embed.embedQuery(params.query);

            // Vector search with time filter
            const results = await s.backend
              .vectorSearch(queryVector, {
                query: params.query,
                maxResults: maxR * 2,
                minScore: 0.1,
                agentId,
              })
              .catch(() => [] as import("./src/storage/backend.js").SearchResult[]);

            // Apply temporal filter
            const filtered = results.filter((r) => {
              const ts = r.chunk.updated_at ? new Date(r.chunk.updated_at).getTime() : 0;
              if (params.after && ts < new Date(params.after).getTime()) return false;
              if (params.before && ts > new Date(params.before).getTime()) return false;
              return true;
            });

            if (filtered.length === 0) {
              return {
                content: [
                  {
                    type: "text" as const,
                    text: `No memories found${params.after ? ` after ${params.after}` : ""}${params.before ? ` before ${params.before}` : ""}.`,
                  },
                ],
                details: {},
              };
            }

            const text = filtered
              .slice(0, maxR)
              .map(
                (r, i) =>
                  `${i + 1}. [${r.chunk.path}] (score: ${r.score.toFixed(2)}, date: ${r.chunk.updated_at ?? "unknown"}) ${r.chunk.text.slice(0, 200)}`,
              )
              .join("\n\n");

            return {
              content: [{ type: "text" as const, text }],
              details: { resultCount: filtered.length },
            };
          },
        };

        const relatedTool = {
          name: "memory_related",
          description:
            "Given a memory chunk ID, find semantically similar memories (neighborhood search). " +
            'Use to explore: "what else do I know about this topic?"',
          label: "Related Memories",
          parameters: Type.Object({
            chunkId: Type.String({ description: "ID of the memory chunk to find neighbors for" }),
            maxResults: Type.Optional(
              Type.Number({ description: "Max related results", minimum: 1, maximum: 20 }),
            ),
          }),
          execute: async (
            _toolCallId: string,
            params: { chunkId: string; maxResults?: number },
          ) => {
            const s = await getState(cfg, api.logger);
            const maxR = params.maxResults ?? 5;

            // Get the source chunk
            const chunks = await s.backend.getByIds([params.chunkId]);
            if (chunks.length === 0) {
              return {
                content: [
                  { type: "text" as const, text: `Chunk ${params.chunkId} not found.` },
                ],
                details: {},
              };
            }

            const sourceChunk = chunks[0]!;
            // Embed the source chunk's text and search for neighbors
            const vector = await s.embed.embedDocument(sourceChunk.text);
            const neighbors = await s.backend
              .vectorSearch(vector, {
                query: sourceChunk.text.slice(0, 200),
                maxResults: maxR + 1, // +1 to exclude self
                minScore: 0.3,
              })
              .catch(() => [] as import("./src/storage/backend.js").SearchResult[]);

            // Filter out the source chunk itself
            const related = neighbors.filter((r) => r.chunk.id !== params.chunkId).slice(0, maxR);

            if (related.length === 0) {
              return {
                content: [
                  { type: "text" as const, text: "No related memories found." },
                ],
                details: {},
              };
            }

            const text = related
              .map(
                (r, i) =>
                  `${i + 1}. [${r.chunk.path}] (similarity: ${r.score.toFixed(2)}) ${r.chunk.text.slice(0, 200)}`,
              )
              .join("\n\n");

            return {
              content: [{ type: "text" as const, text }],
              details: {
                sourceChunk: sourceChunk.path,
                relatedCount: related.length,
              },
            };
          },
        };

        const gateStatusTool = {
          name: "memory_gate_status",
          description:
            "Show the current reranker gate configuration and mode. " +
            "The gate controls when the cross-encoder reranker fires vs when vector-only results are trusted.",
          label: "Gate Status",
          parameters: Type.Object({}),
          execute: async () => {
            const gateMode = cfg.rerank.rerankerGate ?? "hard";
            const threshold = cfg.rerank.rerankerGateThreshold ?? 0.08;
            const lowThreshold = cfg.rerank.rerankerGateLowThreshold ?? 0.02;
            const blendMode = cfg.rerank.blendMode ?? "rrf";
            const rrfK = cfg.rerank.rrfK ?? 60;

            const explanation =
              gateMode === "hard"
                ? `Hard gate: reranker is SKIPPED when top-5 vector spread > ${threshold} (vector is confident) ` +
                  `or < ${lowThreshold} (tied set — reranker would be gambling). ` +
                  `Only fires in the productive [${lowThreshold}, ${threshold}] range.`
                : gateMode === "soft"
                  ? `Soft gate: vector weight is dynamically scaled based on score spread. ` +
                    `High spread → trust vector. Low spread → let reranker help.`
                  : "Gate disabled — reranker fires on every query.";

            const text = [
              `**Reranker Gate Configuration**`,
              `Mode: ${gateMode}`,
              `Threshold (high): ${threshold}`,
              `Threshold (low): ${lowThreshold}`,
              `Blend mode: ${blendMode}`,
              `RRF k: ${rrfK}`,
              ``,
              explanation,
            ].join("\n");

            return {
              content: [{ type: "text" as const, text }],
              details: { gateMode, blendMode },
            };
          },
        };

        // SDK boundary cast — OpenClaw's AnyAgentTool uses `any` internally
        return [
          searchTool,
          getTool,
          storeTool,
          forgetTool,
          referenceSearchTool,
          indexStatusTool,
          forgetByPathTool,
          inspectTool,
          reindexTool,
          mistakesSearchTool,
          mistakesStoreTool,
          rulesStoreTool,
          rulesSearchTool,
          recallDebugTool,
          bulkIngestTool,
          temporalTool,
          relatedTool,
          gateStatusTool,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
        ] as any;
      },
      {
        names: [
          "memory_search",
          "memory_get",
          "memory_store",
          "memory_forget",
          "memory_reference_search",
          "memory_index_status",
          "memory_forget_by_path",
          "memory_inspect",
          "memory_reindex",
          "memory_mistakes_search",
          "memory_mistakes_store",
          "memory_rules_store",
          "memory_rules_search",
          "memory_recall_debug",
          "memory_bulk_ingest",
          "memory_temporal",
          "memory_related",
          "memory_gate_status",
        ],
      },
    );

    // -------------------------------------------------------------------
    // 2. Auto-recall (before_prompt_build)
    // -------------------------------------------------------------------
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- OpenClaw plugin SDK hooks are untyped
    api.on("before_prompt_build", async (event: any, ctx: any) => {
      try {
        const s = await getState(cfg, api.logger);
        const handler = createAutoRecallHandler({
          cfg: cfg.autoRecall,
          backend: s.backend,
          embed: s.cachedEmbed,
          reranker: s.reranker,
          hyde: cfg.hyde,
        });
        return await handler(event, ctx);
      } catch {
        return undefined;
      }
    });

    // -------------------------------------------------------------------
    // 3. Auto-capture (agent_end)
    // -------------------------------------------------------------------
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- OpenClaw plugin SDK hooks are untyped
    api.on("agent_end", async (event: any, ctx: any) => {
      try {
        const s = await getState(cfg, api.logger);
        const handler = createAutoCaptureHandler({
          cfg: cfg.autoCapture,
          globalCfg: cfg,
          backend: s.backend,
          embed: s.queue,
        });
        await handler(event, ctx);
      } catch {
        // Non-fatal
      }
    });

    // -------------------------------------------------------------------
    // 4. After compaction — re-index compacted session
    // -------------------------------------------------------------------
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- OpenClaw plugin SDK hooks are untyped
    api.on("after_compaction", async (event: any, ctx: any) => {
      try {
        if (!event.sessionFile) return;
        const s = await getState(cfg, api.logger);
        const agentId = ctx.agentId ?? "default";
        const workspaceDir =
          ctx.workspaceDir ?? path.join(os.homedir(), ".openclaw", `workspace-${agentId}`);
        const { ingestFile } = await import("./src/ingest/pipeline.js");
        await ingestFile({
          filePath: event.sessionFile,
          agentId,
          workspaceDir,
          backend: s.backend,
          embed: s.queue,
          cfg,
          source: "sessions",
        });
      } catch {
        // Non-fatal
      }
    });

    // -------------------------------------------------------------------
    // 5. File watcher service (workspace memory + sessions + external)
    // -------------------------------------------------------------------
    api.registerService({
      id: "memory-spark-watcher",
      async start(svcCtx) {
        try {
          const s = await getState(cfg, api.logger);
          s.watcher = createWatcher({
            watch: cfg.watch,
            cfg,
            backend: s.backend,
            embed: s.queue,
            queue: s.queue,
            logger: svcCtx.logger,
          });
          await s.watcher.start();
        } catch (err) {
          svcCtx.logger.error(`memory-spark watcher start failed: ${err}`);
        }
      },
      async stop() {
        if (state?.watcher) {
          await state.watcher.stop();
          state.watcher = null;
        }
        if (state?.backend) {
          await state.backend.close();
        }
        state = null;
        initPromise = null;
      },
    });

    // -------------------------------------------------------------------
    // 6. Auto-migration on first gateway start
    // -------------------------------------------------------------------
    api.on("gateway_start", async () => {
      if (!cfg.migrate.autoMigrateOnFirstBoot) return;
      try {
        const statusFile = cfg.migrate.statusFile;
        try {
          const existing = JSON.parse(await fs.readFile(statusFile, "utf-8"));
          if (existing.completedAt) return; // Already migrated
        } catch {
          // No status file — first boot
        }

        api.logger.info("memory-spark: first boot — migration delegated to boot pass");
        import("./tools/migrate.js")
          .then((m) => m.runMigration?.())
          .catch((err) => {
            api.logger.warn(`memory-spark: migration import failed: ${err}`);
          });
      } catch (err) {
        api.logger.warn(`memory-spark: migration check failed: ${err}`);
      }
    });

    // -------------------------------------------------------------------
    // 7. CLI
    // -------------------------------------------------------------------
    api.registerCli(
      ({ program }) => {
        const memoryCmd = program
          .command("memory")
          .description("memory-spark: manage the autonomous memory system");

        memoryCmd
          .command("status")
          .description("Show memory system status")
          .action(async () => {
            try {
              const s = await getState(cfg, api.logger);
              const st = await s.backend.status();
              console.log(`Backend:     ${st.backend}`);
              console.log(`Chunks:      ${st.chunkCount}`);
              console.log(`Ready:       ${st.ready}`);
              console.log(`Embed:       ${s.embed.id}/${s.embed.model} (${s.embed.dims}d)`);
              console.log(`Reranker:    ${cfg.rerank.enabled ? "spark" : "off"}`);
              console.log(
                `AutoRecall:  ${cfg.autoRecall.enabled ? cfg.autoRecall.agents.join(",") : "off"}`,
              );
              console.log(
                `AutoCapture: ${cfg.autoCapture.enabled ? cfg.autoCapture.agents.join(",") : "off"}`,
              );
              console.log(`Watch:       ${cfg.watch.paths.length} external paths`);
              console.log(`LanceDB dir: ${cfg.lancedbDir}`);
            } catch (err) {
              console.error(`Status error: ${err}`);
            }
          });

        memoryCmd
          .command("sync")
          .description("Force re-index all workspace memory files and sessions")
          .action(async () => {
            try {
              const s = await getState(cfg, api.logger);
              if (s.watcher) {
                await s.watcher.stop();
              }
              s.watcher = createWatcher({
                watch: { ...cfg.watch, indexOnBoot: true },
                cfg,
                backend: s.backend,
                embed: s.queue,
                queue: s.queue,
                logger: { info: console.log, warn: console.warn, error: console.error },
              });
              await s.watcher.start();
              console.log("Sync triggered — boot pass running in background.");
            } catch (err) {
              console.error(`Sync error: ${err}`);
            }
          });

        memoryCmd
          .command("migrate")
          .description("Run migration from memory-core to LanceDB")
          .action(async () => {
            try {
              await import("./tools/migrate.js");
            } catch (err) {
              console.error(`Migration error: ${err}`);
            }
          });
      },
      { commands: ["memory"] },
    );

    api.logger.info(
      "memory-spark: plugin registered (workspace auto-discovery + sessions + auto-recall/capture)",
    );
  },
};

export default memorySpark;
