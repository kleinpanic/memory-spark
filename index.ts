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

import { Type } from "@sinclair/typebox";
import type { OpenClawPluginApi, OpenClawPluginConfigSchema } from "openclaw/plugin-sdk";
import os from "node:os";
import path from "node:path";

import { resolveConfig, type MemorySparkConfig } from "./src/config.js";
import { LanceDBBackend } from "./src/storage/lancedb.js";
import { createEmbedProvider, type EmbedProvider } from "./src/embed/provider.js";
import { EmbedQueue } from "./src/embed/queue.js";
import { validateDimsLock } from "./src/embed/dims-lock.js";
import { withCache, type CachedEmbedProvider } from "./src/embed/cached-provider.js";
import { createReranker, type Reranker } from "./src/rerank/reranker.js";
import { MemorySparkManager } from "./src/manager.js";
import { createWatcher, type Watcher } from "./src/ingest/watcher.js";
import { createAutoRecallHandler } from "./src/auto/recall.js";
import { createAutoCaptureHandler } from "./src/auto/capture.js";
import type { StorageBackend } from "./src/storage/backend.js";
import fs from "node:fs/promises";

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

    const backend = new LanceDBBackend(cfg);
    await backend.open();

    const embed = await createEmbedProvider(cfg.embed);
    logger.info(`memory-spark: embed → ${embed.id}/${embed.model} (${embed.dims}d)`);

    // Dimension safety: lock dims on first boot, refuse mismatch on subsequent boots
    const dimsCheck = await validateDimsLock(cfg.lancedbDir, embed.id, embed.model, embed.dims);
    if (!dimsCheck.ok) {
      logger.error(`memory-spark: FATAL — ${dimsCheck.error}`);
      throw new Error(dimsCheck.error);
    }

    // Queue: serialized embed requests with retry/backoff
    const queue = new EmbedQueue(embed, {
      concurrency: 1,
      maxRetries: 3,
      baseDelayMs: 2000,
      maxDelayMs: 30000,
      timeoutMs: 30000,
      unhealthyThreshold: 5,
      unhealthyCooldownMs: 60000,
    }, logger);

    const reranker = await createReranker(cfg.rerank);

    // Cache wraps the queue for query embeddings only (recall).
    // Document embeddings (indexing) bypass the cache via queue directly.
    const cachedEmbed = withCache(queue, {
      enabled: true,
      maxSize: 256,
      ttlMs: 30 * 60 * 1000, // 30 minutes
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
  category: Type.Optional(Type.String({ description: "Category: fact, preference, decision, code-snippet" })),
});

const ForgetParams = Type.Object({
  query: Type.String({ description: "What to forget — matches and removes similar memories" }),
});

const ReferenceSearchParams = Type.Object({
  query: Type.String({ description: "What to search for in reference documentation" }),
  tag: Type.Optional(Type.String({ description: "Filter by tag (e.g. 'internal', 'openclaw')" })),
  maxResults: Type.Optional(Type.Number({ description: "Max results (default 10)" })),
});

const IndexStatusParams = Type.Object({
  agentId: Type.Optional(Type.String({ description: "Agent ID to scope stats to (default: current agent)" })),
});

const ForgetByPathParams = Type.Object({
  path: Type.String({ description: "Relative file path whose indexed chunks should be removed" }),
});

// ---------------------------------------------------------------------------
// Plugin definition
// ---------------------------------------------------------------------------
const memorySpark = {
  id: "memory-spark",
  name: "Memory Spark",
  version: "0.1.0",
  description: "Autonomous Spark-powered memory: LanceDB + local embed/rerank + auto-recall/capture.",
  kind: "memory" as const,
  configSchema: {
    safeParse(value: unknown) {
      if (value === undefined || value === null) return { success: true, data: undefined };
      if (typeof value !== "object" || Array.isArray(value)) {
        return { success: false, error: { issues: [{ path: [], message: "expected config object" }] } };
      }
      // Passthrough — resolveConfig() handles deep merging and type coercion
      return { success: true, data: value };
    },
    jsonSchema: {
      type: "object",
      additionalProperties: true,
      properties: {
        sparkHost: { type: "string", description: "Spark node IP/hostname (overrides SPARK_HOST env)" },
        sparkBearerToken: { type: "string", description: "Spark bearer token (overrides SPARK_BEARER_TOKEN env)" },
        backend: { type: "string", enum: ["lancedb", "sqlite-vec"] },
        lancedbDir: { type: "string", description: "Path to LanceDB data directory" },
        autoRecall: {
          type: "object",
          properties: {
            enabled: { type: "boolean" },
            agents: { type: "array", items: { type: "string" }, description: "Agent IDs or [\"*\"] for all" },
            maxResults: { type: "number" },
            minScore: { type: "number" },
            queryMessageCount: { type: "number" },
          },
        },
        autoCapture: {
          type: "object",
          properties: {
            enabled: { type: "boolean" },
            agents: { type: "array", items: { type: "string" }, description: "Agent IDs or [\"*\"] for all" },
            categories: { type: "array", items: { type: "string" } },
            minConfidence: { type: "number" },
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
        watch: {
          type: "object",
          properties: {
            enabled: { type: "boolean" },
            indexOnBoot: { type: "boolean" },
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
        const workspaceDir = ctx.workspaceDir ?? path.join(os.homedir(), ".openclaw", `workspace-${agentId}`);

        const searchTool = {
          name: "memory_search",
          description: "Search the knowledge base and memory for relevant information.",
          label: "Memory Search",
          parameters: SearchParams,
          execute: async (_toolCallId: string, params: { query: string; maxResults?: number }) => {
            const s = await getState(cfg, api.logger);
            const manager = new MemorySparkManager({
              cfg, agentId, workspaceDir,
              backend: s.backend, embed: s.embed, reranker: s.reranker, queue: s.queue,
            });
            const results = await manager.search(params.query, { maxResults: params.maxResults });
            if (results.length === 0) {
              return { content: [{ type: "text" as const, text: "No relevant memories found." }], details: {} };
            }
            const text = results.map((r, i) =>
              `${i + 1}. [${r.citation ?? r.path}] (score: ${r.score.toFixed(2)}) ${r.snippet}`
            ).join("\n\n");
            return { content: [{ type: "text" as const, text }], details: { resultCount: results.length } };
          },
        };

        const getTool = {
          name: "memory_get",
          description: "Read a section of an indexed file by path and line range.",
          label: "Memory Get",
          parameters: GetParams,
          execute: async (_toolCallId: string, params: { path: string; from?: number; lines?: number }) => {
            const s = await getState(cfg, api.logger);
            const manager = new MemorySparkManager({
              cfg, agentId, workspaceDir,
              backend: s.backend, embed: s.embed, reranker: s.reranker, queue: s.queue,
            });
            const result = await manager.readFile({ relPath: params.path, from: params.from, lines: params.lines });
            return { content: [{ type: "text" as const, text: result.text || "(empty)" }], details: {} };
          },
        };

        const storeTool = {
          name: "memory_store",
          description: "Explicitly store a piece of information in long-term memory. Use for facts, preferences, or decisions the user wants remembered.",
          label: "Memory Store",
          parameters: StoreParams,
          execute: async (_toolCallId: string, params: { text: string; category?: string }) => {
            const s = await getState(cfg, api.logger);
            const { looksLikePromptInjection } = await import("./src/security.js");
            if (looksLikePromptInjection(params.text)) {
              return { content: [{ type: "text" as const, text: "Refused: text contains suspicious patterns." }], details: {} };
            }
            const vector = await s.queue.embedQuery(params.text);
            // Duplicate check
            const existing = await s.backend.vectorSearch(vector, {
              query: params.text, maxResults: 1, minScore: 0.92, agentId, source: "capture",
            }).catch(() => []);
            if (existing.length > 0 && existing[0]!.score >= 0.92) {
              return { content: [{ type: "text" as const, text: "Already stored (similar memory exists)." }], details: {} };
            }
            const { tagEntities } = await import("./src/classify/ner.js");
            const entities = await tagEntities(params.text, cfg).catch(() => [] as string[]);
            const crypto = await import("node:crypto");
            const now = new Date();
            await s.backend.upsert([{
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
            }]);
            return { content: [{ type: "text" as const, text: `Stored in memory: "${params.text.slice(0, 80)}..."` }], details: {} };
          },
        };

        const forgetTool = {
          name: "memory_forget",
          description: "Remove memories matching a query. Use when the user wants to correct or delete stored information.",
          label: "Memory Forget",
          parameters: ForgetParams,
          execute: async (_toolCallId: string, params: { query: string }) => {
            const s = await getState(cfg, api.logger);
            const vector = await s.queue.embedQuery(params.query);
            const matches = await s.backend.vectorSearch(vector, {
              query: params.query, maxResults: 5, minScore: 0.7, agentId, source: "capture",
            }).catch(() => []);
            if (matches.length === 0) {
              return { content: [{ type: "text" as const, text: "No matching memories found to forget." }], details: {} };
            }
            const ids = matches.map((m) => m.chunk.id);
            await s.backend.deleteById(ids);
            const previews = matches.map((m) => `• "${m.chunk.text.slice(0, 60)}..."`).join("\n");
            return { content: [{ type: "text" as const, text: `Forgot ${ids.length} memories:\n${previews}` }], details: { deleted: ids.length } };
          },
        };

        const referenceSearchTool = {
          name: "memory_reference_search",
          description: "Search reference documentation (textbooks, API docs, source code docs). Use instead of web search when relevant reference material has been indexed.",
          label: "Reference Search",
          parameters: ReferenceSearchParams,
          execute: async (_toolCallId: string, params: { query: string; tag?: string; maxResults?: number }) => {
            const s = await getState(cfg, api.logger);
            const limit = params.maxResults ?? 10;
            let queryVector: number[];
            try {
              queryVector = await s.queue.embedQuery(params.query);
            } catch {
              return { content: [{ type: "text" as const, text: "Embedding unavailable — cannot search reference docs." }], details: {} };
            }

            const fetchN = limit * 3;
            const [vectorResults, ftsResults] = await Promise.all([
              s.backend.vectorSearch(queryVector, {
                query: params.query, maxResults: fetchN, agentId, contentType: "reference",
              }).catch(() => []),
              s.backend.ftsSearch(params.query, {
                query: params.query, maxResults: fetchN, agentId, contentType: "reference",
              }).catch(() => []),
            ]);

            // Merge and deduplicate by id
            const seen = new Set<string>();
            const merged = [...vectorResults, ...ftsResults]
              .filter((r) => { if (seen.has(r.chunk.id)) return false; seen.add(r.chunk.id); return true; })
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
                  matchingPrefixes.some((prefix) => r.chunk.path.startsWith(prefix))
                );
              }
            }

            const final = results.slice(0, limit);
            if (final.length === 0) {
              return { content: [{ type: "text" as const, text: "No reference documentation found for that query." }], details: {} };
            }
            const text = final.map((r, i) =>
              `${i + 1}. [${r.chunk.path}:${r.chunk.start_line}] (score: ${r.score.toFixed(2)})\n${r.chunk.text.slice(0, 400)}`
            ).join("\n\n");
            return { content: [{ type: "text" as const, text }], details: { resultCount: final.length } };
          },
        };

        const indexStatusTool = {
          name: "memory_index_status",
          description: "Show memory index statistics: chunk counts by type, index health, and top indexed paths.",
          label: "Index Status",
          parameters: IndexStatusParams,
          execute: async (_toolCallId: string, params: { agentId?: string }) => {
            const s = await getState(cfg, api.logger);
            const targetAgentId = params.agentId ?? agentId;

            // Use LanceDBBackend.getStats() if available
            const lanceStats = (s.backend as import("./src/storage/lancedb.js").LanceDBBackend).getStats?.(targetAgentId);
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

            // ── Config Summary ──
            statsText += `\nConfig:\n`;
            statsText += `  AutoRecall: ${cfg.autoRecall.enabled ? cfg.autoRecall.agents.join(",") : "off"}\n`;
            statsText += `  AutoCapture: ${cfg.autoCapture.enabled ? cfg.autoCapture.agents.join(",") : "off"}\n`;

            return { content: [{ type: "text" as const, text: statsText }], details: {} };
          },
        };

        const forgetByPathTool = {
          name: "memory_forget_by_path",
          description: "Remove all indexed chunks from a specific file path. Use when reference docs are outdated or a file has been deleted.",
          label: "Forget by Path",
          parameters: ForgetByPathParams,
          execute: async (_toolCallId: string, params: { path: string }) => {
            const s = await getState(cfg, api.logger);
            const removed = await s.backend.deleteByPath(params.path, agentId);
            if (removed === 0) {
              return { content: [{ type: "text" as const, text: `No chunks found for path: ${params.path}` }], details: { removed: 0 } };
            }
            return {
              content: [{ type: "text" as const, text: `Removed ${removed} chunks from: ${params.path}` }],
              details: { removed },
            };
          },
        };

        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- OpenClaw plugin SDK expects untyped tool arrays
        return [searchTool, getTool, storeTool, forgetTool, referenceSearchTool, indexStatusTool, forgetByPathTool] as any;
      },
      { names: ["memory_search", "memory_get", "memory_store", "memory_forget", "memory_reference_search", "memory_index_status", "memory_forget_by_path"] },
    );

    // -------------------------------------------------------------------
    // 2. Auto-recall (before_prompt_build)
    // -------------------------------------------------------------------
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- OpenClaw plugin SDK hooks are untyped
    api.on("before_prompt_build", async (event: any, ctx: any) => {
      try {
        const s = await getState(cfg, api.logger);
        const handler = createAutoRecallHandler({
          cfg: cfg.autoRecall, backend: s.backend, embed: s.cachedEmbed, reranker: s.reranker,
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
          cfg: cfg.autoCapture, globalCfg: cfg, backend: s.backend, embed: s.queue,
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
        const workspaceDir = ctx.workspaceDir ?? path.join(os.homedir(), ".openclaw", `workspace-${agentId}`);
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
        import("./scripts/migrate.js")
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
              console.log(`AutoRecall:  ${cfg.autoRecall.enabled ? cfg.autoRecall.agents.join(",") : "off"}`);
              console.log(`AutoCapture: ${cfg.autoCapture.enabled ? cfg.autoCapture.agents.join(",") : "off"}`);
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
              await import("./scripts/migrate.js");
            } catch (err) {
              console.error(`Migration error: ${err}`);
            }
          });
      },
      { commands: ["memory"] },
    );

    api.logger.info("memory-spark: plugin registered (workspace auto-discovery + sessions + auto-recall/capture)");
  },
};

export default memorySpark;
