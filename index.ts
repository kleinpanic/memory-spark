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
import { emptyPluginConfigSchema } from "openclaw/plugin-sdk";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import os from "node:os";
import path from "node:path";

import { resolveConfig, type MemorySparkConfig } from "./src/config.js";
import { LanceDBBackend } from "./src/storage/lancedb.js";
import { createEmbedProvider, type EmbedProvider } from "./src/embed/provider.js";
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

    const reranker = await createReranker(cfg.rerank);

    state = { cfg, backend, embed, reranker, watcher: null };
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

// ---------------------------------------------------------------------------
// Plugin definition
// ---------------------------------------------------------------------------
const memorySpark = {
  id: "memory-spark",
  name: "Memory Spark",
  version: "0.1.0",
  description: "Autonomous Spark-powered memory: LanceDB + local embed/rerank + auto-recall/capture.",
  kind: "memory" as const,
  configSchema: emptyPluginConfigSchema(),

  async register(api: OpenClawPluginApi) {
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
              backend: s.backend, embed: s.embed, reranker: s.reranker,
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
              backend: s.backend, embed: s.embed, reranker: s.reranker,
            });
            const result = await manager.readFile({ relPath: params.path, from: params.from, lines: params.lines });
            return { content: [{ type: "text" as const, text: result.text || "(empty)" }], details: {} };
          },
        };

        return [searchTool, getTool] as any;
      },
      { names: ["memory_search", "memory_get"] },
    );

    // -------------------------------------------------------------------
    // 2. Auto-recall (before_prompt_build)
    // -------------------------------------------------------------------
    api.on("before_prompt_build", async (event: any, ctx: any) => {
      try {
        const s = await getState(cfg, api.logger);
        const handler = createAutoRecallHandler({
          cfg: cfg.autoRecall, backend: s.backend, embed: s.embed, reranker: s.reranker,
        });
        return await handler(event, ctx);
      } catch {
        return undefined;
      }
    });

    // -------------------------------------------------------------------
    // 3. Auto-capture (agent_end)
    // -------------------------------------------------------------------
    api.on("agent_end", async (event: any, ctx: any) => {
      try {
        const s = await getState(cfg, api.logger);
        const handler = createAutoCaptureHandler({
          cfg: cfg.autoCapture, globalCfg: cfg, backend: s.backend, embed: s.embed,
        });
        await handler(event, ctx);
      } catch {
        // Non-fatal
      }
    });

    // -------------------------------------------------------------------
    // 4. After compaction — re-index compacted session
    // -------------------------------------------------------------------
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
          embed: s.embed,
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
            embed: s.embed,
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

        api.logger.info("memory-spark: first boot — starting migration in background");
        // Import and run migration asynchronously (non-blocking)
        import("./scripts/migrate.js").catch((err) => {
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
                embed: s.embed,
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
