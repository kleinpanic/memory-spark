/**
 * memory-spark — OpenClaw Memory Plugin
 *
 * Drop-in replacement for memory-core.
 * kind: "memory" → occupies the memory slot exclusively.
 */

import { Type } from "@sinclair/typebox";
import { emptyPluginConfigSchema } from "openclaw/plugin-sdk";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";

import { resolveConfig, type MemorySparkConfig } from "./src/config.js";
import { LanceDBBackend } from "./src/storage/lancedb.js";
import { createEmbedProvider, type EmbedProvider } from "./src/embed/provider.js";
import { createReranker, type Reranker } from "./src/rerank/reranker.js";
import { MemorySparkManager } from "./src/manager.js";
import { createWatcher, type Watcher } from "./src/ingest/watcher.js";
import { createAutoRecallHandler } from "./src/auto/recall.js";
import { createAutoCaptureHandler } from "./src/auto/capture.js";
import type { StorageBackend } from "./src/storage/backend.js";

// ---------------------------------------------------------------------------
// Singleton plugin state (initialized once)
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

async function getState(cfg: MemorySparkConfig, logger: { info: (m: string) => void; warn: (m: string) => void; error: (m: string) => void }): Promise<PluginState> {
  if (state) return state;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    logger.info("memory-spark: initializing...");

    const backend = new LanceDBBackend(cfg);
    await backend.open();

    const embed = await createEmbedProvider(cfg.embed);
    logger.info(`memory-spark: embed provider → ${embed.id}/${embed.model} (${embed.dims}d)`);

    const reranker = await createReranker(cfg.rerank);
    logger.info(`memory-spark: reranker → ${cfg.rerank.enabled ? "spark" : "passthrough"}`);

    state = { cfg, backend, embed, reranker, watcher: null };
    logger.info("memory-spark: ready");
    return state;
  })();

  return initPromise;
}

// ---------------------------------------------------------------------------
// Tool parameter schemas (TypeBox)
// ---------------------------------------------------------------------------
const SearchParams = Type.Object({
  query: Type.String({ description: "What to search for in the knowledge base" }),
  maxResults: Type.Optional(Type.Number({ description: "Maximum results to return (default 10)" })),
});

const GetParams = Type.Object({
  path: Type.String({ description: "Relative path to the indexed file" }),
  from: Type.Optional(Type.Number({ description: "Start line (1-indexed)" })),
  lines: Type.Optional(Type.Number({ description: "Number of lines to read (default 50)" })),
});

// ---------------------------------------------------------------------------
// Plugin definition
// ---------------------------------------------------------------------------
const memorySpark = {
  id: "memory-spark",
  name: "Memory Spark",
  version: "0.1.0",
  description: "Autonomous Spark-powered memory: LanceDB, local embed+rerank, auto-recall, auto-capture.",
  kind: "memory" as const,
  configSchema: emptyPluginConfigSchema(),

  async register(api: OpenClawPluginApi) {
    const cfg = resolveConfig(api.pluginConfig as Partial<MemorySparkConfig> | undefined);

    // -------------------------------------------------------------------
    // 1. Register memory_search + memory_get tools
    // -------------------------------------------------------------------
    api.registerTool(
      (ctx) => {
        const agentId = ctx.agentId ?? "default";

        const searchTool = {
          name: "memory_search",
          description: "Search the knowledge base and memory for relevant information. Use for factual recall, past decisions, user preferences, and indexed documents.",
          label: "Memory Search",
          parameters: SearchParams,
          execute: async (_toolCallId: string, params: { query: string; maxResults?: number }) => {
            const s = await getState(cfg, api.logger);
            const manager = new MemorySparkManager({
              cfg, agentId, backend: s.backend, embed: s.embed, reranker: s.reranker,
            });
            const results = await manager.search(params.query, { maxResults: params.maxResults });
            if (results.length === 0) {
              return { content: [{ type: "text" as const, text: "No relevant memories found." }], details: {} };
            }
            const text = results.map((r, i) =>
              `${i + 1}. [${r.citation ?? r.path}] ${r.snippet}`
            ).join("\n\n");
            return { content: [{ type: "text" as const, text }], details: { resultCount: results.length } };
          },
        };

        const getTool = {
          name: "memory_get",
          description: "Read a specific section of an indexed file by path and line range.",
          label: "Memory Get",
          parameters: GetParams,
          execute: async (_toolCallId: string, params: { path: string; from?: number; lines?: number }) => {
            const s = await getState(cfg, api.logger);
            const manager = new MemorySparkManager({
              cfg, agentId, backend: s.backend, embed: s.embed, reranker: s.reranker,
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
    // 2. Auto-recall hook (before_prompt_build)
    // -------------------------------------------------------------------
    api.on("before_prompt_build", async (event: any, ctx: any) => {
      try {
        const s = await getState(cfg, api.logger);
        const handler = createAutoRecallHandler({
          cfg: cfg.autoRecall,
          backend: s.backend,
          embed: s.embed,
          reranker: s.reranker,
        });
        return await handler(event, ctx);
      } catch {
        return undefined; // Non-fatal
      }
    });

    // -------------------------------------------------------------------
    // 3. Auto-capture hook (agent_end)
    // -------------------------------------------------------------------
    api.on("agent_end", async (event: any, ctx: any) => {
      try {
        const s = await getState(cfg, api.logger);
        const handler = createAutoCaptureHandler({
          cfg: cfg.autoCapture,
          globalCfg: cfg,
          backend: s.backend,
          embed: s.embed,
        });
        await handler(event, ctx);
      } catch {
        // Always non-fatal
      }
    });

    // -------------------------------------------------------------------
    // 4. File watcher service
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
          svcCtx.logger.error(`memory-spark watcher failed to start: ${err}`);
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
    // 5. CLI: openclaw memory ...
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
              console.log(`Backend:    ${st.backend}`);
              console.log(`Chunks:     ${st.chunkCount}`);
              console.log(`Ready:      ${st.ready}`);
              console.log(`Embed:      ${s.embed.id}/${s.embed.model} (${s.embed.dims}d)`);
              console.log(`Reranker:   ${cfg.rerank.enabled ? "spark" : "off"}`);
              console.log(`AutoRecall: ${cfg.autoRecall.enabled ? cfg.autoRecall.agents.join(",") : "off"}`);
              console.log(`AutoCapture: ${cfg.autoCapture.enabled ? cfg.autoCapture.agents.join(",") : "off"}`);
              console.log(`Watch paths: ${cfg.watch.paths.length}`);
            } catch (err) {
              console.error(`Status error: ${err}`);
            }
          });

        memoryCmd
          .command("sync")
          .description("Force re-index all watched paths")
          .action(async () => {
            console.log("Triggering sync...");
            // Stop and restart watcher to force boot pass
            if (state?.watcher) {
              await state.watcher.stop();
              const s = await getState(cfg, api.logger);
              s.watcher = createWatcher({
                watch: { ...cfg.watch, indexOnBoot: true },
                cfg,
                backend: s.backend,
                embed: s.embed,
                logger: { info: console.log, warn: console.warn, error: console.error },
              });
              await s.watcher.start();
              console.log("Sync started.");
            }
          });
      },
      { commands: ["memory"] },
    );

    api.logger.info("memory-spark: plugin registered");
  },
};

export default memorySpark;
