/**
 * memory-spark — OpenClaw Memory Plugin
 *
 * Drop-in replacement for memory-core.
 * Declare kind: "memory" → occupies the memory slot exclusively.
 *
 * What this plugin registers:
 *   - memory_search tool       (backed by LanceDB + Spark embed + reranker)
 *   - memory_get tool          (backed by same)
 *   - memory CLI commands      (status, sync, ingest, migrate)
 *   - file watcher service     (auto-index watched paths on change)
 *   - before_prompt_build hook (auto-recall: inject memories pre-turn)
 *   - agent_end hook           (auto-capture: extract + store turn facts)
 *   - gateway_start hook       (run migration on first boot if configured)
 *
 * Activation:
 *   openclaw.json → plugins.entries["memory-spark"].enabled = true
 *   AND plugins.entries["memory-core"].enabled = false (or removed from allow list)
 *
 * The plugin self-initializes all components on first registerTool call
 * (lazy init to avoid blocking gateway startup on embed probe).
 */

import type { OpenClawPluginDefinition, OpenClawPluginApi } from "openclaw/plugin-sdk";
import { resolveConfig } from "./src/config.js";
import type { MemorySparkConfig } from "./src/config.js";
import { LanceDBBackend } from "./src/storage/lancedb.js";
import { createEmbedProvider } from "./src/embed/provider.js";
import { createReranker } from "./src/rerank/reranker.js";
import { MemorySparkManager } from "./src/manager.js";
import { createWatcher } from "./src/ingest/watcher.js";
import { createAutoRecallHook } from "./src/auto/recall.js";
import { createAutoCaptureHook } from "./src/auto/capture.js";
import type { StorageBackend } from "./src/storage/backend.js";
import type { EmbedProvider } from "./src/embed/provider.js";
import type { Reranker } from "./src/rerank/reranker.js";

/** Singleton components initialized once per plugin load */
interface PluginState {
  cfg: MemorySparkConfig;
  backend: StorageBackend;
  embed: EmbedProvider;
  reranker: Reranker;
  initialized: boolean;
}

let state: PluginState | null = null;

async function getOrInitState(cfg: MemorySparkConfig, logger: OpenClawPluginApi["logger"]): Promise<PluginState> {
  if (state?.initialized) return state;

  logger.info("memory-spark: initializing components...");

  const backend = new LanceDBBackend(cfg);
  await backend.open();

  const embed = await createEmbedProvider(cfg.embed);
  logger.info(`memory-spark: embed provider ready — ${embed.id}/${embed.model} (${embed.dims}d)`);

  const reranker = await createReranker(cfg.rerank);

  state = { cfg, backend, embed, reranker, initialized: true };
  logger.info("memory-spark: initialized");
  return state;
}

const memorySpark: OpenClawPluginDefinition = {
  id: "memory-spark",
  name: "Memory Spark",
  version: "0.1.0",
  description: "Autonomous Spark-powered memory: LanceDB, local embed+rerank, auto-recall, auto-capture.",
  kind: "memory",

  async register(api: OpenClawPluginApi) {
    const cfg = resolveConfig(api.pluginConfig as Partial<MemorySparkConfig> | undefined);

    // -------------------------------------------------------------------------
    // 1. Register memory_search + memory_get tools
    //    Lazy-init state on first tool instantiation per agent session
    // -------------------------------------------------------------------------
    api.registerTool(
      (ctx) => {
        // Return null during state init failure — OC will skip the tool gracefully
        const agentId = ctx.agentId ?? "default";

        // We can't await in registerTool factory synchronously.
        // Return the tools with an async search that initializes on first call.
        const searchTool = {
          name: "memory_search",
          description: "Search the knowledge base and memory for relevant information. Use for factual recall, past decisions, user preferences, and indexed documents.",
          parameters: {
            type: "object" as const,
            properties: {
              query: { type: "string", description: "What to search for" },
              maxResults: { type: "number", description: "Max results to return (default 10)" },
            },
            required: ["query"],
          },
          async execute(params: { query: string; maxResults?: number }) {
            const s = await getOrInitState(cfg, api.logger);
            const manager = new MemorySparkManager({
              cfg,
              agentId,
              backend: s.backend,
              embed: s.embed,
              reranker: s.reranker,
            });
            const results = await manager.search(params.query, { maxResults: params.maxResults });
            if (results.length === 0) return "No relevant memories found.";
            return results.map((r, i) =>
              `${i + 1}. [${r.citation ?? r.path}] ${r.snippet}`
            ).join("\n\n");
          },
        };

        const getTool = {
          name: "memory_get",
          description: "Read a specific section of an indexed file by path and line range.",
          parameters: {
            type: "object" as const,
            properties: {
              path: { type: "string", description: "Relative path to the file" },
              from: { type: "number", description: "Start line (1-indexed)" },
              lines: { type: "number", description: "Number of lines to read (default 50)" },
            },
            required: ["path"],
          },
          async execute(params: { path: string; from?: number; lines?: number }) {
            const s = await getOrInitState(cfg, api.logger);
            const manager = new MemorySparkManager({
              cfg,
              agentId,
              backend: s.backend,
              embed: s.embed,
              reranker: s.reranker,
            });
            const result = await manager.readFile({ relPath: params.path, from: params.from, lines: params.lines });
            return result.text;
          },
        };

        return [searchTool as never, getTool as never];
      },
      { names: ["memory_search", "memory_get"] },
    );

    // -------------------------------------------------------------------------
    // 2. Auto-recall hook: inject memories before each prompt
    // -------------------------------------------------------------------------
    api.on("before_prompt_build", async (event, ctx) => {
      const s = await getOrInitState(cfg, api.logger).catch(() => null);
      if (!s) return;
      const recallHook = createAutoRecallHook({
        cfg: cfg.autoRecall,
        backend: s.backend,
        embed: s.embed,
        reranker: s.reranker,
      });
      return recallHook(event, ctx);
    });

    // -------------------------------------------------------------------------
    // 3. Auto-capture hook: extract + store facts after each turn
    // -------------------------------------------------------------------------
    api.on("agent_end", async (event, ctx) => {
      const s = await getOrInitState(cfg, api.logger).catch(() => null);
      if (!s) return;
      const captureHook = createAutoCaptureHook({
        cfg: cfg.autoCapture,
        globalCfg: cfg,
        backend: s.backend,
        embed: s.embed,
      });
      return captureHook(event, ctx);
    });

    // -------------------------------------------------------------------------
    // 4. File watcher service (auto-index configured paths)
    // -------------------------------------------------------------------------
    api.registerService({
      id: "memory-spark-watcher",
      async start(svcCtx) {
        const s = await getOrInitState(cfg, api.logger).catch(() => null);
        if (!s) return;
        const watcher = createWatcher({
          watch: cfg.watch,
          backend: s.backend,
          embed: s.embed,
          logger: svcCtx.logger,
        });
        await watcher.start();
        // Store watcher ref on state for cleanup
        (s as PluginState & { watcher?: ReturnType<typeof createWatcher> }).watcher = watcher;
      },
      async stop() {
        const w = (state as (PluginState & { watcher?: ReturnType<typeof createWatcher> }) | null)?.watcher;
        await w?.stop();
      },
    });

    // -------------------------------------------------------------------------
    // 5. First-boot migration (gateway_start)
    // -------------------------------------------------------------------------
    api.on("gateway_start", async (_event, _ctx) => {
      if (!cfg.migrate.autoMigrateOnFirstBoot) return;
      // TODO: check statusFile — if migration not done, run in background
      // import("./scripts/migrate.js").then(...).catch(err => api.logger.warn(...))
    });

    // -------------------------------------------------------------------------
    // 6. CLI: openclaw memory ...
    // -------------------------------------------------------------------------
    api.registerCli(
      ({ program }) => {
        const memoryCmd = program
          .command("memory")
          .description("memory-spark: manage the autonomous memory system");

        memoryCmd
          .command("status")
          .description("Show memory system status and indexing stats")
          .option("--agent <id>", "Show status for a specific agent")
          .action(async (opts) => {
            // TODO: init state, call backend.status(), print table
            console.log("memory-spark status: not yet implemented");
          });

        memoryCmd
          .command("sync")
          .description("Force re-index all watched paths for an agent")
          .option("--agent <id>", "Agent to sync (default: all)")
          .option("--force", "Re-embed even unchanged files")
          .action(async (opts) => {
            // TODO: init state, run boot pass
            console.log("memory-spark sync: not yet implemented");
          });

        memoryCmd
          .command("migrate")
          .description("Run migration from memory-core SQLite-vec to LanceDB")
          .action(async () => {
            // TODO: call migrate.ts main()
            console.log("memory-spark migrate: not yet implemented");
          });
      },
      { commands: ["memory"] },
    );

    api.logger.info("memory-spark: plugin registered");
  },
};

export default memorySpark;
