/**
 * File Watcher — auto-indexes workspace memory, sessions, AND external paths.
 *
 * On start:
 *   1. Auto-discovers ALL agent workspace memory dirs
 *   2. Auto-discovers ALL agent session JSONL dirs
 *   3. Adds user-configured watch.paths
 *   4. Runs boot pass: indexes new/modified files
 *   5. Starts chokidar watcher on all discovered paths
 *
 * This replicates memory-core's behavior: every agent's workspace memory
 * files and past session transcripts are automatically indexed.
 */

import type { WatchConfig, MemorySparkConfig } from "../config.js";
import type { StorageBackend } from "../storage/backend.js";
import type { EmbedProvider } from "../embed/provider.js";
import { EmbedQueue } from "../embed/queue.js";
import type { Embedder } from "./pipeline.js";
import { ingestFile } from "./pipeline.js";
import {
  discoverWorkspaceFiles,
  discoverAllAgents,
  toRelativePath,
  walkSupportedFiles,
} from "./workspace.js";
import { SUPPORTED_EXTS } from "./parsers.js";
import { enforceMistakesFiles } from "../auto/mistakes.js";
import chokidar, { type FSWatcher } from "chokidar";
import { minimatch } from "minimatch";
import path from "node:path";
import os from "node:os";
import fs from "node:fs/promises";

// ---------------------------------------------------------------------------
// Persistent pending-embed queue
// Files that fail to embed (e.g. Spark is down) are recorded here so they
// can be re-processed when the embed service recovers — even across restarts.
// The boot-pass already handles recovery via mtime comparison, but this JSONL
// provides explicit observability and a drain target for within-session recovery.
// ---------------------------------------------------------------------------

const PENDING_QUEUE_PATH = path.join(
  os.homedir(),
  ".openclaw",
  "data",
  "memory-spark",
  "pending-embed.jsonl",
);

interface PendingEntry {
  path: string;
  agentId: string;
  source: "memory" | "sessions" | "ingest";
  failedAt: string;
}

async function appendPending(entry: PendingEntry): Promise<void> {
  try {
    await fs.mkdir(path.dirname(PENDING_QUEUE_PATH), { recursive: true });
    await fs.appendFile(PENDING_QUEUE_PATH, JSON.stringify(entry) + "\n", "utf8");
  } catch {
    /* best-effort — don't let queue write failures break anything */
  }
}

async function clearPendingQueue(): Promise<void> {
  try {
    await fs.unlink(PENDING_QUEUE_PATH);
  } catch {
    /* file may not exist */
  }
}

async function pendingQueueSize(): Promise<number> {
  try {
    const content = await fs.readFile(PENDING_QUEUE_PATH, "utf8");
    return content.trim().split("\n").filter(Boolean).length;
  } catch {
    return 0;
  }
}

export interface WatcherLogger {
  info: (m: string) => void;
  warn: (m: string) => void;
  error: (m: string) => void;
}

export interface Watcher {
  start(): Promise<void>;
  stop(): Promise<void>;
  /** Manually trigger a boot-pass re-scan (e.g. after embed recovery) */
  triggerBootPass(): void;
}

export function createWatcher(opts: {
  watch: WatchConfig;
  cfg: MemorySparkConfig;
  backend: StorageBackend;
  embed: Embedder;
  /** Optional EmbedQueue reference — used to register an onRecovery hook that
   *  auto-triggers a boot-pass when Spark comes back online mid-session. */
  queue?: EmbedQueue;
  logger: WatcherLogger;
}): Watcher {
  let fsWatcher: FSWatcher | null = null;
  let bootPassRunning = false;
  const debounceTimers = new Map<string, ReturnType<typeof setTimeout>>();

  // Map: absolute dir path → agentId
  const dirToAgent = new Map<string, string>();

  function resolveAgentForPath(filePath: string): string {
    for (const [dir, agentId] of dirToAgent) {
      if (filePath.startsWith(dir)) return agentId;
    }
    // Check user-configured watch paths
    for (const wp of opts.watch.paths) {
      if (filePath.startsWith(wp.path)) return wp.agents?.[0] ?? "shared";
    }
    return "shared";
  }

  /** Common root for all reference library paths — set during init */
  let referenceRoot: string | null = null;

  function resolveWorkspaceDir(agentId: string): string {
    // "shared" agent = reference library files → use the common reference root
    if (agentId === "shared" && referenceRoot) return referenceRoot;
    const ocDir = path.join(os.homedir(), ".openclaw");
    return path.join(ocDir, `workspace-${agentId}`);
  }

  function shouldIndex(filePath: string): boolean {
    const ext = path.extname(filePath).replace(".", "").toLowerCase();

    // Session JSONL: only index if explicitly enabled
    if (ext === "jsonl") return opts.watch.indexSessions ?? false;

    if (!SUPPORTED_EXTS.has(ext)) return false;

    // Check exclude patterns (glob) and exact paths
    const agentId = resolveAgentForPath(filePath);
    const wsDir = resolveWorkspaceDir(agentId);
    const relPath = path.relative(wsDir, filePath);

    for (const pattern of opts.watch.excludePatterns ?? []) {
      if (minimatch(relPath, pattern, { dot: true })) return false;
    }
    for (const exact of opts.watch.excludePathsExact ?? []) {
      if (relPath === exact) return false;
    }

    return true;
  }

  function sourceForPath(filePath: string): "memory" | "sessions" | "ingest" {
    if (filePath.includes("/sessions/") && filePath.endsWith(".jsonl")) return "sessions";
    const ocDir = path.join(os.homedir(), ".openclaw");
    if (filePath.startsWith(path.join(ocDir, "workspace-"))) return "memory";
    return "ingest";
  }

  /** Check if a file is under one of the configured reference library paths */
  function isReferencePath(filePath: string): boolean {
    if (!opts.cfg.reference?.enabled || !opts.cfg.reference.paths.length) return false;
    for (const refPath of opts.cfg.reference.paths) {
      const resolved = refPath.startsWith("~/")
        ? path.join(os.homedir(), refPath.slice(2))
        : refPath;
      if (filePath.startsWith(resolved)) return true;
    }
    return false;
  }

  async function handleFileChange(filePath: string): Promise<void> {
    if (!shouldIndex(filePath)) return;

    const agentId = resolveAgentForPath(filePath);
    const source = sourceForPath(filePath);
    const contentType = isReferencePath(filePath) ? ("reference" as const) : ("knowledge" as const);

    const result = await ingestFile({
      filePath,
      agentId,
      workspaceDir: resolveWorkspaceDir(agentId),
      backend: opts.backend,
      embed: opts.embed,
      cfg: opts.cfg,
      source,
      contentType,
      logger: opts.logger,
    });

    // If embed failed (e.g. Spark is down), log to persistent pending queue
    if (result.error && /embed|timeout|connect|ECONNREFUSED/i.test(result.error)) {
      await appendPending({ path: filePath, agentId, source, failedAt: new Date().toISOString() });
    }
  }

  function debouncedHandle(filePath: string): void {
    // Skip watcher events during boot pass to avoid commit conflicts
    if (bootPassRunning) return;
    const existing = debounceTimers.get(filePath);
    if (existing) clearTimeout(existing);

    debounceTimers.set(
      filePath,
      setTimeout(() => {
        debounceTimers.delete(filePath);
        handleFileChange(filePath).catch((err) =>
          opts.logger.error(`memory-spark watcher: ${filePath}: ${err}`),
        );
      }, opts.watch.debounceMs),
    );
  }

  /** Trigger a boot-pass (can be called externally, e.g. after embed recovery) */
  function triggerBootPassNow(): void {
    if (bootPassRunning) {
      opts.logger.info(
        "memory-spark recovery: boot pass already running, skipping duplicate trigger",
      );
      return;
    }
    pendingQueueSize()
      .then((n) => {
        opts.logger.info(
          `memory-spark recovery: embed healthy again — triggering catch-up boot pass (${n} entries in pending queue)`,
        );
      })
      .catch(() => {
        opts.logger.info(
          "memory-spark recovery: embed healthy again — triggering catch-up boot pass",
        );
      });
    discoverAllAgents()
      .then((agents) => {
        bootPassRunning = true;
        runBootPass(agents, opts)
          .then(() => {
            // Clear the pending queue — boot-pass picked up everything via mtime comparison
            clearPendingQueue()
              .then(() => {
                opts.logger.info(
                  "memory-spark recovery: pending-embed.jsonl cleared after successful boot pass",
                );
              })
              .catch(() => {});
          })
          .catch((err) => opts.logger.error(`memory-spark recovery boot pass failed: ${err}`))
          .finally(() => {
            bootPassRunning = false;
          });
      })
      .catch((err) => opts.logger.error(`memory-spark recovery: discoverAllAgents failed: ${err}`));
  }

  return {
    triggerBootPass() {
      triggerBootPassNow();
    },

    async start() {
      const ocDir = path.join(os.homedir(), ".openclaw");
      const watchPaths: string[] = [];

      // 1. Auto-discover ALL agent workspace memory dirs + session dirs
      const agents = await discoverAllAgents();
      for (const agentId of agents) {
        const wsDir = path.join(ocDir, `workspace-${agentId}`);
        const memDir = path.join(wsDir, "memory");
        const sessDir = path.join(ocDir, "agents", agentId, "sessions");

        // Watch workspace root (for MEMORY.md, SOUL.md, etc.)
        try {
          await fs.access(wsDir);
          watchPaths.push(wsDir);
          dirToAgent.set(wsDir, agentId);
        } catch {
          /* skip */
        }

        // Watch memory/ dir
        try {
          await fs.access(memDir);
          watchPaths.push(memDir);
          dirToAgent.set(memDir, agentId);
        } catch {
          /* skip */
        }

        // Watch mistakes/ dir
        const mistakesDir = path.join(wsDir, "mistakes");
        try {
          await fs.access(mistakesDir);
          watchPaths.push(mistakesDir);
          dirToAgent.set(mistakesDir, agentId);
        } catch {
          /* skip */
        }

        // Watch sessions dir
        try {
          await fs.access(sessDir);
          watchPaths.push(sessDir);
          dirToAgent.set(sessDir, agentId);
        } catch {
          /* skip */
        }
      }

      // 2. Add user-configured watch paths
      for (const wp of opts.watch.paths) {
        try {
          await fs.access(wp.path);
          watchPaths.push(wp.path);
          if (wp.agents?.[0]) {
            dirToAgent.set(wp.path, wp.agents[0]);
          }
        } catch {
          /* skip */
        }
      }

      // 2b. Add reference library paths (indexed as content_type: "reference")
      if (opts.cfg.reference?.enabled && opts.cfg.reference.paths.length > 0) {
        const resolvedRefPaths: string[] = [];
        for (const refPath of opts.cfg.reference.paths) {
          const resolved = refPath.startsWith("~/")
            ? path.join(os.homedir(), refPath.slice(2))
            : refPath;
          try {
            await fs.access(resolved);
            resolvedRefPaths.push(resolved);
            watchPaths.push(resolved);
            dirToAgent.set(resolved, "shared");
            opts.logger.info(`memory-spark watcher: added reference path ${resolved}`);
          } catch {
            /* skip */
          }
        }
        // Compute common root for meaningful relative paths (e.g. "ReferenceLibrary/vllm/quickstart.md")
        if (resolvedRefPaths.length > 0) {
          let common = resolvedRefPaths[0]!;
          while (
            common !== "/" &&
            !resolvedRefPaths.every((p) => p.startsWith(common + "/") || p === common)
          ) {
            common = path.dirname(common);
          }
          referenceRoot = common;
          opts.logger.info(`memory-spark watcher: reference root = ${referenceRoot}`);
        }
      }

      // 2c. MISTAKES.md enforcement — create missing ones across all agent workspaces
      const workspaceDirs = agents.map((a) => path.join(ocDir, `workspace-${a}`));
      enforceMistakesFiles(workspaceDirs, opts.logger).catch((err) =>
        opts.logger.error(`memory-spark: MISTAKES.md enforcement failed: ${err}`),
      );

      if (watchPaths.length === 0) {
        opts.logger.info("memory-spark watcher: no paths to watch");
        return;
      }

      opts.logger.info(
        `memory-spark watcher: watching ${watchPaths.length} paths across ${agents.length} agents`,
      );

      // 3. Boot pass
      if (opts.watch.indexOnBoot) {
        bootPassRunning = true;
        runBootPass(agents, opts)
          .catch((err) => opts.logger.error(`memory-spark boot pass failed: ${err}`))
          .finally(() => {
            bootPassRunning = false;
          });
      }

      // 4. Start chokidar
      // NOTE: Do NOT use /(^|[/\\])\./ — it would match .openclaw in the path
      // and silently drop all events. Use a function instead to only ignore
      // hidden basenames, not dotted directories in the watch path.
      const shouldIgnore = (filePath: string): boolean => {
        const basename = path.basename(filePath);
        // Ignore hidden files/dirs (starting with dot) EXCEPT the root watch paths themselves
        if (basename.startsWith(".") && !watchPaths.includes(filePath)) return true;
        // Ignore common noise dirs
        if (basename === "node_modules" || basename === ".git") return true;
        if (filePath.includes("/dist/") || filePath.includes("/archive/")) return true;
        return false;
      };
      fsWatcher = chokidar.watch(watchPaths, {
        ignoreInitial: true,
        persistent: true,
        depth: 5,
        awaitWriteFinish: { stabilityThreshold: 1000, pollInterval: 200 },
        ignored: shouldIgnore,
      });

      fsWatcher.on("ready", () => {
        opts.logger.info("memory-spark watcher: chokidar ready, watching for changes");
      });
      fsWatcher.on("error", (err: unknown) => {
        opts.logger.error(`memory-spark watcher: chokidar error: ${err}`);
      });
      fsWatcher.on("add", (fp: string) => {
        opts.logger.info(`memory-spark watcher: add ${fp}`);
        debouncedHandle(fp);
      });
      fsWatcher.on("change", (fp: string) => {
        opts.logger.info(`memory-spark watcher: change ${fp}`);
        debouncedHandle(fp);
      });
      fsWatcher.on("unlink", async (fp: string) => {
        opts.logger.info(`memory-spark watcher: unlink ${fp}`);
        const agentId = resolveAgentForPath(fp);
        const relPath = toRelativePath(fp, resolveWorkspaceDir(agentId));
        await opts.backend.deleteByPath(relPath, agentId).catch(() => {});
      });

      // 5. Register recovery hook: when Spark comes back online mid-session,
      //    auto-trigger a catch-up boot pass for files written while it was down.
      if (opts.queue) {
        opts.queue.onRecovery(() => triggerBootPassNow());
        opts.logger.info("memory-spark watcher: embed recovery hook registered");
      }
    },

    async stop() {
      for (const timer of debounceTimers.values()) {
        clearTimeout(timer);
      }
      debounceTimers.clear();
      if (fsWatcher) {
        await fsWatcher.close();
        fsWatcher = null;
      }
    },
  };
}

/**
 * Boot pass: discover + index all agent workspace files and sessions.
 */
async function runBootPass(
  agents: string[],
  opts: { cfg: MemorySparkConfig; backend: StorageBackend; embed: Embedder; logger: WatcherLogger },
): Promise<void> {
  // Key by agentId::path to avoid cross-agent false-skip when multiple agents
  // share the same relative path (e.g. all agents have memory/learnings.md).
  const indexed = await opts.backend.listPaths();
  const indexedMap = new Map(indexed.map((i) => [`${i.agentId ?? ""}::${i.path}`, i.updatedAt]));
  let total = 0;
  let ingested = 0;
  let skipped = 0;

  // Build queue of all files to ingest
  const queue: Array<{
    absPath: string;
    agentId: string;
    wsDir: string;
    source: "memory" | "sessions";
    contentType?: "knowledge" | "reference";
  }> = [];

  for (const agentId of agents) {
    const wsFiles = await discoverWorkspaceFiles(agentId);

    for (const absPath of wsFiles.memoryFiles) {
      total++;
      try {
        const stat = await fs.stat(absPath);
        const relPath = toRelativePath(absPath, wsFiles.workspaceDir);
        const existing = indexedMap.get(`${agentId}::${relPath}`);
        if (existing && existing >= stat.mtime.toISOString()) {
          skipped++;
          continue;
        }
        queue.push({ absPath, agentId, wsDir: wsFiles.workspaceDir, source: "memory" });
      } catch {
        skipped++;
      }
    }

    for (const absPath of wsFiles.sessionFiles) {
      total++;
      try {
        const stat = await fs.stat(absPath);
        const relPath = toRelativePath(absPath, wsFiles.workspaceDir);
        const existing = indexedMap.get(`${agentId}::${relPath}`);
        if (existing && existing >= stat.mtime.toISOString()) {
          skipped++;
          continue;
        }
        queue.push({ absPath, agentId, wsDir: wsFiles.workspaceDir, source: "sessions" });
      } catch {
        skipped++;
      }
    }
  }

  // Boot pass: reference library paths (content_type will be set to "reference" by isReferencePath)
  // Use the common reference root for relative path generation so tag filtering works.
  // e.g. ~/Documents/OpenClaw/ as root → path = "ReferenceLibrary/vllm/quickstart.md"
  if (opts.cfg.reference?.enabled && opts.cfg.reference.paths.length > 0) {
    const resolvedRefPaths: string[] = [];
    for (const refPath of opts.cfg.reference.paths) {
      const resolved = refPath.startsWith("~/")
        ? path.join(os.homedir(), refPath.slice(2))
        : refPath;
      resolvedRefPaths.push(resolved);
    }
    // Compute common root (same logic as init, but boot pass runs standalone)
    let refRoot = resolvedRefPaths[0] ?? "/";
    while (
      refRoot !== "/" &&
      !resolvedRefPaths.every((p) => p.startsWith(refRoot + "/") || p === refRoot)
    ) {
      refRoot = path.dirname(refRoot);
    }

    for (const resolved of resolvedRefPaths) {
      try {
        await fs.access(resolved);
        const refFiles = await walkSupportedFiles(resolved);
        for (const absPath of refFiles) {
          total++;
          try {
            const stat = await fs.stat(absPath);
            // Store path relative to common root so tag prefixes match
            // e.g. "ReferenceLibrary/vllm/quickstart.md" or "InternalDocs/HARDWARE.md"
            const relPath = path.relative(refRoot, absPath);
            const key = `shared::${relPath}`;
            const existing = indexedMap.get(key);
            if (existing && existing >= stat.mtime.toISOString()) {
              skipped++;
              continue;
            }
            queue.push({
              absPath,
              agentId: "shared",
              wsDir: refRoot,
              source: "memory",
              contentType: "reference",
            });
          } catch {
            skipped++;
          }
        }
        opts.logger.info(
          `memory-spark boot: added ${refFiles.length} reference files from ${resolved}`,
        );
      } catch {
        opts.logger.warn(`memory-spark boot: reference path not accessible: ${resolved}`);
      }
    }
  }

  opts.logger.info(
    `memory-spark boot pass: ${queue.length} files to index (${skipped} up-to-date, ${total} total)`,
  );

  // Process files sequentially — the EmbedQueue handles backpressure/retry
  const CONCURRENCY = 1;
  let errors = 0;

  // Dynamic timeout: 60s base + 30s per 50KB of file size
  const BASE_TIMEOUT_MS = 60_000;
  const TIMEOUT_PER_50KB = 30_000;
  const MAX_TIMEOUT_MS = 300_000; // 5 min cap

  function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`Timeout after ${ms}ms: ${label}`)), ms);
      promise
        .then((v) => {
          clearTimeout(timer);
          resolve(v);
        })
        .catch((e) => {
          clearTimeout(timer);
          reject(e);
        });
    });
  }

  async function fileTimeout(filePath: string): Promise<number> {
    try {
      const stat = await fs.stat(filePath);
      const sizeKb = stat.size / 1024;
      const timeout = Math.min(
        BASE_TIMEOUT_MS + Math.ceil(sizeKb / 50) * TIMEOUT_PER_50KB,
        MAX_TIMEOUT_MS,
      );
      return timeout;
    } catch {
      return BASE_TIMEOUT_MS;
    }
  }

  for (let i = 0; i < queue.length; i++) {
    const item = queue[i]!;
    try {
      const timeout = await fileTimeout(item.absPath);
      const result = await withTimeout(
        ingestFile({
          filePath: item.absPath,
          agentId: item.agentId,
          workspaceDir: item.wsDir,
          backend: opts.backend,
          embed: opts.embed,
          cfg: opts.cfg,
          source: item.source,
          contentType: item.contentType,
          logger: opts.logger,
        }),
        timeout,
        item.absPath,
      );
      if (result.error) {
        errors++;
        opts.logger.warn(`memory-spark boot: ${path.basename(item.absPath)}: ${result.error}`);
      } else {
        ingested++;
      }
    } catch (err) {
      errors++;
      const msg = err instanceof Error ? err.message : String(err);
      opts.logger.warn(`memory-spark boot: ${path.basename(item.absPath)}: ${msg}`);
    }

    // Progress log every 20 files
    if ((i + 1) % 20 === 0 || i + 1 === queue.length) {
      opts.logger.info(`memory-spark boot: ${ingested}/${queue.length} indexed, ${errors} errors`);
    }
  }

  opts.logger.info(
    `memory-spark boot pass complete: ${ingested}/${queue.length} indexed, ${errors} errors, ${skipped} skipped (${agents.length} agents)`,
  );
}
