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
import { ingestFile } from "./pipeline.js";
import { discoverWorkspaceFiles, discoverAllAgents, toRelativePath } from "./workspace.js";
import { SUPPORTED_EXTS } from "./parsers.js";
import chokidar, { type FSWatcher } from "chokidar";
import path from "node:path";
import os from "node:os";
import fs from "node:fs/promises";

export interface WatcherLogger {
  info: (m: string) => void;
  warn: (m: string) => void;
  error: (m: string) => void;
}

export interface Watcher {
  start(): Promise<void>;
  stop(): Promise<void>;
}

export function createWatcher(opts: {
  watch: WatchConfig;
  cfg: MemorySparkConfig;
  backend: StorageBackend;
  embed: EmbedProvider;
  logger: WatcherLogger;
}): Watcher {
  let fsWatcher: FSWatcher | null = null;
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

  function resolveWorkspaceDir(agentId: string): string {
    const ocDir = path.join(os.homedir(), ".openclaw");
    return path.join(ocDir, `workspace-${agentId}`);
  }

  function shouldIndex(filePath: string): boolean {
    const ext = path.extname(filePath).replace(".", "").toLowerCase();
    if (ext === "jsonl") return true; // Session files
    return SUPPORTED_EXTS.has(ext);
  }

  function sourceForPath(filePath: string): "memory" | "sessions" | "ingest" {
    if (filePath.includes("/sessions/") && filePath.endsWith(".jsonl")) return "sessions";
    const ocDir = path.join(os.homedir(), ".openclaw");
    if (filePath.startsWith(path.join(ocDir, "workspace-"))) return "memory";
    return "ingest";
  }

  async function handleFileChange(filePath: string): Promise<void> {
    if (!shouldIndex(filePath)) return;

    const agentId = resolveAgentForPath(filePath);
    const source = sourceForPath(filePath);

    await ingestFile({
      filePath,
      agentId,
      workspaceDir: resolveWorkspaceDir(agentId),
      backend: opts.backend,
      embed: opts.embed,
      cfg: opts.cfg,
      source,
      logger: opts.logger,
    });
  }

  function debouncedHandle(filePath: string): void {
    const existing = debounceTimers.get(filePath);
    if (existing) clearTimeout(existing);

    debounceTimers.set(
      filePath,
      setTimeout(() => {
        debounceTimers.delete(filePath);
        handleFileChange(filePath).catch((err) =>
          opts.logger.error(`memory-spark watcher: ${filePath}: ${err}`)
        );
      }, opts.watch.debounceMs),
    );
  }

  return {
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
        } catch { /* skip */ }

        // Watch memory/ dir
        try {
          await fs.access(memDir);
          watchPaths.push(memDir);
          dirToAgent.set(memDir, agentId);
        } catch { /* skip */ }

        // Watch sessions dir
        try {
          await fs.access(sessDir);
          watchPaths.push(sessDir);
          dirToAgent.set(sessDir, agentId);
        } catch { /* skip */ }
      }

      // 2. Add user-configured watch paths
      for (const wp of opts.watch.paths) {
        try {
          await fs.access(wp.path);
          watchPaths.push(wp.path);
          if (wp.agents?.[0]) {
            dirToAgent.set(wp.path, wp.agents[0]);
          }
        } catch { /* skip */ }
      }

      if (watchPaths.length === 0) {
        opts.logger.info("memory-spark watcher: no paths to watch");
        return;
      }

      opts.logger.info(`memory-spark watcher: watching ${watchPaths.length} paths across ${agents.length} agents`);

      // 3. Boot pass
      if (opts.watch.indexOnBoot) {
        runBootPass(agents, opts).catch((err) =>
          opts.logger.error(`memory-spark boot pass failed: ${err}`)
        );
      }

      // 4. Start chokidar
      fsWatcher = chokidar.watch(watchPaths, {
        ignoreInitial: true,
        persistent: true,
        depth: 5,
        awaitWriteFinish: { stabilityThreshold: 1000, pollInterval: 200 },
        ignored: [
          /(^|[/\\])\./,
          /node_modules/,
          /\.git/,
          /dist\//,
          /archive\//,
        ],
      });

      fsWatcher.on("add", (fp: string) => debouncedHandle(fp));
      fsWatcher.on("change", (fp: string) => debouncedHandle(fp));
      fsWatcher.on("unlink", async (fp: string) => {
        const agentId = resolveAgentForPath(fp);
        const relPath = toRelativePath(fp, resolveWorkspaceDir(agentId));
        await opts.backend.deleteByPath(relPath, agentId).catch(() => {});
      });
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
  opts: { cfg: MemorySparkConfig; backend: StorageBackend; embed: EmbedProvider; logger: WatcherLogger },
): Promise<void> {
  const indexed = await opts.backend.listPaths();
  const indexedMap = new Map(indexed.map((i) => [i.path, i.updatedAt]));
  let total = 0;
  let ingested = 0;

  for (const agentId of agents) {
    const wsFiles = await discoverWorkspaceFiles(agentId);

    // Index memory files
    for (const absPath of wsFiles.memoryFiles) {
      total++;
      try {
        const stat = await fs.stat(absPath);
        const relPath = toRelativePath(absPath, wsFiles.workspaceDir);
        const existing = indexedMap.get(relPath);
        if (existing && existing >= stat.mtime.toISOString()) continue;

        await ingestFile({
          filePath: absPath,
          agentId,
          workspaceDir: wsFiles.workspaceDir,
          backend: opts.backend,
          embed: opts.embed,
          cfg: opts.cfg,
          source: "memory",
          logger: opts.logger,
        });
        ingested++;
      } catch (err) {
        opts.logger.warn(`memory-spark boot: skip ${absPath}: ${err}`);
      }
    }

    // Index session files
    for (const absPath of wsFiles.sessionFiles) {
      total++;
      try {
        const stat = await fs.stat(absPath);
        const relPath = toRelativePath(absPath, wsFiles.workspaceDir);
        const existing = indexedMap.get(relPath);
        if (existing && existing >= stat.mtime.toISOString()) continue;

        await ingestFile({
          filePath: absPath,
          agentId,
          workspaceDir: wsFiles.workspaceDir,
          backend: opts.backend,
          embed: opts.embed,
          cfg: opts.cfg,
          source: "sessions",
          logger: opts.logger,
        });
        ingested++;
      } catch (err) {
        opts.logger.warn(`memory-spark boot: skip session ${absPath}: ${err}`);
      }
    }
  }

  opts.logger.info(`memory-spark boot pass: indexed ${ingested}/${total} files across ${agents.length} agents`);
}
