/**
 * File Watcher + Boot Pass — auto-indexes watched directories.
 * Uses chokidar for cross-platform file watching.
 */

import type { WatchConfig, WatchPath, MemorySparkConfig } from "../config.js";
import type { StorageBackend } from "../storage/backend.js";
import type { EmbedProvider } from "../embed/provider.js";
import { ingestFile } from "./pipeline.js";
import { SUPPORTED_EXTS } from "./parsers.js";
import chokidar, { type FSWatcher } from "chokidar";
import path from "node:path";
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

  function resolveAgentForPath(filePath: string): string | undefined {
    for (const wp of opts.watch.paths) {
      if (filePath.startsWith(wp.path)) {
        return wp.agents?.[0]; // First agent scope for this path
      }
    }
    return undefined;
  }

  function shouldIndex(filePath: string): boolean {
    const ext = path.extname(filePath).replace(".", "").toLowerCase();
    return SUPPORTED_EXTS.has(ext) && opts.watch.fileTypes.includes(ext);
  }

  async function handleFileChange(filePath: string): Promise<void> {
    if (!shouldIndex(filePath)) return;

    const agentId = resolveAgentForPath(filePath);
    await ingestFile({
      filePath,
      agentId,
      backend: opts.backend,
      embed: opts.embed,
      cfg: opts.cfg,
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
      if (!opts.watch.enabled || opts.watch.paths.length === 0) return;

      const watchPaths = opts.watch.paths.map((p) => p.path);
      opts.logger.info(`memory-spark watcher: watching ${watchPaths.join(", ")}`);

      // Boot pass: index existing files
      if (opts.watch.indexOnBoot) {
        runBootPass(opts.watch.paths, opts.backend, opts.embed, opts.cfg, opts.logger).catch((err) =>
          opts.logger.error(`memory-spark boot pass failed: ${err}`)
        );
      }

      // Start file watcher
      fsWatcher = chokidar.watch(watchPaths, {
        ignoreInitial: true,
        persistent: true,
        awaitWriteFinish: { stabilityThreshold: 1000, pollInterval: 200 },
        ignored: [
          /(^|[\/\\])\./,  // dotfiles
          /node_modules/,
          /\.git/,
        ],
      });

      fsWatcher.on("add", (fp: string) => debouncedHandle(fp));
      fsWatcher.on("change", (fp: string) => debouncedHandle(fp));
      fsWatcher.on("unlink", async (fp: string) => {
        const agentId = resolveAgentForPath(fp);
        await opts.backend.deleteByPath(fp, agentId).catch(() => {});
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
 * Boot pass: walk configured paths, index files not yet in storage or stale.
 */
async function runBootPass(
  paths: WatchPath[],
  backend: StorageBackend,
  embed: EmbedProvider,
  cfg: MemorySparkConfig,
  logger: WatcherLogger,
): Promise<void> {
  const indexed = await backend.listPaths();
  const indexedMap = new Map(indexed.map((i) => [i.path, i.updatedAt]));

  let total = 0;
  let ingested = 0;

  for (const wp of paths) {
    const files = await walkDir(wp.path, cfg.watch.fileTypes);
    total += files.length;

    for (const filePath of files) {
      try {
        const stat = await fs.stat(filePath);
        const mtime = stat.mtime.toISOString();
        const existing = indexedMap.get(filePath);

        // Skip if already indexed and not modified
        if (existing && existing >= mtime) continue;

        await ingestFile({
          filePath,
          agentId: wp.agents?.[0],
          backend,
          embed,
          cfg,
          logger,
        });
        ingested++;
      } catch (err) {
        logger.warn(`memory-spark boot pass: skip ${filePath}: ${err}`);
      }
    }
  }

  logger.info(`memory-spark boot pass: indexed ${ingested}/${total} files`);
}

async function walkDir(dir: string, fileTypes: string[]): Promise<string[]> {
  const results: string[] = [];
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.name.startsWith(".") || entry.name === "node_modules") continue;

      if (entry.isDirectory()) {
        const sub = await walkDir(fullPath, fileTypes);
        results.push(...sub);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).replace(".", "").toLowerCase();
        if (fileTypes.includes(ext)) {
          results.push(fullPath);
        }
      }
    }
  } catch {
    // Directory not accessible
  }
  return results;
}
