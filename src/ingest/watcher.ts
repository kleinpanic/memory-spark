/**
 * File Watcher + Ingest Coordinator
 *
 * Watches configured paths for file changes.
 * On add/change → extracts text → chunks → embeds → stores.
 * On delete → removes chunks from storage.
 *
 * Registered as a plugin service (api.registerService) so it ties to
 * the gateway lifecycle: starts on boot, stops cleanly on shutdown.
 *
 * Uses chokidar for cross-platform fs.watch with proper debouncing.
 *
 * Boot pass: on first start, walks all configured paths and indexes
 * any file not already in storage (or modified since last index time).
 * Delta detection uses file mtime vs stored updatedAt timestamp.
 */

import type { WatchConfig, WatchPath } from "../config.js";
import type { StorageBackend } from "../storage/backend.js";
import type { EmbedProvider } from "../embed/provider.js";
import { ingestFile } from "./pipeline.js";

export interface WatcherOptions {
  watch: WatchConfig;
  backend: StorageBackend;
  embed: EmbedProvider;
  logger: { info: (m: string) => void; warn: (m: string) => void; error: (m: string) => void };
}

export interface Watcher {
  /** Start watching all configured paths. Runs boot-pass if configured. */
  start(): Promise<void>;
  /** Stop all watchers cleanly */
  stop(): Promise<void>;
  /** Force re-index a specific path immediately */
  forceIndex(path: string, agentId?: string): Promise<void>;
  /** Get stats on watched paths */
  stats(): { watchedPaths: number; pendingFiles: number };
}

export function createWatcher(opts: WatcherOptions): Watcher {
  // TODO:
  // let watcher: chokidar.FSWatcher | null = null;
  // const pending = new Map<string, ReturnType<typeof setTimeout>>();

  return {
    async start() {
      if (!opts.watch.enabled) return;

      // TODO:
      // 1. If watch.indexOnBoot: run boot pass across all paths
      // 2. Initialize chokidar watcher on all configured paths
      // 3. On "add" / "change": debounce(debounceMs) → ingestFile()
      // 4. On "unlink": backend.deleteByPath()
      throw new Error("Watcher.start() not yet implemented");
    },

    async stop() {
      // TODO: await watcher?.close()
    },

    async forceIndex(filePath, agentId) {
      // TODO: await ingestFile({ filePath, agentId, backend: opts.backend, embed: opts.embed })
      throw new Error("Watcher.forceIndex() not yet implemented");
    },

    stats() {
      return { watchedPaths: 0, pendingFiles: 0 };
    },
  };
}

/**
 * Boot-pass: walk all configured WatchPaths, compare file mtime to
 * stored updatedAt, queue stale/new files for ingestion.
 *
 * Runs in the background — non-blocking for gateway startup.
 */
export async function runBootPass(
  paths: WatchPath[],
  backend: StorageBackend,
  embed: EmbedProvider,
  logger: WatcherOptions["logger"],
): Promise<void> {
  // TODO:
  // for each path:
  //   glob files matching fileTypes
  //   compare mtime to backend.listPaths()
  //   ingest changed/new files
  //   log summary
  throw new Error("runBootPass() not yet implemented");
}
