/**
 * SQLite-vec Backend — migration source for reading existing memory-core data.
 * Also serves as fallback if LanceDB fails to initialize.
 */

import type {
  StorageBackend,
  MemoryChunk,
  SearchOptions,
  SearchResult,
  BackendStatus,
} from "./backend.js";
import type { MemorySparkConfig } from "../config.js";
import fs from "node:fs/promises";
import path from "node:path";

/**
 * Minimal read-only adapter for existing memory-core SQLite-vec databases.
 * Used primarily during migration to read existing chunk text.
 */
export class SqliteVecBackend implements StorageBackend {
  private cfg: MemorySparkConfig;

  constructor(cfg: MemorySparkConfig) {
    this.cfg = cfg;
  }

  async open(): Promise<void> {
    // Verify directory exists
    try {
      await fs.access(this.cfg.sqliteVecDir);
    } catch {
      // Directory doesn't exist — that's fine, no data to migrate
    }
  }

  async close(): Promise<void> {
    // No-op for read-only adapter
  }

  async upsert(_chunks: MemoryChunk[]): Promise<void> {
    throw new Error("SqliteVecBackend is read-only (migration source)");
  }

  async deleteByPath(_path: string): Promise<number> {
    throw new Error("SqliteVecBackend is read-only");
  }

  async deleteById(_ids: string[]): Promise<void> {
    throw new Error("SqliteVecBackend is read-only");
  }

  async vectorSearch(_queryVector: number[], _opts: SearchOptions): Promise<SearchResult[]> {
    return []; // Not supported — use LanceDB
  }

  async ftsSearch(_query: string, _opts: SearchOptions): Promise<SearchResult[]> {
    return [];
  }

  async listPaths(
    _agentId?: string,
  ): Promise<Array<{ path: string; agentId: string; updatedAt: string; chunkCount: number }>> {
    return [];
  }

  async getById(_id: string): Promise<MemoryChunk | null> {
    return null;
  }

  async readFile(_params: { path: string }): Promise<{ text: string; path: string }> {
    return { text: "", path: _params.path };
  }

  /**
   * Discover all existing memory-core SQLite DBs and list their agent IDs.
   */
  async discoverAgentDbs(): Promise<string[]> {
    try {
      const files = await fs.readdir(this.cfg.sqliteVecDir);
      return files.filter((f) => f.endsWith(".sqlite")).map((f) => path.basename(f, ".sqlite"));
    } catch {
      return [];
    }
  }

  async status(): Promise<BackendStatus> {
    return {
      backend: "sqlite-vec",
      chunkCount: 0,
      tableExists: false,
      ready: false,
      error: "Read-only migration source",
    };
  }
}
