/**
 * SQLite-vec Storage Backend
 *
 * Read-write backend for compatibility + migration source.
 * Wraps the existing memory-core SQLite-vec DBs at ~/.openclaw/memory/*.sqlite
 *
 * Used in two contexts:
 *   1. As a READ source during migration (migrate.ts reads existing vectors)
 *   2. As a fallback backend if LanceDB fails to initialize
 *
 * Schema matches memory-core's existing format so we can read existing DBs
 * without any conversion until we're ready to re-embed into LanceDB.
 *
 * NOTE: Vectors from memory-core (Gemini 3072-dim) are NOT compatible with
 * our Spark embeddings (Qwen3 2560-dim or Nemotron 4096-dim).
 * Migration will re-embed all chunks with the new provider.
 */

import type { StorageBackend, MemoryChunk, SearchOptions, SearchResult, BackendStatus } from "./backend.js";
import type { MemorySparkConfig } from "../config.js";

export class SqliteVecBackend implements StorageBackend {
  private cfg: MemorySparkConfig;
  private agentId: string;
  private dbPath: string;
  private db: unknown = null; // node:sqlite Database

  constructor(cfg: MemorySparkConfig, agentId: string) {
    this.cfg = cfg;
    this.agentId = agentId;
    this.dbPath = `${cfg.sqliteVecDir}/${agentId}.sqlite`;
  }

  async open(): Promise<void> {
    // TODO:
    // const { requireNodeSqlite } = await import("openclaw/plugin-sdk/memory");
    // const sqlite = requireNodeSqlite();
    // this.db = new sqlite.DatabaseSync(this.dbPath);
    // Load sqlite-vec extension, enable WAL mode
    throw new Error("SqliteVecBackend.open() not yet implemented");
  }

  async close(): Promise<void> {
    // TODO: this.db?.close()
    this.db = null;
  }

  async upsert(chunks: MemoryChunk[]): Promise<void> {
    // TODO: INSERT OR REPLACE into chunks + vectors tables
    throw new Error("SqliteVecBackend.upsert() not yet implemented");
  }

  async deleteByPath(path: string): Promise<number> {
    // TODO: DELETE FROM chunks WHERE path = ?
    throw new Error("SqliteVecBackend.deleteByPath() not yet implemented");
  }

  async deleteById(ids: string[]): Promise<void> {
    // TODO: DELETE FROM chunks WHERE id IN (...)
    throw new Error("SqliteVecBackend.deleteById() not yet implemented");
  }

  async vectorSearch(queryVector: number[], opts: SearchOptions): Promise<SearchResult[]> {
    // TODO: sqlite-vec KNN search on vec_chunks virtual table
    throw new Error("SqliteVecBackend.vectorSearch() not yet implemented");
  }

  async ftsSearch(query: string, opts: SearchOptions): Promise<SearchResult[]> {
    // TODO: FTS5 search on chunks_fts virtual table
    throw new Error("SqliteVecBackend.ftsSearch() not yet implemented");
  }

  async listPaths(): Promise<Array<{ path: string; updatedAt: string; chunkCount: number }>> {
    // TODO: SELECT path, max(updated_at), count(*) FROM chunks GROUP BY path
    throw new Error("SqliteVecBackend.listPaths() not yet implemented");
  }

  async getById(id: string): Promise<MemoryChunk | null> {
    // TODO: SELECT * FROM chunks WHERE id = ?
    throw new Error("SqliteVecBackend.getById() not yet implemented");
  }

  async readFile(params: { path: string; from?: number; lines?: number }): Promise<{ text: string; path: string }> {
    // TODO: fetch ordered chunks, reconstruct text
    throw new Error("SqliteVecBackend.readFile() not yet implemented");
  }

  /**
   * Read ALL chunks with their raw vectors from the existing memory-core DB.
   * Used by migrate.ts to extract content for re-embedding.
   */
  async readAllForMigration(): Promise<MemoryChunk[]> {
    // TODO: SELECT all rows including raw vector blobs
    throw new Error("SqliteVecBackend.readAllForMigration() not yet implemented");
  }

  async status(): Promise<BackendStatus> {
    // TODO: COUNT(*) FROM chunks
    return {
      backend: "sqlite-vec",
      chunkCount: 0,
      ready: false,
      error: "not yet implemented",
    };
  }
}
