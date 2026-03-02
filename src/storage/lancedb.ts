/**
 * LanceDB Storage Backend
 *
 * Primary backend. Uses @lancedb/lancedb with Apache Arrow schema.
 * One table per agent (e.g. "memory_main", "memory_school") + "memory_shared" for global.
 *
 * Schema per table:
 *   id         (utf8, not null)
 *   path       (utf8, not null)
 *   source     (utf8)
 *   agent_id   (utf8, nullable)
 *   start_line (int32)
 *   end_line   (int32)
 *   text       (utf8, not null)
 *   fts_text   (utf8)          ← full-text index target
 *   vector     (fixed_size_list<float32>[dims])  ← ANN index
 *   updated_at (utf8)
 *   category   (utf8, nullable)
 *   entities   (list<utf8>, nullable)
 *   confidence (float32, nullable)
 *
 * LanceDB supports:
 *   - Native ANN vector index (IVF_PQ, IVF_HNSW)
 *   - Native FTS index (Tantivy-based)
 *   - Hybrid vector+FTS queries in a single pass
 *
 * Implementation notes:
 *   - vectorDims determined from first embed call (Qwen3-Embedding-4B = 2560 dims)
 *   - ANN index rebuilt on significant size growth (every 10k new chunks)
 *   - FTS index rebuilt on every sync pass (incremental update not yet in LanceDB stable)
 */

import type { StorageBackend, MemoryChunk, SearchOptions, SearchResult, SyncStats, BackendStatus } from "./backend.js";
import type { MemorySparkConfig } from "../config.js";

// TODO: implement when @lancedb/lancedb is installed
// import * as lancedb from "@lancedb/lancedb";

export class LanceDBBackend implements StorageBackend {
  private cfg: MemorySparkConfig;
  private db: unknown = null; // lancedb.Connection

  constructor(cfg: MemorySparkConfig) {
    this.cfg = cfg;
  }

  async open(): Promise<void> {
    // TODO:
    // const dir = expandHome(this.cfg.lancedbDir);
    // await fs.mkdir(dir, { recursive: true });
    // this.db = await lancedb.connect(dir);
    throw new Error("LanceDBBackend.open() not yet implemented");
  }

  async close(): Promise<void> {
    // TODO: lancedb connection has no explicit close, but we clear state
    this.db = null;
  }

  async upsert(chunks: MemoryChunk[]): Promise<void> {
    // TODO:
    // 1. Group chunks by agentId → table name
    // 2. For each table: open or create with schema
    // 3. lancedb merge_insert on id field
    // 4. Rebuild FTS index after upsert
    throw new Error("LanceDBBackend.upsert() not yet implemented");
  }

  async deleteByPath(path: string, agentId?: string): Promise<number> {
    // TODO: table.delete(`path = '${sanitize(path)}'`)
    throw new Error("LanceDBBackend.deleteByPath() not yet implemented");
  }

  async deleteById(ids: string[]): Promise<void> {
    // TODO: table.delete(`id IN (${ids.map(quote).join(",")})`)
    throw new Error("LanceDBBackend.deleteById() not yet implemented");
  }

  async vectorSearch(queryVector: number[], opts: SearchOptions): Promise<SearchResult[]> {
    // TODO:
    // table.vectorSearch(queryVector)
    //   .limit(opts.maxResults ?? 20)
    //   .distanceType("cosine")
    //   .where(agentId filter)
    //   .toArray()
    throw new Error("LanceDBBackend.vectorSearch() not yet implemented");
  }

  async ftsSearch(query: string, opts: SearchOptions): Promise<SearchResult[]> {
    // TODO:
    // table.search(query, "fts_text")
    //   .limit(opts.maxResults ?? 20)
    //   .where(agentId filter)
    //   .toArray()
    throw new Error("LanceDBBackend.ftsSearch() not yet implemented");
  }

  async listPaths(agentId?: string): Promise<Array<{ path: string; updatedAt: string; chunkCount: number }>> {
    // TODO: SELECT path, max(updated_at) as updated_at, count(*) as chunk_count GROUP BY path
    throw new Error("LanceDBBackend.listPaths() not yet implemented");
  }

  async getById(id: string): Promise<MemoryChunk | null> {
    // TODO: table.query().where(`id = '${sanitize(id)}'`).limit(1).toArray()
    throw new Error("LanceDBBackend.getById() not yet implemented");
  }

  async readFile(params: { path: string; from?: number; lines?: number; agentId?: string }): Promise<{ text: string; path: string }> {
    // TODO: fetch chunks for path ordered by start_line, reconstruct text
    throw new Error("LanceDBBackend.readFile() not yet implemented");
  }

  async status(): Promise<BackendStatus> {
    // TODO: count rows across all tables
    return {
      backend: "lancedb",
      chunkCount: 0,
      ready: false,
      error: "not yet implemented",
    };
  }
}
