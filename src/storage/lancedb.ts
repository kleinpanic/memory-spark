/**
 * LanceDB Storage Backend — primary backend for memory-spark.
 */

import * as lancedb from "@lancedb/lancedb";
import type { Table } from "@lancedb/lancedb";
import type {
  StorageBackend, MemoryChunk, SearchOptions, SearchResult, BackendStatus,
} from "./backend.js";
import type { MemorySparkConfig } from "../config.js";
import fs from "node:fs/promises";

const TABLE_NAME = "memory_chunks";

export class LanceDBBackend implements StorageBackend {
  private cfg: MemorySparkConfig;
  private db: lancedb.Connection | null = null;
  private table: Table | null = null;
  private ftsCreated = false;

  constructor(cfg: MemorySparkConfig) {
    this.cfg = cfg;
  }

  async open(): Promise<void> {
    await fs.mkdir(this.cfg.lancedbDir, { recursive: true });
    this.db = await lancedb.connect(this.cfg.lancedbDir);

    const names = await this.db.tableNames();
    if (names.includes(TABLE_NAME)) {
      this.table = await this.db.openTable(TABLE_NAME);
    }
    // Table is created on first upsert when we know the vector dims
  }

  async close(): Promise<void> {
    if (this.table) {
      this.table.close();
      this.table = null;
    }
    this.db = null;
  }

  private ensureTablePromise: Promise<Table> | null = null;

  private async ensureTable(dims: number): Promise<Table> {
    if (this.table) return this.table;
    // Mutex: only one createTable attempt at a time
    if (this.ensureTablePromise) return this.ensureTablePromise;

    this.ensureTablePromise = this._createTable(dims);
    try {
      return await this.ensureTablePromise;
    } finally {
      this.ensureTablePromise = null;
    }
  }

  private async _createTable(dims: number): Promise<Table> {
    if (!this.db) throw new Error("LanceDB not connected");

    // Check if table was created between our check and now
    const names = await this.db.tableNames();
    if (names.includes(TABLE_NAME)) {
      this.table = await this.db.openTable(TABLE_NAME);
      return this.table;
    }

    // Create table with a seed record that includes ALL fields (schema-defining).
    // LanceDB locks schema on creation — missing fields cause append errors.
    const seed: MemoryChunk = {
      id: "__seed__",
      path: "__seed__",
      source: "memory",
      agent_id: "__seed__",
      start_line: 0,
      end_line: 0,
      text: "",
      vector: new Array(dims).fill(0),
      updated_at: new Date().toISOString(),
      category: "",
      entities: "[]",
      confidence: 0,
    };
    this.table = await this.db.createTable(TABLE_NAME, [seed as unknown as Record<string, unknown>]);
    await this.table.delete("id = '__seed__'");
    return this.table;
  }

  async upsert(chunks: MemoryChunk[]): Promise<void> {
    if (chunks.length === 0) return;
    const dims = chunks[0]!.vector.length;
    const table = await this.ensureTable(dims);

    // Normalize: ensure ALL schema fields are present (LanceDB rejects mismatched schemas)
    const normalized = chunks.map((c) => ({
      id: c.id,
      path: c.path,
      source: c.source,
      agent_id: c.agent_id,
      start_line: c.start_line,
      end_line: c.end_line,
      text: c.text,
      vector: c.vector,
      updated_at: c.updated_at,
      category: c.category ?? "",
      entities: c.entities ?? "[]",
      confidence: c.confidence ?? 0,
    }));

    // Retry on commit conflict (concurrent writes from boot pass + watcher)
    const MAX_RETRIES = 3;
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        await table.mergeInsert("id")
          .whenMatchedUpdateAll()
          .whenNotMatchedInsertAll()
          .execute(normalized as unknown as Record<string, unknown>[]);
        return;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("Commit conflict") && attempt < MAX_RETRIES - 1) {
          await new Promise((r) => setTimeout(r, 200 * (attempt + 1)));
          continue;
        }
        throw err;
      }
    }
  }

  async deleteByPath(pathStr: string, agentId?: string): Promise<number> {
    if (!this.table) return 0;
    const before = await this.table.countRows();
    let predicate = `path = '${escapeSql(pathStr)}'`;
    if (agentId) predicate += ` AND agent_id = '${escapeSql(agentId)}'`;
    await this.table.delete(predicate);
    const after = await this.table.countRows();
    return before - after;
  }

  async deleteById(ids: string[]): Promise<void> {
    if (!this.table || ids.length === 0) return;
    const inList = ids.map((id) => `'${escapeSql(id)}'`).join(",");
    await this.table.delete(`id IN (${inList})`);
  }

  async vectorSearch(queryVector: number[], opts: SearchOptions): Promise<SearchResult[]> {
    if (!this.table) return [];
    const limit = opts.maxResults ?? 20;

    let q = this.table.vectorSearch(queryVector)
      .distanceType("cosine")
      .limit(limit);

    const filters: string[] = [];
    if (opts.agentId) filters.push(`agent_id = '${escapeSql(opts.agentId)}'`);
    if (opts.source) filters.push(`source = '${escapeSql(opts.source)}'`);
    if (filters.length > 0) {
      q = q.where(filters.join(" AND "));
    }

    const rows = await q.toArray();
    return rows.map(rowToSearchResult).filter((r) => {
      if (opts.minScore && r.score < opts.minScore) return false;
      return true;
    });
  }

  async ftsSearch(query: string, opts: SearchOptions): Promise<SearchResult[]> {
    if (!this.table) return [];

    // Ensure FTS index exists
    if (!this.ftsCreated) {
      try {
        await this.table.createIndex("text", { config: lancedb.Index.fts() });
        this.ftsCreated = true;
      } catch {
        // Index might already exist or FTS might not be available
        this.ftsCreated = true;
      }
    }

    const limit = opts.maxResults ?? 20;
    try {
      let q = this.table.search(query, "fts", "text").limit(limit);
      const filters: string[] = [];
      if (opts.agentId) filters.push(`agent_id = '${escapeSql(opts.agentId)}'`);
      if (opts.source) filters.push(`source = '${escapeSql(opts.source)}'`);
      if (filters.length > 0) {
        q = q.where(filters.join(" AND "));
      }
      const rows = await q.toArray();
      return rows.map(rowToSearchResult);
    } catch {
      // FTS search failed — return empty (non-fatal)
      return [];
    }
  }

  async listPaths(agentId?: string): Promise<Array<{ path: string; agentId: string; updatedAt: string; chunkCount: number }>> {
    if (!this.table) return [];

    let q = this.table.query().select(["path", "agent_id", "updated_at"]);
    if (agentId) {
      q = q.where(`agent_id = '${escapeSql(agentId)}'`);
    }
    const rows = await q.toArray();

    // Group by agentId::path to avoid cross-agent collisions on same relative paths
    const groups = new Map<string, { path: string; agentId: string; updatedAt: string; count: number }>();
    for (const row of rows) {
      const key = `${row.agent_id}::${row.path}`;
      const existing = groups.get(key);
      if (!existing || row.updated_at > existing.updatedAt) {
        groups.set(key, {
          path: row.path,
          agentId: row.agent_id,
          updatedAt: row.updated_at,
          count: (existing?.count ?? 0) + 1,
        });
      } else {
        existing.count++;
      }
    }

    return Array.from(groups.values()).map((v) => ({
      path: v.path,
      agentId: v.agentId,
      updatedAt: v.updatedAt,
      chunkCount: v.count,
    }));
  }

  async getById(id: string): Promise<MemoryChunk | null> {
    if (!this.table) return null;
    const rows = await this.table.query()
      .where(`id = '${escapeSql(id)}'`)
      .limit(1)
      .toArray();
    return rows.length > 0 ? (rows[0] as MemoryChunk) : null;
  }

  async readFile(params: { path: string; from?: number; lines?: number; agentId?: string }): Promise<{ text: string; path: string }> {
    if (!this.table) return { text: "", path: params.path };

    let q = this.table.query()
      .where(`path = '${escapeSql(params.path)}'`)
      .select(["text", "start_line", "end_line"]);

    if (params.agentId) {
      q = q.where(`agent_id = '${escapeSql(params.agentId)}'`);
    }

    const rows = await q.toArray();
    // Sort by start_line and reconstruct
    rows.sort((a: Record<string, unknown>, b: Record<string, unknown>) =>
      (a.start_line as number) - (b.start_line as number)
    );

    let text = rows.map((r: Record<string, unknown>) => r.text as string).join("\n");

    if (params.from !== undefined) {
      const lines = text.split("\n");
      const start = Math.max(0, (params.from ?? 1) - 1);
      const count = params.lines ?? 50;
      text = lines.slice(start, start + count).join("\n");
    }

    return { text, path: params.path };
  }

  async status(): Promise<BackendStatus> {
    if (!this.table) {
      return { backend: "lancedb", chunkCount: 0, tableExists: false, ready: !!this.db, };
    }
    try {
      const count = await this.table.countRows();
      return {
        backend: "lancedb",
        chunkCount: count,
        tableExists: true,
        ready: true,
      };
    } catch (err) {
      return {
        backend: "lancedb",
        chunkCount: 0,
        tableExists: true,
        ready: false,
        error: String(err),
      };
    }
  }
}

function escapeSql(s: string): string {
  return s.replace(/'/g, "''");
}

function rowToSearchResult(row: Record<string, unknown>): SearchResult {
  // LanceDB returns _distance for vector search (lower = closer)
  // Convert cosine distance to similarity score (0-1)
  const distance = (row._distance as number) ?? 0;
  const score = Math.max(0, 1 - distance);

  const chunk: MemoryChunk = {
    id: row.id as string,
    path: row.path as string,
    source: (row.source as MemoryChunk["source"]) ?? "memory",
    agent_id: (row.agent_id as string) ?? "",
    start_line: (row.start_line as number) ?? 0,
    end_line: (row.end_line as number) ?? 0,
    text: (row.text as string) ?? "",
    vector: (row.vector as number[]) ?? [],
    updated_at: (row.updated_at as string) ?? "",
    category: row.category as string | undefined,
    entities: row.entities as string | undefined,
    confidence: row.confidence as number | undefined,
  };

  return {
    chunk,
    score,
    snippet: chunk.text.slice(0, 500),
  };
}
