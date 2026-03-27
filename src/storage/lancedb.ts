/**
 * LanceDB Storage Backend — primary backend for memory-spark.
 */

import * as lancedb from "@lancedb/lancedb";
import type { Table } from "@lancedb/lancedb";
import type {
  StorageBackend,
  MemoryChunk,
  SearchOptions,
  SearchResult,
  BackendStatus,
} from "./backend.js";
import type { MemorySparkConfig } from "../config.js";
import fs from "node:fs/promises";

const TABLE_NAME = "memory_chunks";

export class LanceDBBackend implements StorageBackend {
  private cfg: MemorySparkConfig;
  private db: lancedb.Connection | null = null;
  private table: Table | null = null;
  private ftsCreated = false;

  // Write mutex — serializes all upsert/delete calls to prevent LanceDB commit conflicts
  // when the boot scanner and file watcher run concurrently at startup.
  private writeLock: Promise<void> = Promise.resolve();
  private withWriteLock<T>(fn: () => Promise<T>): Promise<T> {
    const next = this.writeLock.then(() => fn());
    // Settle the lock chain even if fn throws, so subsequent writes aren't blocked forever
    this.writeLock = next.then(
      () => {},
      () => {},
    );
    return next;
  }

  constructor(cfg: MemorySparkConfig) {
    this.cfg = cfg;
  }

  async open(): Promise<void> {
    await fs.mkdir(this.cfg.lancedbDir, { recursive: true });
    this.db = await lancedb.connect(this.cfg.lancedbDir);

    const names = await this.db.tableNames();
    if (names.includes(TABLE_NAME)) {
      this.table = await this.db.openTable(TABLE_NAME);
      // Schema evolution: add new columns if they don't exist yet
      await this.ensureSchema();
      // Ensure indexes are created for fast ANN + FTS
      await this.ensureIndexes();
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

  /**
   * Track whether the table schema includes the new columns.
   * When true, upsert includes content_type, quality_score, token_count, parent_heading.
   * When false, upsert skips them (old table, pre-overhaul).
   */
  private schemaHasNewColumns = false;

  /**
   * Detect whether the table has the new columns from the RAG overhaul.
   *
   * We do NOT use addColumns() to migrate old tables — it creates columns with
   * different Arrow nullability than seed-record columns, causing schema errors
   * on mergeInsert. Instead:
   * - Old tables: skip new fields in upsert (use scripts/rebuild-table.ts to migrate)
   * - New tables (seed record): include all fields, schema is consistent
   */
  private async ensureSchema(): Promise<void> {
    if (!this.table) return;
    try {
      const schema = await this.table.schema();
      const existingFields = new Set(schema.fields.map((f) => f.name));
      const needed = ["content_type", "quality_score", "token_count", "parent_heading"];
      this.schemaHasNewColumns = needed.every((n) => existingFields.has(n));
    } catch {
      this.schemaHasNewColumns = false;
    }
  }

  /**
   * Create IVF_PQ vector index and FTS index if they don't exist.
   * Wrapped in try/catch — index creation failure doesn't prevent startup.
   */
  private async ensureIndexes(): Promise<void> {
    if (!this.table) return;
    try {
      const indices = await this.table.listIndices();
      const hasVectorIndex = indices.some((i) => i.columns.includes("vector"));
      const hasFtsIndex = indices.some((i) => i.columns.includes("text"));

      if (!hasVectorIndex) {
        try {
          // numSubVectors: 64 → 4096/64 = 64 dims per subvector (SIMD-friendly multiple of 8)
          await this.table.createIndex("vector", {
            config: lancedb.Index.ivfPq({
              numPartitions: 10,
              numSubVectors: 64,
              distanceType: "cosine",
            }),
          });
        } catch {
          // Vector index creation may fail if table is too small (needs ≥ numPartitions rows)
        }
      }

      if (!hasFtsIndex) {
        try {
          await this.table.createIndex("text", { config: lancedb.Index.fts() });
        } catch {
          // FTS index creation failure is non-fatal
        }
      }

      // Mark FTS as ready regardless — ftsSearch() has its own fallback
      this.ftsCreated = true;
    } catch {
      // listIndices() failure is non-fatal
    }
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
      await this.ensureSchema();
      return this.table;
    }

    // Create table with a seed record that includes ALL fields (schema-defining).
    // LanceDB locks schema on creation — missing fields cause append errors.
    const seed = {
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
      content_type: "knowledge",
      quality_score: 0.5,
      token_count: 0,
      parent_heading: "",
    };
    this.table = await this.db.createTable(TABLE_NAME, [
      seed as unknown as Record<string, unknown>,
    ]);
    await this.table.delete("id = '__seed__'");
    // Fresh table has all columns from seed record — mark them as available
    this.schemaHasNewColumns = true;
    return this.table;
  }

  async upsert(chunks: MemoryChunk[]): Promise<void> {
    if (chunks.length === 0) return;
    return this.withWriteLock(() => this._upsert(chunks));
  }

  private async _upsert(chunks: MemoryChunk[]): Promise<void> {
    const dims = chunks[0]!.vector.length;
    const table = await this.ensureTable(dims);

    // Normalize: ensure ALL schema fields are present (LanceDB rejects mismatched schemas).
    // If the table was created before schema evolution (addColumns path), the new columns
    // are nullable. Including non-nullable values in mergeInsert causes Arrow schema errors.
    // In that case, omit the new fields — existing rows keep their defaults.
    const includeNewCols = this.schemaHasNewColumns;
    const normalized = chunks.map((c) => {
      const base: Record<string, unknown> = {
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
      };
      if (includeNewCols) {
        base.content_type = c.content_type ?? "knowledge";
        base.quality_score = c.quality_score ?? 0.5;
        base.token_count = c.token_count ?? 0;
        base.parent_heading = c.parent_heading ?? "";
      }
      return base;
    });

    // Retry on commit conflict (concurrent writes from boot pass + watcher)
    const MAX_RETRIES = 3;
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        await table
          .mergeInsert("id")
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
    return this.withWriteLock(async () => {
      if (!this.table) return 0;
      const before = await this.table.countRows();
      let predicate = `path = '${escapeSql(pathStr)}'`;
      if (agentId) predicate += ` AND agent_id = '${escapeSql(agentId)}'`;
      await this.table.delete(predicate);
      const after = await this.table.countRows();
      return before - after;
    });
  }

  async deleteById(ids: string[]): Promise<void> {
    return this.withWriteLock(async () => {
      if (!this.table || ids.length === 0) return;
      const inList = ids.map((id) => `'${escapeSql(id)}'`).join(",");
      await this.table.delete(`id IN (${inList})`);
    });
  }

  async vectorSearch(queryVector: number[], opts: SearchOptions): Promise<SearchResult[]> {
    if (!this.table) return [];
    const limit = opts.maxResults ?? 20;

    let q = this.table
      .vectorSearch(queryVector)
      .distanceType("cosine")
      .refineFactor(20)
      .limit(limit);

    const filters: string[] = [];
    if (opts.agentId) filters.push(`agent_id = '${escapeSql(opts.agentId)}'`);
    if (opts.source) filters.push(`source = '${escapeSql(opts.source)}'`);
    if (opts.contentType) filters.push(`content_type = '${escapeSql(opts.contentType)}'`);
    // pathContains: LanceDB supports LIKE in WHERE on vector search (unlike FTS)
    if (opts.pathContains) {
      filters.push(`path LIKE '%${escapeSql(opts.pathContains)}%'`);
    }
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

    // Ensure FTS index exists (fallback for newly-created tables)
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
      // LanceDB bug: FTS + .where() causes Arrow cast panic (ExecNode(Take)).
      // Workaround: fetch more results without .where(), then post-filter in JS.
      // This is safe because FTS is a secondary ranking signal, not a primary filter.
      const overFetch = limit * 3; // Fetch extra to compensate for post-filtering
      const q = this.table.search(query, "fts", "text").limit(overFetch);
      let rows = await q.toArray();

      // Post-filter (avoids LanceDB WHERE bug on FTS queries)
      if (opts.agentId) {
        rows = rows.filter((r: Record<string, unknown>) => r["agent_id"] === opts.agentId);
      }
      if (opts.source) {
        rows = rows.filter((r: Record<string, unknown>) => r["source"] === opts.source);
      }
      if (opts.contentType) {
        rows = rows.filter((r: Record<string, unknown>) => r["content_type"] === opts.contentType);
      }
      if (opts.pathContains) {
        const needle = opts.pathContains.toLowerCase();
        rows = rows.filter((r: Record<string, unknown>) =>
          typeof r["path"] === "string" && (r["path"] as string).toLowerCase().includes(needle),
        );
      }

      return rows.slice(0, limit).map(rowToSearchResult);
    } catch {
      // FTS search failed — return empty (non-fatal)
      return [];
    }
  }

  async listPaths(
    agentId?: string,
  ): Promise<Array<{ path: string; agentId: string; updatedAt: string; chunkCount: number }>> {
    if (!this.table) return [];

    let q = this.table.query().select(["path", "agent_id", "updated_at"]);
    if (agentId) {
      q = q.where(`agent_id = '${escapeSql(agentId)}'`);
    }
    const rows = await q.toArray();

    // Group by agentId::path to avoid cross-agent collisions on same relative paths
    const groups = new Map<
      string,
      { path: string; agentId: string; updatedAt: string; count: number }
    >();
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
    const rows = await this.table
      .query()
      .where(`id = '${escapeSql(id)}'`)
      .limit(1)
      .toArray();
    return rows.length > 0 ? (rows[0] as MemoryChunk) : null;
  }

  async readFile(params: {
    path: string;
    from?: number;
    lines?: number;
    agentId?: string;
  }): Promise<{ text: string; path: string }> {
    if (!this.table) return { text: "", path: params.path };

    let q = this.table
      .query()
      .where(`path = '${escapeSql(params.path)}'`)
      .select(["text", "start_line", "end_line"]);

    if (params.agentId) {
      q = q.where(`agent_id = '${escapeSql(params.agentId)}'`);
    }

    const rows = await q.toArray();
    // Sort by start_line and reconstruct
    rows.sort(
      (a: Record<string, unknown>, b: Record<string, unknown>) =>
        (a.start_line as number) - (b.start_line as number),
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
      return { backend: "lancedb", chunkCount: 0, tableExists: false, ready: !!this.db };
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

  /**
   * Aggregate statistics for the memory_index_status tool.
   * Not part of StorageBackend interface — access via cast.
   */
  async getStats(agentId?: string): Promise<{
    totalChunks: number;
    indices: Array<{ name: string; indexType: string; columns: string[] }>;
    topPaths: Array<{ path: string; chunkCount: number }>;
  }> {
    if (!this.table) return { totalChunks: 0, indices: [], topPaths: [] };
    try {
      const filter = agentId ? `agent_id = '${escapeSql(agentId)}'` : undefined;
      const [totalChunks, indices, paths] = await Promise.all([
        this.table.countRows(filter),
        this.table
          .listIndices()
          .catch(() => [] as Array<{ name: string; indexType: string; columns: string[] }>),
        this.listPaths(agentId),
      ]);
      const topPaths = paths
        .sort((a, b) => b.chunkCount - a.chunkCount)
        .slice(0, 10)
        .map((p) => ({ path: p.path, chunkCount: p.chunkCount }));
      return { totalChunks, indices, topPaths };
    } catch {
      return { totalChunks: 0, indices: [], topPaths: [] };
    }
  }
}

function escapeSql(s: string): string {
  return s.replace(/'/g, "''");
}

/**
 * Convert a raw LanceDB row to a SearchResult.
 *
 * Score handling:
 *   - Vector search rows have `_distance` (cosine distance, lower = closer).
 *     Score = 1 - distance, clamped to [0, 1].
 *   - FTS search rows have `_score` (BM25 relevance, higher = better, unbounded).
 *     We normalize `_score` using 1/(1 + exp(-score + 3)) to map to ~[0, 1].
 *     This gives BM25=0 → ~0.05, BM25=3 → ~0.50, BM25=6 → ~0.95.
 *   - If neither is present, score defaults to 0 (unknown relevance).
 *
 * Bug fix (2026-03-27): Previously always used `_distance` conversion.
 * FTS rows don't carry `_distance`, so they collapsed to score 1.0
 * (because 1 - 0 = 1.0), corrupting hybrid merge ranking.
 */
function rowToSearchResult(row: Record<string, unknown>): SearchResult {
  let score: number;
  const distance = row._distance as number | undefined;
  const ftsScore = row._score as number | undefined;

  if (distance != null && !Number.isNaN(distance)) {
    // Vector search: cosine distance → similarity
    score = Math.max(0, 1 - distance);
  } else if (ftsScore != null && !Number.isNaN(ftsScore)) {
    // FTS search: BM25 score → normalized [0, 1] via sigmoid
    score = 1 / (1 + Math.exp(-(ftsScore - 3)));
  } else {
    // Neither present — unknown source, zero score
    score = 0;
  }

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
    content_type: (row.content_type as string | undefined) ?? "knowledge",
    quality_score: row.quality_score as number | undefined,
    token_count: row.token_count as number | undefined,
    parent_heading: row.parent_heading as string | undefined,
  };

  return {
    chunk,
    score,
    snippet: chunk.text.slice(0, 500),
  };
}
