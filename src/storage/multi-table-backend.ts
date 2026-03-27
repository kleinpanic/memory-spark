/**
 * Multi-Table LanceDB Backend — routes operations to appropriate tables.
 *
 * Replaces the single-table LanceDBBackend with a table-per-agent architecture.
 * Each agent gets isolated memory and tool tables. Shared tables hold
 * cross-agent knowledge and mistakes. Reference tables are tool-call-only.
 *
 * Key improvements over single-table:
 * - No FTS WHERE workaround needed (per-agent table = no agent_id filter)
 * - True data isolation between agents
 * - Content-type routing at ingest time
 * - Reference docs separate from auto-recall pool
 *
 * @module storage/multi-table-backend
 */

import * as _lancedb from "@lancedb/lancedb";
import type { Table } from "@lancedb/lancedb";
import type {
  StorageBackend,
  MemoryChunk,
  SearchOptions,
  SearchResult,
  BackendStatus,
} from "./backend.js";
import type { MemorySparkConfig } from "../config.js";
import { TableManager, type TableCategory, type TableNamingConfig } from "./table-manager.js";

/**
 * BM25 sigmoid midpoint — controls where FTS scores map to 0.5.
 * Typical BM25 scores for English text: 0–10, median ~2–4.
 * TODO: Make configurable via config.fts.sigmoidMidpoint
 */
const BM25_SIGMOID_MIDPOINT = 3.0;

/** Result of routing a chunk to the correct table */
export interface RoutingDecision {
  tableName: string;
  category: TableCategory;
}

/**
 * Multi-table LanceDB storage backend.
 *
 * Implements the StorageBackend interface with table-per-agent isolation.
 * The `agentId` in operations determines which table to target.
 * Content type determines whether data goes to memory, tools, or shared tables.
 */
export class MultiTableBackend implements StorageBackend {
  private cfg: MemorySparkConfig;
  private mgr: TableManager;

  constructor(cfg: MemorySparkConfig) {
    this.cfg = cfg;
    this.mgr = new TableManager(cfg.lancedbDir, cfg.tables as Partial<TableNamingConfig>);
  }

  async open(): Promise<void> {
    await this.mgr.open();
  }

  async close(): Promise<void> {
    await this.mgr.close();
  }

  // ── Routing ───────────────────────────────────────────────────────────────

  /**
   * Route a chunk to the correct table based on content_type, source, and path.
   *
   * Routing rules (in priority order):
   * 1. content_type === "tool" → agent_tools
   * 2. path matches MISTAKES.md pattern → shared_mistakes
   * 3. source === "reference" OR content_type === "reference" → reference_library
   * 4. content_type === "reference_code" → reference_code
   * 5. Everything else → agent_memory
   *
   * Shared knowledge routing is handled separately by an explicit API call,
   * not by automatic classification during ingest.
   */
  routeChunk(chunk: MemoryChunk): RoutingDecision {
    const agentId = chunk.agent_id;
    const contentType = chunk.content_type ?? "knowledge";
    const pathLower = (chunk.path ?? "").toLowerCase();
    const basename = pathLower.split("/").pop() ?? "";

    // Tool definitions go to agent-specific tools table
    if (contentType === "tool" || basename === "tools.md" || basename.startsWith("tools-")) {
      return {
        tableName: this.mgr.agentToolsName(agentId),
        category: "agent_tools",
      };
    }

    // Mistakes go to shared mistakes table
    if (
      basename === "mistakes.md" ||
      pathLower.includes("mistakes/") ||
      contentType === "mistake"
    ) {
      return {
        tableName: this.mgr.sharedMistakesName(),
        category: "shared_mistakes",
      };
    }

    // Reference documents (NOT auto-injected)
    if (chunk.source === ("reference" as MemoryChunk["source"]) || contentType === "reference") {
      return {
        tableName: this.mgr.referenceLibraryName(),
        category: "reference_library",
      };
    }

    // Code references
    if (contentType === "reference_code") {
      return {
        tableName: this.mgr.referenceCodeName(),
        category: "reference_code",
      };
    }

    // Default: agent-specific memory
    return {
      tableName: this.mgr.agentMemoryName(agentId),
      category: "agent_memory",
    };
  }

  // ── Write Operations ──────────────────────────────────────────────────────

  async upsert(chunks: MemoryChunk[]): Promise<void> {
    if (chunks.length === 0) return;

    // Group chunks by target table
    const grouped = new Map<string, { chunks: MemoryChunk[]; category: TableCategory }>();
    for (const chunk of chunks) {
      const route = this.routeChunk(chunk);
      const group = grouped.get(route.tableName);
      if (group) {
        group.chunks.push(chunk);
      } else {
        grouped.set(route.tableName, { chunks: [chunk], category: route.category });
      }
    }

    // Write to each table (serialized per-table via write lock)
    const writes = [...grouped.entries()].map(([tableName, { chunks: tableChunks, category }]) =>
      this.mgr.withWriteLock(tableName, async () => {
        const rows = tableChunks.map(chunkToRow);
        const table = await this.getOrCreateTable(tableName, rows, category, tableChunks[0]!.agent_id);

        // Delete existing rows by ID for upsert semantics
        const ids = tableChunks.map((c) => c.id);
        try {
          await table.delete(`id IN (${ids.map((id) => `'${escapeSql(id)}'`).join(",")})`);
        } catch {
          // Table might be empty or IDs don't exist — fine
        }

        await table.add(rows);
      }),
    );

    await Promise.all(writes);
  }

  async deleteByPath(filePath: string, agentId?: string): Promise<number> {
    // Delete from agent's memory table (most common case)
    if (agentId) {
      const tableName = this.mgr.agentMemoryName(agentId);
      return this.deleteFromTable(tableName, `path = '${escapeSql(filePath)}'`);
    }

    // No agentId — delete from all tables (admin operation)
    let total = 0;
    const names = await this.mgr.listTableNames();
    for (const name of names) {
      total += await this.deleteFromTable(name, `path = '${escapeSql(filePath)}'`);
    }
    return total;
  }

  async deleteById(ids: string[]): Promise<void> {
    if (ids.length === 0) return;

    // We don't know which table the IDs belong to — search all open tables
    const names = await this.mgr.listTableNames();
    const idClause = ids.map((id) => `'${escapeSql(id)}'`).join(",");
    for (const name of names) {
      await this.deleteFromTable(name, `id IN (${idClause})`).catch(() => {});
    }
  }

  // ── Read Operations ───────────────────────────────────────────────────────

  /**
   * Vector search — searches the appropriate table(s) based on agentId.
   *
   * When agentId is provided: searches agent's memory table (no WHERE needed!)
   * When agentId is absent: searches all agent memory tables (admin/benchmark mode)
   */
  async vectorSearch(queryVector: number[], opts: SearchOptions): Promise<SearchResult[]> {
    const limit = opts.maxResults ?? 20;
    const minScore = opts.minScore ?? 0;

    // Determine which tables to search
    const tableNames = await this.resolveSearchTables(opts);

    const allResults: SearchResult[] = [];
    for (const tableName of tableNames) {
      const results = await this.vectorSearchTable(tableName, queryVector, limit, minScore, opts);
      allResults.push(...results);
    }

    // Sort by score descending, trim to limit
    allResults.sort((a, b) => b.score - a.score);
    return allResults.slice(0, limit);
  }

  /**
   * FTS search — searches the appropriate table(s).
   *
   * Per-agent tables eliminate the FTS+WHERE panic workaround.
   * No overfetch, no post-filtering needed.
   */
  async ftsSearch(query: string, opts: SearchOptions): Promise<SearchResult[]> {
    const limit = opts.maxResults ?? 20;

    const tableNames = await this.resolveSearchTables(opts);

    const allResults: SearchResult[] = [];
    for (const tableName of tableNames) {
      const results = await this.ftsSearchTable(tableName, query, limit);
      allResults.push(...results);
    }

    allResults.sort((a, b) => b.score - a.score);
    return allResults.slice(0, limit);
  }

  async listPaths(
    agentId?: string,
  ): Promise<Array<{ path: string; agentId: string; updatedAt: string; chunkCount: number }>> {
    const tableName = agentId
      ? this.mgr.agentMemoryName(agentId)
      : null;

    const tables = tableName
      ? [tableName]
      : await this.mgr.listTableNames();

    const groups = new Map<string, { path: string; agentId: string; updatedAt: string; count: number }>();

    for (const name of tables) {
      try {
        const managed = await this.mgr.getTable(name);
        if (!managed?.table) continue;

        const rows = await managed.table
          .query()
          .select(["path", "agent_id", "updated_at"])
          .toArray();

        for (const row of rows) {
          const key = `${row.agent_id}::${row.path}`;
          const existing = groups.get(key);
          if (existing) {
            existing.count++;
            if (row.updated_at > existing.updatedAt) {
              existing.updatedAt = row.updated_at as string;
            }
          } else {
            groups.set(key, {
              path: row.path as string,
              agentId: row.agent_id as string,
              updatedAt: row.updated_at as string,
              count: 1,
            });
          }
        }
      } catch {
        // Table might not exist yet — skip
      }
    }

    return [...groups.values()].map((g) => ({
      path: g.path,
      agentId: g.agentId,
      updatedAt: g.updatedAt,
      chunkCount: g.count,
    }));
  }

  async getById(id: string): Promise<MemoryChunk | null> {
    const names = await this.mgr.listTableNames();
    for (const name of names) {
      try {
        const managed = await this.mgr.getTable(name);
        if (!managed?.table) continue;
        const rows = await managed.table
          .query()
          .where(`id = '${escapeSql(id)}'`)
          .limit(1)
          .toArray();
        if (rows.length > 0) {
          return rowToChunk(rows[0]!);
        }
      } catch {
        continue;
      }
    }
    return null;
  }

  async readFile(params: {
    path: string;
    from?: number;
    lines?: number;
    agentId?: string;
  }): Promise<{ text: string; path: string }> {
    const tableName = params.agentId
      ? this.mgr.agentMemoryName(params.agentId)
      : null;

    const tables = tableName ? [tableName] : await this.mgr.listTableNames();

    for (const name of tables) {
      try {
        const managed = await this.mgr.getTable(name);
        if (!managed?.table) continue;

        const q = managed.table
          .query()
          .where(`path = '${escapeSql(params.path)}'`)
          .select(["text", "start_line", "end_line", "path"]);

        const rows = await q.toArray();
        if (rows.length === 0) continue;

        rows.sort((a, b) => (a.start_line as number) - (b.start_line as number));

        let text = rows.map((r) => r.text).join("\n\n");
        if (params.from || params.lines) {
          const lines = text.split("\n");
          const start = (params.from ?? 1) - 1;
          const end = params.lines ? start + params.lines : lines.length;
          text = lines.slice(start, end).join("\n");
        }

        return { text, path: params.path };
      } catch {
        continue;
      }
    }

    return { text: "", path: params.path };
  }

  async status(): Promise<BackendStatus> {
    const tableStatuses = await this.mgr.status();
    const totalRows = tableStatuses.reduce((sum, t) => sum + t.rowCount, 0);

    return {
      backend: "lancedb",
      chunkCount: totalRows,
      tableExists: tableStatuses.length > 0,
      ready: this.mgr.getConnection() !== null,
    };
  }

  // ── Extended API (multi-table specific) ───────────────────────────────────

  /** Get the underlying TableManager for advanced operations */
  getTableManager(): TableManager {
    return this.mgr;
  }

  /**
   * Search ONLY reference tables (for memory_reference_search tool).
   * This is the dedicated path for reference lookup — never auto-injected.
   */
  async referenceSearch(
    queryVector: number[],
    query: string,
    opts: { maxResults?: number; minScore?: number },
  ): Promise<SearchResult[]> {
    const limit = opts.maxResults ?? 10;
    const minScore = opts.minScore ?? 0;
    const tables = [this.mgr.referenceLibraryName(), this.mgr.referenceCodeName()];

    const allResults: SearchResult[] = [];
    for (const tableName of tables) {
      if (!(await this.mgr.tableExists(tableName))) continue;

      const vectorResults = await this.vectorSearchTable(tableName, queryVector, limit, minScore, { query });
      const ftsResults = await this.ftsSearchTable(tableName, query, limit);
      allResults.push(...vectorResults, ...ftsResults);
    }

    // Dedup by ID, keep highest score
    const deduped = new Map<string, SearchResult>();
    for (const r of allResults) {
      const existing = deduped.get(r.chunk.id);
      if (!existing || r.score > existing.score) {
        deduped.set(r.chunk.id, r);
      }
    }

    return [...deduped.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  /**
   * Search ONLY shared tables (knowledge + mistakes).
   * Used during auto-recall to supplement agent-specific results.
   */
  async sharedSearch(
    queryVector: number[],
    query: string,
    opts: { maxResults?: number; minScore?: number },
  ): Promise<SearchResult[]> {
    const limit = opts.maxResults ?? 10;
    const minScore = opts.minScore ?? 0;
    const tables = [this.mgr.sharedKnowledgeName(), this.mgr.sharedMistakesName()];

    const allResults: SearchResult[] = [];
    for (const tableName of tables) {
      if (!(await this.mgr.tableExists(tableName))) continue;

      const vectorResults = await this.vectorSearchTable(tableName, queryVector, limit, minScore, { query });
      const ftsResults = await this.ftsSearchTable(tableName, query, limit);
      allResults.push(...vectorResults, ...ftsResults);
    }

    const deduped = new Map<string, SearchResult>();
    for (const r of allResults) {
      const existing = deduped.get(r.chunk.id);
      if (!existing || r.score > existing.score) {
        deduped.set(r.chunk.id, r);
      }
    }

    return [...deduped.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  /** Store a chunk explicitly in the shared knowledge table */
  async storeSharedKnowledge(chunks: MemoryChunk[]): Promise<void> {
    const tableName = this.mgr.sharedKnowledgeName();
    await this.mgr.withWriteLock(tableName, async () => {
      const rows = chunks.map(chunkToRow);
      const table = await this.getOrCreateTable(tableName, rows, "shared_knowledge");
      await table.add(rows);
    });
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  /**
   * Resolve which tables to search based on the query options.
   *
   * - If agentId is set: search that agent's memory table
   * - If contentType is "reference": search reference tables
   * - If contentType is "tool": search agent's tools table
   * - Default (no filters): search the calling agent's memory table
   */
  private async resolveSearchTables(opts: SearchOptions): Promise<string[]> {
    if (opts.contentType === "reference") {
      return [this.mgr.referenceLibraryName(), this.mgr.referenceCodeName()];
    }

    if (opts.agentId && opts.contentType === "tool") {
      return [this.mgr.agentToolsName(opts.agentId)];
    }

    if (opts.agentId) {
      return [this.mgr.agentMemoryName(opts.agentId)];
    }

    // No agentId — search all agent memory tables (global search)
    const agents = await this.mgr.discoverAgents();
    return agents.map((a) => this.mgr.agentMemoryName(a));
  }

  private async vectorSearchTable(
    tableName: string,
    queryVector: number[],
    limit: number,
    minScore: number,
    opts: Partial<SearchOptions>,
  ): Promise<SearchResult[]> {
    try {
      const managed = await this.mgr.getTable(tableName);
      if (!managed?.table) return [];

      const q = managed.table.search(queryVector).limit(limit);

      // Apply filters that are safe for vector search (no FTS conflict)
      if (opts.source) {
        q.where(`source = '${escapeSql(opts.source)}'`);
      }
      if (opts.contentType && opts.contentType !== "reference" && opts.contentType !== "tool") {
        q.where(`content_type = '${escapeSql(opts.contentType)}'`);
      }
      if (opts.pathContains) {
        // LanceDB supports LIKE for vector search WHERE
        q.where(`path LIKE '%${escapeSql(opts.pathContains.toLowerCase())}%'`);
      }

      const rows = await q.toArray();
      return rows
        .map(rowToSearchResult)
        .filter((r) => r.score >= minScore);
    } catch {
      return [];
    }
  }

  private async ftsSearchTable(
    tableName: string,
    query: string,
    limit: number,
  ): Promise<SearchResult[]> {
    try {
      const managed = await this.mgr.getTable(tableName);
      if (!managed?.table) return [];

      // No WHERE clause needed — table already scoped to the right data.
      // This is the proper fix for the FTS+WHERE Arrow panic.
      const q = managed.table.search(query, "fts", "text").limit(limit);
      const rows = await q.toArray();
      return rows.map(rowToSearchResult);
    } catch {
      return [];
    }
  }

  private async getOrCreateTable(
    tableName: string,
    rows: Record<string, unknown>[],
    category: TableCategory,
    agentId?: string,
  ): Promise<Table> {
    const managed = await this.mgr.getTable(tableName);
    if (managed?.table) return managed.table;

    // Table doesn't exist — create with initial data
    return this.mgr.createTableWithData(tableName, rows, category, agentId);
  }

  private async deleteFromTable(tableName: string, whereClause: string): Promise<number> {
    try {
      const managed = await this.mgr.getTable(tableName);
      if (!managed?.table) return 0;

      const before = await managed.table.countRows();
      await managed.table.delete(whereClause);
      const after = await managed.table.countRows();
      return before - after;
    } catch {
      return 0;
    }
  }
}

// ── Row conversion utilities ────────────────────────────────────────────────

function chunkToRow(chunk: MemoryChunk): Record<string, unknown> {
  return {
    id: chunk.id,
    text: chunk.text,
    vector: chunk.vector,
    path: chunk.path,
    source: chunk.source,
    agent_id: chunk.agent_id,
    user_id: chunk.user_id ?? "",
    start_line: chunk.start_line,
    end_line: chunk.end_line,
    updated_at: chunk.updated_at,
    category: chunk.category ?? "",
    entities: chunk.entities ?? "[]",
    confidence: chunk.confidence ?? 0,
    content_type: chunk.content_type ?? "knowledge",
    quality_score: chunk.quality_score ?? 0,
    token_count: chunk.token_count ?? 0,
    parent_heading: chunk.parent_heading ?? "",
  };
}

function rowToChunk(row: Record<string, unknown>): MemoryChunk {
  return {
    id: row.id as string,
    text: row.text as string,
    vector: (row.vector as number[]) ?? [],
    path: row.path as string,
    source: (row.source as MemoryChunk["source"]) ?? "memory",
    agent_id: row.agent_id as string,
    user_id: (row.user_id as string) || undefined,
    start_line: (row.start_line as number) ?? 0,
    end_line: (row.end_line as number) ?? 0,
    updated_at: row.updated_at as string,
    category: (row.category as string) || undefined,
    entities: (row.entities as string) || undefined,
    confidence: (row.confidence as number) || undefined,
    content_type: (row.content_type as string) || undefined,
    quality_score: (row.quality_score as number) || undefined,
    token_count: (row.token_count as number) || undefined,
    parent_heading: (row.parent_heading as string) || undefined,
  };
}

/**
 * Convert a raw LanceDB row to a SearchResult.
 *
 * Score handling:
 *   - Vector search: `_distance` (cosine) → `1 - distance`, clamped [0, 1]
 *   - FTS search: `_score` (BM25) → normalized via sigmoid
 *   - Neither: score 0 (unknown)
 */
function rowToSearchResult(row: Record<string, unknown>): SearchResult {
  let score: number;
  const distance = row._distance as number | undefined;
  const ftsScore = row._score as number | undefined;

  if (distance != null && !Number.isNaN(distance)) {
    score = Math.max(0, 1 - distance);
  } else if (ftsScore != null && !Number.isNaN(ftsScore)) {
    score = 1 / (1 + Math.exp(-(ftsScore - BM25_SIGMOID_MIDPOINT)));
  } else {
    score = 0;
  }

  const chunk = rowToChunk(row);
  return {
    chunk,
    score,
    snippet: (chunk.text ?? "").slice(0, 200),
  };
}

function escapeSql(s: string): string {
  return s.replace(/'/g, "''");
}
