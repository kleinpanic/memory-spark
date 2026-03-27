/**
 * Multi-Table Manager for LanceDB.
 *
 * Manages the lifecycle of multiple LanceDB tables:
 *   - Per-agent tables: agent_{id}_memory, agent_{id}_tools
 *   - Shared tables: shared_knowledge, shared_mistakes
 *   - Reference tables: reference_library, reference_code
 *
 * Each table has its own vector index and FTS index.
 * Per-agent tables eliminate the need for WHERE agent_id filters on FTS
 * (which panic in LanceDB due to Arrow cast bug).
 *
 * @module storage/table-manager
 */

import * as lancedb from "@lancedb/lancedb";
import type { Table } from "@lancedb/lancedb";
import fs from "node:fs/promises";

/** Table categories determine behavior and access patterns */
export type TableCategory =
  | "agent_memory"     // Per-agent workspace files, daily notes, captures
  | "agent_tools"      // Per-agent TOOLS.md and tool schemas
  | "shared_knowledge" // Cross-agent facts, infrastructure docs
  | "shared_mistakes"  // MISTAKES.md entries from all agents
  | "reference_library" // PDFs, documentation (NOT auto-injected)
  | "reference_code";   // Code snippets, examples (NOT auto-injected)

/** Configuration for how tables are named */
export interface TableNamingConfig {
  /** Prefix for per-agent tables (default: "agent_") */
  agentPrefix: string;
  /** Suffix for agent memory tables (default: "_memory") */
  memorySuffix: string;
  /** Suffix for agent tool tables (default: "_tools") */
  toolsSuffix: string;
  /** Name for shared knowledge table (default: "shared_knowledge") */
  sharedKnowledge: string;
  /** Name for shared mistakes table (default: "shared_mistakes") */
  sharedMistakes: string;
  /** Name for reference library table (default: "reference_library") */
  referenceLibrary: string;
  /** Name for reference code table (default: "reference_code") */
  referenceCode: string;
}

export const DEFAULT_TABLE_NAMING: TableNamingConfig = {
  agentPrefix: "agent_",
  memorySuffix: "_memory",
  toolsSuffix: "_tools",
  sharedKnowledge: "shared_knowledge",
  sharedMistakes: "shared_mistakes",
  referenceLibrary: "reference_library",
  referenceCode: "reference_code",
};

/** Metadata tracked per open table */
interface ManagedTable {
  /** The LanceDB table handle. Null when table hasn't been created yet (lazy init). */
  table: Table | null;
  category: TableCategory;
  agentId?: string;
  ftsCreated: boolean;
  hasNewSchema: boolean;
  rowCount: number;
}

/**
 * Manages multiple LanceDB tables with proper lifecycle, indexing, and naming.
 *
 * Usage:
 * ```ts
 * const mgr = new TableManager(lancedbDir, naming);
 * await mgr.open();
 *
 * // Get or create per-agent table
 * const agentTable = await mgr.getAgentMemory("meta");
 *
 * // Get shared table
 * const mistakes = await mgr.getSharedMistakes();
 *
 * // Search across multiple tables
 * const results = await mgr.searchAcross(["meta_memory", "shared_knowledge"], ...);
 *
 * await mgr.close();
 * ```
 */
export class TableManager {
  private db: lancedb.Connection | null = null;
  private tables = new Map<string, ManagedTable>();
  private naming: TableNamingConfig;
  private lancedbDir: string;

  /** Write mutex per table — prevents concurrent upsert/delete on same table */
  private writeLocks = new Map<string, Promise<void>>();

  constructor(lancedbDir: string, naming?: Partial<TableNamingConfig>) {
    this.lancedbDir = lancedbDir;
    this.naming = { ...DEFAULT_TABLE_NAMING, ...naming };
  }

  /** Open the LanceDB connection (does not open any tables yet) */
  async open(): Promise<void> {
    await fs.mkdir(this.lancedbDir, { recursive: true });
    this.db = await lancedb.connect(this.lancedbDir);
  }

  /** Close all open tables and the database connection */
  async close(): Promise<void> {
    for (const [, managed] of this.tables) {
      if (managed.table) managed.table.close();
    }
    this.tables.clear();
    this.writeLocks.clear();
    this.db = null;
  }

  /** List all table names currently in the database */
  async listTableNames(): Promise<string[]> {
    if (!this.db) throw new Error("TableManager not open");
    return this.db.tableNames();
  }

  /** Discover which agents have tables in the database */
  async discoverAgents(): Promise<string[]> {
    const names = await this.listTableNames();
    const agents = new Set<string>();
    const prefix = this.naming.agentPrefix;
    const suffix = this.naming.memorySuffix;
    for (const name of names) {
      if (name.startsWith(prefix) && name.endsWith(suffix)) {
        const agentId = name.slice(prefix.length, -suffix.length);
        if (agentId) agents.add(agentId);
      }
    }
    return [...agents].sort();
  }

  // ── Table name resolution ─────────────────────────────────────────────────

  /** Get the table name for an agent's memory */
  agentMemoryName(agentId: string): string {
    return `${this.naming.agentPrefix}${sanitizeId(agentId)}${this.naming.memorySuffix}`;
  }

  /** Get the table name for an agent's tools */
  agentToolsName(agentId: string): string {
    return `${this.naming.agentPrefix}${sanitizeId(agentId)}${this.naming.toolsSuffix}`;
  }

  /** Get the shared knowledge table name */
  sharedKnowledgeName(): string {
    return this.naming.sharedKnowledge;
  }

  /** Get the shared mistakes table name */
  sharedMistakesName(): string {
    return this.naming.sharedMistakes;
  }

  /** Get the reference library table name */
  referenceLibraryName(): string {
    return this.naming.referenceLibrary;
  }

  /** Get the reference code table name */
  referenceCodeName(): string {
    return this.naming.referenceCode;
  }

  // ── Table access ──────────────────────────────────────────────────────────

  /** Get or create an agent's memory table */
  async getAgentMemory(agentId: string): Promise<Table | null> {
    const name = this.agentMemoryName(agentId);
    return (await this.getOrOpenTable(name, "agent_memory", agentId)).table;
  }

  /** Get or create an agent's tools table */
  async getAgentTools(agentId: string): Promise<Table | null> {
    const name = this.agentToolsName(agentId);
    return (await this.getOrOpenTable(name, "agent_tools", agentId)).table;
  }

  /** Get or create the shared knowledge table */
  async getSharedKnowledge(): Promise<Table | null> {
    const name = this.sharedKnowledgeName();
    return (await this.getOrOpenTable(name, "shared_knowledge")).table;
  }

  /** Get or create the shared mistakes table */
  async getSharedMistakes(): Promise<Table | null> {
    const name = this.sharedMistakesName();
    return (await this.getOrOpenTable(name, "shared_mistakes")).table;
  }

  /** Get or create the reference library table */
  async getReferenceLibrary(): Promise<Table | null> {
    const name = this.referenceLibraryName();
    return (await this.getOrOpenTable(name, "reference_library")).table;
  }

  /** Get or create the reference code table */
  async getReferenceCode(): Promise<Table | null> {
    const name = this.referenceCodeName();
    return (await this.getOrOpenTable(name, "reference_code")).table;
  }

  /** Get a managed table by exact name (for direct access) */
  async getTable(name: string): Promise<ManagedTable | undefined> {
    return this.tables.get(name);
  }

  /** Check if a table exists in the database */
  async tableExists(name: string): Promise<boolean> {
    const names = await this.listTableNames();
    return names.includes(name);
  }

  // ── Write serialization ───────────────────────────────────────────────────

  /** Execute a write operation with per-table mutex to prevent LanceDB commit conflicts */
  async withWriteLock<T>(tableName: string, fn: () => Promise<T>): Promise<T> {
    const existing = this.writeLocks.get(tableName) ?? Promise.resolve();
    const next = existing.then(() => fn());
    // Settle the lock chain even if fn throws
    this.writeLocks.set(
      tableName,
      next.then(
        () => {},
        () => {},
      ),
    );
    return next;
  }

  // ── Stats ─────────────────────────────────────────────────────────────────

  /** Get status of all open tables */
  async status(): Promise<
    Array<{
      name: string;
      category: TableCategory;
      agentId?: string;
      rowCount: number;
      ftsCreated: boolean;
    }>
  > {
    const result = [];
    for (const [name, managed] of this.tables) {
      try {
        if (!managed.table) {
          result.push({
            name,
            category: managed.category,
            agentId: managed.agentId,
            rowCount: 0,
            ftsCreated: false,
          });
          continue;
        }
        const count = await managed.table.countRows();
        result.push({
          name,
          category: managed.category,
          agentId: managed.agentId,
          rowCount: count,
          ftsCreated: managed.ftsCreated,
        });
      } catch {
        result.push({
          name,
          category: managed.category,
          agentId: managed.agentId,
          rowCount: managed.rowCount,
          ftsCreated: managed.ftsCreated,
        });
      }
    }
    return result;
  }

  /** Drop a table entirely (DESTRUCTIVE — use with care) */
  async dropTable(name: string): Promise<void> {
    if (!this.db) throw new Error("TableManager not open");
    const managed = this.tables.get(name);
    if (managed) {
      if (managed.table) managed.table.close();
      this.tables.delete(name);
      this.writeLocks.delete(name);
    }
    await this.db.dropTable(name);
  }

  // ── Internal ──────────────────────────────────────────────────────────────

  private async getOrOpenTable(
    name: string,
    category: TableCategory,
    agentId?: string,
  ): Promise<ManagedTable> {
    const existing = this.tables.get(name);
    if (existing) return existing;

    if (!this.db) throw new Error("TableManager not open");

    const names = await this.db.tableNames();
    if (!names.includes(name)) {
      // Table doesn't exist yet — it will be created on first upsert.
      // Return a placeholder — the actual table will be set by createTableWithData
      const managed: ManagedTable = {
        table: null, // Placeholder until first write via createTableWithData
        category,
        agentId,
        ftsCreated: false,
        hasNewSchema: true, // New tables always have full schema
        rowCount: 0,
      };
      this.tables.set(name, managed);
      return managed;
    }

    const table = await this.db.openTable(name);
    // Check schema for new columns
    const schema = await table.schema();

    const managed: ManagedTable = {
      table,
      category,
      agentId,
      ftsCreated: false,
      hasNewSchema: schema.fields.some((f) => f.name === "content_type"),
      rowCount: 0,
    };

    // Try creating FTS index
    try {
      await table.createIndex("text", { config: lancedb.Index.fts() });
      managed.ftsCreated = true;
    } catch {
      // Index might already exist
      managed.ftsCreated = true;
    }

    this.tables.set(name, managed);
    return managed;
  }

  /**
   * Create a table with initial data (required for LanceDB — can't create empty tables).
   * Use this on first upsert when the table doesn't exist yet.
   */
  async createTableWithData(
    name: string,
    data: Record<string, unknown>[],
    category: TableCategory,
    agentId?: string,
  ): Promise<Table> {
    if (!this.db) throw new Error("TableManager not open");

    const table = await this.db.createTable(name, data, { mode: "overwrite" });

    // Create FTS index (best-effort — might already exist)
    try {
      await table.createIndex("text", { config: lancedb.Index.fts() });
    } catch {
      // Index already exists or FTS not supported — non-fatal
    }

    const managed: ManagedTable = {
      table,
      category,
      agentId,
      ftsCreated: true, // Either created above or already existed
      hasNewSchema: true,
      rowCount: data.length,
    };
    this.tables.set(name, managed);
    return table;
  }

  /** Get the LanceDB connection (for advanced operations) */
  getConnection(): lancedb.Connection | null {
    return this.db;
  }

  /** Get the naming configuration */
  getNaming(): TableNamingConfig {
    return { ...this.naming };
  }
}

// ── Utilities ───────────────────────────────────────────────────────────────

/**
 * Sanitize an agent/table ID for use in LanceDB table names.
 * Only allows alphanumeric, hyphens, and underscores.
 */
function sanitizeId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_-]/g, "_").toLowerCase();
}
