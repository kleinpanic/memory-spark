/**
 * StorageBackend interface — both LanceDB and SQLite-vec implement this.
 */

export interface MemoryChunk {
  id: string;
  path: string;
  source: "memory" | "sessions" | "ingest" | "capture";
  agent_id: string;
  /** User/gateway isolation — enables multi-user RAG (Klein, Nicholas, etc.) */
  user_id?: string;
  start_line: number;
  end_line: number;
  text: string;
  vector: number[];
  updated_at: string;
  category?: string;
  entities?: string; // JSON-serialized string[] for storage
  confidence?: number;
  /** Content classification: "knowledge" (default), "reference", "capture" */
  content_type?: string;
  /** Quality score 0-1 from quality gate */
  quality_score?: number;
  /** Estimated token count of this chunk */
  token_count?: number;
  /** Most recent markdown heading (## or ###) above this chunk */
  parent_heading?: string;
  /**
   * Logical pool within the single LanceDB table.
   * Determines auto-injection behavior and access patterns.
   *
   * Values:
   * - "agent_memory" — per-agent workspace files, captures (auto-injected)
   * - "agent_tools" — per-agent tool definitions (auto-injected for tool context)
   * - "shared_knowledge" — cross-agent facts (auto-injected, 0.8x weight)
   * - "shared_mistakes" — cross-agent mistakes (auto-injected, 1.6x boost)
   * - "shared_rules" — global rules & preferences (always injected)
   * - "reference_library" — PDFs, documentation (tool-call only, NOT auto-injected)
   * - "reference_code" — code examples (tool-call only, NOT auto-injected)
   */
  pool?: string;
}

export interface SearchOptions {
  query: string;
  queryVector?: number[];
  maxResults?: number;
  minScore?: number;
  agentId?: string;
  userId?: string; // Filter by user/gateway for multi-user isolation
  source?: string;
  /** Filter by content_type (e.g. "reference", "knowledge") */
  contentType?: string;
  /** Filter results where path contains this substring (case-insensitive) */
  pathContains?: string;
  /** Filter by pool (logical section). Multiple pools: comma-separated */
  pool?: string;
  /** Filter by multiple pools (OR logic) */
  pools?: string[];
}

export interface SearchResult {
  chunk: MemoryChunk;
  score: number;
  snippet: string;
}

export interface BackendStatus {
  backend: "lancedb" | "sqlite-vec";
  chunkCount: number;
  tableExists: boolean;
  vectorDims?: number;
  ready: boolean;
  error?: string;
}

export interface StorageBackend {
  open(): Promise<void>;
  close(): Promise<void>;
  upsert(chunks: MemoryChunk[]): Promise<void>;
  deleteByPath(path: string, agentId?: string): Promise<number>;
  deleteById(ids: string[]): Promise<void>;
  vectorSearch(queryVector: number[], opts: SearchOptions): Promise<SearchResult[]>;
  ftsSearch(query: string, opts: SearchOptions): Promise<SearchResult[]>;
  listPaths(
    agentId?: string,
  ): Promise<Array<{ path: string; agentId: string; updatedAt: string; chunkCount: number }>>;
  getById(id: string): Promise<MemoryChunk | null>;
  readFile(params: {
    path: string;
    from?: number;
    lines?: number;
    agentId?: string;
  }): Promise<{ text: string; path: string }>;
  status(): Promise<BackendStatus>;
}
