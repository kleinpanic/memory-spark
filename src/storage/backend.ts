/**
 * StorageBackend — Abstract interface both LanceDB and SQLite-vec implement.
 *
 * The rest of the plugin never knows which backend is active.
 * Swap by changing config.backend — no other code changes.
 */

export interface MemoryChunk {
  /** Unique stable ID (hash of path+startLine or UUID for captured memories) */
  id: string;
  /** Relative path from workspace/memory root, or agent-scoped ID for captures */
  path: string;
  /** Source type */
  source: "memory" | "sessions" | "ingest" | "capture";
  /** Agent this chunk belongs to (undefined = shared/global) */
  agentId?: string;
  /** Line range in source file (0 for non-file captures) */
  startLine: number;
  endLine: number;
  /** The raw text of this chunk */
  text: string;
  /** Embedding vector (populated before storage, empty on read if not requested) */
  vector?: number[];
  /** BM25-indexable: same as text (stored separately for FTS in LanceDB) */
  ftsText?: string;
  /** ISO timestamp of when this chunk was created/updated */
  updatedAt: string;
  /** Category for captured memories: "fact" | "preference" | "decision" | "code-snippet" */
  category?: string;
  /** NER-extracted entity tags: ["Klein", "OpenClaw", "Spark", ...] */
  entities?: string[];
  /** Zero-shot confidence score (for captured memories) */
  confidence?: number;
}

export interface SearchOptions {
  query: string;
  queryVector?: number[];   // pre-computed embedding if available
  maxResults?: number;
  minScore?: number;
  agentId?: string;
  source?: MemoryChunk["source"] | MemoryChunk["source"][];
  sessionKey?: string;
}

export interface SearchResult {
  chunk: MemoryChunk;
  /** Combined score after hybrid merge + rerank (0–1) */
  score: number;
  /** Snippet with context window around the matched region */
  snippet: string;
}

export interface UpsertOptions {
  /** If true, skip re-embedding and use provided vector */
  skipEmbed?: boolean;
}

export interface SyncStats {
  added: number;
  updated: number;
  removed: number;
  skipped: number;
  errors: string[];
}

export interface BackendStatus {
  backend: "lancedb" | "sqlite-vec";
  tableCount?: number;
  chunkCount: number;
  vectorDims?: number;
  ready: boolean;
  error?: string;
}

/**
 * Core storage interface. Both LanceDB and SQLite-vec back this.
 */
export interface StorageBackend {
  /** Initialize connection, create tables/schemas if needed */
  open(): Promise<void>;

  /** Graceful shutdown */
  close(): Promise<void>;

  /** Insert or update chunks (upsert by id) */
  upsert(chunks: MemoryChunk[], opts?: UpsertOptions): Promise<void>;

  /** Delete chunks by path (when a file is deleted or changed) */
  deleteByPath(path: string, agentId?: string): Promise<number>;

  /** Delete chunks by id */
  deleteById(ids: string[]): Promise<void>;

  /** Vector similarity search (returns top-N by cosine similarity) */
  vectorSearch(queryVector: number[], opts: SearchOptions): Promise<SearchResult[]>;

  /** Full-text search (BM25) */
  ftsSearch(query: string, opts: SearchOptions): Promise<SearchResult[]>;

  /** List all unique paths indexed (for sync/delta checks) */
  listPaths(agentId?: string): Promise<Array<{ path: string; updatedAt: string; chunkCount: number }>>;

  /** Get a single chunk by id */
  getById(id: string): Promise<MemoryChunk | null>;

  /** Read raw text lines from an indexed file (for memory_get tool) */
  readFile(params: { path: string; from?: number; lines?: number; agentId?: string }): Promise<{ text: string; path: string }>;

  /** Get backend status/health */
  status(): Promise<BackendStatus>;
}
