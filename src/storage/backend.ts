/**
 * StorageBackend interface — both LanceDB and SQLite-vec implement this.
 */

export interface MemoryChunk {
  id: string;
  path: string;
  source: "memory" | "sessions" | "ingest" | "capture";
  agent_id: string;
  start_line: number;
  end_line: number;
  text: string;
  vector: number[];
  updated_at: string;
  category?: string;
  entities?: string;    // JSON-serialized string[] for storage
  confidence?: number;
}

export interface SearchOptions {
  query: string;
  queryVector?: number[];
  maxResults?: number;
  minScore?: number;
  agentId?: string;
  source?: string;
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
  listPaths(agentId?: string): Promise<Array<{ path: string; agentId: string; updatedAt: string; chunkCount: number }>>;
  getById(id: string): Promise<MemoryChunk | null>;
  readFile(params: { path: string; from?: number; lines?: number; agentId?: string }): Promise<{ text: string; path: string }>;
  status(): Promise<BackendStatus>;
}
