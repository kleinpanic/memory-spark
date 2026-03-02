/**
 * MemorySparkManager — central coordinator.
 * Implements the search + readFile interface that tools call.
 */

import type { MemorySparkConfig } from "./config.js";
import type { StorageBackend, SearchResult } from "./storage/backend.js";
import type { EmbedProvider } from "./embed/provider.js";
import type { Reranker } from "./rerank/reranker.js";

export interface MemorySearchResult {
  path: string;
  startLine: number;
  endLine: number;
  score: number;
  snippet: string;
  source: "memory" | "sessions";
  citation?: string;
}

export interface ManagerOptions {
  cfg: MemorySparkConfig;
  agentId: string;
  backend: StorageBackend;
  embed: EmbedProvider;
  reranker: Reranker;
}

export class MemorySparkManager {
  private cfg: MemorySparkConfig;
  private agentId: string;
  private backend: StorageBackend;
  private embed: EmbedProvider;
  private reranker: Reranker;

  constructor(opts: ManagerOptions) {
    this.cfg = opts.cfg;
    this.agentId = opts.agentId;
    this.backend = opts.backend;
    this.embed = opts.embed;
    this.reranker = opts.reranker;
  }

  async search(
    query: string,
    opts?: { maxResults?: number; minScore?: number },
  ): Promise<MemorySearchResult[]> {
    const maxResults = opts?.maxResults ?? 10;
    const minScore = opts?.minScore ?? this.cfg.autoRecall.minScore;

    // 1. Embed query
    const queryVector = await this.embed.embedQuery(query);

    // 2. Hybrid search (over-fetch for reranking)
    const fetchN = maxResults * 3;
    const [vectorResults, ftsResults] = await Promise.all([
      this.backend.vectorSearch(queryVector, {
        query,
        maxResults: fetchN,
        minScore,
        agentId: this.agentId,
      }).catch(() => [] as SearchResult[]),
      this.backend.ftsSearch(query, {
        query,
        maxResults: fetchN,
        agentId: this.agentId,
      }).catch(() => [] as SearchResult[]),
    ]);

    // 3. Merge + dedup
    const seen = new Set<string>();
    const merged = [...vectorResults, ...ftsResults]
      .filter((r) => {
        if (seen.has(r.chunk.id)) return false;
        seen.add(r.chunk.id);
        return true;
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, fetchN);

    // 4. Rerank
    const reranked = await this.reranker.rerank(query, merged, maxResults);

    // 5. Map to result format
    return reranked.map((r) => ({
      path: r.chunk.path,
      startLine: r.chunk.start_line,
      endLine: r.chunk.end_line,
      score: r.score,
      snippet: r.snippet || r.chunk.text.slice(0, 500),
      source: (r.chunk.source === "sessions" ? "sessions" : "memory") as "memory" | "sessions",
      citation: `${r.chunk.path}:${r.chunk.start_line}`,
    }));
  }

  async readFile(params: { relPath: string; from?: number; lines?: number }): Promise<{ text: string; path: string }> {
    return this.backend.readFile({
      path: params.relPath,
      from: params.from,
      lines: params.lines,
      agentId: this.agentId,
    });
  }
}
