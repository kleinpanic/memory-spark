/**
 * MemorySparkManager — central coordinator.
 * Implements the search + readFile interface matching OC's MemorySearchManager.
 */

import type { MemorySparkConfig } from "./config.js";
import type { StorageBackend, SearchResult } from "./storage/backend.js";
import type { EmbedProvider } from "./embed/provider.js";
import type { Reranker } from "./rerank/reranker.js";
import { toAbsolutePath } from "./ingest/workspace.js";
import fs from "node:fs/promises";

export interface MemorySearchResult {
  path: string;
  startLine: number;
  endLine: number;
  score: number;
  snippet: string;
  source: "memory" | "sessions";
  citation?: string;
}

export interface MemoryProviderStatus {
  backend: "builtin" | "qmd";
  provider: string;
  model?: string;
  files?: number;
  chunks?: number;
  workspaceDir?: string;
  sources?: Array<"memory" | "sessions">;
  vector?: { enabled: boolean; dims?: number };
}

export interface ManagerOptions {
  cfg: MemorySparkConfig;
  agentId: string;
  workspaceDir: string;
  backend: StorageBackend;
  embed: EmbedProvider;
  reranker: Reranker;
}

export class MemorySparkManager {
  private cfg: MemorySparkConfig;
  private agentId: string;
  private workspaceDir: string;
  private backend: StorageBackend;
  private embed: EmbedProvider;
  private reranker: Reranker;

  constructor(opts: ManagerOptions) {
    this.cfg = opts.cfg;
    this.agentId = opts.agentId;
    this.workspaceDir = opts.workspaceDir;
    this.backend = opts.backend;
    this.embed = opts.embed;
    this.reranker = opts.reranker;
  }

  async search(
    query: string,
    opts?: { maxResults?: number; minScore?: number; sessionKey?: string },
  ): Promise<MemorySearchResult[]> {
    const maxResults = opts?.maxResults ?? 10;
    const minScore = opts?.minScore ?? this.cfg.autoRecall.minScore;

    const queryVector = await this.embed.embedQuery(query);

    const fetchN = maxResults * 3;
    const [vectorResults, ftsResults] = await Promise.all([
      this.backend.vectorSearch(queryVector, {
        query, maxResults: fetchN, minScore, agentId: this.agentId,
      }).catch(() => [] as SearchResult[]),
      this.backend.ftsSearch(query, {
        query, maxResults: fetchN, agentId: this.agentId,
      }).catch(() => [] as SearchResult[]),
    ]);

    const seen = new Set<string>();
    const merged = [...vectorResults, ...ftsResults]
      .filter((r) => {
        if (seen.has(r.chunk.id)) return false;
        seen.add(r.chunk.id);
        return true;
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, fetchN);

    const reranked = await this.reranker.rerank(query, merged, maxResults);

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

  /**
   * Read file content. First tries the indexed chunks in storage,
   * then falls back to reading from disk (using workspace-relative path).
   */
  async readFile(params: { relPath: string; from?: number; lines?: number }): Promise<{ text: string; path: string }> {
    // Try storage first
    const fromStorage = await this.backend.readFile({
      path: params.relPath,
      from: params.from,
      lines: params.lines,
      agentId: this.agentId,
    });

    if (fromStorage.text.trim()) return fromStorage;

    // Fallback: read from disk
    const absPath = toAbsolutePath(params.relPath, this.workspaceDir);
    try {
      const text = await fs.readFile(absPath, "utf-8");
      const lines = text.split("\n");
      const start = Math.max(0, (params.from ?? 1) - 1);
      const count = params.lines ?? 50;
      return {
        text: lines.slice(start, start + count).join("\n"),
        path: params.relPath,
      };
    } catch {
      return { text: "", path: params.relPath };
    }
  }

  status(): MemoryProviderStatus {
    return {
      backend: "builtin",
      provider: `spark:${this.embed.id}`,
      model: this.embed.model,
      workspaceDir: this.workspaceDir,
      sources: ["memory", "sessions"],
      vector: { enabled: true, dims: this.embed.dims },
    };
  }

  async probeEmbeddingAvailability(): Promise<{ ok: boolean; error?: string }> {
    const ok = await this.embed.probe().catch(() => false);
    return { ok, error: ok ? undefined : "Embedding provider unreachable" };
  }

  async probeVectorAvailability(): Promise<boolean> {
    const st = await this.backend.status();
    return st.ready;
  }

  async close(): Promise<void> {
    await this.backend.close();
  }
}
