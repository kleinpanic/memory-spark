/**
 * MemorySparkManager
 *
 * Central coordinator. Implements the OpenClaw MemorySearchManager interface
 * so the plugin can register memory_search and memory_get tools via
 * the standard plugin API.
 *
 * Owns: backend, embed provider, reranker.
 * Delegates: search to backend + reranker, sync to ingest pipeline.
 *
 * One manager instance per agent session (agentId-scoped).
 */

import type { MemorySearchManager, MemorySearchResult, MemoryProviderStatus, MemorySyncProgressUpdate } from "openclaw/plugin-sdk/memory";
import type { MemorySparkConfig } from "./config.js";
import type { StorageBackend } from "./storage/backend.js";
import type { EmbedProvider } from "./embed/provider.js";
import type { Reranker } from "./rerank/reranker.js";

export interface ManagerOptions {
  cfg: MemorySparkConfig;
  agentId: string;
  backend: StorageBackend;
  embed: EmbedProvider;
  reranker: Reranker;
}

export class MemorySparkManager implements MemorySearchManager {
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
    opts?: { maxResults?: number; minScore?: number; sessionKey?: string },
  ): Promise<MemorySearchResult[]> {
    const maxResults = opts?.maxResults ?? 10;
    const minScore = opts?.minScore ?? this.cfg.autoRecall.minScore;

    // 1. Embed query
    const queryVector = await this.embed.embedQuery(query);

    // 2. Hybrid vector + FTS search (over-fetch for reranking)
    const fetchN = maxResults * 3;
    const [vectorResults, ftsResults] = await Promise.all([
      this.backend.vectorSearch(queryVector, { query, maxResults: fetchN, minScore, agentId: this.agentId }),
      this.backend.ftsSearch(query, { query, maxResults: fetchN, agentId: this.agentId }),
    ]);

    // 3. Merge + deduplicate
    const seen = new Set<string>();
    const merged = [...vectorResults, ...ftsResults].filter((r) => {
      if (seen.has(r.chunk.id)) return false;
      seen.add(r.chunk.id);
      return true;
    }).sort((a, b) => b.score - a.score).slice(0, fetchN);

    // 4. Rerank
    const reranked = await this.reranker.rerank(query, merged, maxResults);

    // 5. Map to MemorySearchResult (OC interface)
    return reranked.map((r) => ({
      path: r.chunk.path,
      startLine: r.chunk.startLine,
      endLine: r.chunk.endLine,
      score: r.score,
      snippet: r.snippet || r.chunk.text.slice(0, 500),
      source: (r.chunk.source === "sessions" ? "sessions" : "memory") as "memory" | "sessions",
      citation: formatCitation(r.chunk.path, r.chunk.startLine),
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

  status(): MemoryProviderStatus {
    // TODO: async status, for now return sync stub
    return {
      backend: "builtin",
      provider: "spark",
      model: this.cfg.embed.spark?.model ?? "unknown",
      workspaceDir: this.cfg.lancedbDir,
    };
  }

  async sync(params?: { reason?: string; force?: boolean; progress?: (u: MemorySyncProgressUpdate) => void }): Promise<void> {
    // TODO: trigger watcher boot pass for this agent's paths
    throw new Error("MemorySparkManager.sync() not yet implemented");
  }

  async probeEmbeddingAvailability(): Promise<{ ok: boolean; error?: string }> {
    const ok = await this.embed.probe().catch(() => false);
    return { ok, error: ok ? undefined : "Embedding provider unavailable" };
  }

  async probeVectorAvailability(): Promise<boolean> {
    const status = await this.backend.status();
    return status.ready;
  }

  async close(): Promise<void> {
    await this.backend.close();
  }
}

function formatCitation(path: string, startLine: number): string {
  return `${path}:${startLine}`;
}
