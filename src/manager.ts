/**
 * MemorySparkManager — central coordinator.
 * Implements the search + readFile interface matching OC's MemorySearchManager.
 */

import fs from "node:fs/promises";

import { hybridMerge, applySourceWeighting, applyTemporalDecay, mmrRerank } from "./auto/recall.js";
import type { MemorySparkConfig } from "./config.js";
import type { EmbedProvider } from "./embed/provider.js";
import { EmbedQueue } from "./embed/queue.js";
import { isPathInside, toAbsolutePath } from "./ingest/workspace.js";
import type { Reranker } from "./rerank/reranker.js";
import type { StorageBackend, SearchResult } from "./storage/backend.js";

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
  queue?: EmbedQueue;
}

export class MemorySparkManager {
  private cfg: MemorySparkConfig;
  private agentId: string;
  private workspaceDir: string;
  private backend: StorageBackend;
  private embed: EmbedProvider;
  private reranker: Reranker;
  public queue: EmbedQueue;

  constructor(opts: ManagerOptions) {
    this.cfg = opts.cfg;
    this.agentId = opts.agentId;
    this.workspaceDir = opts.workspaceDir;
    this.backend = opts.backend;
    this.embed = opts.embed;
    this.reranker = opts.reranker;
    this.queue = opts.queue ?? new EmbedQueue(opts.embed);
  }

  async search(
    query: string,
    opts?: { maxResults?: number; minScore?: number; sessionKey?: string },
  ): Promise<MemorySearchResult[]> {
    const maxResults = opts?.maxResults ?? 10;
    const minScore = opts?.minScore ?? this.cfg.autoRecall.minScore;
    // Vector pre-filter floor. The two retrievers score on incompatible
    // scales: vector = cosine similarity (relevant docs typically land
    // 0.2-0.6 per ISSUES.md), FTS = sigmoid(BM25) which saturates to >0.98.
    // A single 0.75 threshold applied to both lets FTS always pass while
    // silently dropping nearly every vector hit — the "hybrid" degrades to
    // BM25-only and semantic recall dies. Use the same 0.7x floor that
    // recall.ts applies (lowThreshold) so the tool path and the auto-recall
    // path filter identically.
    const vectorFloor = Math.max(minScore * 0.7, 0.05);

    // Try to embed the query for vector search. If Spark is temporarily down,
    // degrade gracefully to FTS-only recall — same dims, no provider switch.
    let queryVector: number[] | null = null;
    try {
      queryVector = await this.queue.embedQuery(query);
    } catch {
      // Spark unavailable — vector recall disabled until it recovers.
      // Write-side pending queue will replay when Spark comes back.
    }

    // Use the same pipeline as auto-recall (recall.ts) for consistent results.
    // Pool-aware: search agent's own memory + shared knowledge (not references).
    // LanceDB 0.27+ supports FTS+WHERE natively — no workaround needed.
    const fetchN = maxResults * 4;
    const agentPools = ["agent_memory", "agent_tools", "agent_mistakes"];
    const sharedPools = ["shared_knowledge", "shared_mistakes", "shared_rules"];

    const vectorResults = queryVector
      ? await this.backend
          .vectorSearch(queryVector, {
            query,
            maxResults: fetchN,
            minScore: vectorFloor,
            agentId: this.agentId,
            pools: [...agentPools, ...sharedPools],
          })
          .catch(() => [] as SearchResult[])
      : ([] as SearchResult[]);

    const rawFtsResults = await this.backend
      .ftsSearch(query, {
        query,
        maxResults: fetchN,
        agentId: this.agentId,
        pools: [...agentPools, ...sharedPools],
      })
      .catch(() => [] as SearchResult[]);

    // Filter FTS: exclude sessions source only.
    // Do NOT apply minScore to FTS results — BM25 scores are sigmoid-normalized
    // and nearly all map to >0.98, making minScore non-discriminative (this
    // matches the documented decision in recall.ts). RRF ranks FTS by
    // position, not score magnitude.
    const ftsResults = rawFtsResults.filter((r) => r.chunk.source !== "sessions");

    // Full pipeline: hybridMerge → sourceWeighting → temporalDecay → MMR → rerank
    const merged = hybridMerge(vectorResults, ftsResults, fetchN);
    applySourceWeighting(merged);
    applyTemporalDecay(merged);
    // Adaptive λ (not fixed 0.7): ISSUES.md Issue 6 measured λ=0.7 destroying
    // 28% of recall, and this MMR runs BEFORE the reranker — an over-aggressive
    // λ prunes candidates the cross-encoder never gets to score. recall.ts
    // already moved off fixed 0.7 for the same reason.
    const diverse = mmrRerank(merged, maxResults * 2, "adaptive");
    const reranked = await this.reranker.rerank(query, diverse, maxResults);

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
  async readFile(params: {
    relPath: string;
    from?: number;
    lines?: number;
  }): Promise<{ text: string; path: string }> {
    // Try storage first
    const fromStorage = await this.backend.readFile({
      path: params.relPath,
      from: params.from,
      lines: params.lines,
      agentId: this.agentId,
    });

    if (fromStorage.text.trim()) return fromStorage;

    // Fallback: read from disk — only inside the agent's workspace, so a tool
    // call can't use absolute paths or "../" to read arbitrary files.
    const absPath = toAbsolutePath(params.relPath, this.workspaceDir);
    if (!isPathInside(absPath, this.workspaceDir)) {
      return { text: "", path: params.relPath };
    }
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
