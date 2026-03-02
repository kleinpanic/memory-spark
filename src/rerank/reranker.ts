/**
 * Reranker
 *
 * Post-processes raw vector+FTS hybrid search results using Spark's
 * cross-encoder reranker service (nvidia/llama-nemotron-rerank-1b-v2).
 *
 * Cross-encoders are significantly more accurate than bi-encoders for ranking
 * because they jointly encode the query and each candidate passage together.
 * The cost is higher latency, so we only rerank the top-N candidates (default 20)
 * from the initial vector/FTS retrieval pass.
 *
 * API: Spark reranker is compatible with Cohere's /v1/rerank endpoint.
 * Request body:
 *   {
 *     model: "nvidia/llama-nemotron-rerank-1b-v2",
 *     query: "...",
 *     documents: ["passage1", "passage2", ...],
 *     top_n: 5,
 *     return_documents: false
 *   }
 * Response:
 *   { results: [{ index: 0, relevance_score: 0.95 }, ...] }
 *
 * Fallback: if reranker is unavailable, return input results unchanged.
 * This is safe — reranking improves quality but doesn't break correctness.
 */

import type { RerankConfig } from "../config.js";
import type { SearchResult } from "../storage/backend.js";

export interface Reranker {
  /** Rerank candidates against the query. Returns reordered + filtered results. */
  rerank(query: string, candidates: SearchResult[], topN?: number): Promise<SearchResult[]>;
  /** Check if reranker is reachable */
  probe(): Promise<boolean>;
}

/**
 * Create a Reranker instance. Returns a passthrough (no-op) if not configured or unavailable.
 */
export async function createReranker(cfg: RerankConfig): Promise<Reranker> {
  if (!cfg.enabled || !cfg.spark) {
    return createPassthroughReranker();
  }

  const sparkReranker = createSparkReranker(cfg);
  const ok = await sparkReranker.probe().catch(() => false);
  if (!ok) {
    console.warn("memory-spark: reranker unavailable, falling back to passthrough");
    return createPassthroughReranker();
  }

  return sparkReranker;
}

// ---------------------------------------------------------------------------
// Spark reranker (Cohere /v1/rerank compatible)
// ---------------------------------------------------------------------------

function createSparkReranker(cfg: RerankConfig): Reranker {
  const baseUrl = cfg.spark!.baseUrl;
  const model = cfg.spark!.model;
  const apiKey = cfg.spark!.apiKey ?? "none";

  return {
    async rerank(query, candidates, topN = 5) {
      if (candidates.length === 0) return candidates;

      // TODO:
      // const resp = await fetch(`${baseUrl}/rerank`, {
      //   method: "POST",
      //   headers: { "Authorization": `Bearer ${apiKey}`, "Content-Type": "application/json" },
      //   body: JSON.stringify({
      //     model,
      //     query,
      //     documents: candidates.map(c => c.chunk.text),
      //     top_n: topN,
      //     return_documents: false,
      //   }),
      // });
      // const data = await resp.json();
      // return data.results
      //   .map((r: { index: number; relevance_score: number }) => ({
      //     ...candidates[r.index]!,
      //     score: r.relevance_score,
      //   }))
      //   .sort((a, b) => b.score - a.score);
      throw new Error("SparkReranker.rerank() not yet implemented");
    },
    async probe() {
      // TODO: send a minimal rerank request to check health
      return false;
    },
  };
}

// ---------------------------------------------------------------------------
// Passthrough (no-op) reranker — used as fallback
// ---------------------------------------------------------------------------

function createPassthroughReranker(): Reranker {
  return {
    async rerank(_query, candidates, topN = 5) {
      return candidates.slice(0, topN);
    },
    async probe() {
      return true;
    },
  };
}
