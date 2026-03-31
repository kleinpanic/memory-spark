/**
 * Reranker — Spark 18096 cross-encoder, Cohere /v1/rerank compatible.
 * Falls back to passthrough (no-op) if unavailable.
 */

import type { RerankConfig } from "../config.js";
import type { SearchResult } from "../storage/backend.js";

export interface Reranker {
  rerank(query: string, candidates: SearchResult[], topN?: number): Promise<SearchResult[]>;
  probe(): Promise<boolean>;
}

export async function createReranker(cfg: RerankConfig): Promise<Reranker> {
  if (!cfg.enabled || !cfg.spark) {
    return passthroughReranker();
  }

  const reranker = sparkReranker(cfg);
  const ok = await reranker.probe().catch(() => false);
  if (!ok) {
    return passthroughReranker();
  }
  return reranker;
}

/**
 * Maximum candidates sent to the cross-encoder.
 * Top-30 from first-stage retrieval captures >99% of relevant docs (Nogueira 2020).
 * Reduces reranker latency linearly — O(n) scoring.
 */
const MAX_RERANK_CANDIDATES = 30;

function sparkReranker(cfg: RerankConfig): Reranker {
  const baseUrl = cfg.spark!.baseUrl;
  const model = cfg.spark!.model;
  const apiKey = cfg.spark!.apiKey ?? "none";

  return {
    async rerank(query, candidates, topN = 5) {
      if (candidates.length === 0) return [];

      // Phase 5B: Limit candidate pool to prevent unbounded reranker input
      const pool = candidates.slice(0, MAX_RERANK_CANDIDATES);

      const documents = pool.map((c) => c.chunk.text);

      const t0 = performance.now();
      const resp = await fetch(`${baseUrl}/rerank`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          query,
          documents,
          top_n: topN,
          return_documents: false,
        }),
      });

      if (!resp.ok) {
        // Fallback to passthrough on error
        return pool.slice(0, topN);
      }

      const data = (await resp.json()) as {
        results: Array<{ index: number; relevance_score: number }>;
      };

      const results = data.results
        .map((r) => ({
          ...pool[r.index]!,
          score: r.relevance_score,
        }))
        .sort((a, b) => b.score - a.score);

      // Phase 5C: Score calibration telemetry
      const elapsedMs = performance.now() - t0;
      if (results.length > 0) {
        const scores = results.map((r) => r.score);
        const min = Math.min(...scores);
        const max = Math.max(...scores);
        const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
        const spread = max - min;
        // Debug-level: enable via DEBUG=memory-spark:reranker or similar
        if (process.env.DEBUG?.includes("rerank") || process.env.RERANKER_TELEMETRY) {
          console.log(
            `[reranker] ${pool.length} candidates → ${results.length} results in ${elapsedMs.toFixed(0)}ms | ` +
              `scores: min=${min.toFixed(4)} max=${max.toFixed(4)} mean=${mean.toFixed(4)} spread=${spread.toFixed(4)}`,
          );
        }
      }

      return results;
    },

    async probe() {
      try {
        const resp = await fetch(`${baseUrl}/rerank`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey}`,
          },
          body: JSON.stringify({
            model,
            query: "test",
            documents: ["test document"],
            top_n: 1,
          }),
        });
        return resp.ok;
      } catch {
        return false;
      }
    },
  };
}

function passthroughReranker(): Reranker {
  return {
    async rerank(_query, candidates, topN = 5) {
      return candidates.slice(0, topN);
    },
    async probe() {
      return true;
    },
  };
}
