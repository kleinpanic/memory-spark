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

function sparkReranker(cfg: RerankConfig): Reranker {
  const baseUrl = cfg.spark!.baseUrl;
  const model = cfg.spark!.model;
  const apiKey = cfg.spark!.apiKey ?? "none";

  return {
    async rerank(query, candidates, topN = 5) {
      if (candidates.length === 0) return [];
      if (candidates.length <= topN) {
        // Not enough candidates to rerank, but still score them
      }

      const documents = candidates.map((c) => c.chunk.text);

      const resp = await fetch(`${baseUrl}/rerank`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${apiKey}`,
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
        return candidates.slice(0, topN);
      }

      const data = await resp.json() as {
        results: Array<{ index: number; relevance_score: number }>;
      };

      return data.results
        .map((r) => ({
          ...candidates[r.index]!,
          score: r.relevance_score,
        }))
        .sort((a, b) => b.score - a.score);
    },

    async probe() {
      try {
        const resp = await fetch(`${baseUrl}/rerank`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${apiKey}`,
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
