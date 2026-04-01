/**
 * Reranker — Spark 18096 cross-encoder, Cohere /v1/rerank compatible.
 * Falls back to passthrough (no-op) if unavailable.
 *
 * Includes query normalization: the Nemotron reranker was trained on
 * question–answer pairs.  Declarative statements / claims produce
 * compressed scores with almost no discrimination.  Converting them
 * to interrogative form restores full score spread.
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

      // Normalize declarative queries → questions for better reranker discrimination
      const normalizedQuery = normalizeQueryForReranker(query);

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
          query: normalizedQuery,
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
          const normalized =
            normalizedQuery !== query ? ` (normalized: "${normalizedQuery.slice(0, 60)}…")` : "";
          console.log(
            `[reranker] ${pool.length} candidates → ${results.length} results in ${elapsedMs.toFixed(0)}ms | ` +
              `scores: min=${min.toFixed(4)} max=${max.toFixed(4)} mean=${mean.toFixed(4)} spread=${spread.toFixed(4)}${normalized}`,
          );
        }
      }

      // Score spread guard: if reranker scores are too compressed, it's not
      // discriminating — fall back to input ordering to avoid coin-flip reranks.
      if (results.length >= 2) {
        const scores = results.map((r) => r.score);
        const spread = Math.max(...scores) - Math.min(...scores);
        const minSpread = cfg.spark!.minScoreSpread ?? 0.01;
        if (spread < minSpread) {
          if (process.env.MEMORY_SPARK_DEBUG || process.env.DEBUG?.includes("rerank")) {
            console.debug(
              `[reranker] spread=${spread.toFixed(4)} < minSpread=${minSpread} — falling back to input order`,
            );
          }
          return pool.slice(0, topN);
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

// ── Query Normalizer ─────────────────────────────────────────────────
//
// The Nemotron reranker (llama-nemotron-rerank-1b-v2) was fine-tuned on
// question–answer pairs.  When fed a declarative claim like
// "0-dimensional biomaterials show inductive properties", scores
// compress to a ~0.02 gap between relevant and irrelevant docs.
// Reformatting the same claim as a question restores a 0.77–0.98 gap.
//
// This normalizer detects non-question queries and prepends a minimal
// interrogative prefix.  It's intentionally conservative — production
// agent queries are already questions; this catches edge cases and
// BEIR-style claim inputs.

const QUESTION_STARTERS =
  /^(who|what|which|when|where|why|how|is|are|was|were|do|does|did|can|could|will|would|should|shall|has|have|had|may|might)\b/i;
const ENDS_WITH_QUESTION = /\?\s*$/;

/**
 * Returns true if the query already looks like a question.
 */
export function isQuestion(query: string): boolean {
  const trimmed = query.trim();
  if (ENDS_WITH_QUESTION.test(trimmed)) return true;
  if (QUESTION_STARTERS.test(trimmed)) return true;
  return false;
}

/**
 * Normalize a query for the cross-encoder reranker.
 * If it's already a question, return as-is.
 * If it's a declarative statement / claim, prepend "Is it true that"
 * to convert it into an interrogative the model can discriminate on.
 */
export function normalizeQueryForReranker(query: string): string {
  if (isQuestion(query)) return query;
  const trimmed = query.trim().replace(/\.\s*$/, "");
  return `Is it true that ${trimmed.charAt(0).toLowerCase()}${trimmed.slice(1)}?`;
}
