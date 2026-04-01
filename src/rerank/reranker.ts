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
  const blendAlpha = cfg.scoreBlendAlpha ?? 0;

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

      // ── Phase 9A: Score Interpolation ────────────────────────────────
      //
      // Instead of replacing original scores entirely with reranker scores,
      // blend them: final = α × normalized_original + (1 - α) × reranker_score
      //
      // This prevents catastrophic reranking — when the cross-encoder makes
      // a bad call (32/108 queries in SciFact), the original signal limits
      // the damage. When it makes a good call (20/108), the blend still
      // promotes the better document.
      //
      // α = 0: Pure reranker (backward-compatible default)
      // α = 0.3: Reranker-biased blend (recommended for production)
      // α = 0.5: Equal weight between original and reranker
      //
      const results = blendScores(pool, data.results, blendAlpha);

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
          const blendStr = blendAlpha > 0 ? ` blend=α${blendAlpha}` : "";
          console.log(
            `[reranker] ${pool.length} candidates → ${results.length} results in ${elapsedMs.toFixed(0)}ms | ` +
              `scores: min=${min.toFixed(4)} max=${max.toFixed(4)} mean=${mean.toFixed(4)} spread=${spread.toFixed(4)}${blendStr}${normalized}`,
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

// ── Score Interpolation (Phase 9A) ───────────────────────────────────
//
// Blends original retrieval scores with cross-encoder reranker scores.
// This prevents the reranker from catastrophically overriding good vector
// rankings while still allowing it to refine uncertain ones.
//
// Formula: final = α × norm(original) + (1 - α) × reranker_score
//   where norm(original) maps the original scores to [0, 1] using
//   min-max normalization within the candidate pool.

/**
 * Blend original retrieval scores with reranker relevance scores.
 *
 * @param pool — original candidates (with retrieval scores)
 * @param rerankResults — reranker output (index + relevance_score)
 * @param alpha — blend weight for original scores (0 = pure reranker, 1 = ignore reranker)
 * @returns sorted SearchResult[] with blended scores
 */
export function blendScores(
  pool: SearchResult[],
  rerankResults: Array<{ index: number; relevance_score: number }>,
  alpha: number,
): SearchResult[] {
  if (alpha <= 0) {
    // Pure reranker mode (backward-compatible default)
    return rerankResults
      .map((r) => ({
        ...pool[r.index]!,
        score: r.relevance_score,
      }))
      .sort((a, b) => b.score - a.score);
  }

  // Normalize original scores to [0, 1] within this pool
  const indices = rerankResults.map((r) => r.index);
  const originalScores = indices.map((i) => pool[i]!.score);
  const origMin = Math.min(...originalScores);
  const origMax = Math.max(...originalScores);
  const origRange = origMax - origMin;

  return rerankResults
    .map((r) => {
      const origScore = pool[r.index]!.score;
      // Min-max normalize: maps [origMin, origMax] → [0, 1]
      // If all original scores are identical (range=0), normalize to 0.5
      const normOrig = origRange > 0 ? (origScore - origMin) / origRange : 0.5;
      const blended = alpha * normOrig + (1 - alpha) * r.relevance_score;
      return {
        ...pool[r.index]!,
        score: blended,
      };
    })
    .sort((a, b) => b.score - a.score);
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

const QUESTION_STARTERS = new Set([
  "who","what","which","when","where","why","how",
  "is","are","was","were","do","does","did",
  "can","could","will","would","should","shall",
  "has","have","had","may","might",
]);
const ENDS_WITH_QUESTION = /\?\s*$/;

/**
 * Returns true if the query already looks like a question.
 */
export function isQuestion(query: string): boolean {
  const trimmed = query.trim();
  if (ENDS_WITH_QUESTION.test(trimmed)) return true;
  const firstWord = trimmed.split(/\s/, 1)[0]?.toLowerCase();
  if (firstWord && QUESTION_STARTERS.has(firstWord)) return true;
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
