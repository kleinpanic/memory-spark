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

export interface RerankOptions {
  /** Override the blend alpha for this call (0 = pure reranker, 1 = ignore reranker). */
  alphaOverride?: number;
}

export interface Reranker {
  rerank(query: string, candidates: SearchResult[], topN?: number, options?: RerankOptions): Promise<SearchResult[]>;
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
    async rerank(query, candidates, topN = 5, options?: RerankOptions) {
      if (candidates.length === 0) return [];

      // Phase 10B: per-call alpha override for benchmark flexibility
      const effectiveAlpha = options?.alphaOverride ?? blendAlpha;

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
          // When blending (alpha > 0), score ALL candidates so the blend
          // can promote vector-strong docs the reranker would have excluded.
          // Without blending, respect the original topN to save latency.
          top_n: effectiveAlpha > 0 ? pool.length : topN,
          return_documents: false,
        }),
      });

      if (!resp.ok) {
        // Phase 10B: Always log reranker errors (not gated behind VERBOSE)
        const body = await resp.text().catch(() => "");
        console.error(
          `[reranker] ERROR: ${resp.status} ${resp.statusText} — falling back to input order` +
          (body ? ` | body: ${body.slice(0, 200)}` : ""),
        );
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
      // When blending, we scored all candidates. Now take top-N.
      const results = effectiveAlpha > 0
        ? blendScores(pool, data.results, effectiveAlpha).slice(0, topN)
        : blendScores(pool, data.results, effectiveAlpha);

      // Phase 10A: Telemetry — log both sigmoid and logit-space metrics
      const elapsedMs = performance.now() - t0;
      const wasNormalized = normalizedQuery !== query;

      if (results.length > 0) {
        const scores = results.map((r) => r.score);
        const min = Math.min(...scores);
        const max = Math.max(...scores);
        const spread = max - min;
        const rawSigmoids = data.results.map((r) => r.relevance_score);
        const logits = rawSigmoids.map(recoverLogit);
        const logitMin = Math.min(...logits);
        const logitMax = Math.max(...logits);
        const logitSpread = logitMax - logitMin;
        const normalizedTag = wasNormalized ? " [normalized→interrogative]" : "";
        const blendStr = effectiveAlpha > 0 ? ` blend=α${effectiveAlpha}` : " blend=none";

        // Always-on summary: one line per query
        console.log(
          `[reranker] ${pool.length} candidates → ${results.length} results in ${elapsedMs.toFixed(0)}ms` +
            ` | blended spread=${spread.toFixed(4)} logitSpread=${logitSpread.toFixed(2)}${blendStr}${normalizedTag}`,
        );

        const verbose = process.env.VERBOSE || process.env.DEBUG_PIPELINE;
        if (verbose) {
          if (wasNormalized) {
            console.log(`[reranker]   query: "${query.slice(0, 80)}"`);
            console.log(`[reranker]   normalized: "${normalizedQuery.slice(0, 80)}"`);
          }

          // Per-candidate scoring trace
          console.log(`[reranker]   --- per-candidate scores (top ${Math.min(results.length, 10)}) ---`);
          for (let ci = 0; ci < Math.min(results.length, 10); ci++) {
            const res = results[ci]!;
            const rerankEntry = data.results.find((dr) => pool[dr.index]?.chunk.id === res.chunk.id);
            const origScore = pool.find((p) => p.chunk.id === res.chunk.id)?.score ?? 0;
            const sigmoidScore = rerankEntry?.relevance_score ?? 0;
            const logit = recoverLogit(sigmoidScore);
            const snippet = res.chunk.text.slice(0, 60).replace(/\n/g, " ");
            console.log(
              `[reranker]     [${ci + 1}] id=${res.chunk.id.slice(0, 20)} ` +
                `origScore=${origScore.toFixed(4)} sigmoid=${sigmoidScore.toFixed(4)} ` +
                `logit=${logit.toFixed(2)} blended=${res.score.toFixed(4)} | "${snippet}…"`,
            );
          }
        }
      }

      // Score spread guard: operate on recovered LOGITS, not sigmoid scores.
      // A 0.02 sigmoid spread might represent a 2-point logit spread (real signal).
      // Default threshold: 0.5 logit units ≈ negligible discrimination.
      if (data.results.length >= 2) {
        const logits = data.results.map((r) => recoverLogit(r.relevance_score));
        const logitSpread = Math.max(...logits) - Math.min(...logits);
        // Config uses minScoreSpread for backward compat; interpret as logit threshold
        // in Phase 10A.  Old default 0.01 (sigmoid) → new default 0.5 (logits).
        const minSpread = cfg.spark!.minScoreSpread ?? 0.5;
        if (logitSpread < minSpread) {
          // Always-on: spread guard fired
          console.log(
            `[reranker] spread guard: logitSpread=${logitSpread.toFixed(4)} < minSpread=${minSpread} — falling back to input order`,
          );
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
    async rerank(_query, candidates, topN = 5, _options?: RerankOptions) {
      return candidates.slice(0, topN);
    },
    async probe() {
      return true;
    },
  };
}

// ── Logit Recovery (Phase 10A) ───────────────────────────────────────
//
// vLLM's /rerank endpoint applies sigmoid internally, converting the
// model's raw logits to relevance_score ∈ (0, 1).  For in-domain docs
// the logits cluster in a narrow band (e.g. +1 to +5), which sigmoid
// crushes to ~0.73–0.99 — a 0.26 effective range.  This destroys
// discrimination when we try to blend with vector scores in full [0, 1].
//
// Fix: recover the logits via inverse sigmoid (logit function), then
// min-max normalize them to [0, 1] — same scale as the vector signal.
//
// Math: logit = ln(s / (1 - s))
//       where s = sigmoid-compressed relevance_score from vLLM

/**
 * Recover the raw logit from a sigmoid-compressed relevance score.
 * Clamps to (ε, 1-ε) to avoid ±Infinity at boundaries.
 */
export function recoverLogit(sigmoidScore: number): number {
  const eps = 1e-7;
  const s = Math.max(eps, Math.min(1 - eps, sigmoidScore));
  return Math.log(s / (1 - s));
}

/**
 * Min-max normalize an array of values to [0, 1].
 * If all values are identical, returns 0.5 for each.
 */
function minMaxNormalize(values: number[]): number[] {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min;
  if (range === 0) return values.map(() => 0.5);
  return values.map((v) => (v - min) / range);
}

// ── Score Interpolation (Phase 9A → 10A) ────────────────────────────
//
// Blends original retrieval scores with cross-encoder reranker scores.
// Phase 10A upgrade: recover logits from sigmoid-compressed scores
// before normalizing, so the reranker's discrimination signal survives
// the blend instead of being crushed into a ~0.17 band.
//
// Formula: final = α × norm(original) + (1 - α) × norm(logit(reranker))
//   where both sides are min-max normalized to [0, 1].

/**
 * Blend original retrieval scores with reranker relevance scores.
 *
 * Phase 10A: Recovers reranker logits via inverse sigmoid before
 * normalizing, ensuring both signals compete on equal [0, 1] footing.
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
  // Recover logits from sigmoid-compressed scores
  const logits = rerankResults.map((r) => recoverLogit(r.relevance_score));
  const normLogits = minMaxNormalize(logits);

  if (alpha <= 0) {
    // Pure reranker mode — use recovered logits for better ordering
    return rerankResults
      .map((r, i) => ({
        ...pool[r.index]!,
        score: normLogits[i]!,
      }))
      .sort((a, b) => b.score - a.score);
  }

  // Normalize original scores to [0, 1] within this pool
  const originalScores = rerankResults.map((r) => pool[r.index]!.score);
  const normOrig = minMaxNormalize(originalScores);

  return rerankResults
    .map((r, i) => {
      const blended = alpha * normOrig[i]! + (1 - alpha) * normLogits[i]!;
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
