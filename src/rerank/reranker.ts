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
  /** Override the blend mode for this call. */
  blendModeOverride?: "score" | "rrf";
  /** Override RRF k constant for this call. */
  rrfKOverride?: number;
  /** Override vector weight for RRF for this call. */
  vectorWeightOverride?: number;
  /** Override reranker weight for RRF for this call. */
  rerankerWeightOverride?: number;
  /** Override reranker gate mode for this call. */
  gateOverride?: "off" | "hard" | "soft";
  /** Override reranker gate threshold for this call. */
  gateThresholdOverride?: number;
  /** Override reranker gate low threshold for this call. */
  gateLowThresholdOverride?: number;
}

export interface Reranker {
  rerank(
    query: string,
    candidates: SearchResult[],
    topN?: number,
    options?: RerankOptions,
  ): Promise<SearchResult[]>;
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
  const defaultBlendMode = cfg.blendMode ?? "rrf";
  const defaultRrfK = cfg.rrfK ?? 60;
  const defaultVectorWeight = cfg.rrfVectorWeight ?? 1.0;
  const defaultRerankerWeight = cfg.rrfRerankerWeight ?? 1.0;
  const defaultGateMode = cfg.rerankerGate ?? "hard";
  const defaultGateThreshold = cfg.rerankerGateThreshold ?? 0.08;
  const defaultGateLowThreshold = cfg.rerankerGateLowThreshold ?? 0.02;

  return {
    async rerank(query, candidates, topN = 5, options?: RerankOptions) {
      if (candidates.length === 0) return [];

      // Phase 10B: per-call alpha override for benchmark flexibility
      const effectiveAlpha = options?.alphaOverride ?? blendAlpha;
      const effectiveBlendMode = options?.blendModeOverride ?? defaultBlendMode;
      const effectiveRrfK = options?.rrfKOverride ?? defaultRrfK;
      let effectiveVectorWeight = options?.vectorWeightOverride ?? defaultVectorWeight;
      const effectiveRerankerWeight = options?.rerankerWeightOverride ?? defaultRerankerWeight;
      const effectiveGateMode = options?.gateOverride ?? defaultGateMode;
      const effectiveGateThreshold = options?.gateThresholdOverride ?? defaultGateThreshold;
      const effectiveGateLowThreshold =
        options?.gateLowThresholdOverride ?? defaultGateLowThreshold;

      // Phase 5B: Limit candidate pool to prevent unbounded reranker input
      const pool = candidates.slice(0, MAX_RERANK_CANDIDATES);

      // Phase 12 Fix 2: Dynamic Reranker Gate
      const gate = computeRerankerGate(
        pool,
        effectiveGateMode,
        effectiveGateThreshold,
        effectiveGateLowThreshold,
      );
      if (!gate.shouldRerank) {
        console.log(`[reranker] GATE SKIP: ${gate.reason} — returning vector order`);
        return pool.slice(0, topN);
      }
      if (effectiveBlendMode === "rrf" && gate.vectorWeightMultiplier !== 1.0) {
        // Soft gate: scale vector weight
        effectiveVectorWeight *= gate.vectorWeightMultiplier;
        if (process.env.VERBOSE || process.env.DEBUG_PIPELINE) {
          console.log(
            `[reranker] GATE SOFT: ${gate.reason} → effectiveVectorWeight=${effectiveVectorWeight.toFixed(3)}`,
          );
        }
      }

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
      let results: SearchResult[];
      if (effectiveBlendMode === "rrf") {
        // Phase 12 Fix 1: Rank-based fusion — scale-invariant, no normalization needed
        results = blendByRank(
          pool,
          data.results,
          effectiveRrfK,
          effectiveVectorWeight,
          effectiveRerankerWeight,
        ).slice(0, topN);
      } else {
        // Legacy score-based blending (Phase 9A/10A)
        results =
          effectiveAlpha > 0
            ? blendScores(pool, data.results, effectiveAlpha).slice(0, topN)
            : blendScores(pool, data.results, effectiveAlpha);
      }

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
          console.log(
            `[reranker]   --- per-candidate scores (top ${Math.min(results.length, 10)}) ---`,
          );
          for (let ci = 0; ci < Math.min(results.length, 10); ci++) {
            const res = results[ci]!;
            const rerankEntry = data.results.find(
              (dr) => pool[dr.index]?.chunk.id === res.chunk.id,
            );
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

// ── Dynamic Reranker Gate (Phase 12, Fix 2) ──────────────────────────
//
// Telemetry from 300 SciFact queries showed:
// - 13 queries LOST relevant docs when reranker reshuffled tight clusters
// - 9 queries GAINED relevant docs from reranker promotion
// - Net: reranker is harmful on tight clusters, helpful on clear separations
//
// The gate skips or dampens the reranker based on the top-5 vector score spread:
// - High spread (>0.08): vector is confident → skip reranker (it'll mess this up)
// - Low spread (<0.02): near-tie → skip reranker (it's gambling)
// - Medium spread: let the reranker help break the tie
//
// In "soft" mode, instead of hard skip, the vector weight in RRF is scaled
// by the spread, giving a smooth transition.

export interface RerankerGateResult {
  /** Whether to proceed with reranking */
  shouldRerank: boolean;
  /** For soft gate: dynamic vector weight multiplier (1.0 = full trust in vector) */
  vectorWeightMultiplier: number;
  /** Why the gate fired (for telemetry) */
  reason: string;
}

/**
 * Compute whether to gate (skip) the reranker based on vector score distribution.
 *
 * @param candidates — top candidates sorted by vector score (descending)
 * @param mode — "off" | "hard" | "soft"
 * @param threshold — spread at which vector is "confident" (default 0.08)
 * @param lowThreshold — spread below which candidates are "tied" (default 0.02)
 * @returns gate decision with reason and optional weight multiplier
 */
export function computeRerankerGate(
  candidates: SearchResult[],
  mode: "off" | "hard" | "soft" = "off",
  threshold: number = 0.08,
  lowThreshold: number = 0.02,
): RerankerGateResult {
  if (mode === "off" || candidates.length < 2) {
    return { shouldRerank: true, vectorWeightMultiplier: 1.0, reason: "gate-off" };
  }

  const top5 = candidates.slice(0, Math.min(5, candidates.length));
  const scores = top5.map((c) => c.score);
  const spread = Math.max(...scores) - Math.min(...scores);

  if (mode === "hard") {
    if (spread > threshold) {
      return {
        shouldRerank: false,
        vectorWeightMultiplier: 1.0,
        reason: `hard-gate-high: spread=${spread.toFixed(4)} > ${threshold} (vector confident)`,
      };
    }
    if (spread < lowThreshold) {
      return {
        shouldRerank: false,
        vectorWeightMultiplier: 1.0,
        reason: `hard-gate-low: spread=${spread.toFixed(4)} < ${lowThreshold} (tied set)`,
      };
    }
    return {
      shouldRerank: true,
      vectorWeightMultiplier: 1.0,
      reason: `hard-gate-pass: spread=${spread.toFixed(4)} in [${lowThreshold}, ${threshold}]`,
    };
  }

  // Soft gate: compute a dynamic vector weight multiplier
  // At spread=0: multiplier = 0.5 (let reranker have influence)
  // At spread=threshold: multiplier = 1.0 (full vector trust)
  // Above threshold: multiplier = 1.0 (capped)
  // Below lowThreshold: multiplier ramps up to 1.0 (don't trust reranker on ties either)
  let multiplier: number;
  if (spread >= threshold) {
    multiplier = 1.0;
  } else if (spread <= lowThreshold) {
    // Below low threshold: ramp from 0.8 at lowThreshold to 1.0 at spread=0
    // (ties → trust vector slightly more, but still let reranker try)
    multiplier = 0.8 + 0.2 * (1 - spread / lowThreshold);
  } else {
    // Linear interpolation in the useful range [lowThreshold, threshold]
    // At lowThreshold: multiplier = 0.5 (reranker gets max influence)
    // At threshold: multiplier = 1.0 (reranker has no influence)
    const t = (spread - lowThreshold) / (threshold - lowThreshold);
    multiplier = 0.5 + 0.5 * t;
  }

  return {
    shouldRerank: true,
    vectorWeightMultiplier: multiplier,
    reason: `soft-gate: spread=${spread.toFixed(4)} → vectorMultiplier=${multiplier.toFixed(3)}`,
  };
}

// ── Rank-Based Fusion (Phase 12, Fix 1) ──────────────────────────────
//
// Score-based blending (blendScores) fails because min-max normalization
// after any monotonic transform is a no-op — the ranking is identical
// regardless of whether we apply recoverLogit first. Proven by telemetry:
// Configs M=T and N=Q produce identical NDCG/MRR/Recall/MAP.
//
// RRF (Reciprocal Rank Fusion) works on rank positions, not scores.
// It's scale-invariant and doesn't need normalization. The formula:
//   rrfScore(doc) = Σ weight_i / (k + rank_i)
// where k is a constant (60 is standard) that controls how much
// top ranks dominate over lower ranks.
//
// Key advantage: when vector and reranker AGREE on top-1, that doc
// gets double the RRF credit. When they DISAGREE, neither can
// catastrophically demote the other's pick.

/**
 * Blend vector and reranker results using Reciprocal Rank Fusion.
 *
 * Unlike score-based blending, RRF is scale-invariant — it doesn't matter
 * if reranker scores are 0.83–1.0 or 0–100. It fuses based on rank positions.
 *
 * @param pool — original candidates sorted by vector/retrieval score
 * @param rerankResults — reranker output sorted by relevance_score (descending)
 * @param k — RRF constant (default 60). Lower k = top ranks dominate more.
 * @param vectorWeight — weight for vector rank contribution (default 1.0)
 * @param rerankerWeight — weight for reranker rank contribution (default 1.0)
 * @returns sorted SearchResult[] with RRF fusion scores
 */
export function blendByRank(
  pool: SearchResult[],
  rerankResults: Array<{ index: number; relevance_score: number }>,
  k: number = 60,
  vectorWeight: number = 1.0,
  rerankerWeight: number = 1.0,
): SearchResult[] {
  if (pool.length === 0 || rerankResults.length === 0) return [];

  const rrfScores = new Map<number, number>(); // index → rrf score

  // Vector rank contribution (pool is already sorted by vector score descending)
  for (let rank = 0; rank < pool.length; rank++) {
    const idx = rank; // pool index IS the vector rank
    rrfScores.set(idx, (rrfScores.get(idx) ?? 0) + vectorWeight / (k + rank + 1));
  }

  // Reranker rank contribution (rerankResults sorted by relevance_score descending)
  // Sort by relevance_score to get reranker ranks
  const sortedRerank = [...rerankResults].sort((a, b) => b.relevance_score - a.relevance_score);
  for (let rank = 0; rank < sortedRerank.length; rank++) {
    const idx = sortedRerank[rank]!.index;
    rrfScores.set(idx, (rrfScores.get(idx) ?? 0) + rerankerWeight / (k + rank + 1));
  }

  // Build results sorted by RRF score
  return [...rrfScores.entries()]
    .sort(([, a], [, b]) => b - a)
    .map(([idx, score]) => ({
      ...pool[idx]!,
      score, // RRF fusion score
    }));
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
  "who",
  "what",
  "which",
  "when",
  "where",
  "why",
  "how",
  "is",
  "are",
  "was",
  "were",
  "do",
  "does",
  "did",
  "can",
  "could",
  "will",
  "would",
  "should",
  "shall",
  "has",
  "have",
  "had",
  "may",
  "might",
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
