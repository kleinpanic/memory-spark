/**
 * Auto-Recall — before_prompt_build hook.
 * Matches memory-lancedb: XML-wrapped injection with security preamble.
 * Adds: MMR diversity, temporal decay, prompt injection filtering.
 */

import type { AutoRecallConfig, RecallWeights, HydeConfig } from "../config.js";
import { shouldProcessAgent } from "../config.js";
import type { EmbedLike } from "../embed/cached-provider.js";
import type { EmbedProvider } from "../embed/provider.js";
import type { EmbedQueue } from "../embed/queue.js";
import { generateHypotheticalDocument } from "../hyde/generator.js";
import {
  expandQuery,
  QUERY_EXPANSION_DEFAULTS,
  type QueryExpansionConfig,
} from "../query/expander.js";
import type { Reranker } from "../rerank/reranker.js";
import { looksLikePromptInjection, formatRecalledMemories } from "../security.js";
import type { StorageBackend, SearchResult } from "../storage/backend.js";

type BeforePromptBuildEvent = { prompt: string; messages: unknown[] };
type BeforePromptBuildResult = { systemPrompt?: string; prependContext?: string };
type HookContext = { agentId?: string; sessionKey?: string };

export interface AutoRecallDeps {
  cfg: AutoRecallConfig;
  backend: StorageBackend;
  embed: EmbedProvider | EmbedQueue | EmbedLike;
  reranker: Reranker;
  /** HyDE config — when enabled, queries are expanded via hypothetical document generation */
  hyde?: HydeConfig;
}

export function createAutoRecallHandler(deps: AutoRecallDeps) {
  return async function recallHandler(
    event: BeforePromptBuildEvent,
    ctx: HookContext,
  ): Promise<BeforePromptBuildResult | undefined> {
    const { cfg, backend, embed, reranker } = deps;

    if (!cfg.enabled) return undefined;
    const agentId = ctx.agentId ?? "unknown";
    if (!shouldProcessAgent(agentId, cfg.agents, cfg.ignoreAgents ?? [])) return undefined;

    const rawQueryText = buildQuery(event.messages, cfg.queryMessageCount);
    const queryText = cleanQueryText(rawQueryText);
    // Skip trivial messages that won't produce meaningful recall
    if (!queryText.trim() || queryText.trim().length < 4) return undefined;

    // ── HyDE: Hypothetical Document Embeddings ───────────────────────────
    //
    // Instead of embedding the raw query ("What model does Spark use?"),
    // generate a hypothetical answer document and embed THAT. The hypothetical
    // doc lives in the same semantic space as the stored knowledge, bridging
    // the question-answer gap that hurts naive query embedding.
    //
    // FTS still uses the raw queryText — keyword matching needs real terms.
    //
    const verbose = process.env.VERBOSE || process.env.DEBUG_PIPELINE;

    let queryVector: number[];
    try {
      const hydeConfig = deps.hyde;
      if (hydeConfig?.enabled) {
        console.log(
          `[recall] HyDE: generating hypothetical document for query="${queryText.slice(0, 60)}"`,
        );
        const hypothetical = await generateHypotheticalDocument(queryText, hydeConfig);
        if (hypothetical) {
          console.log(
            `[recall] HyDE: success, doc length=${hypothetical.length} chars — embedding as document`,
          );
          if (verbose) {
            console.log(
              `[recall]   HyDE doc preview: "${hypothetical.slice(0, 120).replace(/\n/g, " ")}…"`,
            );
          }
          // HyDE: embed the hypothetical as a DOCUMENT (no instruction prefix).
          // The hypothetical is a pseudo-document and must land in document embedding
          // space, not query space. This is the correct approach per Gao et al. 2022.
          queryVector = await embed.embedDocument(hypothetical);
        } else {
          // HyDE failed (timeout, too short, etc.) — fall back to raw query
          console.log(
            `[recall] HyDE: generation returned empty/null — falling back to raw query embedding`,
          );
          queryVector = await embed.embedQuery(queryText);
        }
      } else {
        queryVector = await embed.embedQuery(queryText);
      }
    } catch (hydeErr) {
      console.log(
        `[recall] HyDE/embed error: ${hydeErr instanceof Error ? hydeErr.message : String(hydeErr)} — aborting recall`,
      );
      return undefined;
    }

    // ── Pool-Aware Recall Pipeline ─────────────────────────────────────────
    //
    // The recall pipeline queries multiple pools in priority order and merges
    // results. Each pool has different auto-injection behavior:
    //
    //   1. Agent memory + tools   → filtered by agent_id, primary recall
    //   2. Agent mistakes         → filtered by agent_id, 1.6x boost
    //   3. Shared mistakes        → cross-agent, 1.6x boost
    //   4. Shared knowledge       → cross-agent, 0.8x weight
    //   5. Shared rules           → cross-agent, relevance-gated (lowThreshold)
    //   6. Reference pools        → NEVER auto-injected (tool-call only)
    //
    // Within each pool: hybrid search (Vector + FTS) → merge
    // FTS+WHERE is supported in LanceDB 0.27+ (no workaround needed).

    const fetchN = cfg.maxResults * (cfg.overfetchMultiplier ?? 4);
    const lowThreshold = Math.max((cfg.minScore ?? 0.1) * 0.7, 0.05);

    // ── Phase 11B: Multi-Query Expansion ─────────────────────────────────
    // Generate query reformulations and embed them for multi-vector search.
    // Done ONCE here, then passed to poolSearch for all pool searches.
    let allQueryVectors: number[][] = [queryVector];
    const mqConfig = cfg.queryExpansion;
    if (mqConfig?.enabled) {
      const resolvedMqConfig: QueryExpansionConfig = {
        ...QUERY_EXPANSION_DEFAULTS,
        ...mqConfig,
      };
      const t0mq = performance.now();
      const queries = await expandQuery(queryText, resolvedMqConfig);
      if (queries.length > 1) {
        // Embed reformulations in parallel (skip first — it's the original, already embedded)
        const extraVectors = await Promise.all(
          queries.slice(1).map((mq) => embed.embedQuery(mq).catch(() => null)),
        );
        const validExtra = extraVectors.filter((v): v is number[] => v !== null);
        allQueryVectors = [queryVector, ...validExtra];
        console.log(
          `[recall] multi-query: ${queries.length} queries, ${allQueryVectors.length} vectors in ${Math.round(performance.now() - t0mq)}ms`,
        );
        if (verbose) {
          for (const mq of queries.slice(1))
            console.log(`[recall]   reformulation: "${mq.slice(0, 100)}"`);
        }
      }
    }

    // Helper: hybrid search within specific pool(s)
    const poolSearch = async (
      pools: string[],
      filterAgentId?: string,
      limit = fetchN,
      minScore = cfg.minScore,
    ): Promise<SearchResult[]> => {
      const searchOpts = {
        query: queryText,
        maxResults: limit,
        minScore,
        agentId: filterAgentId,
        pools,
      };

      // Multi-vector search: search each query vector and union by chunk ID
      async function searchOne(vec: number[]): Promise<SearchResult[]> {
        try {
          return await backend.vectorSearch(vec, searchOpts);
        } catch {
          return [];
        }
      }
      let vectorResults: SearchResult[];
      if (allQueryVectors.length > 1) {
        const allVecResults = await Promise.all(allQueryVectors.map(searchOne));
        // Union: dedupe by chunk ID, keep highest score
        const best = new Map<string, SearchResult>();
        for (const resultSet of allVecResults) {
          for (const r of resultSet) {
            const existing = best.get(r.chunk.id);
            if (!existing || r.score > existing.score) {
              best.set(r.chunk.id, r);
            }
          }
        }
        vectorResults = [...best.values()].sort((a, b) => b.score - a.score);
      } else {
        vectorResults = await backend
          .vectorSearch(queryVector, searchOpts)
          .catch(() => [] as SearchResult[]);
      }
      const rawFtsResults =
        (cfg.ftsEnabled ?? true)
          ? await backend.ftsSearch(queryText, searchOpts).catch(() => [] as SearchResult[])
          : [];
      // Filter FTS: exclude sessions source only.
      // Do NOT apply minScore to FTS results — BM25 scores are sigmoid-normalized
      // and nearly all map to >0.98, making minScore non-discriminative.
      // RRF handles FTS ranking by position (rank-only), not score magnitude.
      const ftsResults = rawFtsResults.filter((r) => r.chunk.source !== "sessions");
      const merged = hybridMerge(
        vectorResults,
        ftsResults,
        limit,
        60,
        cfg.hybridVectorWeight ?? 1.0,
        cfg.hybridFtsWeight ?? 1.0,
      );
      // Exclude parent chunks from search results — they exist for context
      // expansion only. Children are the precise search targets; after reranking,
      // matched children get expanded to their parent's text for delivery.
      return merged.filter((r) => !r.chunk.is_parent);
    };

    // 1. Agent's own memory + tools (primary recall)
    const agentResults = await poolSearch(["agent_memory", "agent_tools"], agentId, fetchN);
    console.log(
      `[recall] pool agent_memory+agent_tools (agent=${agentId}): ${agentResults.length} results`,
    );
    if (verbose && agentResults.length > 0) {
      for (const r of agentResults.slice(0, 3)) {
        console.log(
          `[recall]   score=${r.score.toFixed(4)} pool=${r.chunk.pool} path=${r.chunk.path} id=${r.chunk.id.slice(0, 20)}`,
        );
      }
    }

    // 2. Agent's own mistakes (per-agent, higher priority than shared)
    const agentMistakes = await poolSearch(["agent_mistakes"], agentId, 5, lowThreshold);
    console.log(`[recall] pool agent_mistakes (agent=${agentId}): ${agentMistakes.length} results`);

    // 3. Shared mistakes (cross-agent)
    const sharedMistakes = await poolSearch(
      ["shared_mistakes"],
      undefined, // No agent filter — all agents' shared mistakes
      5,
      lowThreshold,
    );
    console.log(`[recall] pool shared_mistakes (cross-agent): ${sharedMistakes.length} results`);

    // 4. Shared knowledge (cross-agent)
    const sharedKnowledge = await poolSearch(["shared_knowledge"], undefined, 10, lowThreshold);
    console.log(`[recall] pool shared_knowledge (cross-agent): ${sharedKnowledge.length} results`);

    // 5. Shared rules — relevance-gated like all other pools
    const sharedRules = await poolSearch(["shared_rules"], undefined, 5, lowThreshold);
    console.log(`[recall] pool shared_rules (cross-agent): ${sharedRules.length} results`);

    // Merge all results, deduplicating by chunk ID
    const merged: SearchResult[] = [...agentResults];
    const seenIds = new Set(merged.map((r) => r.chunk.id));

    const mergeIn = (results: SearchResult[]) => {
      for (const r of results) {
        if (!seenIds.has(r.chunk.id)) {
          merged.push(r);
          seenIds.add(r.chunk.id);
        }
      }
    };

    mergeIn(agentMistakes);
    mergeIn(sharedMistakes);
    mergeIn(sharedKnowledge);
    mergeIn(sharedRules);

    console.log(`[recall] merged (all pools, deduped): ${merged.length} candidates`);
    if (merged.length === 0) return undefined;

    // P2-B fix: dedup on RAW cosine scores BEFORE weighting.
    // Previously dedup ran after source weighting, so Jaccard kept the "highest score"
    // among near-duplicate chunks — but those scores were already inflated to 1.0
    // by mistake/capture boosts, making the dedup outcome arbitrary.
    // Now: dedup on real cosine similarity → weighting → decay → reranker.
    const deduped = deduplicateSources(merged);
    console.log(
      `[recall] after source dedup: ${deduped.length} candidates (removed ${merged.length - deduped.length})`,
    );

    // P1-A fix: source weighting runs AFTER dedup, on the cleaned candidate set.
    // Weights are applied WITHOUT Math.min(1.0) clamping — scores are allowed to
    // exceed 1.0 so the reranker gate sees a real spread between boosted and
    // non-boosted chunks. A post-weighting normalization step rescales the top
    // score back to 1.0, preserving relative ordering while keeping gate spread intact.
    applySourceWeighting(deduped, cfg.weights);

    // Temporal decay: boost recent memories (configurable floor + rate)
    applyTemporalDecay(deduped, cfg.temporalDecay);

    // Post-weighting normalization: rescale so top score = 1.0.
    // This ensures minScore downstream filters work correctly AND the gate spread
    // reflects the real relative differences between boosted and unboosted chunks,
    // not a ceiling artifact from Math.min(1.0) clamping.
    normalizeScores(deduped);

    // Cross-encoder rerank — the reranker is the most accurate relevance signal.
    // Give it a large candidate set so it can identify the best documents.
    // Then MMR trims the reranked output for diversity.
    //
    // Phase 13: Gate now sees real score spread (P1-A fix). When multiple mistake
    // chunks are boosted, their spread reflects actual cosine differences rather
    // than collapsing to 0 due to clamping.
    const reranked = await reranker.rerank(queryText, deduped, cfg.maxResults * 2);
    console.log(`[recall] after rerank: ${reranked.length} candidates`);

    // MMR diversity re-ranking on reranker-approved candidates.
    // This prevents MMR from discarding documents the reranker would have rescued.
    // (Pipeline ordering fix: retrieve → rerank → MMR per NVIDIA RAG Blueprint)
    const mmrLambda = cfg.mmrLambda ?? 0.9;

    // Log adaptive lambda details for diagnostics
    const adaptiveLambdaInfo = computeAdaptiveLambda(reranked);
    console.log(
      `[recall] MMR: configured λ=${mmrLambda} | adaptive would be λ=${adaptiveLambdaInfo.lambda} ` +
        `(spread=${adaptiveLambdaInfo.spread.toFixed(3)}, tier=${adaptiveLambdaInfo.tier})`,
    );

    if (verbose && reranked.length > 0) {
      console.log(`[recall]   MMR input scores (top ${Math.min(reranked.length, 5)}):`);
      for (const r of reranked.slice(0, 5)) {
        console.log(
          `[recall]     score=${r.score.toFixed(4)} id=${r.chunk.id.slice(0, 20)} path=${r.chunk.path}`,
        );
      }
    }

    const diverse = mmrRerank(reranked, cfg.maxResults, mmrLambda);
    console.log(
      `[recall] after MMR: ${diverse.length} results (removed ${reranked.length - diverse.length} near-duplicates)`,
    );

    if (verbose && diverse.length > 0) {
      console.log(`[recall]   MMR output (final candidates):`);
      for (const r of diverse) {
        console.log(
          `[recall]     score=${r.score.toFixed(4)} id=${r.chunk.id.slice(0, 20)} path=${r.chunk.path} pool=${r.chunk.pool}`,
        );
      }
    }

    if (diverse.length === 0) return undefined;

    // ── Parent-Child Context Expansion ───────────────────────────────────
    //
    // If child chunks were retrieved (they have parent_id), look up parent
    // chunks and use the PARENT text for context injection instead of the
    // child text. This gives the LLM much more surrounding context while
    // keeping search precision high (small children embed precisely).
    //
    // Dedup: if multiple children share a parent, include parent only once.
    //
    const parentIds = new Set<string>();
    for (const r of diverse) {
      if (r.chunk.parent_id && !r.chunk.is_parent) {
        parentIds.add(r.chunk.parent_id);
      }
    }

    if (parentIds.size > 0) {
      try {
        const parentChunks = await backend.getByIds(Array.from(parentIds));
        const parentMap = new Map(parentChunks.map((p) => [p.id, p]));

        // Replace child text with parent text (keeping child's score and metadata)
        for (const r of diverse) {
          if (r.chunk.parent_id && !r.chunk.is_parent) {
            const parent = parentMap.get(r.chunk.parent_id);
            if (parent) {
              r.chunk.text = parent.text;
              // Mark that we expanded, so we can dedup parent text
              r.chunk.parent_id = `expanded:${r.chunk.parent_id}`;
            }
          }
        }

        // Dedup: if multiple children mapped to the same parent, keep only
        // the highest-scoring one. Use the expanded parent_id as the dedup key
        // (it's "expanded:<parentId>") — far more reliable than text prefix matching.
        const seenParents = new Set<string>();
        const dedupedResults: SearchResult[] = [];
        for (const r of diverse) {
          const expandedTag = r.chunk.parent_id;
          if (expandedTag?.startsWith("expanded:")) {
            if (!seenParents.has(expandedTag)) {
              seenParents.add(expandedTag);
              dedupedResults.push(r);
            }
            // else skip — a higher-scoring sibling already claimed this parent
          } else {
            // Non-expanded chunks (flat or failed lookup) pass through
            dedupedResults.push(r);
          }
        }
        // Use deduplicated results for the rest of the pipeline
        diverse.splice(0, diverse.length, ...dedupedResults);
      } catch {
        // Parent lookup failed — continue with child text (graceful degradation)
      }
    }

    // LCM + recency dedup — skip chunks that overlap heavily with:
    // 1. Recent conversation messages
    // 2. LCM summary content already in context
    // This prevents memory-spark from injecting something LCM already provides.
    const allContextTexts: string[] = [];

    // Recent messages
    const recentTexts = event.messages
      .slice(-cfg.queryMessageCount)
      .map(extractMessageText)
      .map(cleanQueryText)
      .filter(Boolean);
    allContextTexts.push(...recentTexts);

    // LCM summaries — extract text from any <summary> or <content> blocks in messages
    for (const msg of event.messages) {
      const text = extractMessageText(msg);
      // Find LCM summary content blocks
      const summaryMatches = text.match(/<content>([\s\S]*?)<\/content>/g);
      if (summaryMatches) {
        for (const match of summaryMatches) {
          const inner = match.replace(/<\/?content>/g, "").trim();
          if (inner.length > 50) allContextTexts.push(cleanQueryText(inner));
        }
      }
    }

    const dedupThreshold = cfg.dedupOverlapThreshold ?? 0.4;
    const deduplicated = diverse.filter((r) => {
      const chunkTokens = new Set(
        (r.chunk.text.match(/\b\w{4,}\b/g) ?? []).map((w) => w.toLowerCase()),
      );
      if (chunkTokens.size === 0) return true;
      for (const ctx of allContextTexts) {
        const ctxTokens = new Set((ctx.match(/\b\w{4,}\b/g) ?? []).map((w) => w.toLowerCase()));
        let overlap = 0;
        for (const t of chunkTokens) {
          if (ctxTokens.has(t)) overlap++;
        }
        if (overlap / chunkTokens.size > dedupThreshold) return false;
      }
      return true;
    });

    // Filter prompt injection + format with rich metadata
    const safeMemories = deduplicated
      .filter((r) => !looksLikePromptInjection(r.chunk.text))
      .map((r) => ({
        // Prefix with "memory-spark:" so agents know which plugin this came from
        source: `memory-spark:${r.chunk.source}:${r.chunk.path}`,
        // No hard truncation here — token budget enforcement below handles sizing.
        // Parent chunks (~2000 tokens) need to flow through intact after expansion.
        text: r.chunk.text,
        score: r.score,
        // Use the ORIGINAL content timestamp, not when it was re-indexed.
        // For captures, updated_at is when the capture was stored (correct).
        // For file-sourced chunks, updated_at is file mtime (correct).
        // The bug was that garbage captures had stale content with fresh timestamps.
        updatedAt: r.chunk.updated_at,
        contentType: r.chunk.content_type ?? "knowledge",
        agentId: r.chunk.agent_id,
        path: r.chunk.path,
      }));

    if (safeMemories.length === 0) return undefined;

    // Token budget enforcement
    const maxTokens = cfg.maxInjectionTokens ?? 2000;
    let totalTokens = 0;
    const budgeted: typeof safeMemories = [];
    for (const mem of safeMemories) {
      const tokens = Math.ceil(mem.text.split(/\s+/).length * 1.3);
      if (totalTokens + tokens > maxTokens) break;
      totalTokens += tokens;
      budgeted.push(mem);
    }

    if (budgeted.length === 0) return undefined;

    return { prependContext: formatRecalledMemories(budgeted) };
  };
}

/**
 * Hybrid merge — preserves original vector similarity scores.
 *
 * The old rrfMerge() replaced all scores with 1/(k+rank), destroying the
 * cosine similarity signal from the embedding model. A chunk with cosine 0.85
 * became 0.0167 — same as a junk chunk at FTS rank 0. This caused the full
 * pipeline to perform WORSE than vanilla vector search.
 *
 * This version:
 * 1. Keeps original vector scores (cosine similarity) as the base
 * 2. Adds a small RRF rank boost for chunks found by both sources
 * 3. FTS-only chunks get a normalized score based on their FTS rank
 *
 * Chunks appearing in BOTH lists get a bonus (evidence from two signals).
 */
/**
 * Reciprocal Rank Fusion (RRF) — combines vector and FTS result lists using
 * rank positions only, avoiding the scale-mismatch problem of mixing cosine
 * similarity with BM25 scores.
 *
 * Formula per document: RRF(d) = Σ 1/(k + rank_i) for each list containing d
 * where rank_i is 1-indexed.
 *
 * Documents found in BOTH lists get the sum of both reciprocal ranks, naturally
 * promoting dual-evidence matches. Output scores are normalized to [0, 1] so
 * downstream minScore filters work correctly (top result = 1.0).
 *
 * @param vectorResults — results from vector (semantic) search, ordered by relevance
 * @param ftsResults — results from full-text (BM25) search, ordered by relevance
 * @param limit — maximum results to return
 * @param k — RRF smoothing constant (default 60, same as Elasticsearch/Azure)
 * @see https://plg.uwaterloo.ca/~grcorcor/topicmodels/rrf.pdf
 */
/**
 * Compute overlap ratio between two result sets (by document ID).
 * Returns a value in [0, 1] representing how many of the top-K vector
 * results also appear anywhere in the FTS results.
 */
export function computeOverlap(
  vectorResults: SearchResult[],
  ftsResults: SearchResult[],
  topK = 10,
): number {
  const vecIds = new Set(vectorResults.slice(0, topK).map((r) => r.chunk.id));
  const ftsIds = new Set(ftsResults.map((r) => r.chunk.id));
  if (vecIds.size === 0) return 0;
  let overlap = 0;
  for (const id of vecIds) {
    if (ftsIds.has(id)) overlap++;
  }
  return overlap / vecIds.size;
}

/**
 * Overlap-Aware Adaptive Hybrid Merge (Phase 8 Fix 1).
 *
 * Instead of static RRF weights, dynamically adjusts vector vs FTS influence
 * based on how much the two retrieval systems agree at query time.
 *
 * - High overlap (>0.6): Standard RRF — both systems agree, dual evidence is meaningful.
 * - Medium overlap (0.3–0.6): Vector-biased RRF — trust vector more, FTS supplements.
 * - Low overlap (<0.3): Vector-primary — FTS-only docs heavily demoted, only dual-evidence FTS promoted.
 *
 * When mode="static", falls back to original fixed-weight behavior for backward compat.
 */
export function hybridMerge(
  vectorResults: SearchResult[],
  ftsResults: SearchResult[],
  limit: number,
  k = 60,
  vectorWeight = 1.0,
  ftsWeight = 1.0,
  mode: "adaptive" | "static" = "static",
): SearchResult[] {
  // Compute overlap to determine fusion strategy
  const overlapRatio = computeOverlap(vectorResults, ftsResults);

  // Adaptive weight selection based on overlap
  let effectiveVecWeight = vectorWeight;
  let effectiveFtsWeight = ftsWeight;
  let strategy: string;

  if (mode === "adaptive") {
    if (overlapRatio > 0.6) {
      // High agreement — standard RRF, both systems see the same docs
      effectiveVecWeight = 1.0;
      effectiveFtsWeight = 1.0;
      strategy = "high-overlap-balanced";
    } else if (overlapRatio > 0.3) {
      // Medium agreement — vector primary, FTS supplements
      effectiveVecWeight = 1.5;
      effectiveFtsWeight = 0.5;
      strategy = "medium-overlap-vec-biased";
    } else {
      // Low agreement — vector dominant, FTS barely contributes
      // FTS-only docs get heavily penalized; only dual-evidence docs benefit
      effectiveVecWeight = 2.0;
      effectiveFtsWeight = 0.3;
      strategy = "low-overlap-vec-primary";
    }
  } else {
    strategy = "static";
  }

  const merged = new Map<string, { result: SearchResult; rrfScore: number; sources: number }>();

  // Vector results: weighted RRF score from rank position only (1-indexed per the paper)
  vectorResults.forEach((r, idx) => {
    merged.set(r.chunk.id, {
      result: r,
      rrfScore: effectiveVecWeight * (1 / (k + idx + 1)),
      sources: 1,
    });
  });

  // FTS results: weighted RRF score (sums when document found in both lists)
  ftsResults.forEach((r, idx) => {
    const id = r.chunk.id;
    const ftsRrf = effectiveFtsWeight * (1 / (k + idx + 1));
    const existing = merged.get(id);
    if (existing) {
      existing.rrfScore += ftsRrf;
      existing.sources = 2;
    } else {
      merged.set(id, {
        result: r,
        rrfScore: ftsRrf,
        sources: 1,
      });
    }
  });

  // Sort by RRF score descending, normalize to [0, 1] for downstream minScore filters
  const sorted = Array.from(merged.values()).sort((a, b) => b.rrfScore - a.rrfScore);
  const maxRrf = sorted[0]?.rrfScore ?? 1;

  if (process.env.MEMORY_SPARK_DEBUG) {
    const dualEvidence = sorted.filter((s) => s.sources === 2).length;
    console.debug(
      `[hybridMerge] strategy=${strategy} overlap=${overlapRatio.toFixed(2)} total=${sorted.length} dualEvidence=${dualEvidence} vecW=${effectiveVecWeight} ftsW=${effectiveFtsWeight}`,
    );
  }

  return sorted.slice(0, limit).map((s) => ({
    ...s.result,
    score: maxRrf > 0 ? s.rrfScore / maxRrf : 0,
  }));
}

/**
 * Reranker-as-Fusioner (Phase 8 Fix 2).
 *
 * Instead of RRF → Reranker (reranker cleans up RRF mess), this takes the
 * raw union of vector and FTS results and lets the cross-encoder score
 * each (query, doc) pair independently. The reranker IS the fusion step.
 *
 * This follows the NVIDIA RAG Blueprint: retrieve broadly, rerank precisely.
 */
export function prepareRerankerFusion(
  vectorResults: SearchResult[],
  ftsResults: SearchResult[],
  limit: number,
): SearchResult[] {
  // Deduplicate by ID, keeping the version with the higher original score
  const seen = new Map<string, SearchResult>();
  for (const r of vectorResults) {
    seen.set(r.chunk.id, r);
  }
  for (const r of ftsResults) {
    const existing = seen.get(r.chunk.id);
    if (!existing || r.score > existing.score) {
      seen.set(r.chunk.id, r);
    }
  }

  // Sort by original score descending (this ordering doesn't matter much
  // since the reranker will rescore everything, but limits candidate count)
  const candidates = Array.from(seen.values()).sort((a, b) => b.score - a.score);

  if (process.env.MEMORY_SPARK_DEBUG) {
    const vecOnly = vectorResults.filter(
      (r) => !ftsResults.some((f) => f.chunk.id === r.chunk.id),
    ).length;
    const ftsOnly = ftsResults.filter(
      (r) => !vectorResults.some((v) => v.chunk.id === r.chunk.id),
    ).length;
    const both = candidates.length - vecOnly - ftsOnly;
    console.debug(
      `[rerankerFusion] candidates=${candidates.length} vecOnly=${vecOnly} ftsOnly=${ftsOnly} both=${both}`,
    );
  }

  return candidates.slice(0, limit);
}

/**
 * Temporal decay: boost recent memories, apply gentle decay for old ones.
 * Uses exponential decay with a configurable floor.
 *
 * Formula: floor + (1 - floor) * exp(-rate * ageDays)
 * Default: floor=0.8, rate=0.03
 *   0 days=1.0, 7 days=0.96, 30 days=0.89, 90 days=0.81, 365 days=0.80
 *
 * @param results — search results to apply decay to (mutated in place)
 * @param opts — optional floor (default 0.8) and rate (default 0.03)
 */
export function applyTemporalDecay(
  results: SearchResult[],
  opts?: { floor?: number; rate?: number },
): void {
  const floor = opts?.floor ?? 0.8;
  const rate = opts?.rate ?? 0.03;
  const now = Date.now();
  for (const r of results) {
    const rawTime = r.chunk.updated_at ? new Date(r.chunk.updated_at).getTime() : NaN;
    // Guard against NaN from invalid/missing timestamps — use score as-is (decay = 1.0)
    if (Number.isNaN(rawTime)) continue;
    const ageDays = Math.max(0, (now - rawTime) / (86400 * 1000));
    const decay = floor + (1 - floor) * Math.exp(-rate * ageDays);
    r.score *= decay;
  }
}

/**
 * Source-level deduplication: collapse near-identical chunks from the same source.
 * Groups results by parent_id (or path if no parent). Within each group,
 * if two chunks have Jaccard token overlap > 0.85, keep only the higher-scoring one.
 * Cross-source chunks are never deduped (different sources = different context).
 *
 * Jaccard on tokens IS the right metric here (unlike MMR) — we're checking for
 * near-identical text, not semantic similarity.
 */
export function deduplicateSources(results: SearchResult[]): SearchResult[] {
  if (results.length <= 1) return results;

  // Group by source key: parent_id if available, else path
  const groups = new Map<string, SearchResult[]>();
  for (const r of results) {
    const key = r.chunk.parent_id ?? r.chunk.path ?? r.chunk.id;
    const group = groups.get(key);
    if (group) group.push(r);
    else groups.set(key, [r]);
  }

  const kept: SearchResult[] = [];
  for (const group of groups.values()) {
    if (group.length <= 1) {
      kept.push(...group);
      continue;
    }

    // Sort by score descending within group
    group.sort((a, b) => b.score - a.score);

    // Greedy dedup: keep a chunk unless it's >85% token overlap with an already-kept chunk
    const keptInGroup: SearchResult[] = [];
    const keptTokens: Set<string>[] = [];

    for (const r of group) {
      const tokens = new Set((r.chunk.text.match(/\b\w{3,}\b/g) ?? []).map((w) => w.toLowerCase()));
      let isDuplicate = false;
      for (const existingTokens of keptTokens) {
        const sim = jaccardSimilaritySet(tokens, existingTokens);
        if (sim > 0.85) {
          isDuplicate = true;
          break;
        }
      }
      if (!isDuplicate) {
        keptInGroup.push(r);
        keptTokens.push(tokens);
      }
    }
    kept.push(...keptInGroup);
  }

  return kept;
}

// Standalone Jaccard for source dedup (not exported — internal utility)
function jaccardSimilaritySet(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 1;
  let intersection = 0;
  for (const token of a) {
    if (b.has(token)) intersection++;
  }
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

/**
 * Maximum Marginal Relevance — ensures result diversity.
 * Iteratively selects results that are relevant but dissimilar to already-selected.
 * Uses cosine similarity on embedding vectors (semantic diversity).
 * Falls back to Jaccard similarity on token sets when vectors are unavailable.
 */
/**
 * Compute adaptive MMR lambda based on the score distribution (Phase 8 Fix 3).
 *
 * Wide spread → ranking is confident → high lambda (trust relevance, ~0.95)
 * Medium spread → moderate confidence → balanced lambda (~0.85)
 * Tight cluster → ranker can't distinguish → lower lambda (~0.7, diversity helps)
 *
 * Returns computed lambda and the spread for telemetry.
 */
export function computeAdaptiveLambda(
  results: SearchResult[],
  opts?: {
    highSpreadThreshold?: number;
    lowSpreadThreshold?: number;
    highLambda?: number;
    midLambda?: number;
    lowLambda?: number;
  },
): { lambda: number; spread: number; tier: "wide" | "medium" | "tight" } {
  if (results.length <= 1) return { lambda: 0.9, spread: 0, tier: "wide" };

  const scores = results.map((r) => r.score);
  const maxScore = Math.max(...scores);
  const minScore = Math.min(...scores);
  const spread = maxScore - minScore;

  const highThresh = opts?.highSpreadThreshold ?? 0.3;
  const lowThresh = opts?.lowSpreadThreshold ?? 0.1;

  if (spread > highThresh) {
    return { lambda: opts?.highLambda ?? 0.95, spread, tier: "wide" };
  } else if (spread > lowThresh) {
    return { lambda: opts?.midLambda ?? 0.85, spread, tier: "medium" };
  } else {
    return { lambda: opts?.lowLambda ?? 0.7, spread, tier: "tight" };
  }
}

/**
 * MMR reranking with optional adaptive lambda (Phase 8).
 *
 * When lambda is a number, uses fixed lambda (backward compat).
 * When lambda is "adaptive", computes lambda from score distribution.
 */
export function mmrRerank(
  results: SearchResult[],
  limit: number,
  lambda: number | "adaptive",
): SearchResult[] {
  if (results.length <= 1) return results;

  // Resolve lambda — adaptive or fixed
  let effectiveLambda: number;
  let adaptiveInfo: { spread: number; tier: string } | undefined;
  if (lambda === "adaptive") {
    const computed = computeAdaptiveLambda(results);
    effectiveLambda = computed.lambda;
    adaptiveInfo = { spread: computed.spread, tier: computed.tier };
  } else {
    effectiveLambda = lambda;
  }

  // Determine similarity strategy: cosine on vectors if available, else Jaccard on tokens.
  // Defense-in-depth: Arrow Vector objects pass .length > 0 but vec[0] is undefined.
  // Require typeof check to ensure we have actual JS number arrays before cosine path.
  const hasVectors = results.every(
    (r) => r.vector && r.vector.length > 0 && typeof r.vector[0] === "number",
  );
  const tokenSets = hasVectors ? undefined : results.map((r) => tokenize(r.chunk.text));

  if (process.env.MEMORY_SPARK_DEBUG) {
    const strategy = hasVectors ? "cosine" : "jaccard";
    const vecType = results[0]?.vector?.constructor?.name ?? "none";
    const lambdaStr = adaptiveInfo
      ? `adaptive(λ=${effectiveLambda} spread=${adaptiveInfo.spread.toFixed(3)} tier=${adaptiveInfo.tier})`
      : `fixed(λ=${effectiveLambda})`;
    console.debug(
      `[mmrRerank] strategy=${strategy} vectorType=${vecType} candidates=${results.length} limit=${limit} ${lambdaStr}`,
    );
  }

  function similarity(i: number, j: number): number {
    if (hasVectors) {
      return cosineSimilarity(results[i]!.vector!, results[j]!.vector!);
    }
    return jaccardSimilarity(tokenSets![i]!, tokenSets![j]!);
  }

  const selectedIndices: number[] = [];
  const remaining = new Set(results.map((_, i) => i));

  // Always pick the highest-scoring result first
  let bestIdx = 0;
  let bestScore = -Infinity;
  for (const i of remaining) {
    if (results[i]!.score > bestScore) {
      bestScore = results[i]!.score;
      bestIdx = i;
    }
  }
  selectedIndices.push(bestIdx);
  remaining.delete(bestIdx);

  while (selectedIndices.length < limit && remaining.size > 0) {
    let bestMmrScore = -Infinity;
    let bestMmrIdx = -1;

    for (const i of remaining) {
      const relevance = results[i]!.score;

      // Max similarity to any already-selected result
      let maxSim = 0;
      for (const sIdx of selectedIndices) {
        const sim = similarity(i, sIdx);
        if (sim > maxSim) maxSim = sim;
      }

      const mmrScore = effectiveLambda * relevance - (1 - effectiveLambda) * maxSim;
      if (mmrScore > bestMmrScore) {
        bestMmrScore = mmrScore;
        bestMmrIdx = i;
      }
    }

    if (bestMmrIdx === -1) break;
    selectedIndices.push(bestMmrIdx);
    remaining.delete(bestMmrIdx);
  }

  return selectedIndices.map((i) => results[i]!);
}

/**
 * Cosine similarity between two vectors. Returns value in [-1, 1].
 * For normalized embeddings (unit vectors), this equals the dot product.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

function tokenize(text: string): Set<string> {
  return new Set(text.toLowerCase().match(/\b\w{3,}\b/g) ?? []);
}

// Reuse the standalone Jaccard defined earlier
const jaccardSimilarity = jaccardSimilaritySet;

/**
 * Clean query text — strip Discord metadata, timestamps, and injected blocks
 * so the embedding vector represents the actual conversational content.
 * @public — exported for external test/debug tooling
 */
export function cleanQueryText(text: string): string {
  // Strip Discord conversation metadata blocks
  text = text.replace(/```json\s*\{[\s\S]*?"message_id"[\s\S]*?\}\s*```/g, "");
  text = text.replace(/Conversation info \(untrusted metadata\):[\s\S]*?```\s*/g, "");
  text = text.replace(/Sender \(untrusted metadata\):[\s\S]*?```\s*/g, "");
  text = text.replace(
    /Untrusted context \(metadata[^)]*\):[\s\S]*?<<<END_EXTERNAL_UNTRUSTED_CONTENT[^>]*>>>/g,
    "",
  );
  text = text.replace(
    /<<<EXTERNAL_UNTRUSTED_CONTENT[^>]*>>>[\s\S]*?<<<END_EXTERNAL_UNTRUSTED_CONTENT[^>]*>>>/g,
    "",
  );

  // Strip timestamp headers
  text = text.replace(/\[\w{3} \d{4}-\d{2}-\d{2} \d{2}:\d{2} \w+\]/g, "");

  // Strip <relevant-memories> blocks (avoid recursive recall)
  text = text.replace(/<relevant-memories>[\s\S]*?<\/relevant-memories>/g, "");

  // Strip LCM summary blocks
  text = text.replace(/<summary id="sum_[a-f0-9]+"[\s\S]*?<\/summary>/g, "");

  // Strip oc-tasks injection blocks
  text = text.replace(/## Current Task Queue[\s\S]*?(?=\n## [^C]|\n---|$)/g, "");
  text = text.replace(/### 🔄 In Progress[\s\S]*?(?=\n## |\n---|$)/g, "");

  // Strip media attachment lines
  text = text.replace(/\[media attached:[^\]]*\][^\n]*/g, "");

  // Strip [System: ...] prefixes
  text = text.replace(/\[System:[^\]]*\]\s*/g, "");

  // Strip HEARTBEAT_OK / NO_REPLY
  text = text.replace(/^(HEARTBEAT_OK|HEARTBEAT_DISABLED|NO_REPLY)\s*$/gm, "");

  // Strip XML/HTML comments
  text = text.replace(/<!--[\s\S]*?-->/g, "");

  // Collapse excessive whitespace
  return text.replace(/\n{3,}/g, "\n\n").trim();
}

/**
 * Source weighting — adjust scores based on content source and path.
 * Fully configurable via AutoRecallConfig.weights.
 * Falls back to sensible defaults if no weights config is provided.
 */
const DEFAULT_WEIGHTS: RecallWeights = {
  sources: { capture: 1.5, memory: 1.0, sessions: 0.5, reference: 1.0 },
  paths: {
    "MEMORY.md": 1.4,
    "TOOLS.md": 1.3,
    "AGENTS.md": 1.2,
    "SOUL.md": 1.2,
    "USER.md": 1.3,
    "memory/learnings.md": 0.1,
  },
  pathPatterns: {
    mistakes: 1.6,
    "memory/archive/": 0.4,
  },
};

export function applySourceWeighting(results: SearchResult[], weights?: RecallWeights): void {
  const w = weights ?? DEFAULT_WEIGHTS;

  for (const r of results) {
    const source = r.chunk.source;
    const chunkPath = r.chunk.path;

    // Source-level weight
    let weight = (w.sources as Record<string, number>)[source] ?? 1.0;

    // Exact path match
    if (w.paths[chunkPath] !== undefined) {
      weight *= w.paths[chunkPath];
    } else {
      // Pattern match (substring)
      let patternMatched = false;
      for (const [pattern, patternWeight] of Object.entries(w.pathPatterns)) {
        if (chunkPath.toLowerCase().includes(pattern.toLowerCase())) {
          weight *= patternWeight;
          patternMatched = true;
          break;
        }
      }
      // No match = 1.0x (no change)
      if (!patternMatched) weight *= 1.0;
    }

    // P1-A fix: NO Math.min(1.0) clamping here.
    // Scores are allowed to exceed 1.0 after weighting so the reranker gate
    // sees real spread between boosted (mistake/capture) and regular chunks.
    // normalizeScores() rescales to [0, 1] after all weighting is complete.
    r.score = r.score * weight;
  }
}

/**
 * Rescale scores so the highest score = 1.0, preserving relative ordering.
 * Called after source weighting + temporal decay so the gate spread reflects
 * real differences between chunks rather than clamping artifacts.
 * No-op if results is empty or all scores are 0.
 */
export function normalizeScores(results: SearchResult[]): void {
  if (results.length === 0) return;
  const maxScore = Math.max(...results.map((r) => r.score));
  if (maxScore <= 0) return;
  for (const r of results) {
    r.score = r.score / maxScore;
  }
}

function buildQuery(messages: unknown[], maxMessages: number): string {
  if (!Array.isArray(messages)) return "";
  return messages
    .slice(-maxMessages)
    .map(extractMessageText)
    .filter(Boolean)
    .join("\n")
    .slice(0, 2000);
}

function extractMessageText(msg: unknown): string {
  if (typeof msg === "string") return msg;
  if (!msg || typeof msg !== "object") return "";
  const obj = msg as Record<string, unknown>;
  const content = obj.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((c) => {
        if (typeof c === "string") return c;
        if (c && typeof c === "object" && "text" in c) return (c as { text: string }).text;
        return "";
      })
      .filter(Boolean)
      .join(" ");
  }
  return "";
}
