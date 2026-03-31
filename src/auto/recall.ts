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
    let queryVector: number[];
    try {
      const hydeConfig = deps.hyde;
      if (hydeConfig?.enabled) {
        const hypothetical = await generateHypotheticalDocument(queryText, hydeConfig);
        if (hypothetical) {
          // HyDE: embed the hypothetical as a DOCUMENT (no instruction prefix).
          // The hypothetical is a pseudo-document and must land in document embedding
          // space, not query space. This is the correct approach per Gao et al. 2022.
          queryVector = await embed.embedDocument(hypothetical);
        } else {
          // HyDE failed (timeout, too short, etc.) — fall back to raw query
          queryVector = await embed.embedQuery(queryText);
        }
      } else {
        queryVector = await embed.embedQuery(queryText);
      }
    } catch {
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
    //   5. Shared rules           → cross-agent, ALWAYS injected (no relevance gate)
    //   6. Reference pools        → NEVER auto-injected (tool-call only)
    //
    // Within each pool: hybrid search (Vector + FTS) → merge
    // FTS+WHERE is supported in LanceDB 0.27+ (no workaround needed).

    const fetchN = cfg.maxResults * (cfg.overfetchMultiplier ?? 4);
    const lowThreshold = Math.max((cfg.minScore ?? 0.1) * 0.7, 0.05);

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
      const vectorResults = await backend
        .vectorSearch(queryVector, searchOpts)
        .catch(() => [] as SearchResult[]);
      const rawFtsResults =
        (cfg.ftsEnabled ?? true)
          ? await backend.ftsSearch(queryText, searchOpts).catch(() => [] as SearchResult[])
          : [];
      // Filter FTS: exclude sessions source only.
      // Do NOT apply minScore to FTS results — BM25 scores are sigmoid-normalized
      // and nearly all map to >0.98, making minScore non-discriminative.
      // RRF handles FTS ranking by position (rank-only), not score magnitude.
      const ftsResults = rawFtsResults.filter(
        (r) => r.chunk.source !== "sessions",
      );
      const merged = hybridMerge(vectorResults, ftsResults, limit);
      // Exclude parent chunks from search results — they exist for context
      // expansion only. Children are the precise search targets; after reranking,
      // matched children get expanded to their parent's text for delivery.
      return merged.filter((r) => !r.chunk.is_parent);
    };

    // 1. Agent's own memory + tools (primary recall)
    const agentResults = await poolSearch(["agent_memory", "agent_tools"], agentId, fetchN);

    // 2. Agent's own mistakes (per-agent, higher priority than shared)
    const agentMistakes = await poolSearch(["agent_mistakes"], agentId, 5, lowThreshold);

    // 3. Shared mistakes (cross-agent)
    const sharedMistakes = await poolSearch(
      ["shared_mistakes"],
      undefined, // No agent filter — all agents' shared mistakes
      5,
      lowThreshold,
    );

    // 4. Shared knowledge (cross-agent)
    const sharedKnowledge = await poolSearch(["shared_knowledge"], undefined, 10, lowThreshold);

    // 5. Shared rules — relevance-gated like all other pools
    const sharedRules = await poolSearch(["shared_rules"], undefined, 5, lowThreshold);

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

    if (merged.length === 0) return undefined;

    // Source weighting EARLY — penalize garbage before expensive stages
    // so session chunks and archive content don't waste reranker slots.
    applySourceWeighting(merged, cfg.weights);

    // Temporal decay: boost recent memories (configurable floor + rate)
    applyTemporalDecay(merged, cfg.temporalDecay);

    // MMR diversity re-ranking
    const diverse = mmrRerank(merged, cfg.maxResults * 2, cfg.mmrLambda ?? 0.7);

    // Cross-encoder rerank
    const reranked = await reranker.rerank(queryText, diverse, cfg.maxResults);
    if (reranked.length === 0) return undefined;

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
    for (const r of reranked) {
      if (r.chunk.parent_id && !r.chunk.is_parent) {
        parentIds.add(r.chunk.parent_id);
      }
    }

    if (parentIds.size > 0) {
      try {
        const parentChunks = await backend.getByIds(Array.from(parentIds));
        const parentMap = new Map(parentChunks.map((p) => [p.id, p]));

        // Replace child text with parent text (keeping child's score and metadata)
        for (const r of reranked) {
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
        const dedupedReranked: SearchResult[] = [];
        for (const r of reranked) {
          const expandedTag = r.chunk.parent_id;
          if (expandedTag?.startsWith("expanded:")) {
            if (!seenParents.has(expandedTag)) {
              seenParents.add(expandedTag);
              dedupedReranked.push(r);
            }
            // else skip — a higher-scoring sibling already claimed this parent
          } else {
            // Non-expanded chunks (flat or failed lookup) pass through
            dedupedReranked.push(r);
          }
        }
        // Use deduplicated results for the rest of the pipeline
        reranked.splice(0, reranked.length, ...dedupedReranked);
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
    const deduplicated = reranked.filter((r) => {
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
export function hybridMerge(
  vectorResults: SearchResult[],
  ftsResults: SearchResult[],
  limit: number,
  k = 60,
): SearchResult[] {
  const merged = new Map<string, { result: SearchResult; rrfScore: number; sources: number }>();

  // Vector results: RRF score from rank position only (1-indexed per the paper)
  vectorResults.forEach((r, idx) => {
    merged.set(r.chunk.id, {
      result: r,
      rrfScore: 1 / (k + idx + 1),
      sources: 1,
    });
  });

  // FTS results: add RRF score (sums when document found in both lists)
  ftsResults.forEach((r, idx) => {
    const id = r.chunk.id;
    const ftsRrf = 1 / (k + idx + 1);
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

  return sorted.slice(0, limit).map((s) => ({
    ...s.result,
    score: maxRrf > 0 ? s.rrfScore / maxRrf : 0,
  }));
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
 * Maximum Marginal Relevance — ensures result diversity.
 * Iteratively selects results that are relevant but dissimilar to already-selected.
 * Uses cosine similarity on embedding vectors (semantic diversity).
 * Falls back to Jaccard similarity on token sets when vectors are unavailable.
 */
export function mmrRerank(results: SearchResult[], limit: number, lambda: number): SearchResult[] {
  if (results.length <= 1) return results;

  // Determine similarity strategy: cosine on vectors if available, else Jaccard on tokens.
  const hasVectors = results.every((r) => r.vector && r.vector.length > 0);
  const tokenSets = hasVectors ? undefined : results.map((r) => tokenize(r.chunk.text));

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

      const mmrScore = lambda * relevance - (1 - lambda) * maxSim;
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

function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 1;
  let intersection = 0;
  for (const token of a) {
    if (b.has(token)) intersection++;
  }
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

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

    r.score = Math.min(1.0, r.score * weight);
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
