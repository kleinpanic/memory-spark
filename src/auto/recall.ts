/**
 * Auto-Recall — before_prompt_build hook.
 * Matches memory-lancedb: XML-wrapped injection with security preamble.
 * Adds: MMR diversity, temporal decay, prompt injection filtering.
 */

import type { AutoRecallConfig } from "../config.js";
import { shouldProcessAgent } from "../config.js";
import type { StorageBackend, SearchResult } from "../storage/backend.js";
import type { EmbedProvider } from "../embed/provider.js";
import type { EmbedQueue } from "../embed/queue.js";
import type { Reranker } from "../rerank/reranker.js";
import { looksLikePromptInjection, formatRecalledMemories } from "../security.js";

type BeforePromptBuildEvent = { prompt: string; messages: unknown[] };
type BeforePromptBuildResult = { systemPrompt?: string; prependContext?: string };
type HookContext = { agentId?: string; sessionKey?: string };

export interface AutoRecallDeps {
  cfg: AutoRecallConfig;
  backend: StorageBackend;
  embed: EmbedProvider | EmbedQueue;
  reranker: Reranker;
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

    const queryText = buildQuery(event.messages, cfg.queryMessageCount);
    if (!queryText.trim()) return undefined;

    let queryVector: number[];
    try {
      queryVector = await embed.embedQuery(queryText);
    } catch {
      return undefined;
    }

    // Hybrid search: vector + FTS
    const fetchN = cfg.maxResults * 4;
    const [vectorResults, ftsResults] = await Promise.all([
      backend.vectorSearch(queryVector, {
        query: queryText, maxResults: fetchN, minScore: cfg.minScore, agentId,
      }).catch(() => [] as SearchResult[]),
      backend.ftsSearch(queryText, {
        query: queryText, maxResults: fetchN, agentId,
      }).catch(() => [] as SearchResult[]),
    ]);

    // RRF (Reciprocal Rank Fusion) merge
    const merged = rrfMerge(vectorResults, ftsResults, fetchN);
    if (merged.length === 0) return undefined;

    // Temporal decay: boost recent memories
    applyTemporalDecay(merged);

    // MMR diversity re-ranking
    const diverse = mmrRerank(merged, cfg.maxResults * 2, 0.7);

    // Cross-encoder rerank
    const reranked = await reranker.rerank(queryText, diverse, cfg.maxResults);
    if (reranked.length === 0) return undefined;

    // Filter prompt injection + format with security preamble
    const safeMemories = reranked
      .filter((r) => !looksLikePromptInjection(r.chunk.text))
      .map((r) => ({
        source: `${r.chunk.source}:${r.chunk.path}:${r.chunk.start_line}`,
        text: r.chunk.text.slice(0, 500),
      }));

    if (safeMemories.length === 0) return undefined;

    return { prependContext: formatRecalledMemories(safeMemories) };
  };
}

/**
 * Reciprocal Rank Fusion — proper hybrid merge.
 * RRF(d) = Σ 1 / (k + rank_i(d)) across all ranking lists.
 */
function rrfMerge(vectorResults: SearchResult[], ftsResults: SearchResult[], limit: number, k = 60): SearchResult[] {
  const scores = new Map<string, { result: SearchResult; rrfScore: number }>();

  // Score from vector ranking
  vectorResults.forEach((r, rank) => {
    const id = r.chunk.id;
    const existing = scores.get(id);
    const rrfScore = 1 / (k + rank);
    if (existing) {
      existing.rrfScore += rrfScore;
    } else {
      scores.set(id, { result: r, rrfScore });
    }
  });

  // Score from FTS ranking
  ftsResults.forEach((r, rank) => {
    const id = r.chunk.id;
    const existing = scores.get(id);
    const rrfScore = 1 / (k + rank);
    if (existing) {
      existing.rrfScore += rrfScore;
    } else {
      scores.set(id, { result: r, rrfScore });
    }
  });

  return Array.from(scores.values())
    .sort((a, b) => b.rrfScore - a.rrfScore)
    .slice(0, limit)
    .map((s) => ({ ...s.result, score: s.rrfScore }));
}

/**
 * Temporal decay: boost recent memories, penalize old ones.
 * Half-life: 30 days. A 60-day-old memory gets 0.25x weight.
 */
function applyTemporalDecay(results: SearchResult[], halfLifeDays = 30): void {
  const now = Date.now();
  for (const r of results) {
    const updatedAt = r.chunk.updated_at ? new Date(r.chunk.updated_at).getTime() : now;
    const ageDays = Math.max(0, (now - updatedAt) / (86400 * 1000));
    const decay = Math.pow(0.5, ageDays / halfLifeDays);
    r.score *= decay;
  }
}

/**
 * Maximum Marginal Relevance — ensures result diversity.
 * Iteratively selects results that are relevant but dissimilar to already-selected.
 * Uses Jaccard similarity on token sets (fast, no vectors needed).
 */
function mmrRerank(results: SearchResult[], limit: number, lambda: number): SearchResult[] {
  if (results.length <= 1) return results;

  const tokenSets = results.map((r) => tokenize(r.chunk.text));
  const selected: SearchResult[] = [];
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
  selected.push(results[bestIdx]!);
  remaining.delete(bestIdx);

  while (selected.length < limit && remaining.size > 0) {
    let bestMmrScore = -Infinity;
    let bestMmrIdx = -1;

    for (const i of remaining) {
      const relevance = results[i]!.score;

      // Max similarity to any already-selected result
      let maxSim = 0;
      for (const s of selected) {
        const sIdx = results.indexOf(s);
        const sim = jaccardSimilarity(tokenSets[i]!, tokenSets[sIdx]!);
        if (sim > maxSim) maxSim = sim;
      }

      const mmrScore = lambda * relevance - (1 - lambda) * maxSim;
      if (mmrScore > bestMmrScore) {
        bestMmrScore = mmrScore;
        bestMmrIdx = i;
      }
    }

    if (bestMmrIdx === -1) break;
    selected.push(results[bestMmrIdx]!);
    remaining.delete(bestMmrIdx);
  }

  return selected;
}

function tokenize(text: string): Set<string> {
  return new Set(
    text.toLowerCase().match(/\b\w{3,}\b/g) ?? []
  );
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
