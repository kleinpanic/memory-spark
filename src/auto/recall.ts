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

    const rawQueryText = buildQuery(event.messages, cfg.queryMessageCount);
    const queryText = cleanQueryText(rawQueryText);
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
      backend
        .vectorSearch(queryVector, {
          query: queryText,
          maxResults: fetchN,
          minScore: cfg.minScore,
          agentId,
        })
        .catch(() => [] as SearchResult[]),
      backend
        .ftsSearch(queryText, {
          query: queryText,
          maxResults: fetchN,
          agentId,
        })
        .catch(() => [] as SearchResult[]),
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

    // Source weighting — boost captures & MEMORY.md, penalize archives
    applySourceWeighting(reranked);

    // Re-sort after source weighting
    reranked.sort((a, b) => b.score - a.score);

    // LCM recency suppression — skip chunks that overlap heavily with recent messages
    const recentTexts = event.messages
      .slice(-cfg.queryMessageCount)
      .map(extractMessageText)
      .map(cleanQueryText)
      .filter(Boolean);

    const deduplicated = reranked.filter((r) => {
      const chunkAge = Date.now() - new Date(r.chunk.updated_at).getTime();
      if (chunkAge > 2 * 60 * 60 * 1000) return true; // older than 2h — keep
      const chunkTokens = new Set(
        (r.chunk.text.match(/\b\w{4,}\b/g) ?? []).map((w) => w.toLowerCase()),
      );
      if (chunkTokens.size === 0) return true;
      for (const msg of recentTexts) {
        const msgTokens = new Set((msg.match(/\b\w{4,}\b/g) ?? []).map((w) => w.toLowerCase()));
        let overlap = 0;
        for (const t of chunkTokens) {
          if (msgTokens.has(t)) overlap++;
        }
        if (overlap / chunkTokens.size > 0.4) return false; // >40% overlap = redundant
      }
      return true;
    });

    // Filter prompt injection + format with rich metadata
    const safeMemories = deduplicated
      .filter((r) => !looksLikePromptInjection(r.chunk.text))
      .map((r) => ({
        source: `${r.chunk.source}:${r.chunk.path}:${r.chunk.start_line}`,
        text: r.chunk.text.slice(0, 500),
        score: r.score,
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
 * Reciprocal Rank Fusion — proper hybrid merge.
 * RRF(d) = Σ 1 / (k + rank_i(d)) across all ranking lists.
 */
function rrfMerge(
  vectorResults: SearchResult[],
  ftsResults: SearchResult[],
  limit: number,
  k = 60,
): SearchResult[] {
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
 * Temporal decay: boost recent memories, apply gentle decay for old ones.
 * Uses exponential decay with a floor at 0.8 — old-but-gold knowledge
 * never drops below 80% weight (previously at 0.25x for 60-day-old chunks).
 *
 * Formula: 0.8 + 0.2 * exp(-0.03 * ageDays)
 * 0 days=1.0, 7 days=0.96, 30 days=0.89, 90 days=0.81, 365 days=0.80
 */
function applyTemporalDecay(results: SearchResult[]): void {
  const now = Date.now();
  for (const r of results) {
    const updatedAt = r.chunk.updated_at ? new Date(r.chunk.updated_at).getTime() : now;
    const ageDays = Math.max(0, (now - updatedAt) / (86400 * 1000));
    const decay = 0.8 + 0.2 * Math.exp(-0.03 * ageDays);
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
 */
function cleanQueryText(text: string): string {
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

  // Strip XML/HTML comments
  text = text.replace(/<!--[\s\S]*?-->/g, "");

  // Collapse excessive whitespace
  return text.replace(/\n{3,}/g, "\n\n").trim();
}

/**
 * Source weighting — adjust scores based on content source and path.
 * Captures and curated knowledge get boosted, archives and stale content penalized.
 */
function applySourceWeighting(results: SearchResult[]): void {
  for (const r of results) {
    const source = r.chunk.source;
    const chunkPath = r.chunk.path;

    let weight = 1.0;
    // Source-level weights
    if (source === "capture") weight = 1.5;
    else if (source === "sessions") weight = 0.5;

    // Path-level refinements
    if (chunkPath === "MEMORY.md") weight *= 1.4;
    else if (chunkPath.toLowerCase().includes("mistakes")) weight *= 1.6;
    else if (chunkPath.startsWith("memory/archive/")) weight *= 0.4;
    else if (chunkPath === "memory/learnings.md") weight *= 0.1;

    r.score *= weight;
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
