/**
 * Auto-Recall — before_prompt_build hook.
 * Matches memory-lancedb: XML-wrapped injection with security preamble.
 * Adds: MMR diversity, temporal decay, prompt injection filtering.
 */

import type { AutoRecallConfig, RecallWeights } from "../config.js";
import { shouldProcessAgent } from "../config.js";
import type { StorageBackend, SearchResult } from "../storage/backend.js";
import type { EmbedProvider } from "../embed/provider.js";
import type { EmbedQueue } from "../embed/queue.js";
import type { EmbedLike } from "../embed/cached-provider.js";
import type { Reranker } from "../rerank/reranker.js";
import { looksLikePromptInjection, formatRecalledMemories } from "../security.js";

type BeforePromptBuildEvent = { prompt: string; messages: unknown[] };
type BeforePromptBuildResult = { systemPrompt?: string; prependContext?: string };
type HookContext = { agentId?: string; sessionKey?: string };

export interface AutoRecallDeps {
  cfg: AutoRecallConfig;
  backend: StorageBackend;
  embed: EmbedProvider | EmbedQueue | EmbedLike;
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
    const [vectorResults, rawFtsResults] = await Promise.all([
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

    // Filter FTS results: exclude sessions source (keyword matches on chat logs
    // are almost always garbage) and apply a minimum score threshold
    const ftsResults = rawFtsResults.filter(
      (r) => r.chunk.source !== "sessions" && r.score >= (cfg.minScore ?? 0.1),
    );

    // Hybrid merge: preserve original vector scores + RRF rank boost.
    // Unlike pure RRF (which destroys cosine similarity by replacing scores
    // with 1/(k+rank)), this preserves the embedding quality signal.
    const merged = hybridMerge(vectorResults, ftsResults, fetchN);
    if (merged.length === 0) return undefined;

    // Source weighting EARLY — penalize garbage before expensive stages
    // so session chunks and archive content don't waste reranker slots.
    applySourceWeighting(merged, cfg.weights);

    // Temporal decay: boost recent memories
    applyTemporalDecay(merged);

    // MMR diversity re-ranking
    const diverse = mmrRerank(merged, cfg.maxResults * 2, 0.7);

    // Cross-encoder rerank
    const reranked = await reranker.rerank(queryText, diverse, cfg.maxResults);
    if (reranked.length === 0) return undefined;

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
        // Prefix with "memory-spark:" so agents know which plugin this came from
        source: `memory-spark:${r.chunk.source}:${r.chunk.path}`,
        text: r.chunk.text.slice(0, 500),
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
export function hybridMerge(
  vectorResults: SearchResult[],
  ftsResults: SearchResult[],
  limit: number,
  k = 60,
): SearchResult[] {
  const merged = new Map<string, { result: SearchResult; score: number; sources: number }>();

  // Vector results: use original cosine similarity as base score
  vectorResults.forEach((r, _rank) => {
    const id = r.chunk.id;
    merged.set(id, {
      result: r,
      score: r.score, // preserve cosine similarity
      sources: 1,
    });
  });

  // FTS results: boost existing vector entries, or add FTS-only entries
  // with a decaying score based on FTS rank (max ~0.5 for rank 0)
  ftsResults.forEach((r, rank) => {
    const id = r.chunk.id;
    const existing = merged.get(id);
    const ftsRankScore = 0.5 / (1 + rank * 0.1); // 0.50, 0.45, 0.42, ...

    if (existing) {
      // Found in both vector AND FTS — boost score (dual evidence)
      existing.score += 0.1 + (1 / (k + rank)) * 2;
      existing.sources = 2;
    } else {
      // FTS-only: use a moderate score — these lack semantic similarity
      merged.set(id, {
        result: r,
        score: ftsRankScore,
        sources: 1,
      });
    }
  });

  return Array.from(merged.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((s) => ({ ...s.result, score: s.score }));
}

/**
 * Temporal decay: boost recent memories, apply gentle decay for old ones.
 * Uses exponential decay with a floor at 0.8 — old-but-gold knowledge
 * never drops below 80% weight (previously at 0.25x for 60-day-old chunks).
 *
 * Formula: 0.8 + 0.2 * exp(-0.03 * ageDays)
 * 0 days=1.0, 7 days=0.96, 30 days=0.89, 90 days=0.81, 365 days=0.80
 */
export function applyTemporalDecay(results: SearchResult[]): void {
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
export function mmrRerank(results: SearchResult[], limit: number, lambda: number): SearchResult[] {
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
    "mistakes": 1.6,
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
    if (w.paths[chunkPath] != null) {
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
