/**
 * Auto-Recall Hook
 *
 * Registered on the "before_prompt_build" plugin hook.
 * Fires before EVERY agent turn for configured agents.
 *
 * What it does:
 *   1. Takes the last N conversation messages (queryMessageCount, default 4)
 *   2. Concatenates them into a composite query string
 *   3. Embeds the query
 *   4. Runs the full search pipeline (vector → FTS hybrid → rerank)
 *   5. Returns { prependContext: "..." } injecting top-K memories
 *      directly into the system prompt before the LLM sees the prompt
 *
 * The injected block looks like:
 *   ---
 *   [Memory — recalled from knowledge base]
 *   1. (source: school/2026-02-15.md:42) Klein prefers TypeScript strict mode...
 *   2. (source: ingest/syllabus.pdf:12) Final exam covers chapters 4–7...
 *   ---
 *
 * Agent never needs to call memory_search explicitly.
 * The context is just there, silently.
 *
 * Configured per-agent via autoRecall.agents (supports ["*"] for all).
 * No-ops for agents not in the list — zero overhead for heartbeat/cron agents.
 */

import type { AutoRecallConfig } from "../config.js";
import type { StorageBackend } from "../storage/backend.js";
import type { EmbedProvider } from "../embed/provider.js";
import type { Reranker } from "../rerank/reranker.js";
import type {
  PluginHookBeforePromptBuildEvent,
  PluginHookBeforePromptBuildResult,
  PluginHookAgentContext,
} from "openclaw/plugin-sdk";

export interface AutoRecallOptions {
  cfg: AutoRecallConfig;
  backend: StorageBackend;
  embed: EmbedProvider;
  reranker: Reranker;
}

/**
 * Factory: returns the before_prompt_build hook handler.
 * Register with: api.on("before_prompt_build", autoRecallHook(...))
 */
export function createAutoRecallHook(opts: AutoRecallOptions) {
  return async function autoRecallHook(
    event: PluginHookBeforePromptBuildEvent,
    ctx: PluginHookAgentContext,
  ): Promise<PluginHookBeforePromptBuildResult | void> {
    const { cfg, backend, embed, reranker } = opts;

    // 1. Check if this agent has auto-recall enabled
    if (!cfg.enabled) return;
    const agentId = ctx.agentId ?? "unknown";
    const agentEnabled = cfg.agents.includes("*") || cfg.agents.includes(agentId);
    if (!agentEnabled) return;

    // 2. Build query from recent messages
    const recentMessages = Array.isArray(event.messages)
      ? event.messages.slice(-(cfg.queryMessageCount))
      : [event.prompt];

    const queryText = buildQueryText(recentMessages);
    if (!queryText.trim()) return;

    // 3. Embed the query
    let queryVector: number[];
    try {
      queryVector = await embed.embedQuery(queryText);
    } catch {
      return; // embedding failed — skip recall silently
    }

    // 4. Search: vector + FTS hybrid
    const [vectorResults, ftsResults] = await Promise.all([
      backend.vectorSearch(queryVector, {
        query: queryText,
        maxResults: cfg.maxResults * 2, // over-fetch for reranking
        minScore: cfg.minScore,
        agentId,
      }).catch(() => []),
      backend.ftsSearch(queryText, {
        query: queryText,
        maxResults: cfg.maxResults * 2,
        agentId,
      }).catch(() => []),
    ]);

    // 5. Merge + deduplicate
    const merged = mergeResults(vectorResults, ftsResults, cfg.maxResults * 2);
    if (merged.length === 0) return;

    // 6. Rerank
    const reranked = await reranker.rerank(queryText, merged, cfg.maxResults);

    // 7. Format injection block
    const prependContext = formatMemoryBlock(reranked);
    return { prependContext };
  };
}

/** Build a single query string from recent conversation messages */
function buildQueryText(messages: unknown[]): string {
  // TODO: extract text content from AgentMessage objects
  // For now, JSON stringify as fallback
  return messages
    .map((m) => {
      if (typeof m === "string") return m;
      if (m && typeof m === "object" && "content" in m) {
        const content = (m as { content: unknown }).content;
        if (typeof content === "string") return content;
        if (Array.isArray(content)) {
          return content
            .map((c: unknown) => (typeof c === "object" && c !== null && "text" in c ? (c as { text: string }).text : ""))
            .join(" ");
        }
      }
      return "";
    })
    .filter(Boolean)
    .join("\n")
    .slice(0, 2000); // cap query length
}

/** Simple score-based merge + dedup by chunk id */
function mergeResults(
  vector: ReturnType<StorageBackend["vectorSearch"]> extends Promise<infer T> ? T : never,
  fts: ReturnType<StorageBackend["ftsSearch"]> extends Promise<infer T> ? T : never,
  limit: number,
) {
  // TODO: proper hybrid merge with RRF (Reciprocal Rank Fusion)
  // For now: combine, deduplicate by id, sort by score
  const seen = new Set<string>();
  const all = [...vector, ...fts].filter((r) => {
    if (seen.has(r.chunk.id)) return false;
    seen.add(r.chunk.id);
    return true;
  });
  return all.sort((a, b) => b.score - a.score).slice(0, limit);
}

/** Format memory results as a clean injection block */
function formatMemoryBlock(results: Awaited<ReturnType<Reranker["rerank"]>>): string {
  if (results.length === 0) return "";

  const lines = results.map((r, i) => {
    const source = `${r.chunk.source}/${r.chunk.path}:${r.chunk.startLine}`;
    return `${i + 1}. (${source}) ${r.chunk.text.slice(0, 400).replace(/\n/g, " ")}`;
  });

  return `---\n[Memory — recalled from knowledge base]\n${lines.join("\n")}\n---`;
}
