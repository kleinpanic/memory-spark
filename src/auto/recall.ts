/**
 * Auto-Recall — before_prompt_build hook.
 * Embeds recent conversation, searches memory, returns { prependContext }.
 */

import type { AutoRecallConfig } from "../config.js";
import type { StorageBackend, SearchResult } from "../storage/backend.js";
import type { EmbedProvider } from "../embed/provider.js";
import type { Reranker } from "../rerank/reranker.js";

// Structural types matching OC's hook events (not exported from plugin-sdk)
type BeforePromptBuildEvent = { prompt: string; messages: unknown[] };
type BeforePromptBuildResult = { systemPrompt?: string; prependContext?: string };
type HookContext = { agentId?: string; sessionKey?: string };

export interface AutoRecallDeps {
  cfg: AutoRecallConfig;
  backend: StorageBackend;
  embed: EmbedProvider;
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
    if (!cfg.agents.includes("*") && !cfg.agents.includes(agentId)) return undefined;

    // Build query from recent messages
    const queryText = buildQuery(event.messages, cfg.queryMessageCount);
    if (!queryText.trim()) return undefined;

    let queryVector: number[];
    try {
      queryVector = await embed.embedQuery(queryText);
    } catch {
      return undefined; // Embed failed — skip silently
    }

    // Search: vector + FTS hybrid
    const fetchN = cfg.maxResults * 3;
    const [vectorResults, ftsResults] = await Promise.all([
      backend.vectorSearch(queryVector, {
        query: queryText,
        maxResults: fetchN,
        minScore: cfg.minScore,
        agentId,
      }).catch(() => [] as SearchResult[]),
      backend.ftsSearch(queryText, {
        query: queryText,
        maxResults: fetchN,
        agentId,
      }).catch(() => [] as SearchResult[]),
    ]);

    // Merge + dedup
    const seen = new Set<string>();
    const merged = [...vectorResults, ...ftsResults]
      .filter((r) => {
        if (seen.has(r.chunk.id)) return false;
        seen.add(r.chunk.id);
        return true;
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, fetchN);

    if (merged.length === 0) return undefined;

    // Rerank
    const reranked = await reranker.rerank(queryText, merged, cfg.maxResults);
    if (reranked.length === 0) return undefined;

    // Format injection block
    const lines = reranked.map((r, i) => {
      const src = `${r.chunk.source}:${r.chunk.path}:${r.chunk.start_line}`;
      const text = r.chunk.text.slice(0, 400).replace(/\n/g, " ").trim();
      return `${i + 1}. (${src}) ${text}`;
    });

    return {
      prependContext: `---\n[Memory — recalled from knowledge base]\n${lines.join("\n")}\n---`,
    };
  };
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
