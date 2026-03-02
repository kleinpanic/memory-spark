/**
 * Auto-Capture Hook
 *
 * Registered on the "agent_end" plugin hook.
 * Fires after every completed agent turn for configured agents.
 * Runs async and non-blocking — never delays the agent response.
 *
 * What it does:
 *   1. Takes the completed turn (last user message + last assistant message)
 *   2. Sends to Spark zero-shot classifier (18113) with capture category labels
 *   3. If category != "none" AND confidence >= minConfidence:
 *      a. NER tag the text for entity metadata
 *      b. Embed the text
 *      c. Store as a MemoryChunk with source="capture" + category + timestamp
 *
 * Categories stored:
 *   "fact"         → factual statement about user, system, or world
 *   "preference"   → Klein expressed a preference
 *   "decision"     → something was decided or committed to
 *   "code-snippet" → useful code / commands referenced
 *
 * Example captures:
 *   "Klein prefers TypeScript strict mode" → fact, stored with entity: Klein
 *   "We decided to use LanceDB as primary" → decision, stored with entity: LanceDB
 *   "scp -r user@host:/path /local/path"  → code-snippet
 *
 * Captured memories appear in auto-recall searches just like indexed files.
 * They have path = "capture/<agentId>/<YYYY-MM-DD>" for grouping.
 */

import type { AutoCaptureConfig, MemorySparkConfig } from "../config.js";
import type { StorageBackend, MemoryChunk } from "../storage/backend.js";
import type { EmbedProvider } from "../embed/provider.js";
import type {
  PluginHookAgentEndEvent,
  PluginHookAgentContext,
} from "openclaw/plugin-sdk";
import { classifyForCapture } from "../classify/zero-shot.js";
import { tagEntities } from "../classify/ner.js";
import crypto from "node:crypto";

export interface AutoCaptureOptions {
  cfg: AutoCaptureConfig;
  globalCfg: MemorySparkConfig;
  backend: StorageBackend;
  embed: EmbedProvider;
}

/**
 * Factory: returns the agent_end hook handler.
 * Register with: api.on("agent_end", autoCaptureHook(...))
 */
export function createAutoCaptureHook(opts: AutoCaptureOptions) {
  return async function autoCaptureHook(
    event: PluginHookAgentEndEvent,
    ctx: PluginHookAgentContext,
  ): Promise<void> {
    const { cfg, globalCfg, backend, embed } = opts;

    if (!cfg.enabled || !event.success) return;
    const agentId = ctx.agentId ?? "unknown";
    const agentEnabled = cfg.agents.includes("*") || cfg.agents.includes(agentId);
    if (!agentEnabled) return;

    // Extract last user+assistant turn from messages
    const captureText = extractTurnText(event.messages);
    if (!captureText || captureText.length < 20) return;

    // Classify — best-effort, non-blocking
    try {
      const result = await classifyForCapture(captureText, globalCfg, cfg.minConfidence);
      if (result.label === "none") return;
      if (!cfg.categories.includes(result.label)) return;

      // NER tag + embed in parallel
      const [entities, vector] = await Promise.all([
        tagEntities(captureText, globalCfg).catch(() => [] as string[]),
        embed.embedQuery(captureText),
      ]);

      const now = new Date();
      const dateStr = now.toISOString().slice(0, 10);
      const chunk: MemoryChunk = {
        id: crypto.randomUUID().slice(0, 16),
        path: `capture/${agentId}/${dateStr}`,
        source: "capture",
        agentId,
        startLine: 0,
        endLine: 0,
        text: captureText,
        ftsText: captureText,
        vector,
        updatedAt: now.toISOString(),
        category: result.label,
        entities,
        confidence: result.score,
      };

      await backend.upsert([chunk]);
    } catch {
      // Always non-fatal — capture failure must never surface to user
    }
  };
}

/** Extract the last user + assistant message text from a turn */
function extractTurnText(messages: unknown[]): string {
  if (!Array.isArray(messages) || messages.length === 0) return "";

  // Find last assistant message
  const lastAssistant = [...messages].reverse().find(
    (m): m is { role: string; content: unknown } =>
      typeof m === "object" && m !== null && (m as { role?: string }).role === "assistant",
  );

  // Find last user message before it
  const lastUser = [...messages].reverse().find(
    (m): m is { role: string; content: unknown } =>
      typeof m === "object" && m !== null && (m as { role?: string }).role === "user",
  );

  const parts: string[] = [];
  if (lastUser) parts.push(extractContent(lastUser.content));
  if (lastAssistant) parts.push(extractContent(lastAssistant.content));
  return parts.filter(Boolean).join("\n\n").slice(0, 2000);
}

function extractContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((c) => (typeof c === "object" && c !== null && "text" in c ? (c as { text: string }).text : ""))
      .filter(Boolean)
      .join(" ");
  }
  return "";
}
