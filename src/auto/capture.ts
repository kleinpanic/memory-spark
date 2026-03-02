/**
 * Auto-Capture — agent_end hook.
 * Classifies conversation turns and stores relevant facts/preferences.
 * Fire-and-forget: never throws, never blocks.
 */

import type { AutoCaptureConfig, MemorySparkConfig } from "../config.js";
import type { StorageBackend, MemoryChunk } from "../storage/backend.js";
import type { EmbedProvider } from "../embed/provider.js";
import { classifyForCapture } from "../classify/zero-shot.js";
import { tagEntities } from "../classify/ner.js";
import crypto from "node:crypto";

// Structural types matching OC's hook events
type AgentEndEvent = { messages: unknown[]; success: boolean; error?: string; durationMs?: number };
type HookContext = { agentId?: string; sessionKey?: string };

export interface AutoCaptureDeps {
  cfg: AutoCaptureConfig;
  globalCfg: MemorySparkConfig;
  backend: StorageBackend;
  embed: EmbedProvider;
}

export function createAutoCaptureHandler(deps: AutoCaptureDeps) {
  return async function captureHandler(
    event: AgentEndEvent,
    ctx: HookContext,
  ): Promise<void> {
    const { cfg, globalCfg, backend, embed } = deps;

    if (!cfg.enabled || !event.success) return;
    const agentId = ctx.agentId ?? "unknown";
    if (!cfg.agents.includes("*") && !cfg.agents.includes(agentId)) return;

    const turnText = extractTurnText(event.messages);
    if (!turnText || turnText.length < 30) return;

    try {
      // Classify
      const result = await classifyForCapture(turnText, globalCfg, cfg.minConfidence);
      if (result.label === "none") return;
      if (!cfg.categories.includes(result.label)) return;

      // NER + embed in parallel
      const [entities, vector] = await Promise.all([
        tagEntities(turnText, globalCfg).catch(() => [] as string[]),
        embed.embedQuery(turnText),
      ]);

      const now = new Date();
      const dateStr = now.toISOString().slice(0, 10);
      const chunk: MemoryChunk = {
        id: crypto.randomUUID().slice(0, 16),
        path: `capture/${agentId}/${dateStr}`,
        source: "capture",
        agent_id: agentId,
        start_line: 0,
        end_line: 0,
        text: turnText,
        vector,
        updated_at: now.toISOString(),
        category: result.label,
        entities: JSON.stringify(entities),
        confidence: result.score,
      };

      await backend.upsert([chunk]);
    } catch {
      // Always non-fatal
    }
  };
}

function extractTurnText(messages: unknown[]): string {
  if (!Array.isArray(messages) || messages.length === 0) return "";

  // Find last assistant + last user messages
  const reversed = [...messages].reverse();
  const lastAssistant = reversed.find(isRole("assistant"));
  const lastUser = reversed.find(isRole("user"));

  const parts: string[] = [];
  if (lastUser) parts.push(extractContent(lastUser));
  if (lastAssistant) parts.push(extractContent(lastAssistant));
  return parts.filter(Boolean).join("\n\n").slice(0, 2000);
}

function isRole(role: string) {
  return (m: unknown): m is Record<string, unknown> =>
    typeof m === "object" && m !== null && (m as Record<string, unknown>).role === role;
}

function extractContent(msg: Record<string, unknown>): string {
  const content = msg.content;
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
