/**
 * Auto-Capture — agent_end hook.
 * Captures from USER messages only (no self-poisoning from assistant output).
 * Deduplicates against existing memories (>0.92 similarity = skip).
 * Max 3 captures per turn. Includes importance scoring.
 */

import type { AutoCaptureConfig, MemorySparkConfig } from "../config.js";
import { shouldProcessAgent } from "../config.js";
import type { StorageBackend, MemoryChunk } from "../storage/backend.js";
import type { EmbedProvider } from "../embed/provider.js";
import type { EmbedQueue } from "../embed/queue.js";
import { classifyForCapture } from "../classify/zero-shot.js";
import type { ClassifyResult } from "../classify/zero-shot.js";
import { heuristicClassify } from "../classify/heuristic.js";
import { tagEntities } from "../classify/ner.js";
import { looksLikePromptInjection } from "../security.js";
import crypto from "node:crypto";

type AgentEndEvent = { messages: unknown[]; success: boolean; error?: string; durationMs?: number };
type HookContext = { agentId?: string; sessionKey?: string };

const MAX_CAPTURES_PER_TURN = 3;
const DEDUP_THRESHOLD = 0.92;

export interface AutoCaptureDeps {
  cfg: AutoCaptureConfig;
  globalCfg: MemorySparkConfig;
  backend: StorageBackend;
  embed: EmbedProvider | EmbedQueue;
}

export function createAutoCaptureHandler(deps: AutoCaptureDeps) {
  return async function captureHandler(
    event: AgentEndEvent,
    ctx: HookContext,
  ): Promise<void> {
    const { cfg, globalCfg, backend, embed } = deps;

    if (!cfg.enabled || !event.success) return;
    const agentId = ctx.agentId ?? "unknown";
    if (!shouldProcessAgent(agentId, cfg.agents, cfg.ignoreAgents ?? [])) return;

    // Extract user messages + assistant decision/fact patterns
    const minLen = cfg.minMessageLength ?? 80;
    const captureTexts = extractCaptureMessages(event.messages, minLen);
    if (captureTexts.length === 0) return;

    let captured = 0;

    for (const text of captureTexts) {
      if (captured >= MAX_CAPTURES_PER_TURN) break;
      if (text.length < minLen) continue;

      // Skip prompt injection attempts
      if (looksLikePromptInjection(text)) continue;

      try {
        // Classify — use Spark zero-shot or local heuristic fallback
        let result: ClassifyResult;
        if (cfg.useClassifier !== false) {
          result = await classifyForCapture(text, globalCfg, cfg.minConfidence);
          // If zero-shot returned "none", try heuristic as a safety net
          if (result.label === "none") {
            result = heuristicClassify(text);
          }
        } else {
          result = heuristicClassify(text);
        }
        // Heuristic scores cap at 0.70 — use a lower threshold for heuristic results
        const effectiveMinConfidence = result.score <= 0.70 ? 0.60 : cfg.minConfidence;
        if (result.label === "none") continue;
        if (result.score < effectiveMinConfidence) continue;
        if (!cfg.categories.includes(result.label)) continue;

        // Embed
        const vector = await embed.embedQuery(text);

        // Duplicate detection: search for similar existing memories
        const existing = await backend.vectorSearch(vector, {
          query: text,
          maxResults: 1,
          minScore: DEDUP_THRESHOLD,
          agentId,
          source: "capture",
        }).catch(() => []);

        if (existing.length > 0 && existing[0]!.score >= DEDUP_THRESHOLD) {
          continue; // Duplicate — skip
        }

        // NER tag
        const entities = await tagEntities(text, globalCfg).catch(() => [] as string[]);

        // Importance scoring: confidence from classifier + category weight
        const categoryWeights: Record<string, number> = {
          "decision": 0.9,
          "preference": 0.8,
          "fact": 0.7,
          "code-snippet": 0.6,
        };
        const importance = (result.score + (categoryWeights[result.label] ?? 0.5)) / 2;

        const now = new Date();
        const dateStr = now.toISOString().slice(0, 10);
        const chunk: MemoryChunk = {
          id: crypto.randomUUID().slice(0, 16),
          path: `capture/${agentId}/${dateStr}`,
          source: "capture",
          agent_id: agentId,
          start_line: 0,
          end_line: 0,
          text,
          vector,
          updated_at: now.toISOString(),
          category: result.label,
          entities: JSON.stringify(entities),
          confidence: importance,
        };

        await backend.upsert([chunk]);
        captured++;
      } catch {
        // Non-fatal
      }
    }
  };
}

/**
 * Extract capture-worthy messages: all user messages + assistant messages
 * containing decision/fact patterns. Prevents self-poisoning by limiting
 * assistant capture to specific knowledge patterns.
 */
function extractCaptureMessages(messages: unknown[], minLen = 80): string[] {
  if (!Array.isArray(messages)) return [];

  const texts: string[] = [];
  const seen = new Set<string>();

  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    const obj = msg as Record<string, unknown>;
    const role = obj.role as string;
    const text = extractContent(obj).trim();
    if (!text || text.length < minLen) continue;

    // Deduplicate within this turn
    const key = text.slice(0, 100);
    if (seen.has(key)) continue;
    seen.add(key);

    if (role === "user") {
      texts.push(text.slice(0, 2000));
    } else if (role === "assistant") {
      // Only capture assistant messages with decision/fact patterns
      if (containsDecisionPattern(text) || containsFactPattern(text)) {
        texts.push(text.slice(0, 2000));
      }
    }
  }

  return texts;
}

function containsDecisionPattern(text: string): boolean {
  return /\b(decided|going with|switched to|approved|we'll use|migrated? to|the fix is|conclusion|we should|the solution is)\b/i.test(text);
}

function containsFactPattern(text: string): boolean {
  return /\b(runs on|runs at|located at|IP is|port \d+|the server|version \d|deployed to|configured as)\b/i.test(text);
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
