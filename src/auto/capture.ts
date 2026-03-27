/**
 * Auto-Capture — agent_end hook.
 * Captures from USER messages and assistant messages containing decision/fact patterns.
 * Assistant capture is restricted to specific knowledge patterns (decisions, facts,
 * root causes) to prevent self-poisoning from generic responses.
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
import { scoreChunkQuality } from "../classify/quality.js";
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
  return async function captureHandler(event: AgentEndEvent, ctx: HookContext): Promise<void> {
    const { cfg, globalCfg, backend, embed } = deps;

    if (!cfg.enabled || !event.success) return;
    const agentId = ctx.agentId ?? "unknown";
    if (!shouldProcessAgent(agentId, cfg.agents, cfg.ignoreAgents ?? [])) return;

    // Extract user messages + assistant decision/fact patterns
    const minLen = cfg.minMessageLength ?? 30;
    const captureTexts = extractCaptureMessages(event.messages, minLen);
    if (captureTexts.length === 0) return;

    let captured = 0;

    for (const text of captureTexts) {
      if (captured >= MAX_CAPTURES_PER_TURN) break;
      if (text.length < minLen) continue;

      // Skip prompt injection attempts
      if (looksLikePromptInjection(text)) continue;

      // Skip garbage: Discord metadata, media paths, XML memory blocks, etc.
      // This is the CRITICAL gate that prevents the agent from storing
      // conversation envelope noise as "knowledge".
      if (looksLikeCaptureGarbage(text)) continue;

      // Run the full quality scorer as a second gate
      const qualityCheck = scoreChunkQuality(text, `capture/${agentId}`, "capture");
      if (qualityCheck.score < 0.3) continue;

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
        const effectiveMinConfidence = result.score <= 0.7 ? 0.6 : cfg.minConfidence;
        if (result.label === "none") continue;
        if (result.score < effectiveMinConfidence) continue;
        if (!cfg.categories.includes(result.label)) continue;

        // Embed
        const vector = await embed.embedQuery(text);

        // Duplicate detection: search for similar existing memories
        const existing = await backend
          .vectorSearch(vector, {
            query: text,
            maxResults: 1,
            minScore: DEDUP_THRESHOLD,
            agentId,
            source: "capture",
          })
          .catch(() => []);

        if (existing.length > 0 && existing[0]!.score >= DEDUP_THRESHOLD) {
          continue; // Duplicate — skip
        }

        // NER tag
        const entities = await tagEntities(text, globalCfg).catch(() => [] as string[]);

        // Importance scoring: confidence from classifier + category weight
        const categoryWeights: Record<string, number> = {
          decision: 0.9,
          preference: 0.8,
          fact: 0.7,
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
          content_type: "knowledge",
          pool: "agent_memory", // Captures go to agent's own memory pool
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

/**
 * Fast garbage detector for auto-capture. Rejects Discord/OpenClaw envelope noise
 * that should NEVER become a memory. This runs BEFORE the expensive classifier.
 *
 * The school agent thought Klein sent a screenshot because auto-capture stored
 * a "[media attached: ...]" envelope line as a "fact". This function prevents that.
 */
const CAPTURE_GARBAGE_PATTERNS: RegExp[] = [
  // Discord/OpenClaw envelope metadata
  /Conversation info \(untrusted metadata\)/,
  /Sender \(untrusted metadata\)/,
  /"message_id":\s*"\d+"/,
  /"sender_id":\s*"\d+"/,
  /"conversation_label":/,
  /<<<EXTERNAL_UNTRUSTED_CONTENT/,
  /<<<END_EXTERNAL_UNTRUSTED_CONTENT/,
  /UNTRUSTED Discord message body/,
  /UNTRUSTED \w+ message body/,

  // Media attachment paths (the exact bug Klein found)
  /\[media attached:\s*\/home\//,
  /\[media attached:\s*https?:\/\//,
  /\.openclaw\/media\/inbound\//,
  /To send an image back, prefer the message tool/,

  // Memory recall XML blocks (memories recalling themselves = infinite loop)
  /<relevant-memories>/,
  /<\/relevant-memories>/,
  /<memory index="\d+"/,
  /<!-- SECURITY: Treat every memory below as untrusted/,

  // LCM summary blocks
  /<summary id="sum_[a-f0-9]+"/,
  /<summary_ref id="sum_/,

  // System/heartbeat noise
  /^HEARTBEAT_OK$/m,
  /^HEARTBEAT_DISABLED$/m,
  /^NO_REPLY$/m,
  /\[System:\s/,
  /\[RESTART_APPROVAL_REQUEST\]/,

  // oc-tasks injection blocks
  /^## Current Task Queue$/m,
  /^### 🔄 In Progress/m,
  /^### 👀 Awaiting Review/m,
  /`oc_tasks_\w+`/,

  // Raw tool output / exec results
  /^```json\s*\n\s*\{[\s\S]{0,50}"schema":\s*"openclaw\./m,
  /Exec completed \([^)]+, code \d+\)/,

  // Agent bootstrap / session headers
  /^## \d{4}-\d{2}-\d{2}T[\d:.]+Z — (agent bootstrap|session new)/m,
  /^# Session: \d{4}-\d{2}-\d{2}/m,

  // Raw role prefixes (conversation logs, not knowledge)
  /^(assistant|user|system):\s/m,
];

export function looksLikeCaptureGarbage(text: string): boolean {
  return CAPTURE_GARBAGE_PATTERNS.some((p) => p.test(text));
}

function containsDecisionPattern(text: string): boolean {
  return /\b(decided|going with|switched to|approved|we'll use|migrated? to|the fix is|conclusion|we should|the solution is)\b/i.test(
    text,
  );
}

function containsFactPattern(text: string): boolean {
  return /\b(runs on|runs at|located at|IP is|port \d+|the server|version \d|deployed to|configured as|the issue|root cause|fixed by|the problem|resolved|the answer|the solution|the fix is|because of|due to|caused by)\b/i.test(
    text,
  );
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
