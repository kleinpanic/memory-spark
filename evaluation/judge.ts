/**
 * LLM-as-Judge Module for memory-spark evaluation
 *
 * Uses Nemotron-Super-120B on Spark to score relevance of retrieved
 * chunks against queries. Scores are 1-5 (RAGAS-style grading).
 *
 * Key design decisions:
 * - Non-thinking mode (enable_thinking: false) to minimize latency
 * - Strict JSON output format to avoid conversational filler
 * - Parallel HTTP calls (3x) for throughput
 * - Calibration validation before full evaluation
 */

import type { SearchResult } from "../src/storage/backend.js";

// ── Config ──────────────────────────────────────────────────────────────────

const SPARK_HOST = process.env.SPARK_HOST ?? "10.99.1.1"; // eslint-disable-line sonarjs/no-hardcoded-ip -- env-overridable fallback for local LAN
const SPARK_LLM_URL = `http://${SPARK_HOST}:18080/v1/chat/completions`;
const SPARK_API_KEY = "610b70253998918445408763682298af4c1f492fae39c9289f938f55f7c47310";
const MODEL = "nemotron-super";
const MAX_PARALLEL = 3;
const TIMEOUT_MS = 30_000;

// ── Types ───────────────────────────────────────────────────────────────────

export interface JudgeResult {
  queryId: string;
  query: string;
  chunkId: string;
  chunkText: string;
  score: number; // 1-5
  rawResponse: string;
}

export interface JudgeConfig {
  sparkUrl?: string;
  apiKey?: string;
  model?: string;
  maxParallel?: number;
}

// ── Prompt ──────────────────────────────────────────────────────────────────

const JUDGE_SYSTEM_PROMPT = `You are a relevance judge for a personal AI assistant's memory retrieval system. The system serves ONE user (Klein), so documents from the user's workspace (USER.md, MEMORY.md, AGENTS.md, etc.) are inherently about that user even if they don't mention their name explicitly.

Score on a 1-5 scale:
1 = Completely irrelevant — the document has nothing to do with the query
2 = Marginally relevant — mentions a related topic but doesn't answer the query
3 = Partially relevant — contains some useful information but incomplete or indirect
4 = Highly relevant — directly addresses the query with useful information
5 = Perfectly relevant — exactly answers the query with complete, precise information

IMPORTANT context: This is a PERSONAL assistant system. When the query asks about "Klein's timezone" and the document says "Timezone: America/New_York", that IS about Klein (the sole user). Score based on information content, not whether the user's name appears literally.

Respond with ONLY a JSON object: {"score": N, "reason": "one sentence"}
No other text. No markdown. No explanation outside the JSON.`;

function buildJudgePrompt(query: string, document: string): string {
  // Truncate very long documents to avoid token overflow
  const maxDocLen = 2000;
  const truncated =
    document.length > maxDocLen ? document.slice(0, maxDocLen) + "\n...[truncated]" : document;

  return `Query: "${query}"

Retrieved Document:
---
${truncated}
---

Rate the relevance of this document to the query. Respond with {"score": N, "reason": "..."}`;
}

// ── API Call ────────────────────────────────────────────────────────────────

async function callJudge(
  query: string,
  document: string,
  config: Required<JudgeConfig>,
): Promise<{ score: number; reason: string; raw: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const resp = await fetch(config.sparkUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${config.apiKey}`,
      },
      body: JSON.stringify({
        model: config.model,
        messages: [
          { role: "system", content: JUDGE_SYSTEM_PROMPT },
          { role: "user", content: buildJudgePrompt(query, document) },
        ],
        max_tokens: 100,
        temperature: 0.0,
        stream: false,
        enable_thinking: false,
      }),
      signal: controller.signal,
    });

    if (!resp.ok) {
      const errText = await resp.text().catch(() => "");
      throw new Error(`Judge API error ${resp.status}: ${errText.slice(0, 200)}`);
    }

    const data = (await resp.json()) as {
      choices: Array<{ message: { content: string } }>;
    };
    const raw = data.choices?.[0]?.message?.content ?? "";

    // Parse score from JSON response — handle various formats
    const parsed = parseJudgeResponse(raw);
    return { ...parsed, raw };
  } finally {
    clearTimeout(timer);
  }
}

function parseJudgeResponse(raw: string): { score: number; reason: string } {
  // Try direct JSON parse
  try {
    const obj = JSON.parse(raw.trim());
    if (typeof obj.score === "number" && obj.score >= 1 && obj.score <= 5) {
      return { score: Math.round(obj.score), reason: obj.reason ?? "" };
    }
  } catch {
    // Fall through to regex extraction — JSON parse failed
  }

  // Extract from markdown code block
  const codeBlock = raw.match(/```(?:json)?\s*({[^}]+})\s*```/);
  if (codeBlock) {
    try {
      const obj = JSON.parse(codeBlock[1]!);
      if (typeof obj.score === "number") {
        return { score: Math.round(Math.min(5, Math.max(1, obj.score))), reason: obj.reason ?? "" };
      }
    } catch {
      // code block wasn't valid JSON — continue to bare number extraction
    }
  }

  // Extract bare number
  const numMatch = raw.match(/\b([1-5])\b/);
  if (numMatch) {
    return { score: parseInt(numMatch[1]!, 10), reason: "extracted from raw response" };
  }

  // Default: unparseable
  return { score: 3, reason: "PARSE_FAILED: defaulted to 3" };
}

// ── Batch Judging ───────────────────────────────────────────────────────────

interface JudgeItem {
  queryId: string;
  query: string;
  chunkId: string;
  chunkText: string;
}

async function judgeParallel(
  items: JudgeItem[],
  config: Required<JudgeConfig>,
  onProgress?: (completed: number, total: number) => void,
): Promise<JudgeResult[]> {
  const results: JudgeResult[] = [];
  let completed = 0;

  // Process in parallel batches
  for (let i = 0; i < items.length; i += config.maxParallel) {
    const batch = items.slice(i, i + config.maxParallel);
    const batchResults = await Promise.all(
      batch.map(async (item) => {
        try {
          const { score, raw } = await callJudge(item.query, item.chunkText, config);
          return {
            queryId: item.queryId,
            query: item.query,
            chunkId: item.chunkId,
            chunkText: item.chunkText.slice(0, 100),
            score,
            rawResponse: raw,
          } satisfies JudgeResult;
        } catch (e) {
          return {
            queryId: item.queryId,
            query: item.query,
            chunkId: item.chunkId,
            chunkText: item.chunkText.slice(0, 100),
            score: 0, // 0 = judge error
            rawResponse: `ERROR: ${e}`,
          } satisfies JudgeResult;
        }
      }),
    );
    results.push(...batchResults);
    completed += batch.length;
    onProgress?.(completed, items.length);
  }

  return results;
}

// ── Calibration ─────────────────────────────────────────────────────────────

interface CalibrationPair {
  query: string;
  document: string;
  expectedScore: number; // human-expected score (1-5)
  tolerance: number; // acceptable deviation (e.g., 1 means ±1)
}

const CALIBRATION_SET: CalibrationPair[] = [
  // Perfectly relevant
  {
    query: "What is Klein's timezone?",
    document: "# USER.md\n- **Timezone:** America/New_York\n- **Location:** Blacksburg, VA",
    expectedScore: 5,
    tolerance: 1,
  },
  {
    query: "How many agents does Klein have?",
    document:
      "## System Facts\n- Loop agents: `meta`, `dev`, `main`, `school`\n- Assignment agents: `immune`, `research`, `recovery`, `ghost`\n- Task orchestration: `taskmaster` via `oc-tasks`",
    expectedScore: 5,
    tolerance: 1,
  },
  // Highly relevant
  {
    query: "What model does the dev agent use?",
    document:
      "## Agent Model Policy\n- **Dev coding tasks:** always `opus` for complex/architectural work, `sonnet` for moderate tasks. NEVER flash/gemini for coding.",
    expectedScore: 5,
    tolerance: 1,
  },
  // Partially relevant
  {
    query: "How does the cron system work?",
    document:
      "## Open Risks\n- All 21 crons currently DISABLED — need Phase 1 review before re-enabling\n- main + dev heartbeats to be disabled (next config pass)",
    expectedScore: 3,
    tolerance: 1,
  },
  // Marginally relevant
  {
    query: "What is Klein's favorite food?",
    document:
      "# USER.md\n- **Name:** kleinpanic\n- **Timezone:** America/New_York\n- **Location:** Blacksburg, VA\n- Focus: development + system administration.",
    expectedScore: 2,
    tolerance: 1,
  },
  // Completely irrelevant
  {
    query: "How to configure nginx proxy buffering?",
    document:
      "# SOUL.md\nYou are KleinClaw-Meta, the configuration architect and AI systems expert.\nYou're Klein's trusted configuration architect with deep technical expertise.",
    expectedScore: 1,
    tolerance: 1,
  },
  {
    query: "What is the WireGuard IP for mt?",
    document:
      "## Voice Bridge Setup\nThe oc-voice-bridge service handles Discord voice conversations via STT and TTS pipelines.",
    expectedScore: 1,
    tolerance: 1,
  },
  // Edge cases
  {
    query: "What is oc-restart?",
    document:
      "| `oc-restart` | `~/.openclaw/hooks/oc-restart` | ONLY way to restart gateway. Validates config, posts Discord card, deferred restart after Klein approves. |",
    expectedScore: 5,
    tolerance: 1,
  },
  {
    query: "What is the Spark node IP address?",
    document:
      "### Important Relationships\n- **Spark Node (10.99.1.1):** Primary provider for vLLM, embeddings, reranking, and OCR.",
    expectedScore: 5,
    tolerance: 1,
  },
  {
    query: "How do I install Python packages?",
    document:
      "## Workspace Integrity Rules\n### File Categories\n| Category | Files | Rule |\n| Sacred | SOUL.md, USER.md, IDENTITY.md | Klein-only edits |",
    expectedScore: 1,
    tolerance: 1,
  },
  // 10 more for robustness
  {
    query: "What port does the embed service run on?",
    document:
      "### Service Configuration\nNginx buffers are increased to 256k. Streaming is restored. Embed: port 18091. Rerank: port 18096.",
    expectedScore: 5,
    tolerance: 1,
  },
  {
    query: "Who is Mika?",
    document:
      "## Active Agents (non-standard)\n- **Mika** — personal companion agent, WhatsApp DM, `xai/grok-3-mini`. Klein knows about it.",
    expectedScore: 5,
    tolerance: 1,
  },
  {
    query: "What is the temporal decay formula?",
    document:
      "**Temporal Decay:** `0.8 + 0.2 * exp(-0.03 * ageDays)` with an 80% floor to preserve older critical knowledge.",
    expectedScore: 5,
    tolerance: 1,
  },
  {
    query: "How do I set up Docker on my Mac?",
    document:
      "## Host Disambiguation\n- **broklein** = the machine OpenClaw runs on. Runs Debian sid.\n- **kernelpanic** = Klein's main Debian laptop.",
    expectedScore: 1,
    tolerance: 1,
  },
  {
    query: "What is the force_nonempty_content bug?",
    document:
      "## Forge Chat CLI\nThe forge chat command provides an interactive terminal for chatting with models loaded in vLLM slots on the Spark node.",
    expectedScore: 2,
    tolerance: 1,
  },
  {
    query: "What is the Teleport cluster name?",
    document:
      "## Teleport Cluster Notes\n- Cluster: `kleinpanic-homelab` v18.2.4 at `tp.kleinpanic.com`\n- CA pin: `sha256:298fd061c1aa7728dc7a13db89195b064364abd913d3c7744af5bfae39f40077`",
    expectedScore: 5,
    tolerance: 1,
  },
  {
    query: "What are the WIP limits for agents?",
    document:
      '### Work on in_progress tasks\n- Log progress: `oc-tasks comment <id> "what you did"`\n- Respect WIP limit (max 3 in_progress)',
    expectedScore: 4,
    tolerance: 1,
  },
  {
    query: "How does the memory-spark plugin handle authentication?",
    document:
      "## Safety-Critical Rules\n- Never use `config.patch` for `agents.list` array mutations\n- Config edits: validate JSON → `openclaw doctor` → restart → verify",
    expectedScore: 1,
    tolerance: 1,
  },
  {
    query: "What machine is OpenClaw running on?",
    document:
      "## Host Disambiguation\n- **broklein** = the machine OpenClaw runs on. NOT Klein's daily driver. Runs Debian sid.",
    expectedScore: 5,
    tolerance: 1,
  },
  {
    query: "What is the mt server layout?",
    document:
      "## mt /srv/ Layout\n- `/srv/vcs/` — VCS stack: `gitea/` (Gitea+nginx+PG+act-runner)\n- `/srv/oc/` — OpenClaw services: `mcp/` (FastMCP stack)",
    expectedScore: 5,
    tolerance: 1,
  },
];

export async function runCalibration(
  config?: JudgeConfig,
): Promise<{ passed: number; total: number; details: CalibrationResult[] }> {
  const cfg: Required<JudgeConfig> = {
    sparkUrl: config?.sparkUrl ?? SPARK_LLM_URL,
    apiKey: config?.apiKey ?? SPARK_API_KEY,
    model: config?.model ?? MODEL,
    maxParallel: config?.maxParallel ?? MAX_PARALLEL,
  };

  console.log("\n🔍 Judge Calibration (20 pairs, known ground truth)\n");

  const details: CalibrationResult[] = [];
  let passed = 0;

  for (let i = 0; i < CALIBRATION_SET.length; i += cfg.maxParallel) {
    const batch = CALIBRATION_SET.slice(i, i + cfg.maxParallel);
    const batchResults = await Promise.all(
      batch.map(async (pair) => {
        const { score, reason } = await callJudge(pair.query, pair.document, cfg);
        const diff = Math.abs(score - pair.expectedScore);
        const ok = diff <= pair.tolerance;
        return { pair, score, reason, ok, diff };
      }),
    );

    for (const r of batchResults) {
      if (r.ok) {
        passed++;
        console.log(`  ✅ [${r.score}/${r.pair.expectedScore}] "${r.pair.query.slice(0, 50)}"`);
      } else {
        console.log(
          `  ❌ [${r.score}/${r.pair.expectedScore}] "${r.pair.query.slice(0, 50)}" — ${r.reason}`,
        );
      }
      details.push(r);
    }
  }

  const passRate = passed / CALIBRATION_SET.length;
  console.log(
    `\n  Calibration: ${passed}/${CALIBRATION_SET.length} (${(passRate * 100).toFixed(0)}%)`,
  );
  if (passRate < 0.8) {
    console.log("  ⚠️  Judge calibration below 80% — results may be unreliable!");
  } else {
    console.log("  ✅ Judge calibration passed — proceeding with evaluation");
  }

  return { passed, total: CALIBRATION_SET.length, details };
}

interface CalibrationResult {
  pair: CalibrationPair;
  score: number;
  reason: string;
  ok: boolean;
  diff: number;
}

// ── Public API ──────────────────────────────────────────────────────────────

export async function judgeRetrievalResults(
  queryId: string,
  query: string,
  results: SearchResult[],
  config?: JudgeConfig,
): Promise<JudgeResult[]> {
  const cfg: Required<JudgeConfig> = {
    sparkUrl: config?.sparkUrl ?? SPARK_LLM_URL,
    apiKey: config?.apiKey ?? SPARK_API_KEY,
    model: config?.model ?? MODEL,
    maxParallel: config?.maxParallel ?? MAX_PARALLEL,
  };

  const items: JudgeItem[] = results.map((r) => ({
    queryId,
    query,
    chunkId: r.chunk.id,
    chunkText: r.chunk.text,
  }));

  return judgeParallel(items, cfg);
}

/**
 * Compute average judge score for a set of retrieval results.
 * Returns a normalized 0-1 score (divides by 5).
 */
export function computeJudgeNDCG(judgeResults: JudgeResult[], k: number = 10): number {
  if (judgeResults.length === 0) return 0;
  const topK = judgeResults.slice(0, k);

  // DCG: sum of score / log2(rank + 1)
  let dcg = 0;
  for (let i = 0; i < topK.length; i++) {
    dcg += topK[i]!.score / Math.log2(i + 2);
  }

  // Ideal DCG: all scores = 5
  let idcg = 0;
  for (let i = 0; i < topK.length; i++) {
    idcg += 5 / Math.log2(i + 2);
  }

  return idcg > 0 ? dcg / idcg : 0;
}

/**
 * Compute mean average score across all queries.
 */
export function computeMeanJudgeScore(allResults: Map<string, JudgeResult[]>): number {
  let totalScore = 0;
  let totalCount = 0;
  for (const results of allResults.values()) {
    for (const r of results) {
      if (r.score > 0) {
        // Skip errors (score=0)
        totalScore += r.score;
        totalCount++;
      }
    }
  }
  return totalCount > 0 ? totalScore / totalCount : 0;
}
