#!/usr/bin/env npx tsx
/**
 * E2E Agent Benchmark — tests whether recalled memories actually help agents answer correctly.
 *
 * This is the definitive test. IR metrics (NDCG, MRR) measure retrieval quality,
 * but this measures: "Does the agent produce a better answer WITH memory than WITHOUT?"
 *
 * Usage:
 *   MEMORY_SPARK_DATA_DIR=./test-data SPARK_HOST=127.0.0.1 npx tsx evaluation/e2e-benchmark.ts
 *   MEMORY_SPARK_DATA_DIR=./test-data SPARK_HOST=127.0.0.1 npx tsx evaluation/e2e-benchmark.ts --model codex
 *
 * Models:
 *   --model nemotron  (default) — Nemotron-Super-120B via Spark vLLM (local, free)
 *   --model codex     — OpenAI Codex via API (requires OPENAI_API_KEY)
 *
 * Output: evaluation/results/e2e-{model}-{timestamp}.json
 */

import fs from "node:fs/promises";
import path from "node:path";

import { resolveConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { createReranker } from "../src/rerank/reranker.js";
import type { SearchResult } from "../src/storage/backend.js";
import {
  hybridMerge,
  applyTemporalDecay,
  applySourceWeighting,
  mmrRerank,
} from "../src/auto/recall.js";
import { formatRecalledMemories } from "../src/security.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface GoldenQuery {
  id: string;
  category: string;
  query: string;
  gold_answer: string;
  relevant: Array<{ path_contains: string; snippet_contains?: string; grade: number }>;
}

interface GoldenDataset {
  categories: string[];
  queries: GoldenQuery[];
}

interface JudgmentResult {
  score: "correct" | "partial" | "wrong";
  reasoning: string;
}

interface E2EQueryResult {
  queryId: string;
  category: string;
  query: string;
  goldAnswer: string;
  // Memory ON
  withMemory: {
    answer: string;
    judgment: JudgmentResult;
    retrievedCount: number;
    latencyMs: number;
  };
  // Memory OFF
  withoutMemory: {
    answer: string;
    judgment: JudgmentResult;
    latencyMs: number;
  };
}

interface E2EResults {
  model: string;
  timestamp: string;
  queryCount: number;
  withMemory: {
    correct: number;
    partial: number;
    wrong: number;
    accuracy: number; // (correct + 0.5*partial) / total
  };
  withoutMemory: {
    correct: number;
    partial: number;
    wrong: number;
    accuracy: number;
  };
  improvement: number; // withMemory.accuracy - withoutMemory.accuracy
  perCategory: Record<string, { withMemory: number; withoutMemory: number; delta: number }>;
  perQuery: E2EQueryResult[];
}

// ---------------------------------------------------------------------------
// LLM Clients
// ---------------------------------------------------------------------------

interface LLMClient {
  name: string;
  generate(systemPrompt: string, userPrompt: string): Promise<string>;
}

function createNemotronClient(sparkHost: string, token?: string): LLMClient {
  const baseUrl = `http://${sparkHost}:18096/v1`;
  return {
    name: "nemotron-super-120b",
    async generate(systemPrompt: string, userPrompt: string): Promise<string> {
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const resp = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          model: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt },
          ],
          max_tokens: 1024,
          temperature: 0.1,
        }),
        signal: AbortSignal.timeout(120000),
      });
      if (!resp.ok) throw new Error(`Nemotron error: ${resp.status} ${await resp.text()}`);
      const data = (await resp.json()) as {
        choices: Array<{ message: { content: string } }>;
      };
      return data.choices?.[0]?.message?.content?.trim() ?? "";
    },
  };
}

function createCodexClient(): LLMClient {
  const apiKey = process.env["OPENAI_API_KEY"];
  if (!apiKey) throw new Error("OPENAI_API_KEY required for codex model");

  return {
    name: "gpt-5.3-codex",
    async generate(systemPrompt: string, userPrompt: string): Promise<string> {
      const resp = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: "gpt-5.3-codex",
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt },
          ],
          max_tokens: 1024,
          temperature: 0.1,
        }),
        signal: AbortSignal.timeout(60000),
      });
      if (!resp.ok) throw new Error(`Codex error: ${resp.status} ${await resp.text()}`);
      const data = (await resp.json()) as {
        choices: Array<{ message: { content: string } }>;
      };
      return data.choices?.[0]?.message?.content?.trim() ?? "";
    },
  };
}

// ---------------------------------------------------------------------------
// Retrieval (same pipeline as production recall.ts)
// ---------------------------------------------------------------------------

async function retrieveContext(
  query: string,
  backend: LanceDBBackend,
  embedQuery: (text: string) => Promise<number[]>,
  reranker: { rerank: (q: string, r: SearchResult[], n: number) => Promise<SearchResult[]> },
): Promise<{ memories: string; count: number; latencyMs: number }> {
  const start = performance.now();
  const fetchN = 20;
  const minScore = 0.2;

  const queryVector = await embedQuery(query);

  const [vectorResults, rawFtsResults] = await Promise.all([
    backend.vectorSearch(queryVector, { query, maxResults: fetchN, minScore }).catch(() => []),
    backend.ftsSearch(query, { query, maxResults: fetchN }).catch(() => []),
  ]);

  const ftsResults = (rawFtsResults as SearchResult[]).filter(
    (r) => r.chunk.source !== "sessions" && r.score >= minScore,
  );

  const merged = hybridMerge(vectorResults as SearchResult[], ftsResults, fetchN);
  applySourceWeighting(merged);
  applyTemporalDecay(merged);
  const diverse = mmrRerank(merged, 10, 0.7);
  const final = await reranker.rerank(query, diverse, 5);

  const memories = final.map((r) => ({
    source: `${r.chunk.source}:${r.chunk.path}:${r.chunk.start_line}`,
    text: r.chunk.text.slice(0, 500),
    score: r.score,
    updatedAt: r.chunk.updated_at,
    contentType: r.chunk.content_type ?? "knowledge",
    agentId: r.chunk.agent_id,
    path: r.chunk.path,
  }));

  const formatted = formatRecalledMemories(memories);
  const latencyMs = performance.now() - start;

  return { memories: formatted, count: final.length, latencyMs };
}

// ---------------------------------------------------------------------------
// Judge
// ---------------------------------------------------------------------------

async function judgeAnswer(
  llm: LLMClient,
  query: string,
  goldAnswer: string,
  actualAnswer: string,
): Promise<JudgmentResult> {
  const judgePrompt = `You are an evaluation judge. Compare the actual answer to the gold (correct) answer for the given question.

Score:
- "correct" if the actual answer contains the key facts from the gold answer
- "partial" if the actual answer is partially right but missing important details
- "wrong" if the actual answer is incorrect or doesn't address the question

Respond with ONLY valid JSON: {"score": "correct"|"partial"|"wrong", "reasoning": "brief explanation"}`;

  const userPrompt = `Question: ${query}

Gold Answer: ${goldAnswer}

Actual Answer: ${actualAnswer}`;

  try {
    const response = await llm.generate(judgePrompt, userPrompt);
    // Extract JSON from response (model might wrap it in markdown)
    const jsonMatch = response.match(/\{[\s\S]*?"score"[\s\S]*?\}/);
    if (!jsonMatch) return { score: "wrong", reasoning: "Judge failed to produce valid JSON" };
    const parsed = JSON.parse(jsonMatch[0]) as JudgmentResult;
    if (!["correct", "partial", "wrong"].includes(parsed.score)) {
      return { score: "wrong", reasoning: `Invalid score: ${parsed.score}` };
    }
    return parsed;
  } catch (err) {
    return { score: "wrong", reasoning: `Judge error: ${err}` };
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const dataDir = process.env["MEMORY_SPARK_DATA_DIR"];
  if (!dataDir) {
    console.error("ERROR: Set MEMORY_SPARK_DATA_DIR to avoid touching production.");
    process.exit(1);
  }

  // Parse args
  const args = process.argv.slice(2);
  let modelName = "nemotron";
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--model" && args[i + 1]) modelName = args[++i]!;
  }

  console.log(`\n=== E2E Agent Benchmark ===`);
  console.log(`Model: ${modelName}`);
  console.log(`Data dir: ${path.resolve(dataDir)}\n`);

  // Load golden dataset
  const gtPath = path.join("evaluation", "ground-truth.json");
  const gt = JSON.parse(await fs.readFile(gtPath, "utf8")) as GoldenDataset;

  // Filter to queries that have gold_answer
  const queries = gt.queries.filter((q) => (q as any).gold_answer);
  if (queries.length === 0) {
    console.error("No queries with gold_answer found in ground-truth.json.");
    console.error("Add 'gold_answer' field to queries to run E2E benchmark.");
    console.error("\nExample:");
    console.error(
      JSON.stringify(
        {
          id: "safety-001",
          query: "How do I restart the OpenClaw gateway?",
          gold_answer:
            "Use oc-restart at ~/.openclaw/hooks/oc-restart. Never use systemctl restart directly.",
          relevant: [{ path_contains: "AGENTS.md", snippet_contains: "oc-restart", grade: 3 }],
        },
        null,
        2,
      ),
    );
    process.exit(1);
  }

  console.log(`Queries with gold answers: ${queries.length}/${gt.queries.length}\n`);

  // Initialize retrieval stack
  const cfg = resolveConfig();
  const backend = new LanceDBBackend(cfg);
  await backend.open();
  const status = await backend.status();
  console.log(`LanceDB: ${status.chunkCount} chunks`);
  if (status.chunkCount === 0) {
    console.error("Index is empty. Run standalone-index.ts first.");
    await backend.close();
    process.exit(1);
  }

  const provider = await createEmbedProvider(cfg.embed);
  const embedQuery = (text: string) => provider.embedQuery(text);

  let reranker;
  try {
    reranker = await createReranker(cfg.rerank);
  } catch {
    console.warn("Reranker unavailable — using passthrough");
    reranker = {
      async rerank(_q: string, results: SearchResult[], topN: number) {
        return results.slice(0, topN);
      },
    };
  }

  // Initialize LLM
  const sparkHost = process.env["SPARK_HOST"] ?? "127.0.0.1";
  const sparkToken =
    process.env["SPARK_BEARER_TOKEN"] ??
    (() => {
      try {
        const content = require("fs").readFileSync(
          path.join(process.env["HOME"]!, ".openclaw", ".env"),
          "utf-8",
        );
        return content.match(/SPARK_BEARER_TOKEN=["']?([^"'\s\n]+)/)?.[1];
      } catch {
        return undefined;
      }
    })();

  let llm: LLMClient;
  if (modelName === "codex") {
    llm = createCodexClient();
  } else {
    llm = createNemotronClient(sparkHost, sparkToken);
  }
  console.log(`LLM: ${llm.name}\n`);

  // Run benchmark
  const results: E2EQueryResult[] = [];
  const systemPromptBase =
    "You are an OpenClaw agent assistant. Answer the question concisely based on your knowledge.";

  for (let i = 0; i < queries.length; i++) {
    const q = queries[i]!;
    console.log(`[${i + 1}/${queries.length}] ${q.id}: ${q.query.slice(0, 60)}...`);

    // WITH memory
    const retrieval = await retrieveContext(q.query, backend, embedQuery, reranker).catch(() => ({
      memories: "",
      count: 0,
      latencyMs: 0,
    }));

    const withMemorySystem = retrieval.memories
      ? `${systemPromptBase}\n\n${retrieval.memories}`
      : systemPromptBase;

    const withMemoryStart = performance.now();
    let withMemoryAnswer: string;
    try {
      withMemoryAnswer = await llm.generate(withMemorySystem, q.query);
    } catch (err) {
      withMemoryAnswer = `[ERROR: ${err}]`;
    }
    const withMemoryLatency = performance.now() - withMemoryStart;

    // WITHOUT memory
    const withoutMemoryStart = performance.now();
    let withoutMemoryAnswer: string;
    try {
      withoutMemoryAnswer = await llm.generate(systemPromptBase, q.query);
    } catch (err) {
      withoutMemoryAnswer = `[ERROR: ${err}]`;
    }
    const withoutMemoryLatency = performance.now() - withoutMemoryStart;

    // Judge both
    const [withJudgment, withoutJudgment] = await Promise.all([
      judgeAnswer(llm, q.query, (q as any).gold_answer, withMemoryAnswer),
      judgeAnswer(llm, q.query, (q as any).gold_answer, withoutMemoryAnswer),
    ]);

    results.push({
      queryId: q.id,
      category: q.category,
      query: q.query,
      goldAnswer: (q as any).gold_answer,
      withMemory: {
        answer: withMemoryAnswer.slice(0, 1000),
        judgment: withJudgment,
        retrievedCount: retrieval.count,
        latencyMs: Number((retrieval.latencyMs + withMemoryLatency).toFixed(1)),
      },
      withoutMemory: {
        answer: withoutMemoryAnswer.slice(0, 1000),
        judgment: withoutJudgment,
        latencyMs: Number(withoutMemoryLatency.toFixed(1)),
      },
    });

    const icon = withJudgment.score === "correct" ? "✅" : withJudgment.score === "partial" ? "🟡" : "❌";
    const deltaIcon =
      withJudgment.score === withoutJudgment.score
        ? "="
        : ["correct", "partial", "wrong"].indexOf(withJudgment.score) <
            ["correct", "partial", "wrong"].indexOf(withoutJudgment.score)
          ? "↑"
          : "↓";
    console.log(
      `  ${icon} with=${withJudgment.score} without=${withoutJudgment.score} ${deltaIcon} (${retrieval.count} memories)`,
    );
  }

  // Compile results
  function accuracy(items: E2EQueryResult[], field: "withMemory" | "withoutMemory"): number {
    const total = items.length;
    if (total === 0) return 0;
    let score = 0;
    for (const item of items) {
      if (item[field].judgment.score === "correct") score += 1;
      else if (item[field].judgment.score === "partial") score += 0.5;
    }
    return score / total;
  }

  function count(items: E2EQueryResult[], field: "withMemory" | "withoutMemory", score: string) {
    return items.filter((r) => r[field].judgment.score === score).length;
  }

  const perCategory: E2EResults["perCategory"] = {};
  const categories = [...new Set(results.map((r) => r.category))];
  for (const cat of categories) {
    const catResults = results.filter((r) => r.category === cat);
    const withAcc = accuracy(catResults, "withMemory");
    const withoutAcc = accuracy(catResults, "withoutMemory");
    perCategory[cat] = { withMemory: withAcc, withoutMemory: withoutAcc, delta: withAcc - withoutAcc };
  }

  const e2eResults: E2EResults = {
    model: llm.name,
    timestamp: new Date().toISOString(),
    queryCount: results.length,
    withMemory: {
      correct: count(results, "withMemory", "correct"),
      partial: count(results, "withMemory", "partial"),
      wrong: count(results, "withMemory", "wrong"),
      accuracy: accuracy(results, "withMemory"),
    },
    withoutMemory: {
      correct: count(results, "withoutMemory", "correct"),
      partial: count(results, "withoutMemory", "partial"),
      wrong: count(results, "withoutMemory", "wrong"),
      accuracy: accuracy(results, "withoutMemory"),
    },
    improvement: accuracy(results, "withMemory") - accuracy(results, "withoutMemory"),
    perCategory,
    perQuery: results,
  };

  // Write results
  const resultsDir = path.join("evaluation", "results");
  await fs.mkdir(resultsDir, { recursive: true });
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const outPath = path.join(resultsDir, `e2e-${modelName}-${ts}.json`);
  await fs.writeFile(outPath, JSON.stringify(e2eResults, null, 2));

  // Print summary
  console.log(`\n${"=".repeat(60)}`);
  console.log(`E2E RESULTS — ${llm.name}`);
  console.log(`${"=".repeat(60)}`);
  console.log(`Queries: ${results.length}`);
  console.log();
  console.log(`WITH MEMORY:    ${(e2eResults.withMemory.accuracy * 100).toFixed(1)}% accuracy`);
  console.log(
    `  ✅ ${e2eResults.withMemory.correct} correct | 🟡 ${e2eResults.withMemory.partial} partial | ❌ ${e2eResults.withMemory.wrong} wrong`,
  );
  console.log();
  console.log(`WITHOUT MEMORY: ${(e2eResults.withoutMemory.accuracy * 100).toFixed(1)}% accuracy`);
  console.log(
    `  ✅ ${e2eResults.withoutMemory.correct} correct | 🟡 ${e2eResults.withoutMemory.partial} partial | ❌ ${e2eResults.withoutMemory.wrong} wrong`,
  );
  console.log();
  const delta = e2eResults.improvement * 100;
  const deltaStr = delta > 0 ? `+${delta.toFixed(1)}%` : `${delta.toFixed(1)}%`;
  console.log(`IMPROVEMENT: ${deltaStr} ${delta >= 10 ? "✅ PASS" : delta > 0 ? "🟡 MARGINAL" : "❌ FAIL"}`);
  console.log();
  console.log("Per category:");
  for (const [cat, m] of Object.entries(perCategory)) {
    const d = m.delta * 100;
    console.log(
      `  ${cat.padEnd(15)} with=${(m.withMemory * 100).toFixed(0)}% without=${(m.withoutMemory * 100).toFixed(0)}% delta=${d > 0 ? "+" : ""}${d.toFixed(0)}%`,
    );
  }
  console.log(`\nSaved: ${outPath}`);

  await backend.close();
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
