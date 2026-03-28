/**
 * memory-spark Integration Tests (Vitest)
 *
 * Tests the FULL pipeline against real Spark endpoints in a throwaway LanceDB.
 * Requires: Spark node reachable at SPARK_HOST (env var, default 10.99.1.1)
 *
 * Run:   npx vitest run tests/integration.test.ts
 * Skip:  Set SKIP_INTEGRATION=1 to skip (CI without Spark access)
 */

import { describe, it, beforeAll, afterAll, expect } from "vitest";
import * as fs from "node:fs";
import * as path from "node:path";

import { resolveConfig, type MemorySparkConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider, type EmbedProvider } from "../src/embed/provider.js";
import { EmbedQueue } from "../src/embed/queue.js";
import { createReranker } from "../src/rerank/reranker.js";
import { MemorySparkManager } from "../src/manager.js";
import { ingestFile } from "../src/ingest/pipeline.js";
import type { MemoryChunk } from "../src/storage/backend.js";

// ── Config ──────────────────────────────────────────────────────────────────

const SKIP = process.env["SKIP_INTEGRATION"] === "1";
const SPARK_HOST = process.env["SPARK_HOST"] ?? "10.99.1.1";
const SPARK_TOKEN = (() => {
  if (process.env["SPARK_BEARER_TOKEN"]) return process.env["SPARK_BEARER_TOKEN"];
  try {
    const envFile = fs.readFileSync(path.join(process.env["HOME"]!, ".openclaw", ".env"), "utf-8");
    return envFile.match(/SPARK_BEARER_TOKEN=["']?([^"'\s\n]+)/)?.[1] ?? "none";
  } catch {
    return "none";
  }
})();

const TEST_DIR = `/tmp/memory-spark-integ-vitest-${Date.now()}`;
const TEST_DB = path.join(TEST_DIR, "lancedb");

function buildConfig(): MemorySparkConfig {
  return resolveConfig({
    backend: "lancedb",
    lancedbDir: TEST_DB,
    sparkHost: SPARK_HOST,
    sparkBearerToken: SPARK_TOKEN,
  } as Parameters<typeof resolveConfig>[0]);
}

// ── Test Corpus ─────────────────────────────────────────────────────────────

const CORPUS = [
  {
    path: "memory/spark-setup.md",
    text: `# DGX Spark Node Setup\nThe DGX Spark node is at 192.0.2.1.\nEmbed service: port 18091. Rerank: port 18096. vLLM: port 18080.\nGPU memory utilization: 0.65 to leave headroom.\nRuns Nemotron-Super-120B for inference.`,
  },
  {
    path: "memory/voice-bridge.md",
    text: `# Voice Bridge Setup\nConnects Discord voice channels to OpenClaw agents.\nSTT: Parakeet CTC model at http://192.0.2.1:18094.\nTTS: Kokoro on Spark.\nKnown issue: latency spikes under heavy load.`,
  },
  {
    path: "memory/user-preferences.md",
    text: `# User Preferences\nPrimary language: TypeScript.\nEditor: Neovim with Lua config.\nPrefers functional patterns over OOP.\nWants peer-like interaction, direct and concise.`,
  },
  {
    path: "AGENTS.md",
    text: `# Dev Agent\nPrimary coding agent. Complex tasks: claude-opus-4.\nSimple edits: claude-sonnet-4.\nNever use flash for coding. Always create oc-task before work.`,
  },
  {
    path: "MISTAKES.md",
    text: `# Common Mistakes\n- NEVER use config.patch for agents.list mutations\n- NEVER edit openclaw.json directly during heartbeat\n- Always validate JSON before restart\n- Model alias fields are functional config — do not remove`,
  },
];

const QUERIES = [
  { query: "What port does Spark embed use?", expectPath: "memory/spark-setup.md" },
  { query: "How does voice chat work?", expectPath: "memory/voice-bridge.md" },
  { query: "What language does the user prefer?", expectPath: "memory/user-preferences.md" },
  { query: "Which model for coding tasks?", expectPath: "AGENTS.md" },
  { query: "What mistakes should I avoid?", expectPath: "MISTAKES.md" },
];

// ── Shared state ────────────────────────────────────────────────────────────

let cfg: MemorySparkConfig;
let backend: LanceDBBackend;
let embed: EmbedProvider;
let queue: EmbedQueue;

// ── Tests ───────────────────────────────────────────────────────────────────

describe.skipIf(SKIP)("Integration: Spark Connectivity", () => {
  it("embed endpoint responds", async () => {
    const resp = await fetch(`http://${SPARK_HOST}:18091/v1/models`, {
      headers: { Authorization: `Bearer ${SPARK_TOKEN}` },
      signal: AbortSignal.timeout(5000),
    });
    expect(resp.ok).toBe(true);
  });

  it("reranker endpoint responds", async () => {
    const resp = await fetch(`http://${SPARK_HOST}:18096/v1/rerank`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${SPARK_TOKEN}` },
      body: JSON.stringify({ query: "test", documents: ["hello"] }),
      signal: AbortSignal.timeout(5000),
    });
    expect(resp.ok).toBe(true);
  });

  it("NER endpoint responds", async () => {
    const resp = await fetch(`http://${SPARK_HOST}:18112/v1/extract`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${SPARK_TOKEN}` },
      body: JSON.stringify({ text: "Klein uses OpenClaw in Blacksburg VA" }),
      signal: AbortSignal.timeout(5000),
    });
    expect(resp.ok).toBe(true);
    const data = (await resp.json()) as { entities: unknown[] };
    expect(data.entities.length).toBeGreaterThan(0);
  });

  it("zero-shot classifier responds", async () => {
    const resp = await fetch(`http://${SPARK_HOST}:18113/v1/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${SPARK_TOKEN}` },
      body: JSON.stringify({
        text: "We decided to use LanceDB",
        labels: ["fact", "decision", "preference"],
      }),
      signal: AbortSignal.timeout(5000),
    });
    expect(resp.ok).toBe(true);
    const data = (await resp.json()) as { labels: string[]; scores: number[] };
    expect(data.labels.length).toBeGreaterThan(0);
  });
});

describe.skipIf(SKIP)("Integration: Ingest + Search Pipeline", { timeout: 120_000 }, () => {
  beforeAll(async () => {
    fs.mkdirSync(TEST_DB, { recursive: true });
    cfg = buildConfig();
    backend = new LanceDBBackend(cfg);
    await backend.open();
    embed = await createEmbedProvider(cfg.embed);
    queue = new EmbedQueue(embed, { concurrency: 1, maxRetries: 2, timeoutMs: 30000 });

    // Write corpus to temp files and ingest
    const workspaceDir = path.join(TEST_DIR, "workspace");
    fs.mkdirSync(path.join(workspaceDir, "memory"), { recursive: true });
    for (const doc of CORPUS) {
      const fullPath = path.join(workspaceDir, doc.path);
      fs.mkdirSync(path.dirname(fullPath), { recursive: true });
      fs.writeFileSync(fullPath, doc.text);
    }

    for (const doc of CORPUS) {
      await ingestFile({
        filePath: path.join(workspaceDir, doc.path),
        workspaceDir,
        source: "memory",
        agentId: "test",
        embed: queue,
        backend,
        cfg,
      });
    }
  }, 120_000);

  afterAll(async () => {
    await backend?.close();
    fs.rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it("corpus is indexed", async () => {
    const status = await backend.status();
    expect(status.chunkCount).toBeGreaterThan(0);
    expect(status.ready).toBe(true);
  });

  it("vector search returns results above minScore", async () => {
    const vec = await embed.embedQuery("What port does Spark embed use?");
    const results = await backend.vectorSearch(vec, { query: "embed port", maxResults: 5 });
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]!.score).toBeGreaterThan(0.2);
  });

  it("FTS search works with WHERE clause", async () => {
    const results = await backend.ftsSearch("Nemotron-Super-120B", {
      query: "Nemotron",
      maxResults: 5,
      agentId: "test",
    });
    expect(results.length).toBeGreaterThan(0);
  });

  it("FTS search works for keyword lookup", async () => {
    const results = await backend.ftsSearch("TypeScript Neovim", {
      query: "TypeScript",
      maxResults: 5,
    });
    expect(results.length).toBeGreaterThan(0);
  });

  it("correct document ranked top-1 for each query", async () => {
    let correct = 0;
    for (const q of QUERIES) {
      const vec = await embed.embedQuery(q.query);
      const results = await backend.vectorSearch(vec, { query: q.query, maxResults: 3 });
      if (results[0]?.chunk.path === q.expectPath) correct++;
    }
    // Allow 1 miss out of 5 — 80% accuracy minimum
    expect(correct).toBeGreaterThanOrEqual(4);
  });

  it("schema has pool column on fresh table", async () => {
    const vec = await embed.embedQuery("schema test");
    const results = await backend.vectorSearch(vec, { query: "schema", maxResults: 1 });
    // Pool should be set by resolvePool during upsert
    expect(results[0]?.chunk.pool).toBeTruthy();
  });

  it("discoverAgents returns test agent", async () => {
    const agents = await backend.discoverAgents();
    expect(agents).toContain("test");
  });

  it("reranker improves relevance ordering", async () => {
    const reranker = await createReranker(cfg.rerank);
    const vec = await embed.embedQuery("GPU memory utilization setting");
    const raw = await backend.vectorSearch(vec, { query: "GPU memory", maxResults: 10 });
    expect(raw.length).toBeGreaterThan(0);

    const reranked = await reranker.rerank("GPU memory utilization setting", raw, 5);
    expect(reranked.length).toBeGreaterThan(0);
    // Reranked top result should be about Spark setup (mentions GPU memory)
    expect(reranked[0]!.chunk.text).toMatch(/GPU|memory|utilization/i);
  });
});
