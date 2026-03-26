#!/usr/bin/env npx tsx
/**
 * memory-spark Integration Tests
 *
 * Tests the FULL pipeline against real Spark endpoints in a throwaway LanceDB.
 * Catches the class of bugs that unit tests miss:
 *   - Vector space alignment (query vs document embeddings)
 *   - Score distribution vs minScore thresholds
 *   - End-to-end ingest → search → rerank → return
 *   - FTS fallback when embed is down
 *   - Schema evolution on fresh tables
 *
 * Prerequisites: Spark node reachable at $SPARK_HOST (default: 127.0.0.1)
 * Usage:
 *   npx tsx test-integration.ts                    # all suites
 *   npx tsx test-integration.ts --suite scores     # single suite
 *   npx tsx test-integration.ts --verbose          # show all scores
 */

import { chunkDocument } from "./src/embed/chunker.js";
import { createEmbedProvider, type EmbedProvider } from "./src/embed/provider.js";
import { LanceDBBackend } from "./src/storage/lancedb.js";
import { createReranker, type Reranker } from "./src/rerank/reranker.js";
import { MemorySparkManager } from "./src/manager.js";
import { resolveConfig, type MemorySparkConfig } from "./src/config.js";
import { ingestFile } from "./src/ingest/pipeline.js";
import { EmbedQueue } from "./src/embed/queue.js";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

// ── Config ──────────────────────────────────────────────────────────────────
const SPARK_HOST = process.env.SPARK_HOST ?? "localhost";
const SPARK_TOKEN = (() => {
  if (process.env["SPARK_BEARER_TOKEN"]) return process.env["SPARK_BEARER_TOKEN"];
  try {
    const envFile = fs.readFileSync(path.join(os.homedir(), ".openclaw", ".env"), "utf-8");
    return envFile.match(/SPARK_BEARER_TOKEN=["']?([^"'\s\n]+)/)?.[1] ?? "none";
  } catch { return "none"; }
})();

const VERBOSE = process.argv.includes("--verbose");
const SUITE_FILTER = (() => {
  const idx = process.argv.indexOf("--suite");
  return idx >= 0 ? process.argv[idx + 1] : null;
})();

const TEST_DIR = `/tmp/memory-spark-integ-${Date.now()}`;
const TEST_DB = path.join(TEST_DIR, "lancedb");

function buildConfig(overrides?: Partial<MemorySparkConfig>): MemorySparkConfig {
  return resolveConfig({
    backend: "lancedb",
    lancedbDir: TEST_DB,
    sparkHost: SPARK_HOST,
    sparkBearerToken: SPARK_TOKEN,
    embed: {
      provider: "spark",
      spark: {
        baseUrl: `http://${SPARK_HOST}:18091/v1`,
        apiKey: SPARK_TOKEN,
        model: "nvidia/llama-embed-nemotron-8b",
        dimensions: 4096,
      },
    },
    rerank: {
      enabled: true,
      spark: {
        baseUrl: `http://${SPARK_HOST}:18096/v1`,
        apiKey: SPARK_TOKEN,
        model: "nvidia/llama-nemotron-rerank-1b-v2",
      },
      topN: 10,
    },
    autoRecall: {
      enabled: true,
      agents: ["*"],
      maxResults: 5,
      minScore: 0.3,
      queryMessageCount: 3,
    },
    ...overrides,
  });
}

// ── Test corpus ─────────────────────────────────────────────────────────────
// Realistic documents that would live in an agent workspace
const CORPUS: Array<{ path: string; text: string }> = [
  {
    path: "memory/spark-setup.md",
    text: `# DGX Spark Node Setup

The DGX Spark node is the primary AI inference server located at 192.0.2.1.
It runs 8 microservices: embed, rerank, TTS, STT, OCR, NER, zero-shot, and summarizer.
The vLLM server hosts Nemotron-Super-120B for general inference.
GPU memory utilization is set to 0.65 to leave headroom for other services.

## Access
- SSH: \`ssh dgx\` (alias in ~/.ssh/config)
- WireGuard tunnel: 192.0.2.2 → 192.0.2.1
- Nginx proxy: external port 18080 → internal port 8080

## Services
| Service | Internal | External |
|---------|----------|----------|
| Embed   | :8091    | :18091   |
| Rerank  | :8096    | :18096   |
| vLLM    | :8080    | :18080   |
`,
  },
  {
    path: "memory/voice-bridge.md",
    text: `# Voice Bridge Setup

The voice bridge connects Discord voice channels to OpenClaw agents.
It uses oc-voice-bridge which handles STT (speech-to-text) via Spark's
Parakeet CTC model and TTS (text-to-speech) via Kokoro on the Spark node.

## Configuration
- Service: oc-voice-bridge running as systemd user service
- STT endpoint: http://192.0.2.1:18094/v1/audio/transcriptions
- TTS endpoint: Kokoro TTS on Spark
- Discord bot token: stored in ~/.openclaw/.env

## Known Issues
- Latency spikes when Spark is under heavy load
- Voice activity detection sometimes clips first word
`,
  },
  {
    path: "memory/klein-preferences.md",
    text: `# User Preferences

## Coding
- Primary language: TypeScript for all agent tools and plugins
- Editor: Neovim with Lua config
- Prefers functional patterns over OOP
- Uses zsh with starship prompt

## Communication
- Wants peer-like interaction, not subservient
- Direct and concise preferred over verbose

## System
- Runs Debian sid on dev-host (OpenClaw host)
- Runs Debian stable on workstation (daily driver)
- Uses tailscale for mesh networking
`,
  },
  {
    path: "AGENTS.md",
    text: `# AGENTS.md - Dev Agent

## Role
Primary coding agent. Handles feature implementation, bug fixes, and code review.

## Models
- Complex tasks: claude-opus-4 (never use flash for coding)
- Simple edits: claude-sonnet-4
- Research: gemini-flash (read-only, summarization)

## Rules
- Always create oc-task before starting work
- Run tests before marking tasks done
- Never edit openclaw.json directly
`,
  },
  {
    path: "memory/teleport-setup.md",
    text: `# Teleport Cluster Configuration

Cluster name: example-homelab v18.2.4
Auth server: tp.example.internal → edge → wg-teleport → 192.0.2.10
CA pin: sha256:298fd061c1aa7728dc7a13db89195b064364abd913d3c7744af5bfae39f40077

## Nodes
- dev-host: v18.7.3 (ahead of auth server)
- worker: v18.7.2 (ahead of auth server)
- mt: pending enrollment

## Known Issues
- SAN mismatch: cert covers mt-teleport-auth, localhost, teleport.example.internal
  but NOT tp.example.internal
- tsh login requires --insecure flag due to SAN mismatch
- Node agents use ca_pin which bypasses SAN check
`,
  },
];

// ── Queries and expected matches ────────────────────────────────────────────
const QUERIES: Array<{ query: string; expectPath: string; label: string }> = [
  { query: "What port does the Spark embed service use?", expectPath: "memory/spark-setup.md", label: "Spark port question" },
  { query: "How does voice chat work with Discord?", expectPath: "memory/voice-bridge.md", label: "Voice bridge question" },
  { query: "What coding language does the user prefer?", expectPath: "memory/klein-preferences.md", label: "User language preference" },
  { query: "Which model should I use for coding tasks?", expectPath: "AGENTS.md", label: "Model selection for coding" },
  { query: "Teleport cluster CA pin hash", expectPath: "memory/teleport-setup.md", label: "Teleport CA pin" },
  { query: "GPU memory utilization setting", expectPath: "memory/spark-setup.md", label: "GPU memory config" },
  { query: "User's preferred editor and programming language", expectPath: "memory/klein-preferences.md", label: "User editor preference" },
  { query: "STT speech to text endpoint", expectPath: "memory/voice-bridge.md", label: "STT endpoint" },
];

// ── Test framework ──────────────────────────────────────────────────────────
interface TestResult {
  suite: string;
  test: string;
  status: "PASS" | "FAIL" | "SKIP";
  detail?: string;
}

const results: TestResult[] = [];

function log(suite: string, test: string, pass: boolean, detail?: string) {
  const icon = pass ? "✅" : "❌";
  console.log(`  ${icon} ${test}${detail && VERBOSE ? ` — ${detail}` : ""}`);
  results.push({ suite, test, status: pass ? "PASS" : "FAIL", detail });
}

function skip(suite: string, test: string, reason: string) {
  console.log(`  ⏭️  ${test} — ${reason}`);
  results.push({ suite, test, status: "SKIP", detail: reason });
}

function section(name: string) {
  console.log(`\n━━ ${name} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
}

function shouldRun(suite: string): boolean {
  return !SUITE_FILTER || suite === SUITE_FILTER;
}

// ── Suites ──────────────────────────────────────────────────────────────────

async function suiteConnectivity(cfg: MemorySparkConfig) {
  if (!shouldRun("connectivity")) return;
  section("Suite: Connectivity (Spark endpoints reachable)");

  // Embed: GET /v1/models works
  try {
    const resp = await fetch(`http://${SPARK_HOST}:18091/v1/models`, {
      headers: { Authorization: `Bearer ${SPARK_TOKEN}` },
      signal: AbortSignal.timeout(5000),
    });
    log("connectivity", `Embed (:18091)`, resp.ok, `status=${resp.status}`);
  } catch (err: any) {
    log("connectivity", `Embed (:18091)`, false, err.message);
  }

  // Rerank: POST /v1/rerank (no /v1/models route)
  try {
    const resp = await fetch(`http://${SPARK_HOST}:18096/v1/rerank`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${SPARK_TOKEN}` },
      body: JSON.stringify({ query: "test", documents: ["hello"], model: "nvidia/llama-nemotron-rerank-1b-v2" }),
      signal: AbortSignal.timeout(5000),
    });
    const data = await resp.json() as any;
    log("connectivity", `Rerank (:18096)`, resp.ok && !!data.results, `status=${resp.status}`);
  } catch (err: any) {
    log("connectivity", `Rerank (:18096)`, false, err.message);
  }
}

async function suiteIngestPipeline(cfg: MemorySparkConfig, embed: EmbedProvider) {
  if (!shouldRun("ingest")) return;
  section("Suite: Ingest Pipeline (chunk + embed + store)");

  const backend = new LanceDBBackend(cfg);
  await backend.open();
  const queue = new EmbedQueue(embed, { batchSize: 50, maxRetries: 2 });

  // Write corpus to temp files (ingestFile reads from disk)
  const workspaceDir = path.join(TEST_DIR, "workspace");
  fs.mkdirSync(path.join(workspaceDir, "memory"), { recursive: true });
  for (const doc of CORPUS) {
    const fullPath = path.join(workspaceDir, doc.path);
    fs.mkdirSync(path.dirname(fullPath), { recursive: true });
    fs.writeFileSync(fullPath, doc.text);
  }

  let totalChunks = 0;
  for (const doc of CORPUS) {
    try {
      const result = await ingestFile({
        filePath: path.join(workspaceDir, doc.path),
        workspaceDir,
        source: "memory",
        agentId: "test",
        embed: queue,
        backend,
        cfg,
      });
      const count = result.chunksAdded;
      totalChunks += count;
      log("ingest", `Ingested ${doc.path}`, count > 0, `${count} chunks`);
    } catch (err: any) {
      log("ingest", `Ingested ${doc.path}`, false, err.message);
    }
  }

  const status = await backend.status();
  log("ingest", `Total chunks in DB`, status.chunkCount > 0, `${status.chunkCount} rows`);
  log("ingest", `All docs produced chunks`, totalChunks >= CORPUS.length, `${totalChunks} total`);

  await backend.close();
}

async function suiteScoreDistribution(cfg: MemorySparkConfig, embed: EmbedProvider) {
  if (!shouldRun("scores")) return;
  section("Suite: Score Distribution (the bug that killed us)");

  const backend = new LanceDBBackend(cfg);
  await backend.open();

  // The critical test: embed a query and check if top results have scores
  // above the production minScore threshold.
  // Production threshold: vector search pre-filter before reranking.
  // Reranker handles final relevance scoring, so this is a coarse gate.
  // With 4096-dim embeddings, cosine similarity of 0.2+ indicates topical relevance.
  const PRODUCTION_MIN_SCORE = 0.2;

  let allAboveThreshold = true;
  let anyResultFound = true;
  const scoreReport: string[] = [];

  for (const q of QUERIES) {
    try {
      const qvec = await embed.embedQuery(q.query);
      const results = await backend.vectorSearch(qvec, {
        query: q.query,
        maxResults: 5,
        // NO minScore filter — we want to see raw scores
      });

      if (results.length === 0) {
        log("scores", q.label, false, "No results at all — empty vector search");
        anyResultFound = false;
        continue;
      }

      const topScore = results[0]!.score;
      const topPath = results[0]!.chunk.path;
      const matchesExpected = topPath === q.expectPath;

      const aboveThreshold = topScore >= PRODUCTION_MIN_SCORE;
      if (!aboveThreshold) allAboveThreshold = false;

      const detail = `top_score=${topScore.toFixed(4)} path=${topPath} expected=${q.expectPath} match=${matchesExpected}`;
      scoreReport.push(detail);

      log("scores", `${q.label}: score above ${PRODUCTION_MIN_SCORE}`, aboveThreshold, detail);

      if (VERBOSE) {
        for (const r of results) {
          console.log(`      score=${r.score.toFixed(4)} path=${r.chunk.path} text="${r.chunk.text.slice(0, 60)}..."`);
        }
      }
    } catch (err: any) {
      log("scores", q.label, false, err.message);
    }
  }

  // Summary assertions
  log("scores", "ALL queries return at least one result", anyResultFound);
  log("scores", `ALL top scores above production threshold (${PRODUCTION_MIN_SCORE})`, allAboveThreshold,
    allAboveThreshold ? "Vector space is aligned" : "Some queries scored below threshold — check if minScore is too aggressive or if corpus coverage is thin");

  await backend.close();
}

async function suiteRelevanceAccuracy(cfg: MemorySparkConfig, embed: EmbedProvider) {
  if (!shouldRun("relevance")) return;
  section("Suite: Relevance Accuracy (correct doc ranked #1)");

  const backend = new LanceDBBackend(cfg);
  await backend.open();

  let correctTop1 = 0;
  let correctTop3 = 0;

  for (const q of QUERIES) {
    try {
      const qvec = await embed.embedQuery(q.query);
      const results = await backend.vectorSearch(qvec, {
        query: q.query,
        maxResults: 5,
      });

      const top1Match = results[0]?.chunk.path === q.expectPath;
      const top3Match = results.slice(0, 3).some((r) => r.chunk.path === q.expectPath);

      if (top1Match) correctTop1++;
      if (top3Match) correctTop3++;

      log("relevance", `${q.label} → top-1 correct`, top1Match,
        `got=${results[0]?.chunk.path ?? "none"} expected=${q.expectPath}`);
    } catch (err: any) {
      log("relevance", q.label, false, err.message);
    }
  }

  const accuracy1 = correctTop1 / QUERIES.length;
  const accuracy3 = correctTop3 / QUERIES.length;
  log("relevance", `Recall@1 ≥ 0.6`, accuracy1 >= 0.6, `${(accuracy1 * 100).toFixed(0)}%`);
  log("relevance", `Recall@3 ≥ 0.8`, accuracy3 >= 0.8, `${(accuracy3 * 100).toFixed(0)}%`);

  await backend.close();
}

async function suiteManagerE2E(cfg: MemorySparkConfig) {
  if (!shouldRun("manager")) return;
  section("Suite: Manager E2E (full search path)");

  const backend = new LanceDBBackend(cfg);
  await backend.open();
  const embed2 = await createEmbedProvider(cfg.embed);
  const reranker = await createReranker(cfg.rerank);

  const mgr = new MemorySparkManager({
    cfg,
    agentId: "test",
    workspaceDir: os.homedir(),
    backend,
    embed: embed2,
    reranker,
  });

  // The manager.search() call is what the plugin tool uses.
  // This is the exact code path that was returning empty.
  for (const q of QUERIES.slice(0, 4)) {
    try {
      const results = await mgr.search(q.query, { maxResults: 5 });
      const found = results.length > 0;
      const topMatch = results[0]?.path === q.expectPath;
      log("manager", `search("${q.query.slice(0, 40)}...")`, found,
        `${results.length} results, top=${results[0]?.path ?? "none"} match=${topMatch}`);

      if (VERBOSE && results.length > 0) {
        for (const r of results) {
          console.log(`      score=${r.score.toFixed(4)} path=${r.path} snippet="${r.snippet.slice(0, 60)}..."`);
        }
      }
    } catch (err: any) {
      log("manager", `search("${q.query.slice(0, 40)}...")`, false, err.message);
    }
  }

  await mgr.close();
}

async function suiteFTSFallback(cfg: MemorySparkConfig) {
  if (!shouldRun("fts")) return;
  section("Suite: FTS Fallback (keyword search)");

  const backend = new LanceDBBackend(cfg);
  await backend.open();

  const ftsQueries = [
    { query: "Nemotron-Super-120B", label: "Exact model name" },
    { query: "TypeScript", label: "Language keyword" },
    { query: "ca_pin sha256", label: "Technical identifier" },
  ];

  for (const q of ftsQueries) {
    try {
      const results = await backend.ftsSearch(q.query, {
        query: q.query,
        maxResults: 5,
        agentId: "test",
      });
      log("fts", q.label, results.length > 0, `${results.length} results`);
    } catch (err: any) {
      // FTS may legitimately fail if index isn't ready
      log("fts", q.label, false, err.message);
    }
  }

  await backend.close();
}

async function suiteSchemaEvolution(cfg: MemorySparkConfig) {
  if (!shouldRun("schema")) return;
  section("Suite: Schema Evolution (new columns on fresh table)");

  // Use a completely separate DB dir
  const schemaCfg = resolveConfig({
    ...cfg,
    lancedbDir: path.join(TEST_DIR, "schema-test-db"),
  });

  const backend = new LanceDBBackend(schemaCfg);
  await backend.open();

  const status = await backend.status();
  log("schema", "Fresh table created", status.chunkCount >= 0);

  // Check if the table has all expected columns
  try {
    // LanceDB doesn't have a direct schema inspect API, so we insert + query
    const embed2 = await createEmbedProvider(schemaCfg.embed);
    const vec = await embed2.embedQuery("schema test");
    await backend.upsert([{
      id: "schema-test-001",
      path: "test.md",
      source: "memory" as const,
      agent_id: "test",
      start_line: 1,
      end_line: 1,
      text: "Schema test document.",
      vector: vec,
      updated_at: new Date().toISOString(),
      category: "fact",
      entities: "[]",
      confidence: 0.9,
      content_type: "knowledge",
      quality_score: 1,
      token_count: 5,
      parent_heading: "Test Section",
    }]);

    const results = await backend.vectorSearch(vec, { query: "schema", maxResults: 1 });
    const row = results[0]?.chunk;
    log("schema", "content_type stored", row?.content_type === "knowledge");
    log("schema", "quality_score stored", (row as any)?.quality_score === 1);
    log("schema", "token_count stored", (row as any)?.token_count === 5);
    log("schema", "parent_heading stored", (row as any)?.parent_heading === "Test Section");
  } catch (err: any) {
    log("schema", "Schema columns", false, err.message);
  }

  await backend.close();
}

// ── Main ────────────────────────────────────────────────────────────────────
async function main() {
  console.log("╔══════════════════════════════════════════════════════════════╗");
  console.log("║        memory-spark Integration Test Suite                  ║");
  console.log("╠══════════════════════════════════════════════════════════════╣");
  console.log(`║  Spark: ${SPARK_HOST}    Token: ${SPARK_TOKEN.slice(0, 6)}...`);
  console.log(`║  DB:    ${TEST_DB}`);
  console.log(`║  Suite: ${SUITE_FILTER ?? "all"}    Verbose: ${VERBOSE}`);
  console.log("╚══════════════════════════════════════════════════════════════╝");

  fs.mkdirSync(TEST_DB, { recursive: true });

  const cfg = buildConfig();

  // 1. Check connectivity first — abort early if Spark is down
  await suiteConnectivity(cfg);
  const connectFails = results.filter((r) => r.suite === "connectivity" && r.status === "FAIL");
  if (connectFails.length > 0) {
    console.log("\n🛑 Spark connectivity failed — cannot run remaining tests.");
    printSummary();
    process.exit(1);
  }

  // 2. Create shared embed provider
  const embed = await createEmbedProvider(cfg.embed);

  // 3. Ingest test corpus
  await suiteIngestPipeline(cfg, embed);
  const ingestFails = results.filter((r) => r.suite === "ingest" && r.status === "FAIL");
  if (ingestFails.length > 0) {
    console.log("\n🛑 Ingest failed — cannot run search tests.");
    printSummary();
    process.exit(1);
  }

  // 4. Score distribution — THE critical test
  await suiteScoreDistribution(cfg, embed);

  // 5. Relevance accuracy
  await suiteRelevanceAccuracy(cfg, embed);

  // 6. Manager E2E — the exact code path the plugin uses
  await suiteManagerE2E(cfg);

  // 7. FTS fallback
  await suiteFTSFallback(cfg);

  // 8. Schema evolution
  await suiteSchemaEvolution(cfg);

  // Cleanup
  fs.rmSync(TEST_DIR, { recursive: true, force: true });

  printSummary();
}

function printSummary() {
  const passed = results.filter((r) => r.status === "PASS").length;
  const failed = results.filter((r) => r.status === "FAIL").length;
  const skipped = results.filter((r) => r.status === "SKIP").length;

  console.log("\n╔══════════════════════════════════════════════════════════════╗");
  console.log(`║  PASS: ${passed}  |  FAIL: ${failed}  |  SKIP: ${skipped}  |  TOTAL: ${results.length}`);
  console.log("╚══════════════════════════════════════════════════════════════╝");

  if (failed > 0) {
    console.log("\n❌ Failed tests:");
    for (const r of results.filter((r) => r.status === "FAIL")) {
      console.log(`   ${r.suite} > ${r.test}`);
      if (r.detail) console.log(`     ${r.detail}`);
    }
    console.log("\n🚫 DO NOT restart the gateway — fix these first.");
    process.exit(1);
  } else {
    console.log("\n✅ All integration tests passed — safe to restart + rebuild.");
    process.exit(0);
  }
}

main().catch((err) => {
  console.error("💥 Integration test suite crashed:", err);
  process.exit(1);
});
