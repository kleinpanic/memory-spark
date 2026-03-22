/**
 * memory-spark Benchmark Suite
 *
 * Performance and scalability validation for:
 * - Embedding latency and throughput
 * - Reranker latency
 * - Vector search latency at scale
 * - End-to-end recall latency
 * - Multi-user isolation
 *
 * Usage: npx tsx benchmark.ts
 */

import { createEmbedProvider } from "./src/embed/provider.js";
import { createReranker } from "./src/rerank/reranker.js";
import { LanceDBBackend } from "./src/storage/lancedb.js";
import { resolveConfig } from "./src/config.js";
import type { MemorySparkConfig } from "./src/config.js";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// ── Config ────────────────────────────────────────────────────────────────
const SPARK_HOST = process.env["SPARK_HOST"] ?? "10.x.x.x";
const TOKEN = (() => {
  if (process.env["SPARK_BEARER_TOKEN"]) return process.env["SPARK_BEARER_TOKEN"];
  try {
    const f = fs.readFileSync(path.join(os.homedir(), ".openclaw", ".env"), "utf-8");
    return f.match(/SPARK_BEARER_TOKEN=["']?([^"'\s\n]+)/)?.[1] ?? "none";
  } catch { return "none"; }
})();

const cfg: MemorySparkConfig = resolveConfig({
  lancedbDir: "/tmp/memory-spark-bench/lancedb",
  embed: {
    provider: "spark",
    spark: { baseUrl: `http://${SPARK_HOST}:18091/v1`, apiKey: TOKEN, model: "nvidia/llama-embed-nemotron-8b", dimensions: 4096 },
  },
  rerank: {
    enabled: true,
    spark: { baseUrl: `http://${SPARK_HOST}:18096/v1`, apiKey: TOKEN, model: "nvidia/llama-nemotron-rerank-1b-v2" },
  },
  spark: {
    embed: `http://${SPARK_HOST}:18091/v1`,
    rerank: `http://${SPARK_HOST}:18096/v1`,
    ocr: `http://${SPARK_HOST}:18097`,
    glmOcr: `http://${SPARK_HOST}:18080/v1`,
    ner: `http://${SPARK_HOST}:18112`,
    zeroShot: `http://${SPARK_HOST}:18113`,
    summarizer: `http://${SPARK_HOST}:18110`,
    stt: `http://${SPARK_HOST}:18094`,
  },
});

// ── Benchmark helpers ──────────────────────────────────────────────────────
interface BenchResult {
  name: string;
  iterations: number;
  totalMs: number;
  avgMs: number;
  minMs: number;
  maxMs: number;
  p50Ms: number;
  p95Ms: number;
  p99Ms: number;
  opsPerSec: number;
}

function formatMs(ms: number): string {
  return ms < 1000 ? `${ms.toFixed(1)}ms` : `${(ms / 1000).toFixed(2)}s`;
}

function printResult(r: BenchResult) {
  console.log(`  ${r.name}`);
  console.log(`    Iterations: ${r.iterations}`);
  console.log(`    Avg: ${formatMs(r.avgMs)} | Min: ${formatMs(r.minMs)} | Max: ${formatMs(r.maxMs)}`);
  console.log(`    P50: ${formatMs(r.p50Ms)} | P95: ${formatMs(r.p95Ms)} | P99: ${formatMs(r.p99Ms)}`);
  console.log(`    Throughput: ${r.opsPerSec.toFixed(1)} ops/sec`);
}

async function bench(name: string, fn: () => Promise<void>, iterations: number): Promise<BenchResult> {
  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await fn();
    times.push(performance.now() - start);
  }
  times.sort((a, b) => a - b);
  const totalMs = times.reduce((a, b) => a + b, 0);
  return {
    name,
    iterations,
    totalMs,
    avgMs: totalMs / iterations,
    minMs: times[0]!,
    maxMs: times[times.length - 1]!,
    p50Ms: times[Math.floor(iterations * 0.5)]!,
    p95Ms: times[Math.floor(iterations * 0.95)]!,
    p99Ms: times[Math.floor(iterations * 0.99)]!,
    opsPerSec: (iterations / totalMs) * 1000,
  };
}

// ── Benchmark suites ────────────────────────────────────────────────────────

async function benchEmbed(embed: Awaited<ReturnType<typeof createEmbedProvider>>): Promise<BenchResult[]> {
  console.log("\n=== Embedding Benchmarks ===");
  const results: BenchResult[] = [];

  // Single query latency
  results.push(await bench("Single embedQuery()", async () => {
    await embed.embedQuery("Klein prefers TypeScript for agent tools");
  }, 50));

  // Batch throughput (10 texts)
  const batch10 = Array(10).fill("This is a test document about memory systems and RAG architectures.");
  results.push(await bench("embedBatch(10)", async () => {
    await embed.embedBatch(batch10);
  }, 20));

  // Batch throughput (100 texts)
  const batch100 = Array(100).fill("This is a test document about memory systems and RAG architectures.");
  results.push(await bench("embedBatch(100)", async () => {
    await embed.embedBatch(batch100);
  }, 10));

  return results;
}

async function benchRerank(reranker: Awaited<ReturnType<typeof createReranker>>): Promise<BenchResult[]> {
  console.log("\n=== Reranker Benchmarks ===");
  const results: BenchResult[] = [];

  const candidates = Array(50).fill(null).map((_, i) => ({
    chunk: {
      id: `c${i}`,
      path: `test/${i}.md`,
      source: "memory" as const,
      agent_id: "bench",
      start_line: 1,
      end_line: 5,
      text: `Document ${i} about TypeScript, memory systems, and RAG.`,
      updated_at: new Date().toISOString(),
      vector: [],
    },
    score: Math.random(),
  }));

  results.push(await bench("Rerank 50 candidates", async () => {
    await reranker.rerank("TypeScript memory RAG", candidates, 10);
  }, 30));

  return results;
}

async function benchVectorSearch(backend: LanceDBBackend, embed: Awaited<ReturnType<typeof createEmbedProvider>>): Promise<BenchResult[]> {
  console.log("\n=== Vector Search Benchmarks ===");
  const results: BenchResult[] = [];

  // Seed with 10k chunks
  console.log("  Seeding 10k chunks...");
  const seedChunks = [];
  for (let i = 0; i < 10000; i++) {
    const vec = Array(4096).fill(0).map(() => Math.random() * 0.01 - 0.005);
    seedChunks.push({
      id: `bench-${i}`,
      path: `bench/doc${Math.floor(i / 100)}.md`,
      source: "memory" as const,
      agent_id: "bench",
      user_id: i < 5000 ? "klein" : "nicholas",
      start_line: (i % 100) * 10 + 1,
      end_line: (i % 100) * 10 + 10,
      text: `Document chunk ${i} about memory, RAG, embeddings, and retrieval.`,
      vector: vec,
      updated_at: new Date().toISOString(),
    });
  }
  await backend.upsert(seedChunks);
  console.log("  Seeded.");

  const queryVec = await embed.embedQuery("memory retrieval RAG embeddings");

  // Search latency at 10k scale
  results.push(await bench("Vector search (10k chunks)", async () => {
    await backend.vectorSearch(queryVec, { query: "memory retrieval", maxResults: 10 });
  }, 100));

  // Search with user_id filter
  results.push(await bench("Vector search + user_id filter", async () => {
    await backend.vectorSearch(queryVec, { query: "memory retrieval", maxResults: 10, userId: "klein" });
  }, 50));

  // FTS search
  results.push(await bench("FTS search (10k chunks)", async () => {
    await backend.ftsSearch("memory retrieval", { query: "memory retrieval", maxResults: 10 });
  }, 50));

  return results;
}

async function benchEndToEnd(
  embed: Awaited<ReturnType<typeof createEmbedProvider>>,
  reranker: Awaited<ReturnType<typeof createReranker>>,
  backend: LanceDBBackend,
): Promise<BenchResult[]> {
  console.log("\n=== End-to-End Recall Latency ===");
  const results: BenchResult[] = [];

  results.push(await bench("Full recall pipeline (embed + search + rerank)", async () => {
    const vec = await embed.embedQuery("TypeScript agent memory system");
    const candidates = await backend.vectorSearch(vec, { query: "TypeScript agent memory", maxResults: 20 });
    await reranker.rerank("TypeScript agent memory", candidates, 5);
  }, 30));

  return results;
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
  console.log("=== memory-spark Benchmark Suite ===");
  console.log(`Spark host: ${SPARK_HOST}`);
  console.log(`DB: ${cfg.lancedbDir}\n`);

  fs.rmSync("/tmp/memory-spark-bench", { recursive: true, force: true });
  fs.mkdirSync(cfg.lancedbDir, { recursive: true });

  const embed = await createEmbedProvider(cfg.embed);
  console.log(`Embed provider: ${embed.id} (${embed.dims}d)`);

  const reranker = await createReranker(cfg.rerank);
  console.log(`Reranker: ready`);

  const backend = new LanceDBBackend(cfg);
  await backend.open();
  console.log(`Backend: open\n`);

  const allResults: BenchResult[] = [];

  try {
    allResults.push(...await benchEmbed(embed));
    allResults.push(...await benchRerank(reranker));
    allResults.push(...await benchVectorSearch(backend, embed));
    allResults.push(...await benchEndToEnd(embed, reranker, backend));
  } finally {
    await backend.close();
  }

  // Summary
  console.log("\n════════════════════════════════════════════════════════════");
  console.log("=== Benchmark Summary ===\n");
  for (const r of allResults) {
    printResult(r);
    console.log();
  }

  // Export JSON
  const reportPath = "/tmp/memory-spark-bench/report.json";
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, JSON.stringify(allResults, null, 2));
  console.log(`Report saved to: ${reportPath}`);
}

main().catch((err) => {
  console.error("Benchmark failed:", err);
  process.exit(1);
});
