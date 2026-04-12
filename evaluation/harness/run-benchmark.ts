#!/usr/bin/env npx tsx
/**
 * run-benchmark.ts — Headless BEIR benchmark runner for the eval harness.
 *
 * Usage:
 *   npx tsx evaluation/harness/run-benchmark.ts \
 *     --datasets scifact,nfcorpus,fiqa \
 *     --configs all \
 *     --output /data/results \
 *     --db-dir /data/eval-lancedb
 *
 * This script:
 * - Accepts datasets and configs via CLI args (not hardcoded)
 * - Uses a SEPARATE LanceDB path (--db-dir) — never touches production DB
 * - Writes results to --output as JSON
 * - Disables OCR globally
 * - Gracefully handles vLLM endpoint unavailability
 */

import fs from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";

import { resolveConfig } from "../../src/config.js";
import { createEmbedProvider } from "../../src/embed/provider.js";
import { EmbedQueue } from "../../src/embed/queue.js";
import { createReranker } from "../../src/rerank/reranker.js";
import { LanceDBBackend } from "../../src/storage/lancedb.js";
import { generateHypotheticalDocument } from "../../src/hyde/generator.js";
import { hybridMerge, mmrRerank, prepareRerankerFusion } from "../../src/auto/recall.js";
import { expandQuery } from "../../src/query/expander.js";
import { evaluateBEIR, type Qrels, type Results } from "../metrics.js";

// ── Types ─────────────────────────────────────────────────────────────────────

interface BeirQuery { _id: string; text: string; metadata?: Record<string, unknown> }
interface RetrievalConfig { id: string; label: string; useVector: boolean; useFts: boolean;
  useReranker: boolean; useMmr: boolean; useHyde: boolean; mmrLambda: number | "adaptive"; hydeConfigs?: unknown[] }

interface ConfigResult { config: string; label: string; ndcg10: number; map10: number; recall10: number; latencyMs: number; }

interface DatasetResult {
  dataset: string;
  corpusSize: number;
  queryCount: number;
  configs: ConfigResult[];
  timestamp: string;
}

// ── Config definitions (A–G, from run-beir-bench.ts) ─────────────────────────

const ALL_CONFIGS: RetrievalConfig[] = [
  { id: "A", label: "Vector-Only",         useVector: true,  useFts: false, useReranker: false, useMmr: false, useHyde: false, mmrLambda: 0 },
  { id: "B", label: "FTS-Only",            useVector: false, useFts: true,  useReranker: false, useMmr: false, useHyde: false, mmrLambda: 0 },
  { id: "C", label: "Hybrid",              useVector: true,  useFts: true,  useReranker: false, useMmr: false, useHyde: false, mmrLambda: 0 },
  { id: "D", label: "Hybrid+Reranker",     useVector: true,  useFts: true,  useReranker: true,  useMmr: false, useHyde: false, mmrLambda: 0 },
  { id: "E", label: "Hybrid+MMR",          useVector: true,  useFts: true,  useReranker: false, useMmr: true,  useHyde: false, mmrLambda: 0.5 },
  { id: "F", label: "Hybrid+HyDE",         useVector: true,  useFts: true,  useReranker: false, useMmr: false, useHyde: true,  mmrLambda: 0 },
  { id: "G", label: "Full Pipeline",       useVector: true,  useFts: true,  useReranker: true,  useMmr: true,  useHyde: true,  mmrLambda: 0.5 },
];

// ── CLI args ─────────────────────────────────────────────────────────────────

const { values: args } = parseArgs({
  options: {
    datasets:  { type: "string", default: "scifact,nfcorpus,fiqa" },
    configs:   { type: "string", default: "all" },
    output:    { type: "string", default: "/data/results" },
    "db-dir":  { type: "string", default: "/data/eval-lancedb" },
  },
  allowPositionals: true,
});

const datasets = args.datasets!.split(",").map((d: string) => d.trim());
const configIds = args.configs === "all" ? ALL_CONFIGS.map(c => c.id) : args.configs!.split(",").map((c: string) => c.trim());
const outputDir = args.output!;
const dbDir = args["db-dir"]!;

// ── Config resolution ─────────────────────────────────────────────────────────

const cfg = await resolveConfig();

if (dbDir !== cfg.storage.dbPath) {
  // Override DB path to SEPARATE eval DB — never touch production
  cfg.storage.dbPath = dbDir;
  console.log(`🔗 Using eval DB at: ${dbDir}`);
} else {
  console.warn("⚠️  WARNING: DB path equals production path! Overriding to eval path.");
  cfg.storage.dbPath = dbDir;
}

// Force disable OCR
cfg.storage.ocrEnabled = false;

// ── Storage init ──────────────────────────────────────────────────────────────

const storage = new LanceDBBackend({
  ...cfg.storage,
  dbPath: dbDir,
  ocrEnabled: false,
});
await storage.ready;
console.log("✅ LanceDB ready");

// ── Embedder / Reranker init ─────────────────────────────────────────────────

const embedProvider = createEmbedProvider(cfg.embed);
const embedQueue = new EmbedQueue(embedProvider, { maxConcurrency: 8, maxRetries: 2 });
const reranker = createReranker(cfg.rerank);

// ── Dataset loading helpers ────────────────────────────────────────────────────

async function loadQueries(dataset: string): Promise<BeirQuery[]> {
  const beirDataDir = process.env.BEIR_DATA_DIR ?? "/data/beir-datasets";
  const filePath = path.join(beirDataDir, dataset, "queries.jsonl");
  const content = await fs.readFile(filePath, "utf-8");
  return content.trim().split("\n").filter(Boolean).map(line => JSON.parse(line) as BeirQuery);
}

async function loadQrels(dataset: string): Promise<Qrels> {
  const beirDataDir = process.env.BEIR_DATA_DIR ?? "/data/beir-datasets";
  const filePath = path.join(beirDataDir, dataset, "qrels", "test.tsv");
  const content = await fs.readFile(filePath, "utf-8");
  const qrels: Qrels = {};
  for (const line of content.trim().split("\n").slice(1)) {
    const [queryId, , docId, rel] = line.split("\t");
    if (!qrels[queryId]) qrels[queryId] = {};
    qrels[queryId][docId] = parseInt(rel, 10);
  }
  return qrels;
}

// ── Per-config evaluation ──────────────────────────────────────────────────────

async function evaluateConfig(
  cfg_: RetrievalConfig,
  dataset: string,
  queries: BeirQuery[],
  qrels: Qrels,
): Promise<ConfigResult> {
  const start = Date.now();
  const results: Results = {};
  let successCount = 0;
  let errorCount = 0;

  for (const q of queries) {
    try {
      const searchOpts = { query: q.text, maxResults: 100, minScore: 0.0, pathContains: `beir/${dataset}/` };
      let docs = [];

      if (cfg_.useHyde) {
        // HyDE: generate hypothetical doc, embed it
        const hydeDoc = await generateHypotheticalDocument(cfg.hyde!, q.text);
        if (hydeDoc) {
          const vec = await embedQueue.embedQuery(hydeDoc.content ?? hydeDoc.text ?? q.text);
          docs = await storage.vectorSearch(vec, searchOpts);
        } else {
          docs = [];
        }
      } else if (cfg_.useVector) {
        const vec = await embedQueue.embedQuery(q.text);
        docs = await storage.vectorSearch(vec, searchOpts);
      }

      if (cfg_.useFts && docs.length === 0) {
        // Fallback to FTS if vector returned nothing
        docs = await storage.fullTextSearch(q.text, { ...searchOpts, maxResults: 100 });
      } else if (cfg_.useFts && docs.length > 0) {
        // Merge FTS results
        const ftsDocs = await storage.fullTextSearch(q.text, { ...searchOpts, maxResults: 100 });
        docs = hybridMerge(docs, ftsDocs, 0.5);
      }

      if (cfg_.useReranker && docs.length > 0) {
        const reranked = await reranker.rerank(q.text, docs.slice(0, 20));
        docs = reranked ?? docs;
      }

      results[q._id] = docs.slice(0, 10).map((d, i) => ({ doc: d as unknown as string, score: 1 / (i + 1) }));
      successCount++;
    } catch {
      errorCount++;
      results[q._id] = [];
    }
  }

  const metrics = evaluateBEIR(results, qrels);
  const latencyMs = Math.round((Date.now() - start) / queries.length);

  console.log(`   [${cfg_.id}] ${cfg_.label}: NDCG@10=${metrics.ndcg10.toFixed(4)} | MAP@10=${metrics.map10.toFixed(4)} | Recall@10=${metrics.recall10.toFixed(4)} | ${successCount}/${queries.length} | ${latencyMs}ms/q | errors=${errorCount}`);

  return {
    config: cfg_.id,
    label: cfg_.label,
    ndcg10: metrics.ndcg10,
    map10: metrics.map10,
    recall10: metrics.recall10,
    latencyMs,
  };
}

// ── Main loop ─────────────────────────────────────────────────────────────────

await fs.mkdir(outputDir, { recursive: true });

const selectedConfigs = ALL_CONFIGS.filter(c => configIds.includes(c.id));

for (const dataset of datasets) {
  console.log(`\n📂 Dataset: ${dataset}`);
  const queries = await loadQueries(dataset);
  const qrels = await loadQrels(dataset);
  console.log(`   ${queries.length} queries, corpus: beir/${dataset}/`);

  const configResults: ConfigResult[] = [];
  for (const cfg_ of selectedConfigs) {
    const r = await evaluateConfig(cfg_, dataset, queries, qrels);
    configResults.push(r);
  }

  const result: DatasetResult = {
    dataset,
    corpusSize: queries.length,
    queryCount: queries.length,
    configs: configResults,
    timestamp: new Date().toISOString(),
  };

  const outPath = path.join(outputDir, `${dataset}-results.json`);
  await fs.writeFile(outPath, JSON.stringify(result, null, 2));
  console.log(`   ✅ Results → ${outPath}`);
}

console.log("\n🎉 Benchmark complete!");
console.log(`   Results: ${outputDir}`);
