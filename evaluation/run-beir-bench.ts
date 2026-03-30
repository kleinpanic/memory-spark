#!/usr/bin/env npx tsx
/**
 * BEIR Benchmark Runner — Test A/B/C/D/E/F/G configurations against BEIR.
 *
 * Configurations:
 *   A: Vector-Only        — Pure semantic search baseline
 *   B: FTS-Only           — Pure keyword search baseline
 *   C: Hybrid             — Vector + FTS combined (no reranker, no MMR)
 *   D: Hybrid + Reranker  — With Nemotron cross-encoder reranking
 *   E: Hybrid + MMR       — With Maximal Marginal Relevance diversity
 *   F: Hybrid + HyDE      — With Hypothetical Document Embeddings
 *   G: Full Pipeline      — All features: Vector + FTS + Reranker + MMR
 *
 * Usage:
 *   npx tsx evaluation/run-beir-bench.ts                    # All configs
 *   npx tsx evaluation/run-beir-bench.ts --dataset scifact  # Specific dataset
 *   npx tsx evaluation/run-beir-bench.ts --config A         # Specific config
 *
 * Output:
 *   - JSON results in evaluation/results/
 *   - Telemetry logs for each query showing stage contributions
 *   - Summary table with NDCG@10, MAP@10, Recall@10
 */

import fs from "node:fs/promises";
import path from "node:path";

import {
  hybridMerge,
  mmrRerank,
} from "../src/auto/recall.js";
import { resolveConfig, type HydeConfig } from "../src/config.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { EmbedQueue } from "../src/embed/queue.js";
import { generateHypotheticalDocument } from "../src/hyde/generator.js";
import { createReranker } from "../src/rerank/reranker.js";
import type { SearchResult } from "../src/storage/backend.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";

import { evaluateBEIR, type Qrels, type Results } from "./metrics.js";

// ── Types ───────────────────────────────────────────────────────────────────

interface BeirQuery {
  _id: string;
  text: string;
  metadata?: Record<string, unknown>;
}

interface RetrievalConfig {
  id: string;
  label: string;
  useVector: boolean;
  useFts: boolean;
  useReranker: boolean;
  useMmr: boolean;
  useHyde: boolean;
  mmrLambda: number;
  maxResults: number;
}

interface QueryTelemetry {
  queryId: string;
  queryText: string;
  config: string;
  stages: {
    vector?: { count: number; top1Id: string; top1Score: number };
    fts?: { count: number; top1Id: string; top1Score: number };
    hybrid?: { count: number; top1Id: string; top1Score: number };
    reranker?: { top1Id: string; top1Score: number; reorderCount: number };
    mmr?: { removedNearDuplicates: number };
    hyde?: { hypotheticalDoc: string };
  };
  finalResults: Array<{ id: string; score: number; text: string }>;
  latencyMs: number;
}

// ── A/B/C/D/E/F/G Configurations ─────────────────────────────────────────────

const CONFIGS: RetrievalConfig[] = [
  {
    id: "A",
    label: "Vector-Only",
    useVector: true,
    useFts: false,
    useReranker: false,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.7,
    maxResults: 10,
  },
  {
    id: "B",
    label: "FTS-Only",
    useVector: false,
    useFts: true,
    useReranker: false,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.7,
    maxResults: 10,
  },
  {
    id: "C",
    label: "Hybrid",
    useVector: true,
    useFts: true,
    useReranker: false,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.7,
    maxResults: 10,
  },
  {
    id: "D",
    label: "Hybrid + Reranker",
    useVector: true,
    useFts: true,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.7,
    maxResults: 10,
  },
  {
    id: "E",
    label: "Hybrid + MMR",
    useVector: true,
    useFts: true,
    useReranker: false,
    useMmr: true,
    useHyde: false,
    mmrLambda: 0.7,
    maxResults: 10,
  },
  {
    id: "F",
    label: "Hybrid + HyDE",
    useVector: true,
    useFts: true,
    useReranker: false,
    useMmr: false,
    useHyde: true,
    mmrLambda: 0.7,
    maxResults: 10,
  },
  {
    id: "G",
    label: "Full Pipeline",
    useVector: true,
    useFts: true,
    useReranker: true,
    useMmr: true,
    useHyde: false,
    mmrLambda: 0.7,
    maxResults: 10,
  },
];

// ── Functions ───────────────────────────────────────────────────────────────

async function loadQueries(dataset: string): Promise<BeirQuery[]> {
  const file = path.join(import.meta.dirname!, "beir-datasets", dataset, "queries.jsonl");
  const content = await fs.readFile(file, "utf-8");
  return content
    .trim()
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as BeirQuery);
}

async function loadQrels(dataset: string): Promise<Qrels> {
  const file = path.join(import.meta.dirname!, "beir-datasets", dataset, "qrels", "test.tsv");
  const content = await fs.readFile(file, "utf-8");
  const lines = content.trim().split("\n");
  const qrels: Qrels = {};

  for (const line of lines) {
    if (!line.trim() || line.startsWith("query-id")) continue;
    const parts = line.split("\t");
    if (parts.length < 3) continue;
    const [queryId, corpusId, score] = parts;
    if (!qrels[queryId!]) qrels[queryId!] = {};
    qrels[queryId!]![corpusId!] = parseInt(score!, 10);
  }
  return qrels;
}

function stripBeirPrefix(id: string): string {
  return id.replace(/^beir-(scifact|nfcorpus|fiqa)-/, "");
}

async function runRetrieval(
  queries: BeirQuery[],
  qrels: Qrels,
  backend: LanceDBBackend,
  embed: EmbedQueue,
  reranker: Awaited<ReturnType<typeof createReranker>> | null,
  hydeConfig: HydeConfig | undefined,
  config: RetrievalConfig,
): Promise<{ results: Results; telemetry: QueryTelemetry[] }> {
  const results: Results = {};
  const telemetry: QueryTelemetry[] = [];
  const k = config.maxResults;
  const evalQueries = queries.filter((q) => qrels[q._id] && Object.keys(qrels[q._id]!).length > 0);

  for (let i = 0; i < evalQueries.length; i++) {
    const q = evalQueries[i]!;
    const startTime = Date.now();
    const tel: QueryTelemetry = {
      queryId: q._id,
      queryText: q.text,
      config: config.id,
      stages: {},
      finalResults: [],
      latencyMs: 0,
    };

    if ((i + 1) % 25 === 0 || i === 0) {
      process.stdout.write(`\r    [${i + 1}/${evalQueries.length}]`);
    }

    // Get query vector
    let queryVector = await embed.embedQuery(q.text);

    // HyDE: generate hypothetical document
    if (config.useHyde && hydeConfig?.enabled) {
      try {
        const hypothetical = await generateHypotheticalDocument(q.text, hydeConfig);
        if (hypothetical) {
          tel.stages.hyde = { hypotheticalDoc: hypothetical.slice(0, 200) + "..." };
          const hydeVector = await embed.embedQuery(hypothetical);
          // Average query and HyDE vectors
          queryVector = queryVector.map((v, idx) => (v + hydeVector[idx]) / 2);
        }
      } catch {
        // Fall back to raw query
      }
    }

    let candidates: SearchResult[] = [];

    // Vector search
    if (config.useVector) {
      const vResults = await backend
        .vectorSearch(queryVector, { query: q.text, maxResults: k * 4, minScore: 0.0 })
        .catch(() => []);
      if (vResults.length > 0) {
        tel.stages.vector = {
          count: vResults.length,
          top1Id: vResults[0]!.chunk.id,
          top1Score: vResults[0]!.score,
        };
      }
      candidates.push(...vResults);
    }

    // FTS search
    if (config.useFts) {
      const fResults = await backend
        .ftsSearch(q.text, { query: q.text, maxResults: k * 4 })
        .catch(() => []);
      if (fResults.length > 0) {
        tel.stages.fts = {
          count: fResults.length,
          top1Id: fResults[0]!.chunk.id,
          top1Score: fResults[0]!.score,
        };
      }
      candidates.push(...fResults);
    }

    // Hybrid merge - combine vector and FTS results
    if (config.useVector && config.useFts && candidates.length > 0) {
      const vectorResults = candidates.filter((r) => r.score > 0 && r.chunk.id.startsWith("beir-"));
      const ftsResults = candidates.filter((r) => r.score > 0);
      candidates = hybridMerge(vectorResults, ftsResults, k * 2);
      if (candidates.length > 0) {
        tel.stages.hybrid = {
          count: candidates.length,
          top1Id: candidates[0]!.chunk.id,
          top1Score: candidates[0]!.score,
        };
      }
    } else {
      // Dedupe by ID
      const seen = new Set<string>();
      candidates = candidates.filter((r) => {
        if (seen.has(r.chunk.id)) return false;
        seen.add(r.chunk.id);
        return true;
      });
    }

    // Reranker
    if (config.useReranker && reranker && candidates.length > 0) {
      const beforeOrder = candidates.slice(0, 5).map((c) => c.chunk.id);
      candidates = await reranker.rerank(q.text, candidates, k);
      const afterOrder = candidates.slice(0, 5).map((c) => c.chunk.id);
      const reorderCount = beforeOrder.filter((id, idx) => id !== afterOrder[idx]).length;
      tel.stages.reranker = {
        top1Id: candidates[0]?.chunk.id ?? "",
        top1Score: candidates[0]?.score ?? 0,
        reorderCount,
      };
    }

    // MMR diversity
    if (config.useMmr && candidates.length > 0) {
      const beforeCount = candidates.length;
      candidates = mmrRerank(candidates, k, config.mmrLambda);
      tel.stages.mmr = { removedNearDuplicates: beforeCount - candidates.length };
    }

    // Record final results
    tel.finalResults = candidates.slice(0, k).map((r) => ({
      id: stripBeirPrefix(r.chunk.id),
      score: r.score,
      text: r.chunk.text.slice(0, 100) + "...",
    }));
    tel.latencyMs = Date.now() - startTime;
    telemetry.push(tel);

    // Convert results for BEIR metrics - Results is Record<queryId, Record<docId, score>>
    results[q._id] = {};
    for (const r of candidates.slice(0, k)) {
      results[q._id]![stripBeirPrefix(r.chunk.id)] = r.score;
    }
  }

  console.log("");
  return { results, telemetry };
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const datasetArg = args.includes("--dataset") ? args[args.indexOf("--dataset") + 1] : "scifact";
  const configArg = args.includes("--config") ? args[args.indexOf("--config") + 1] : null;
  const enableHyde = args.includes("--hyde");

  console.log("═══════════════════════════════════════════");
  console.log(`  BEIR Benchmark: ${datasetArg}`);
  console.log("═══════════════════════════════════════════\n");

  // Config for testDbBEIR
  const lancedbDir = process.env.BEIR_LANCEDB_DIR || "/home/node/.openclaw/data/testDbBEIR/lancedb";
  console.log(`[INFO] Using lancedbDir: ${lancedbDir}`);

  const cfg = resolveConfig({ lancedbDir } as Parameters<typeof resolveConfig>[0]);

  // Initialize
  const backend = new LanceDBBackend(cfg);
  await backend.open();

  const provider = await createEmbedProvider(cfg.embed);
  const embed = new EmbedQueue(provider, { concurrency: 1, maxRetries: 2, timeoutMs: 30000 });

  const reranker = cfg.rerank.enabled ? await createReranker(cfg.rerank) : null;
  console.log(`[INFO] Reranker: ${reranker ? "enabled" : "disabled"}`);

  // HyDE config
  const hydeConfig = enableHyde ? cfg.hyde : undefined;
  console.log(`[INFO] HyDE: ${hydeConfig?.enabled ? "enabled" : "disabled"}`);

  // Load dataset
  console.log(`[INFO] Loading ${datasetArg} queries and qrels...`);
  const queries = await loadQueries(datasetArg);
  const qrels = await loadQrels(datasetArg);
  console.log(`[INFO] ${queries.length} queries, ${Object.keys(qrels).length} with judgments`);

  // Filter configs
  const configs = configArg ? CONFIGS.filter((c) => c.id === configArg.toUpperCase()) : CONFIGS;

  // Results directory
  const resultsDir = path.join(import.meta.dirname!, "results");
  await fs.mkdir(resultsDir, { recursive: true });

  // Run benchmarks
  const allResults: { config: string; label: string; ndcg: number; map: number; recall: number }[] = [];
  const allTelemetry: QueryTelemetry[] = [];

  for (const config of configs) {
    console.log(`\n▶ Config ${config.id}: ${config.label}`);

    const { results, telemetry } = await runRetrieval(
      queries,
      qrels,
      backend,
      embed,
      reranker,
      hydeConfig,
      config,
    );

    const metrics = evaluateBEIR(qrels, results, [10]);
    console.log(`  NDCG@10: ${metrics.ndcg["@10"].toFixed(4)}`);
    console.log(`  MAP@10:  ${metrics.map["@10"].toFixed(4)}`);
    console.log(`  Recall@10: ${metrics.recall["@10"].toFixed(4)}`);

    allResults.push({
      config: config.id,
      label: config.label,
      ndcg: metrics.ndcg["@10"],
      map: metrics.map["@10"],
      recall: metrics.recall["@10"],
    });

    allTelemetry.push(...telemetry);

    // Save individual config results
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const resultsFile = path.join(resultsDir, `beir-${datasetArg}-${config.id}-${timestamp}.json`);
    await fs.writeFile(resultsFile, JSON.stringify({ config, metrics, results }, null, 2));
  }

  await backend.close();

  // Summary table
  console.log("\n═══════════════════════════════════════════");
  console.log("  Summary: BEIR " + datasetArg);
  console.log("═══════════════════════════════════════════\n");

  console.log("ID | Config               | NDCG@10 | MAP@10  | Recall@10");
  console.log("---|----------------------|---------|---------|----------");
  for (const r of allResults) {
    console.log(
      `${r.config}  | ${r.label.padEnd(20)} | ${r.ndcg.toFixed(4)}  | ${r.map.toFixed(4)}  | ${r.recall.toFixed(4)}`,
    );
  }

  // Save telemetry for audit
  const telemetryFile = path.join(resultsDir, `beir-${datasetArg}-telemetry-${Date.now()}.json`);
  await fs.writeFile(telemetryFile, JSON.stringify(allTelemetry, null, 2));
  console.log(`\n[INFO] Telemetry saved to: ${telemetryFile}`);

  // Summary file
  const summaryFile = path.join(resultsDir, `beir-${datasetArg}-summary-${Date.now()}.json`);
  await fs.writeFile(summaryFile, JSON.stringify({ dataset: datasetArg, results: allResults }, null, 2));

  console.log("\n✅ BEIR benchmark complete");
}

main().catch((err) => {
  console.error("\n❌ FATAL:", err);
  process.exit(1);
});
