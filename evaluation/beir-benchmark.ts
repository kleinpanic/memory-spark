#!/usr/bin/env npx tsx
/**
 * BEIR Standard Benchmark Runner
 *
 * Downloads and evaluates against official BEIR datasets (SciFact, NFCorpus, etc.)
 * to compare our retrieval pipeline against published baselines.
 *
 * Pipeline: Ingest corpus → Embed all docs → Index in LanceDB → Run queries → BEIR metrics
 *
 * Usage:
 *   npx tsx evaluation/beir-benchmark.ts                          # SciFact (default)
 *   npx tsx evaluation/beir-benchmark.ts --dataset nfcorpus       # NFCorpus
 *   npx tsx evaluation/beir-benchmark.ts --skip-index              # Reuse existing index
 *   npx tsx evaluation/beir-benchmark.ts --batch-size 32           # Embed batch size
 */

import fs from "node:fs/promises";
import path from "node:path";

import {
  hybridMerge,
  applySourceWeighting,
  applyTemporalDecay,
  mmrRerank,
} from "../src/auto/recall.js";
import { resolveConfig } from "../src/config.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { EmbedQueue } from "../src/embed/queue.js";
import { createReranker } from "../src/rerank/reranker.js";
import type { SearchResult, MemoryChunk } from "../src/storage/backend.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";

import { evaluateBEIR, formatBEIRResults, type Qrels, type Results } from "./metrics.js";

// ── BEIR Dataset Loading ────────────────────────────────────────────────────

interface BeirCorpusDoc {
  _id: string;
  title: string;
  text: string;
  metadata?: Record<string, unknown>;
}

interface BeirQuery {
  _id: string;
  text: string;
  metadata?: Record<string, unknown>;
}

async function loadJsonl<T>(filePath: string): Promise<T[]> {
  const content = await fs.readFile(filePath, "utf-8");
  return content
    .trim()
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as T);
}

async function loadQrels(filePath: string): Promise<Qrels> {
  const content = await fs.readFile(filePath, "utf-8");
  const lines = content.trim().split("\n");
  const qrels: Qrels = {};

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!.trim();
    if (!line || line.startsWith("query-id")) continue; // skip header
    const parts = line.split("\t");
    if (parts.length < 3) continue;
    const [queryId, corpusId, score] = parts;
    if (!qrels[queryId!]) qrels[queryId!] = {};
    qrels[queryId!]![corpusId!] = parseInt(score!, 10);
  }
  return qrels;
}

// ── Indexing ────────────────────────────────────────────────────────────────

async function indexCorpus(
  corpus: BeirCorpusDoc[],
  backend: LanceDBBackend,
  embed: EmbedQueue,
  batchSize: number,
): Promise<void> {
  const total = corpus.length;
  let indexed = 0;
  let failed = 0;
  const startTime = Date.now();

  for (let i = 0; i < total; i += batchSize) {
    const batch = corpus.slice(i, i + batchSize);
    const texts = batch.map((d) => `${d.title}\n${d.text}`.slice(0, 2000));

    let vectors: number[][];
    try {
      vectors = await embed.embedBatch(texts);
    } catch (batchErr) {
      // Fallback to sequential if batch fails
      console.warn(`[WARN] Batch embed failed, falling back to sequential:`, batchErr);
      vectors = [];
      for (const t of texts) {
        try {
          vectors.push(await embed.embedQuery(t));
        } catch {
          vectors.push([]);
          failed++;
        }
      }
    }

    const chunks: MemoryChunk[] = [];
    for (let j = 0; j < batch.length; j++) {
      const doc = batch[j]!;
      const vec = vectors[j];
      if (!vec || vec.length === 0) continue;
      chunks.push({
        id: doc._id,
        path: `beir/scifact/${doc._id}`,
        source: "ingest",
        agent_id: "beir",
        start_line: 0,
        end_line: 0,
        text: `${doc.title}\n${doc.text}`,
        vector: vec,
        updated_at: new Date().toISOString(),
        category: "knowledge",
        content_type: "knowledge",
        pool: "agent_memory",
      });
    }

    if (chunks.length > 0) {
      await backend.upsert(chunks);
    }

    indexed += chunks.length;
    const elapsed = (Date.now() - startTime) / 1000;
    const rate = indexed / elapsed;
    const eta = ((total - indexed) / rate).toFixed(0);
    process.stdout.write(
      `\r  Indexed: ${indexed}/${total} (${failed} failed) — ${rate.toFixed(1)} docs/s, ETA: ${eta}s`,
    );
  }
  console.log("");
}

// ── Retrieval ───────────────────────────────────────────────────────────────

interface RetrievalOpts {
  useVector: boolean;
  useFts: boolean;
  useReranker: boolean;
  useSourceWeight: boolean;
  useTemporalDecay: boolean;
  useMmr: boolean;
  maxResults: number;
}

const FULL_PIPELINE: RetrievalOpts = {
  useVector: true,
  useFts: true,
  useReranker: true,
  useSourceWeight: false, // disabled for BEIR — no source metadata on external corpus
  useTemporalDecay: false, // disabled for BEIR — no temporal metadata on external corpus
  useMmr: true,
  maxResults: 10,
};

async function runRetrieval(
  queries: BeirQuery[],
  qrels: Qrels,
  backend: LanceDBBackend,
  embed: EmbedQueue,
  reranker: Awaited<ReturnType<typeof createReranker>> | null,
  opts: RetrievalOpts,
): Promise<Results> {
  const results: Results = {};
  const k = opts.maxResults;
  // Only eval queries that have relevance judgments
  const evalQueries = queries.filter((q) => qrels[q._id] && Object.keys(qrels[q._id]!).length > 0);

  for (let i = 0; i < evalQueries.length; i++) {
    const q = evalQueries[i]!;
    if ((i + 1) % 25 === 0 || i === 0) {
      process.stdout.write(`\r    [${i + 1}/${evalQueries.length}]`);
    }

    const queryVector = await embed.embedQuery(q.text);
    let candidates: SearchResult[] = [];

    if (opts.useVector) {
      const vResults = await backend
        .vectorSearch(queryVector, { query: q.text, maxResults: k * 4, minScore: 0.0 })
        .catch(() => []);
      candidates.push(...vResults);
    }

    if (opts.useFts) {
      const fResults = await backend
        .ftsSearch(q.text, { query: q.text, maxResults: k * 4 })
        .catch(() => []);
      if (candidates.length > 0 && fResults.length > 0) {
        candidates = hybridMerge(candidates, fResults, k * 4);
      } else if (fResults.length > 0) {
        candidates = fResults;
      }
    }

    if (opts.useSourceWeight) applySourceWeighting(candidates);
    if (opts.useTemporalDecay) applyTemporalDecay(candidates);
    if (opts.useMmr) candidates = mmrRerank(candidates, k * 2, 0.7);

    if (opts.useReranker && reranker) {
      try {
        candidates = await reranker.rerank(q.text, candidates.slice(0, 20), k);
      } catch {
        candidates = candidates.slice(0, k);
      }
    } else {
      candidates = candidates.slice(0, k);
    }

    // Map back to corpus IDs — for BEIR, chunk.id === corpus doc _id
    const queryResults: Record<string, number> = {};
    for (const r of candidates) {
      const docId = r.chunk.id;
      queryResults[docId] = Math.max(queryResults[docId] ?? 0, r.score);
    }
    results[q._id] = queryResults;
  }

  process.stdout.write("\r" + " ".repeat(40) + "\r");
  return results;
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const datasetName = args.includes("--dataset") ? args[args.indexOf("--dataset") + 1]! : "scifact";
  const skipIndex = args.includes("--skip-index");
  const batchSize = args.includes("--batch-size")
    ? parseInt(args[args.indexOf("--batch-size") + 1]!, 10)
    : 16;
  const quick = args.includes("--quick");

  const datasetDir = path.join(import.meta.dirname!, "beir-datasets", datasetName);
  const indexDir = path.join(import.meta.dirname!, "beir-datasets", `${datasetName}-index`);

  console.log("═══════════════════════════════════════════════════");
  console.log(`  BEIR Benchmark: ${datasetName}`);
  console.log("═══════════════════════════════════════════════════");

  // Load dataset
  console.log("\n▶ Loading dataset...");
  const corpus = await loadJsonl<BeirCorpusDoc>(path.join(datasetDir, "corpus.jsonl"));
  const queries = await loadJsonl<BeirQuery>(path.join(datasetDir, "queries.jsonl"));
  const qrels = await loadQrels(path.join(datasetDir, "qrels", "test.tsv"));
  const evalQueryCount = Object.keys(qrels).length;
  console.log(
    `  Corpus: ${corpus.length} docs | Queries: ${queries.length} | Eval queries (with qrels): ${evalQueryCount}`,
  );

  // Setup
  const cfg = resolveConfig({ lancedbDir: indexDir } as Parameters<typeof resolveConfig>[0]);
  const backend = new LanceDBBackend(cfg);
  await backend.open();
  const provider = await createEmbedProvider(cfg.embed);
  const embed = new EmbedQueue(provider, { concurrency: 1, maxRetries: 3, timeoutMs: 60000 });
  const reranker = cfg.rerank.enabled && !quick ? await createReranker(cfg.rerank) : null;

  // Index
  if (!skipIndex) {
    const status = await backend.status();
    if (status.chunkCount >= corpus.length * 0.9) {
      console.log(`\n▶ Index already has ${status.chunkCount} chunks (≥90% of corpus). Skipping.`);
    } else {
      console.log(`\n▶ Indexing ${corpus.length} docs (batch size: ${batchSize})...`);
      const t0 = Date.now();
      await indexCorpus(corpus, backend, embed, batchSize);
      console.log(`  Done in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    }
  }

  const status = await backend.status();
  console.log(`\n  Index: ${status.chunkCount} chunks | Reranker: ${reranker ? "on" : "off"}`);

  // Run ablations
  const allResults: Record<string, ReturnType<typeof evaluateBEIR>> = {};

  const runConfig = async (name: string, label: string, opts: Partial<RetrievalOpts>) => {
    const t0 = Date.now();
    process.stdout.write(`\n  ${label}...`);
    const fullOpts = { ...FULL_PIPELINE, ...opts };
    const results = await runRetrieval(queries, qrels, backend, embed, reranker, fullOpts);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    allResults[name] = evaluateBEIR(qrels, results);
    console.log(` (${elapsed}s)`);
    console.log(formatBEIRResults(allResults[name]!));
  };

  console.log("\n📊 BEIR Evaluation\n");

  // Baselines
  await runConfig("vector_only", "Vector-Only", {
    useFts: false,
    useReranker: false,
    useMmr: false,
  });
  await runConfig("fts_only", "FTS-Only", { useVector: false, useReranker: false, useMmr: false });
  await runConfig("hybrid", "Hybrid (Vector + FTS)", { useReranker: false });
  await runConfig("hybrid_no_mmr", "Hybrid − MMR", { useReranker: false, useMmr: false });

  if (reranker) {
    await runConfig("full_pipeline", "Full Pipeline (with Reranker)", {});
  }

  // Summary comparison
  console.log("\n═══════════════════════════════════════════════════");
  console.log("  Summary (NDCG@10)");
  console.log("═══════════════════════════════════════════════════\n");
  for (const [name, metrics] of Object.entries(allResults)) {
    console.log(`  ${name.padEnd(25)} NDCG@10=${(metrics.ndcg["@10"] ?? 0).toFixed(4)}`);
  }

  // Published baselines for reference
  console.log("\n  ── Published Baselines (SciFact) ──");
  console.log("  BM25 (Anserini)            NDCG@10=0.6647");
  console.log("  DPR (NQ-trained)           NDCG@10=0.3183");
  console.log("  ANCE (MS MARCO)            NDCG@10=0.5070");
  console.log("  TAS-B                      NDCG@10=0.6434");
  console.log("  ColBERT v2                 NDCG@10=0.6710");

  // Save results
  const outputDir = path.join(import.meta.dirname!, "results");
  await fs.mkdir(outputDir, { recursive: true });
  const ts = new Date().toISOString().slice(0, 19).replace(/:/g, "-");
  const outputPath = path.join(outputDir, `beir-${datasetName}-${ts}.json`);
  await fs.writeFile(
    outputPath,
    JSON.stringify(
      {
        dataset: datasetName,
        timestamp: new Date().toISOString(),
        corpusSize: corpus.length,
        queryCount: queries.length,
        evalQueryCount,
        indexChunks: status.chunkCount,
        rerankerEnabled: !!reranker,
        results: allResults,
      },
      null,
      2,
    ),
  );
  console.log(`\n📄 Results: ${outputPath}`);

  await backend.close();
  return allResults;
}

/**
 * Multi-dataset runner: iterates over all available BEIR datasets in beir-datasets/
 * and produces a summary table.
 *
 * Usage: npx tsx evaluation/beir-benchmark.ts --multi [--quick]
 */
async function runMulti() {
  const args = process.argv.slice(2);
  const quick = args.includes("--quick");
  const datasetsDir = path.join(import.meta.dirname!, "beir-datasets");
  const entries = await fs.readdir(datasetsDir, { withFileTypes: true });
  const datasets = entries
    .filter((e) => e.isDirectory() && !e.name.endsWith("-index"))
    .map((e) => e.name)
    .sort();

  if (datasets.length === 0) {
    console.log("No BEIR datasets found. Run: bash evaluation/scripts/download-beir.sh --small");
    return;
  }

  console.log("═══════════════════════════════════════════════════");
  console.log(`  BEIR Multi-Dataset Benchmark (${datasets.length} datasets)`);
  console.log("═══════════════════════════════════════════════════\n");

  const summary: Record<string, Record<string, { ndcg10: number; mrr10: number }>> = {};

  for (const ds of datasets) {
    const corpusPath = path.join(datasetsDir, ds, "corpus.jsonl");
    try {
      await fs.access(corpusPath);
    } catch {
      console.log(`⏭️  Skipping ${ds} (no corpus.jsonl)`);
      continue;
    }

    // Re-run main() logic for each dataset by manipulating argv
    const origArgv = process.argv;
    process.argv = [origArgv[0]!, origArgv[1]!, "--dataset", ds, ...(quick ? ["--quick"] : [])];
    try {
      const results = await main();
      if (results) {
        summary[ds] = {};
        for (const [config, metrics] of Object.entries(results)) {
          summary[ds]![config] = {
            ndcg10: (metrics as Record<string, Record<string, number>>)["10"]?.ndcg ?? 0,
            mrr10: (metrics as Record<string, Record<string, number>>)["10"]?.mrr ?? 0,
          };
        }
      }
    } catch (err) {
      console.error(`❌ ${ds} failed:`, err);
    }
    process.argv = origArgv;
  }

  // Print summary table
  console.log("\n═══════════════════════════════════════════════════");
  console.log("  BEIR Cross-Dataset Summary");
  console.log("═══════════════════════════════════════════════════\n");

  // Collect all config names
  const allConfigs = new Set<string>();
  for (const dsResults of Object.values(summary)) {
    for (const config of Object.keys(dsResults)) allConfigs.add(config);
  }

  console.log(`  ${"Config".padEnd(30)}  ${datasets.map((d) => d.padStart(12)).join("  ")}`);
  console.log("  " + "─".repeat(30 + datasets.length * 14));

  for (const config of allConfigs) {
    const values = datasets.map((ds) => {
      const v = summary[ds]?.[config]?.ndcg10;
      return v !== undefined ? v.toFixed(4).padStart(12) : "     N/A    ";
    });
    console.log(`  ${config.padEnd(30)}  ${values.join("  ")}`);
  }
  console.log();
}

if (process.argv.includes("--multi")) {
  runMulti().catch((err) => {
    console.error("FATAL:", err);
    process.exit(1);
  });
} else {
  main().catch((err) => {
    console.error("FATAL:", err);
    process.exit(1);
  });
}
