#!/usr/bin/env npx tsx
/**
 * BEIR Corpus Indexer — Index BEIR datasets to testDbBEIR for benchmarking.
 *
 * Usage:
 *   npx tsx evaluation/index-beir.ts                    # Index all datasets
 *   npx tsx evaluation/index-beir.ts --dataset scifact  # Index specific dataset
 *   npx tsx evaluation/index-beir.ts --resume           # Resume from checkpoint
 *
 * Features:
 *   - Hash-based checkpointing: skip unchanged docs
 *   - Resume from crash: load checkpoint, continue where left off
 *   - Batch embedding with Spark
 *   - Progress logging to file
 */

import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";

import { resolveConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { EmbedQueue } from "../src/embed/queue.js";
import type { MemoryChunk } from "../src/storage/backend.js";

// ── Types ───────────────────────────────────────────────────────────────────

interface BeirCorpusDoc {
  _id: string;
  title: string;
  text: string;
  metadata?: Record<string, unknown>;
}

interface Checkpoint {
  dataset: string;
  indexed: Record<string, { hash: string; chunks: number }>;
  updatedAt: string;
  completed: boolean;
}

// ── Config ──────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const datasetArg = args.includes("--dataset") ? args[args.indexOf("--dataset") + 1] : null;
const resume = args.includes("--resume");
const datasetsDir = path.join(import.meta.dirname!, "beir-datasets");
const checkpointDir = path.join(import.meta.dirname!, ".checkpoints");

const DATASETS = datasetArg ? [datasetArg] : ["scifact", "nfcorpus", "fiqa"];

const logger = {
  info: (m: string) => console.log(`[INFO] ${m}`),
  warn: (m: string) => console.warn(`[WARN] ${m}`),
  error: (m: string) => console.error(`[ERR]  ${m}`),
};

// ── Functions ───────────────────────────────────────────────────────────────

async function hashText(text: string): Promise<string> {
  return crypto.createHash("sha256").update(text).digest("hex").slice(0, 16);
}

async function loadCheckpoint(dataset: string): Promise<Checkpoint> {
  const file = path.join(checkpointDir, `${dataset}.json`);
  try {
    const data = await fs.readFile(file, "utf-8");
    return JSON.parse(data) as Checkpoint;
  } catch {
    return { dataset, indexed: {}, updatedAt: new Date().toISOString(), completed: false };
  }
}

async function saveCheckpoint(cp: Checkpoint): Promise<void> {
  await fs.mkdir(checkpointDir, { recursive: true });
  const file = path.join(checkpointDir, `${cp.dataset}.json`);
  await fs.writeFile(file, JSON.stringify(cp, null, 2));
}

async function loadCorpus(dataset: string): Promise<BeirCorpusDoc[]> {
  const file = path.join(datasetsDir, dataset, "corpus.jsonl");
  const content = await fs.readFile(file, "utf-8");
  return content
    .trim()
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as BeirCorpusDoc);
}

async function indexDataset(
  dataset: string,
  backend: LanceDBBackend,
  embed: EmbedQueue,
): Promise<{ indexed: number; skipped: number; failed: number }> {
  logger.info(`Loading ${dataset} corpus...`);
  const corpus = await loadCorpus(dataset);
  const total = corpus.length;

  // Load or create checkpoint
  let checkpoint = resume ? await loadCheckpoint(dataset) : { dataset, indexed: {}, updatedAt: new Date().toISOString(), completed: false };

  // If completed and resuming, skip
  if (checkpoint.completed) {
    logger.info(`${dataset} already completed, skipping`);
    return { indexed: 0, skipped: total, failed: 0 };
  }

  let indexed = 0;
  let skipped = 0;
  let failed = 0;
  const startTime = Date.now();
  const batchSize = 25;

  for (let i = 0; i < total; i += batchSize) {
    const batch = corpus.slice(i, i + batchSize);

    // Check which docs need indexing
    const toIndex: { doc: BeirCorpusDoc; hash: string }[] = [];
    for (const doc of batch) {
      const hash = await hashText(`${doc.title}\n${doc.text}`);
      const existing = checkpoint.indexed[doc._id];
      if (existing && existing.hash === hash) {
        skipped++;
      } else {
        toIndex.push({ doc, hash });
      }
    }

    if (toIndex.length === 0) continue;

    // Embed batch
    const texts = toIndex.map((d) => `${d.doc.title}\n${d.doc.text}`.slice(0, 2000));
    let vectors: number[][];

    try {
      vectors = await embed.embedBatch(texts);
    } catch (batchErr) {
      logger.warn(`Batch embed failed, falling back to sequential: ${batchErr}`);
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

    // Create chunks
    const chunks: MemoryChunk[] = [];
    for (let j = 0; j < toIndex.length; j++) {
      const { doc, hash } = toIndex[j]!;
      const vec = vectors[j];
      if (!vec || vec.length === 0) {
        failed++;
        continue;
      }

      chunks.push({
        id: `beir-${dataset}-${doc._id}`,
        path: `beir/${dataset}/${doc._id}`,
        source: "ingest",
        agent_id: "beir",
        start_line: 0,
        end_line: 0,
        text: `${doc.title}\n${doc.text}`,
        vector: vec,
        updated_at: new Date().toISOString(),
        category: "knowledge",
        content_type: "knowledge",
        pool: "reference_library", // BEIR data is reference, not auto-injected
      });

      checkpoint.indexed[doc._id] = { hash, chunks: 1 };
    }

    if (chunks.length > 0) {
      await backend.upsert(chunks);
      indexed += chunks.length;
    }

    // Save checkpoint every batch
    checkpoint.updatedAt = new Date().toISOString();
    await saveCheckpoint(checkpoint);

    // Progress
    const elapsed = (Date.now() - startTime) / 1000;
    const rate = indexed / elapsed || 1;
    const eta = Math.round((total - i - batchSize) / rate);
    process.stdout.write(
      `\r  [${dataset}] ${indexed}/${total} indexed, ${skipped} skipped, ${failed} failed — ETA: ${eta}s   `,
    );
  }

  // Mark completed
  checkpoint.completed = true;
  await saveCheckpoint(checkpoint);

  console.log("");
  logger.info(`${dataset} complete: ${indexed} indexed, ${skipped} skipped, ${failed} failed`);

  return { indexed, skipped, failed };
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
  console.log("═══════════════════════════════════════════");
  console.log("  BEIR Corpus Indexer");
  console.log("═══════════════════════════════════════════\n");

  // Config for testDbBEIR
  const lancedbDir = process.env.BEIR_LANCEDB_DIR || "/home/node/.openclaw/data/testDbBEIR/lancedb";
  logger.info(`Using lancedbDir: ${lancedbDir}`);

  const cfg = resolveConfig({ lancedbDir } as Parameters<typeof resolveConfig>[0]);

  // Initialize backend and embedding
  const backend = new LanceDBBackend(cfg);
  await backend.open();
  logger.info("Backend opened");

  const provider = await createEmbedProvider(cfg.embed);
  logger.info(`Embed provider: ${provider.id} / ${provider.model}`);

  const embed = new EmbedQueue(provider, { concurrency: 1, maxRetries: 2, timeoutMs: 30000 }, logger);

  // Index each dataset
  const results: Record<string, { indexed: number; skipped: number; failed: number }> = {};
  for (const dataset of DATASETS) {
    if (!["scifact", "nfcorpus", "fiqa"].includes(dataset)) {
      logger.warn(`Unknown dataset: ${dataset}, skipping`);
      continue;
    }
    results[dataset] = await indexDataset(dataset, backend, embed);
  }

  await backend.close();

  console.log("\n═══════════════════════════════════════════");
  console.log("  Summary");
  console.log("═══════════════════════════════════════════\n");

  let totalIndexed = 0;
  let totalSkipped = 0;
  let totalFailed = 0;
  for (const [ds, r] of Object.entries(results)) {
    console.log(`  ${ds}: ${r.indexed} indexed, ${r.skipped} skipped, ${r.failed} failed`);
    totalIndexed += r.indexed;
    totalSkipped += r.skipped;
    totalFailed += r.failed;
  }

  console.log(`\n  Total: ${totalIndexed} indexed, ${totalSkipped} skipped, ${totalFailed} failed`);
  console.log("\n✅ BEIR indexing complete");
}

main().catch((err) => {
  console.error("\n❌ FATAL:", err);
  process.exit(1);
});
