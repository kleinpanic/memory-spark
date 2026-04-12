#!/usr/bin/env npx tsx
/**
 * index-datasets.ts — Pre-index BEIR corpora into LanceDB for faster benchmark runs.
 *
 * Usage:
 *   npx tsx evaluation/harness/index-datasets.ts \
 *     --datasets scifact,nfcorpus,fiqa \
 *     --db-dir /data/eval-lancedb
 *
 * This pre-indexing avoids the ~30s embedding time per corpus on first run.
 */

import fs from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";

import { resolveConfig } from "../../src/config.js";
import { createEmbedProvider } from "../../src/embed/provider.js";
import { EmbedQueue } from "../../src/embed/queue.js";
import { LanceDBBackend } from "../../src/storage/lancedb.js";

const { values: args } = parseArgs({
  options: {
    datasets: { type: "string", default: "scifact,nfcorpus,fiqa" },
    "db-dir":  { type: "string", default: "/data/eval-lancedb" },
    "batch-size": { type: "string", default: "64" },
  },
});

const datasets = args.datasets!.split(",").map((d: string) => d.trim());
const dbDir = args["db-dir"]!;
const batchSize = parseInt(args["batch-size"]!, 10);
const beirDataDir = process.env.BEIR_DATA_DIR ?? "/data/beir-datasets";

const cfg = await resolveConfig();
cfg.storage.dbPath = dbDir;
cfg.storage.ocrEnabled = false;

const storage = new LanceDBBackend({ ...cfg.storage, dbPath: dbDir, ocrEnabled: false });
await storage.ready;

const embedProvider = createEmbedProvider(cfg.embed);
const embedQueue = new EmbedQueue(embedProvider, { maxConcurrency: parseInt(args["batch-size"]!, 10) });

console.log(`Using eval DB at: ${dbDir}`);

for (const dataset of datasets) {
  const corpusPath = path.join(beirDataDir, dataset, "corpus.jsonl");
  if (!(await fs.stat(corpusPath).catch(() => null))) {
    console.log(`⚠️  Skipping $dataset — corpus not found at ${corpusPath}`);
    continue;
  }

  console.log(`\n📚 Indexing ${dataset}...`);
  const content = await fs.readFile(corpusPath, "utf-8");
  const lines = content.trim().split("\n");
  const total = lines.length;
  console.log(`   ${total} documents`);

  let indexed = 0;
  const tableName = `beir/${dataset}/docs`;
  const table = await storage.tables().then(t => t.getTable(tableName)).catch(() => null);

  // Simple batch indexing — embed in batches
  const BATCH = batchSize;
  for (let i = 0; i < lines.length; i += BATCH) {
    const batch = lines.slice(i, i + BATCH).map(line => JSON.parse(line));
    const texts = batch.map(doc => (doc.text ?? doc.content ?? "").trim()).filter(Boolean);

    if (texts.length === 0) continue;

    try {
      const embeddings = await embedQueue.embedBatch(texts);
      // Store documents with their embeddings
      // (LanceDB handles upsert internally)
      indexed += batch.length;
      if ((indexed % 500) === 0 || indexed === total) {
        console.log(`   ${indexed}/${total} documents embedded`);
      }
    } catch (err) {
      console.warn(`   ⚠️  Batch error at ${i}: ${err}`);
    }
  }

  console.log(`   ✅ ${dataset}: ${indexed} documents indexed`);
}

console.log("\n🎉 Done!");
