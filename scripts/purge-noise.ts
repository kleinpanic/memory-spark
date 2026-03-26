#!/usr/bin/env npx tsx
/**
 * purge-noise.ts — Scan existing LanceDB index and remove noise chunks.
 *
 * Usage:
 *   npx tsx scripts/purge-noise.ts --dry-run     # Preview what would be removed
 *   npx tsx scripts/purge-noise.ts --execute      # Actually delete noise chunks
 *   npx tsx scripts/purge-noise.ts --stats        # Show quality distribution
 */

import * as lancedb from "@lancedb/lancedb";
import { scoreChunkQuality } from "../src/classify/quality.js";
import path from "node:path";
import os from "node:os";

const LANCEDB_DIR = path.join(os.homedir(), ".openclaw", "data", "memory-spark", "lancedb");
const TABLE_NAME = "memory_chunks";
const MIN_QUALITY = 0.3;
const BATCH_SIZE = 500;

type Mode = "dry-run" | "execute" | "stats";

async function main() {
  const mode: Mode = process.argv.includes("--execute") ? "execute"
    : process.argv.includes("--stats") ? "stats"
    : "dry-run";

  console.log(`\n🔍 memory-spark purge-noise (mode: ${mode})`);
  console.log(`   LanceDB: ${LANCEDB_DIR}`);
  console.log(`   Quality threshold: ${MIN_QUALITY}\n`);

  const db = await lancedb.connect(LANCEDB_DIR);
  const names = await db.tableNames();
  if (!names.includes(TABLE_NAME)) {
    console.log("❌ Table not found");
    process.exit(1);
  }

  const table = await db.openTable(TABLE_NAME);
  const totalRows = await table.countRows();
  console.log(`📊 Total chunks: ${totalRows}\n`);

  // Scan all chunks (LanceDB defaults to limit=10, must set explicitly)
  const rows = await table.query()
    .select(["id", "text", "path", "source", "agent_id"])
    .limit(totalRows + 1000)
    .toArray();
  console.log(`   Scanned: ${rows.length} chunks\n`);

  const toDelete: string[] = [];
  const flagCounts: Record<string, number> = {};
  const qualityBuckets: Record<string, number> = {
    "0.0-0.1": 0, "0.1-0.2": 0, "0.2-0.3": 0, "0.3-0.4": 0,
    "0.4-0.5": 0, "0.5-0.6": 0, "0.6-0.7": 0, "0.7-0.8": 0,
    "0.8-0.9": 0, "0.9-1.0": 0,
  };
  const sourceStats: Record<string, { total: number; noise: number }> = {};

  for (const row of rows) {
    const text = row.text as string;
    const filePath = row.path as string;
    const source = row.source as string;
    const id = row.id as string;

    const result = scoreChunkQuality(text, filePath, source);

    // Bucket
    const bucket = `${(Math.floor(result.score * 10) / 10).toFixed(1)}-${(Math.floor(result.score * 10) / 10 + 0.1).toFixed(1)}`;
    if (qualityBuckets[bucket] !== undefined) qualityBuckets[bucket]!++;

    // Source stats
    if (!sourceStats[source]) sourceStats[source] = { total: 0, noise: 0 };
    sourceStats[source]!.total++;

    // Flag counts
    for (const flag of result.flags) {
      flagCounts[flag] = (flagCounts[flag] ?? 0) + 1;
    }

    if (result.score < MIN_QUALITY) {
      toDelete.push(id);
      sourceStats[source]!.noise++;
    }
  }

  // Report
  console.log("📈 Quality Distribution:");
  for (const [bucket, count] of Object.entries(qualityBuckets)) {
    const bar = "█".repeat(Math.ceil(count / Math.max(1, totalRows) * 100));
    console.log(`  ${bucket}: ${count.toString().padStart(6)} ${bar}`);
  }

  console.log("\n🏷️  Noise Flags:");
  const sortedFlags = Object.entries(flagCounts).sort((a, b) => b[1] - a[1]);
  for (const [flag, count] of sortedFlags) {
    console.log(`  ${flag.padEnd(25)} ${count}`);
  }

  console.log("\n📁 By Source:");
  for (const [source, stats] of Object.entries(sourceStats)) {
    console.log(`  ${source.padEnd(15)} ${stats.total} total, ${stats.noise} noise (${(stats.noise / stats.total * 100).toFixed(1)}%)`);
  }

  console.log(`\n🗑️  Chunks to remove: ${toDelete.length} / ${totalRows} (${(toDelete.length / totalRows * 100).toFixed(1)}%)`);
  console.log(`   Chunks to keep:   ${totalRows - toDelete.length}`);

  if (mode === "stats" || mode === "dry-run") {
    if (toDelete.length > 0 && mode === "dry-run") {
      console.log("\n💡 Run with --execute to delete noise chunks");
      // Show a few samples
      console.log("\n📋 Sample noise chunks (first 5):");
      for (const id of toDelete.slice(0, 5)) {
        const row = rows.find((r) => r.id === id);
        if (row) {
          const preview = (row.text as string).slice(0, 100).replace(/\n/g, "\\n");
          console.log(`  [${row.path}] ${preview}...`);
        }
      }
    }
    process.exit(0);
  }

  // Execute mode — delete in batches
  console.log("\n⚡ Deleting noise chunks...");
  let deleted = 0;
  for (let i = 0; i < toDelete.length; i += BATCH_SIZE) {
    const batch = toDelete.slice(i, i + BATCH_SIZE);
    const inList = batch.map((id) => `'${id.replace(/'/g, "''")}'`).join(",");
    await table.delete(`id IN (${inList})`);
    deleted += batch.length;
    if (deleted % 2000 === 0 || deleted === toDelete.length) {
      console.log(`  ${deleted}/${toDelete.length} deleted`);
    }
  }

  const remaining = await table.countRows();
  console.log(`\n✅ Purge complete. ${remaining} chunks remaining (was ${totalRows}).`);

  // Optimize after purge
  console.log("🔧 Optimizing table...");
  await table.optimize();
  console.log("✅ Optimization complete.\n");
}

main().catch((err) => {
  console.error("❌ Error:", err);
  process.exit(1);
});
