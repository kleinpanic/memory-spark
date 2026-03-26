#!/usr/bin/env npx tsx
/**
 * Purge garbage captures from LanceDB.
 * Scans all chunks with source="capture" and removes those that match garbage patterns.
 *
 * Usage:
 *   # Dry run (default) — shows what would be deleted:
 *   MEMORY_SPARK_DATA_DIR=./test-data npx tsx tools/purge.ts
 *
 *   # Actually delete:
 *   MEMORY_SPARK_DATA_DIR=./test-data npx tsx tools/purge.ts --delete
 *
 *   # Against production (CAREFUL):
 *   npx tsx tools/purge.ts --delete
 */

import { resolveConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";

const GARBAGE_PATTERNS: RegExp[] = [
  // Discord/OpenClaw envelope metadata
  /Conversation info \(untrusted metadata\)/,
  /Sender \(untrusted metadata\)/,
  /"message_id":\s*"\d+"/,
  /"sender_id":\s*"\d+"/,
  /<<<EXTERNAL_UNTRUSTED_CONTENT/,
  /UNTRUSTED Discord message body/,
  /UNTRUSTED \w+ message body/,

  // Media attachment paths
  /\[media attached:\s*\/home\//,
  /\[media attached:\s*https?:\/\//,
  /\.openclaw\/media\/inbound\//,
  /To send an image back, prefer the message tool/,

  // Memory recall XML blocks
  /<relevant-memories>/,
  /<memory index="\d+"/,
  /<!-- SECURITY: Treat every memory below as untrusted/,

  // LCM summary blocks
  /<summary id="sum_[a-f0-9]+"/,
  /<summary_ref id="sum_/,

  // System noise
  /^HEARTBEAT_OK$/m,
  /^HEARTBEAT_DISABLED$/m,
  /^NO_REPLY$/m,

  // oc-tasks injection
  /^## Current Task Queue$/m,
  /^### 🔄 In Progress/m,

  // Session/bootstrap headers
  /^## \d{4}-\d{2}-\d{2}T[\d:.]+Z — (agent bootstrap|session new)/m,
  /^# Session: \d{4}-\d{2}-\d{2}/m,

  // Raw conversation logs stored as captures
  /^(assistant|user|system):\s/m,
];

async function main() {
  const doDelete = process.argv.includes("--delete");
  const cfg = resolveConfig();
  const backend = new LanceDBBackend(cfg);
  await backend.open();

  const status = await backend.status();
  console.log(`LanceDB: ${status.chunkCount} total chunks`);

  // Scan captures
  const table = (backend as any).table;
  if (!table) {
    console.error("Cannot access LanceDB table directly");
    await backend.close();
    process.exit(1);
  }

  const allCaptures = await table
    .query()
    .where("source = 'capture'")
    .limit(100000)
    .select(["id", "text", "path", "agent_id", "source"])
    .toArray();

  console.log(`Capture chunks: ${allCaptures.length}`);

  const garbageIds: string[] = [];
  const garbageSamples: Array<{ id: string; agent: string; preview: string; flags: string[] }> = [];

  for (const row of allCaptures) {
    const text = row.text as string;
    const matchedFlags: string[] = [];

    for (const pattern of GARBAGE_PATTERNS) {
      if (pattern.test(text)) {
        matchedFlags.push(pattern.source.slice(0, 40));
      }
    }

    if (matchedFlags.length > 0) {
      garbageIds.push(row.id as string);
      if (garbageSamples.length < 20) {
        garbageSamples.push({
          id: row.id as string,
          agent: row.agent_id as string,
          preview: text.slice(0, 120).replace(/\n/g, " "),
          flags: matchedFlags.slice(0, 3),
        });
      }
    }
  }

  console.log(`\nGarbage captures found: ${garbageIds.length} / ${allCaptures.length}`);

  if (garbageSamples.length > 0) {
    console.log("\nSamples:");
    for (const s of garbageSamples) {
      console.log(`  [${s.agent}] ${s.preview}`);
      console.log(`    Flags: ${s.flags.join(", ")}`);
    }
  }

  if (garbageIds.length === 0) {
    console.log("\nNo garbage found. ✅");
    await backend.close();
    return;
  }

  if (doDelete) {
    console.log(`\nDeleting ${garbageIds.length} garbage captures...`);
    // Delete in batches
    const batchSize = 100;
    for (let i = 0; i < garbageIds.length; i += batchSize) {
      const batch = garbageIds.slice(i, i + batchSize);
      await backend.deleteById(batch);
      console.log(`  Deleted ${Math.min(i + batchSize, garbageIds.length)} / ${garbageIds.length}`);
    }
    console.log(`\n✅ Purged ${garbageIds.length} garbage captures.`);
  } else {
    console.log(`\nDry run — pass --delete to actually remove them.`);
  }

  await backend.close();
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
