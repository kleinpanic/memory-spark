#!/usr/bin/env npx tsx
/**
 * Rebuild LanceDB table — drops existing table and lets next boot pass
 * create a fresh one with the correct schema (including new columns).
 *
 * The boot pass re-indexes all workspace files from disk, so no data is lost.
 * New embeddings will use contextual prefixes (Anthropic technique).
 *
 * Usage: npx tsx scripts/rebuild-table.ts [--confirm]
 */

import * as lancedb from "@lancedb/lancedb";
import path from "node:path";
import os from "node:os";
import fs from "node:fs/promises";

const DB_PATH = path.join(os.homedir(), ".openclaw", "data", "memory-spark", "lancedb");
const TABLE_NAME = "memory_chunks";

async function main() {
  const confirm = process.argv.includes("--confirm");

  const db = await lancedb.connect(DB_PATH);
  const names = await db.tableNames();

  if (!names.includes(TABLE_NAME)) {
    console.log("No table found — nothing to rebuild.");
    return;
  }

  const table = await db.openTable(TABLE_NAME);
  const count = await table.countRows();
  const schema = await table.schema();

  console.log(`Table: ${TABLE_NAME}`);
  console.log(`Rows: ${count}`);
  console.log(
    `Columns: ${schema.fields.map((f) => `${f.name}(${f.nullable ? "null" : "notnull"})`).join(", ")}`,
  );
  console.log(`DB path: ${DB_PATH}`);

  if (!confirm) {
    console.log("\nDry run — add --confirm to actually drop the table.");
    console.log(
      "After dropping, restart the gateway and the boot pass will rebuild from disk files.",
    );
    table.close();
    return;
  }

  // Close the table handle before dropping
  table.close();

  // Drop the table
  await db.dropTable(TABLE_NAME);
  console.log(`\n✅ Table "${TABLE_NAME}" dropped.`);
  console.log("Restart the gateway to trigger a fresh boot pass with the new schema.");
  console.log("All workspace files will be re-indexed with contextual embeddings.");

  // Also reset the dims-lock so it re-detects on first embed
  const dimsLock = path.join(DB_PATH, "..", "dims-lock.json");
  try {
    await fs.unlink(dimsLock);
    console.log("dims-lock.json removed (will re-detect on first embed).");
  } catch {
    /* doesn't exist */
  }
}

main().catch(console.error);
