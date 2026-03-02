/**
 * Migration Script: memory-core SQLite-vec → memory-spark LanceDB
 *
 * Run via: node dist/scripts/migrate.js
 * Or automatically on first boot if config.migrate.autoMigrateOnFirstBoot = true.
 *
 * What it does:
 *   1. Discover all existing memory-core SQLite-vec DBs (~/.openclaw/memory/*.sqlite)
 *   2. For each agent DB:
 *      a. Read all chunk text (vectors are discarded — incompatible dims)
 *      b. Re-chunk with new chunker (consistent sizing)
 *      c. Re-embed with configured embed provider (Spark or fallback)
 *      d. Store in LanceDB under the agent's table
 *   3. Write migration completion status to statusFile
 *
 * Progress: writes JSON progress file every 100 chunks so it's resumable.
 * Idempotent: on resume, skips paths already present in LanceDB.
 *
 * This script is intentionally separate from the plugin runtime so it can
 * be run standalone, inspected, and re-run without affecting the live system.
 *
 * Estimated time (rough):
 *   ~1000 existing chunks × ~50ms/embed = ~50s for all agents at Spark speed.
 *   LanceDB write is negligible.
 */

import fs from "node:fs/promises";
import path from "node:path";

interface MigrationStatus {
  version: 1;
  startedAt: string;
  completedAt?: string;
  agents: Record<string, {
    status: "pending" | "in_progress" | "done" | "error";
    chunksRead: number;
    chunksWritten: number;
    error?: string;
  }>;
}

async function main() {
  console.log("memory-spark migration: starting");

  // TODO:
  // 1. Load config from ~/.openclaw/openclaw.json
  // 2. Initialize LanceDB backend
  // 3. Initialize embed provider
  // 4. Discover SQLite-vec DBs in sqliteVecDir
  // 5. For each DB: SqliteVecBackend.readAllForMigration() → re-chunk → re-embed → upsert
  // 6. Write status file on completion

  console.log("TODO: migration not yet implemented");
}

main().catch((err) => {
  console.error("migration failed:", err);
  process.exit(1);
});
