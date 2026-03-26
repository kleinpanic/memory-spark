/**
 * Migration script — imports existing memory-core data into LanceDB.
 * Called on first boot via gateway_start hook.
 *
 * This is a no-op now because the boot pass in watcher.ts handles
 * all file discovery and indexing. The migration is effectively
 * the boot pass itself — it discovers all workspace files and
 * session transcripts and indexes them with Spark embeddings.
 *
 * This file exists to prevent the dynamic import error in index.ts.
 */

export async function runMigration(): Promise<void> {
  // Migration is handled by the boot pass in watcher.ts
  // This file exists as a stub to prevent import errors
  console.log("memory-spark migration: delegated to boot pass (watcher auto-discovery)");
}

export default runMigration;
