#!/usr/bin/env npx tsx
/**
 * Curated Benchmark Reindex — indexes ONLY the core agents for evaluation.
 *
 * Usage:
 *   MEMORY_SPARK_DATA_DIR=./test-data npx tsx scripts/reindex-benchmark.ts
 *
 * Features:
 *   - Hash-based checkpointing: skip unchanged files, re-index modified
 *   - Resume from crash: load checkpoint, continue where left off
 *   - Progress tracking: save every 100 files
 *
 * Core agents: meta, main, dev, school, ghost, recovery, research, taskmaster
 * Plus: shared reference library
 */

import path from "node:path";
import fs from "node:fs/promises";
import os from "node:os";
import crypto from "node:crypto";

import { resolveConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { EmbedQueue } from "../src/embed/queue.js";
import { ingestFile } from "../src/ingest/pipeline.js";
import {
  discoverWorkspaceFiles,
  walkSupportedFiles,
  toRelativePath,
} from "../src/ingest/workspace.js";

/** Only these agents are indexed for benchmarking */
const CORE_AGENTS = [
  "meta",
  "main",
  "dev",
  "school",    // Cortex
  "ghost",
  "recovery",
  "research",
  "taskmaster",
  "immune",    // Security/audit agent — has MISTAKES.md, TOOLS.md, distinct role
];

/** Save checkpoint every N files */
const CHECKPOINT_INTERVAL = 100;

/** Checkpoint file name */
const CHECKPOINT_FILE = "checkpoint.json";

const logger = {
  info: (m: string) => console.log(`[INFO] ${m}`),
  warn: (m: string) => console.warn(`[WARN] ${m}`),
  error: (m: string) => console.error(`[ERR]  ${m}`),
};

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoint Types
// ─────────────────────────────────────────────────────────────────────────────

interface FileEntry {
  /** SHA-256 hash of file content (first 16 chars) */
  hash: string;
  /** File modification time (seconds since epoch) */
  mtime: number;
  /** File size in bytes */
  size: number;
  /** Number of chunks added */
  chunksAdded: number;
}

interface Checkpoint {
  /** Map of absolute path → file entry */
  indexed: Record<string, FileEntry>;
  /** When checkpoint was last updated */
  updatedAt: string;
  /** Total files in queue */
  total: number;
  /** True when all files processed successfully */
  completed: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoint Functions
// ─────────────────────────────────────────────────────────────────────────────

/** Compute SHA-256 hash of file content (first 16 chars for brevity) */
async function hashFile(absPath: string): Promise<string> {
  const content = await fs.readFile(absPath);
  return crypto.createHash("sha256").update(content).digest("hex").slice(0, 16);
}

/** Get file stats: mtime, size */
async function getFileStats(absPath: string): Promise<{ mtime: number; size: number }> {
  const stat = await fs.stat(absPath);
  return {
    mtime: Math.floor(stat.mtimeMs / 1000),
    size: stat.size,
  };
}

/** Load checkpoint from disk, or return empty checkpoint */
async function loadCheckpoint(checkpointPath: string): Promise<Checkpoint> {
  try {
    const data = await fs.readFile(checkpointPath, "utf-8");
    const cp = JSON.parse(data) as Checkpoint;
    logger.info(`Loaded checkpoint: ${Object.keys(cp.indexed).length} files, completed=${cp.completed}`);
    return cp;
  } catch {
    return {
      indexed: {},
      updatedAt: new Date().toISOString(),
      total: 0,
      completed: false,
    };
  }
}

/** Save checkpoint to disk */
async function saveCheckpoint(
  checkpointPath: string,
  checkpoint: Checkpoint
): Promise<void> {
  checkpoint.updatedAt = new Date().toISOString();
  await fs.writeFile(checkpointPath, JSON.stringify(checkpoint, null, 2));
}

/** Check if file should be skipped (unchanged since last index) */
async function shouldSkipFile(
  absPath: string,
  entry: FileEntry | undefined
): Promise<boolean> {
  if (!entry) return false; // Never indexed

  // Fast check: mtime + size
  const stats = await getFileStats(absPath);
  if (stats.mtime !== entry.mtime || stats.size !== entry.size) {
    return false; // File changed
  }

  // File unchanged, skip
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

interface QueueItem {
  absPath: string;
  agentId: string;
  wsDir: string;
  source: "memory" | "sessions";
  contentType?: "knowledge" | "reference";
}

async function main() {
  const dataDir = process.env["MEMORY_SPARK_DATA_DIR"];
  if (!dataDir) {
    console.error("ERROR: Set MEMORY_SPARK_DATA_DIR to avoid touching production.");
    console.error("  MEMORY_SPARK_DATA_DIR=./test-data npx tsx scripts/reindex-benchmark.ts");
    process.exit(1);
  }

  console.log(`\n════════════════════════════════════════════════`);
  console.log(`  memory-spark Curated Benchmark Reindex`);
  console.log(`════════════════════════════════════════════════`);
  console.log(`Data dir: ${path.resolve(dataDir)}`);
  console.log(`Core agents: ${CORE_AGENTS.join(", ")}`);
  console.log(`\n`);

  // Ensure data dir exists
  await fs.mkdir(path.resolve(dataDir, "lancedb"), { recursive: true });

  // Checkpoint path
  const checkpointPath = path.resolve(dataDir, CHECKPOINT_FILE);

  // Load existing checkpoint
  const checkpoint = await loadCheckpoint(checkpointPath);

  // If checkpoint says completed, ask user what to do
  if (checkpoint.completed) {
    console.log(`\n⚠️  Checkpoint shows indexing completed at ${checkpoint.updatedAt}`);
    console.log(`   Delete ${checkpointPath} to force re-index.\n`);
    // Continue anyway - user might want to verify
  }

  // Wipe old index for a clean benchmark baseline (only if no checkpoint)
  const indexPath = path.resolve(dataDir, "lancedb", "memory_chunks.lance");
  if (Object.keys(checkpoint.indexed).length === 0) {
    try {
      await fs.access(indexPath);
      const backupName = `memory_chunks.lance.pre-benchmark.${Date.now()}`;
      console.log(`Backing up existing index → ${backupName}`);
      await fs.rename(indexPath, path.resolve(dataDir, "lancedb", backupName));
    } catch {
      // No existing index — fine
    }

    // Remove dims-lock to allow fresh creation
    try { await fs.unlink(path.resolve(dataDir, "lancedb", "dims-lock.json")); } catch { /* ok */ }
  }

  const cfg = resolveConfig();
  console.log(`Spark host: ${cfg.embed.spark?.baseUrl ?? "not configured"}`);

  // Initialize backend + embed
  const backend = new LanceDBBackend(cfg);
  await backend.open();

  let provider;
  try {
    provider = await createEmbedProvider(cfg.embed);
    console.log(`Embed provider: ${provider.id} (${provider.model}, ${provider.dims}d)`);
  } catch (err) {
    console.error(`FATAL: Cannot connect to embed provider: ${err}`);
    process.exit(1);
  }

  const queue = new EmbedQueue(provider, { concurrency: 1, timeoutMs: 60000 }, logger);

  // Verify core agents exist
  const ocDir = path.join(os.homedir(), ".openclaw");
  const existingAgents: string[] = [];
  for (const agent of CORE_AGENTS) {
    const wsDir = path.join(ocDir, `workspace-${agent}`);
    try {
      await fs.access(wsDir);
      existingAgents.push(agent);
    } catch {
      logger.warn(`Agent workspace not found: ${wsDir} — skipping ${agent}`);
    }
  }
  console.log(`\nFound ${existingAgents.length}/${CORE_AGENTS.length} agent workspaces`);

  const fileQueue: QueueItem[] = [];

  // Index core agents
  for (const agentId of existingAgents) {
    const wsFiles = await discoverWorkspaceFiles(agentId);
    for (const absPath of wsFiles.memoryFiles) {
      fileQueue.push({
        absPath,
        agentId,
        wsDir: wsFiles.workspaceDir,
        source: "memory",
      });
    }
    logger.info(`${agentId}: ${wsFiles.memoryFiles.length} files queued`);
  }

  // Index shared reference library
  if (cfg.reference?.enabled && cfg.reference.paths.length > 0) {
    const resolvedRefPaths: string[] = [];
    for (const refPath of cfg.reference.paths) {
      const resolved = refPath.startsWith("~/")
        ? path.join(os.homedir(), refPath.slice(2))
        : refPath;
      resolvedRefPaths.push(resolved);
    }
    let refRoot = resolvedRefPaths[0] ?? "/";
    while (
      refRoot !== "/" &&
      !resolvedRefPaths.every((p) => p.startsWith(refRoot + "/") || p === refRoot)
    ) {
      refRoot = path.dirname(refRoot);
    }

    for (const resolved of resolvedRefPaths) {
      try {
        await fs.access(resolved);
        const refFiles = await walkSupportedFiles(resolved);
        for (const absPath of refFiles) {
          fileQueue.push({
            absPath,
            agentId: "shared",
            wsDir: refRoot,
            source: "memory",
            contentType: "reference",
          });
        }
        logger.info(`Reference: ${refFiles.length} files from ${resolved}`);
      } catch {
        logger.warn(`Reference path not accessible: ${resolved}`);
      }
    }
  }

  console.log(`\n════════════════════════════════════════════════`);
  console.log(`  Total files discovered: ${fileQueue.length}`);
  console.log(`════════════════════════════════════════════════\n`);

  if (fileQueue.length === 0) {
    console.log("Nothing to index. Exiting.");
    await backend.close();
    return;
  }

  // Filter queue: skip unchanged files
  const toProcess: QueueItem[] = [];
  let skipped = 0;

  for (const item of fileQueue) {
    const entry = checkpoint.indexed[item.absPath];
    const skip = await shouldSkipFile(item.absPath, entry);
    if (skip) {
      skipped++;
    } else {
      toProcess.push(item);
    }
  }

  console.log(`Files unchanged (skipped): ${skipped}`);
  console.log(`Files to process: ${toProcess.length}`);
  checkpoint.total = fileQueue.length;

  if (toProcess.length === 0) {
    console.log("\n✅ All files already indexed and unchanged. Nothing to do.");
    checkpoint.completed = true;
    await saveCheckpoint(checkpointPath, checkpoint);
    await backend.close();
    return;
  }

  console.log("");

  // Process sequentially
  let ingested = 0;
  let errors = 0;
  const startTime = Date.now();

  for (let i = 0; i < toProcess.length; i++) {
    const item = toProcess[i]!;
    const basename = path.basename(item.absPath);

    try {
      const result = await ingestFile({
        filePath: item.absPath,
        agentId: item.agentId,
        workspaceDir: item.wsDir,
        backend,
        embed: queue,
        cfg,
        source: item.source,
        contentType: item.contentType,
        logger,
      });

      if (result.error) {
        errors++;
        if (errors <= 10) logger.warn(`${item.agentId}/${basename}: ${result.error}`);
      } else {
        ingested++;

        // Add to checkpoint
        const stats = await getFileStats(item.absPath);
        const hash = await hashFile(item.absPath);
        checkpoint.indexed[item.absPath] = {
          hash,
          mtime: stats.mtime,
          size: stats.size,
          chunksAdded: result.chunksAdded,
        };

        // Save checkpoint every N files
        if (ingested % CHECKPOINT_INTERVAL === 0) {
          await saveCheckpoint(checkpointPath, checkpoint);
          console.log(`  [CHECKPOINT] Saved at ${ingested}/${toProcess.length} processed`);
        }
      }
    } catch (err) {
      errors++;
      if (errors <= 10) logger.error(`${item.agentId}/${basename}: ${err}`);
    }

    // Progress every 25 files
    if ((i + 1) % 25 === 0 || i + 1 === toProcess.length) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const rate = ((i + 1) / ((Date.now() - startTime) / 1000)).toFixed(1);
      console.log(
        `  [${i + 1}/${toProcess.length}] ${ingested} indexed, ${errors} errors (${elapsed}s, ${rate} files/s)`
      );
    }
  }

  // Final checkpoint
  checkpoint.completed = errors === 0;
  await saveCheckpoint(checkpointPath, checkpoint);

  // Final stats
  const status = await backend.status();
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\n════════════════════════════════════════════════`);
  console.log(`  Benchmark Reindex Complete`);
  console.log(`════════════════════════════════════════════════`);
  console.log(`  Processed: ${ingested}/${toProcess.length} files`);
  console.log(`  Skipped:   ${skipped} (unchanged)`);
  console.log(`  Errors:    ${errors}`);
  console.log(`  Chunks:    ${status.chunkCount}`);
  console.log(`  Time:      ${elapsed}s`);
  console.log(`  Agents:    ${existingAgents.join(", ")} + shared`);
  console.log(`  Completed: ${checkpoint.completed}`);
  console.log(`════════════════════════════════════════════════\n`);

  await backend.close();
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
