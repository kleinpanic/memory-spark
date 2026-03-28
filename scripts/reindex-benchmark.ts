#!/usr/bin/env npx tsx
/**
 * Curated Benchmark Reindex — indexes ONLY the core agents for evaluation.
 *
 * Usage:
 *   MEMORY_SPARK_DATA_DIR=./test-data npx tsx scripts/reindex-benchmark.ts
 *
 * Core agents: meta, main, dev, school, ghost, recovery, research, taskmaster
 * Plus: shared reference library
 *
 * This produces a clean, scoped index that matches the golden dataset coverage.
 */

import path from "node:path";
import fs from "node:fs/promises";
import os from "node:os";

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

const logger = {
  info: (m: string) => console.log(`[INFO] ${m}`),
  warn: (m: string) => console.warn(`[WARN] ${m}`),
  error: (m: string) => console.error(`[ERR]  ${m}`),
};

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

  // Wipe old index for a clean benchmark baseline
  const indexPath = path.resolve(dataDir, "lancedb", "memory_chunks.lance");
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

  interface QueueItem {
    absPath: string;
    agentId: string;
    wsDir: string;
    source: "memory" | "sessions";
    contentType?: "knowledge" | "reference";
  }

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
  console.log(`  Total files to index: ${fileQueue.length}`);
  console.log(`════════════════════════════════════════════════\n`);

  if (fileQueue.length === 0) {
    console.log("Nothing to index. Exiting.");
    await backend.close();
    return;
  }

  // Process sequentially
  let ingested = 0;
  let errors = 0;
  const startTime = Date.now();

  for (let i = 0; i < fileQueue.length; i++) {
    const item = fileQueue[i]!;
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
      }
    } catch (err) {
      errors++;
      if (errors <= 10) logger.error(`${item.agentId}/${basename}: ${err}`);
    }

    // Progress every 25 files
    if ((i + 1) % 25 === 0 || i + 1 === fileQueue.length) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const rate = ((i + 1) / ((Date.now() - startTime) / 1000)).toFixed(1);
      console.log(`  [${i + 1}/${fileQueue.length}] ${ingested} indexed, ${errors} errors (${elapsed}s, ${rate} files/s)`);
    }
  }

  // Final stats
  const status = await backend.status();
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\n════════════════════════════════════════════════`);
  console.log(`  Benchmark Reindex Complete`);
  console.log(`════════════════════════════════════════════════`);
  console.log(`  Indexed: ${ingested}/${fileQueue.length} files`);
  console.log(`  Errors:  ${errors}`);
  console.log(`  Chunks:  ${status.chunkCount}`);
  console.log(`  Time:    ${elapsed}s`);
  console.log(`  Agents:  ${existingAgents.join(", ")} + shared`);
  console.log(`════════════════════════════════════════════════\n`);

  await backend.close();
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
