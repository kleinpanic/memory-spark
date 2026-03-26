#!/usr/bin/env npx tsx
/**
 * Standalone Indexer — replicates the gateway boot-pass without requiring OpenClaw.
 *
 * Usage:
 *   MEMORY_SPARK_DATA_DIR=./test-data npx tsx tools/indexer.ts
 *
 * This discovers all agent workspaces + reference library paths,
 * runs the full ingest pipeline (parse → chunk → quality → NER → embed → store),
 * and writes to the LanceDB at $MEMORY_SPARK_DATA_DIR/lancedb/.
 *
 * Requires: SPARK_HOST or SPARK_BEARER_TOKEN env vars (or ~/.openclaw/.env).
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
  discoverAllAgents,
  discoverWorkspaceFiles,
  walkSupportedFiles,
  toRelativePath,
} from "../src/ingest/workspace.js";

const logger = {
  info: (m: string) => console.log(`[INFO] ${m}`),
  warn: (m: string) => console.warn(`[WARN] ${m}`),
  error: (m: string) => console.error(`[ERR]  ${m}`),
};

async function main() {
  const dataDir = process.env["MEMORY_SPARK_DATA_DIR"];
  if (!dataDir) {
    console.error("ERROR: Set MEMORY_SPARK_DATA_DIR to avoid touching production.");
    console.error("  MEMORY_SPARK_DATA_DIR=./test-data npx tsx tools/indexer.ts");
    process.exit(1);
  }

  console.log(`\n=== memory-spark Standalone Indexer ===`);
  console.log(`Data dir: ${path.resolve(dataDir)}`);
  console.log(`LanceDB:  ${path.resolve(dataDir, "lancedb")}\n`);

  // Ensure data dir exists
  await fs.mkdir(path.resolve(dataDir, "lancedb"), { recursive: true });

  const cfg = resolveConfig();
  console.log(`Spark host: ${cfg.embed.spark?.baseUrl ?? "not configured"}`);
  console.log(`Embed model: ${cfg.embed.spark?.model ?? "unknown"}`);

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

  // Build file queue
  const agents = await discoverAllAgents();
  console.log(`\nDiscovered agents: ${agents.join(", ")}`);

  interface QueueItem {
    absPath: string;
    agentId: string;
    wsDir: string;
    source: "memory" | "sessions";
    contentType?: "knowledge" | "reference";
  }

  const fileQueue: QueueItem[] = [];
  const indexed = await backend.listPaths();
  const indexedMap = new Map(indexed.map((i) => [`${i.agentId}::${i.path}`, i.updatedAt]));
  let skipped = 0;

  for (const agentId of agents) {
    const wsFiles = await discoverWorkspaceFiles(agentId);

    for (const absPath of wsFiles.memoryFiles) {
      try {
        const stat = await fs.stat(absPath);
        const relPath = toRelativePath(absPath, wsFiles.workspaceDir);
        const existing = indexedMap.get(`${agentId}::${relPath}`);
        if (existing && existing >= stat.mtime.toISOString()) {
          skipped++;
          continue;
        }
        fileQueue.push({ absPath, agentId, wsDir: wsFiles.workspaceDir, source: "memory" });
      } catch {
        skipped++;
      }
    }

    // Skip sessions by default (they're noise)
    // for (const absPath of wsFiles.sessionFiles) { ... }
  }

  // Reference library
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
          try {
            const stat = await fs.stat(absPath);
            const relPath = path.relative(refRoot, absPath);
            const key = `shared::${relPath}`;
            const existing = indexedMap.get(key);
            if (existing && existing >= stat.mtime.toISOString()) {
              skipped++;
              continue;
            }
            fileQueue.push({
              absPath,
              agentId: "shared",
              wsDir: refRoot,
              source: "memory",
              contentType: "reference",
            });
          } catch {
            skipped++;
          }
        }
        logger.info(`Reference: ${refFiles.length} files from ${resolved}`);
      } catch {
        logger.warn(`Reference path not accessible: ${resolved}`);
      }
    }
  }

  console.log(`\nFiles to index: ${fileQueue.length} (${skipped} up-to-date)\n`);

  if (fileQueue.length === 0) {
    console.log("Nothing to index. Done.");
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
        logger.warn(`${basename}: ${result.error}`);
      } else {
        ingested++;
      }
    } catch (err) {
      errors++;
      logger.error(`${basename}: ${err}`);
    }

    // Progress every 10 files
    if ((i + 1) % 10 === 0 || i + 1 === fileQueue.length) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`  [${i + 1}/${fileQueue.length}] ${ingested} indexed, ${errors} errors (${elapsed}s)`);
    }
  }

  // Final stats
  const status = await backend.status();
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\n=== Indexing Complete ===`);
  console.log(`Indexed: ${ingested}/${fileQueue.length} files`);
  console.log(`Errors: ${errors}`);
  console.log(`Skipped: ${skipped} (up-to-date)`);
  console.log(`Total chunks in LanceDB: ${status.chunkCount}`);
  console.log(`Time: ${elapsed}s`);

  await backend.close();
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
