/**
 * Migration: existing workspace memory → LanceDB
 * Reads all agent workspace memory .md files + session transcripts,
 * chunks, embeds, stores in LanceDB with relative paths.
 */

import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { resolveConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { ingestFile } from "../src/ingest/pipeline.js";
import { discoverAllAgents, discoverWorkspaceFiles } from "../src/ingest/workspace.js";

interface MigrationStatus {
  version: 1;
  startedAt: string;
  completedAt?: string;
  agents: Record<
    string,
    { status: string; memoryFiles: number; sessionFiles: number; chunks: number; error?: string }
  >;
}

export async function runMigration(): Promise<void> {
  const cfg = resolveConfig();
  const statusFile = cfg.migrate.statusFile;

  // Check if already migrated
  try {
    const existing = JSON.parse(await fs.readFile(statusFile, "utf-8")) as MigrationStatus;
    if (existing.completedAt) {
      console.log(`Migration already completed at ${existing.completedAt}`);
      return;
    }
  } catch {
    // First run
  }

  console.log("memory-spark migration: starting");

  const backend = new LanceDBBackend(cfg);
  await backend.open();

  const embed = await createEmbedProvider(cfg.embed);
  console.log(`Embed: ${embed.id}/${embed.model} (${embed.dims}d)`);

  const status: MigrationStatus = {
    version: 1,
    startedAt: new Date().toISOString(),
    agents: {},
  };

  const agents = await discoverAllAgents();
  console.log(`Found ${agents.length} agents: ${agents.join(", ")}`);

  for (const agentId of agents) {
    try {
      const wsFiles = await discoverWorkspaceFiles(agentId);
      let totalChunks = 0;

      // Index memory files
      for (const absPath of wsFiles.memoryFiles) {
        try {
          const result = await ingestFile({
            filePath: absPath,
            agentId,
            workspaceDir: wsFiles.workspaceDir,
            backend,
            embed,
            cfg,
            source: "memory",
          });
          totalChunks += result.chunksAdded;
          if (result.chunksAdded > 0) {
            console.log(`  ${agentId}: ${result.filePath} → ${result.chunksAdded} chunks`);
          }
        } catch (err) {
          console.warn(`  SKIP ${absPath}: ${err}`);
        }
      }

      // Index session files
      for (const absPath of wsFiles.sessionFiles) {
        try {
          const result = await ingestFile({
            filePath: absPath,
            agentId,
            workspaceDir: wsFiles.workspaceDir,
            backend,
            embed,
            cfg,
            source: "sessions",
          });
          totalChunks += result.chunksAdded;
        } catch {
          // Skip bad sessions silently
        }
      }

      status.agents[agentId] = {
        status: "done",
        memoryFiles: wsFiles.memoryFiles.length,
        sessionFiles: wsFiles.sessionFiles.length,
        chunks: totalChunks,
      };
      console.log(
        `${agentId}: ${wsFiles.memoryFiles.length} memory + ${wsFiles.sessionFiles.length} sessions → ${totalChunks} chunks`,
      );
    } catch (err) {
      status.agents[agentId] = {
        status: "error",
        memoryFiles: 0,
        sessionFiles: 0,
        chunks: 0,
        error: String(err),
      };
      console.error(`${agentId} FAILED: ${err}`);
    }
  }

  status.completedAt = new Date().toISOString();
  await fs.mkdir(path.dirname(statusFile), { recursive: true });
  await fs.writeFile(statusFile, JSON.stringify(status, null, 2));

  await backend.close();
  console.log(`Migration complete → ${statusFile}`);
}

// Run if executed directly
runMigration().catch((err) => {
  console.error("Migration failed:", err);
  process.exit(1);
});
