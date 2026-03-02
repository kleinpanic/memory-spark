/**
 * Migration: memory-core SQLite-vec → memory-spark LanceDB
 *
 * Reads existing memory workspace .md files for each agent,
 * chunks and embeds them with the configured provider, stores in LanceDB.
 *
 * Run via: node dist/scripts/migrate.js
 * Or auto-run on first boot if config.migrate.autoMigrateOnFirstBoot = true
 */

import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { resolveConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { chunkDocument } from "../src/embed/chunker.js";
import { tagEntities } from "../src/classify/ner.js";
import crypto from "node:crypto";

interface MigrationStatus {
  version: 1;
  startedAt: string;
  completedAt?: string;
  agents: Record<string, { status: string; files: number; chunks: number; error?: string }>;
}

async function main() {
  const cfg = resolveConfig();
  const statusFile = cfg.migrate.statusFile;

  // Check if already migrated
  try {
    const existing = JSON.parse(await fs.readFile(statusFile, "utf-8")) as MigrationStatus;
    if (existing.completedAt) {
      console.log(`Migration already completed at ${existing.completedAt}. Skipping.`);
      return;
    }
  } catch {
    // No status file = first run
  }

  console.log("memory-spark migration: starting");

  const backend = new LanceDBBackend(cfg);
  await backend.open();

  const embed = await createEmbedProvider(cfg.embed);
  console.log(`Using embed provider: ${embed.id}/${embed.model}`);

  const status: MigrationStatus = {
    version: 1,
    startedAt: new Date().toISOString(),
    agents: {},
  };

  // Discover agent workspace directories with memory folders
  const openclawDir = path.join(os.homedir(), ".openclaw");
  const entries = await fs.readdir(openclawDir, { withFileTypes: true });
  const workspaceDirs = entries
    .filter((e) => e.isDirectory() && e.name.startsWith("workspace-"))
    .map((e) => ({
      agentId: e.name.replace("workspace-", ""),
      memoryDir: path.join(openclawDir, e.name, "memory"),
      workspaceDir: path.join(openclawDir, e.name),
    }));

  // Also include the shared memory dir
  workspaceDirs.push({
    agentId: "shared",
    memoryDir: path.join(openclawDir, "memory"),
    workspaceDir: openclawDir,
  });

  for (const { agentId, memoryDir, workspaceDir } of workspaceDirs) {
    try {
      const stat = await fs.stat(memoryDir).catch(() => null);
      if (!stat?.isDirectory()) continue;

      console.log(`Migrating agent: ${agentId} (${memoryDir})`);

      // Find all .md files in memory dir
      const mdFiles = await walkMdFiles(memoryDir);

      // Also find MEMORY.md, SOUL.md, USER.md in workspace root
      const rootFiles = ["MEMORY.md", "SOUL.md", "USER.md", "AGENTS.md", "HEARTBEAT.md"];
      for (const f of rootFiles) {
        const fp = path.join(workspaceDir, f);
        try {
          await fs.access(fp);
          mdFiles.push(fp);
        } catch {
          // File doesn't exist
        }
      }

      let totalChunks = 0;

      for (const filePath of mdFiles) {
        try {
          const text = await fs.readFile(filePath, "utf-8");
          if (!text.trim()) continue;

          const chunks = chunkDocument({
            text,
            path: filePath,
            source: "memory",
            ext: "md",
          });

          if (chunks.length === 0) continue;

          // NER (best-effort)
          const entitiesPerChunk = await Promise.all(
            chunks.map((c) => tagEntities(c.text, cfg).catch(() => [] as string[]))
          );

          // Embed
          const vectors = await embed.embedBatch(chunks.map((c) => c.text));

          const now = new Date().toISOString();
          const memoryChunks = chunks.map((raw, i) => ({
            id: crypto.createHash("sha1").update(`${agentId}:${filePath}:${raw.startLine}`).digest("hex").slice(0, 16),
            path: filePath,
            source: "memory" as const,
            agent_id: agentId,
            start_line: raw.startLine,
            end_line: raw.endLine,
            text: raw.text,
            vector: vectors[i]!,
            updated_at: now,
            entities: JSON.stringify(entitiesPerChunk[i] ?? []),
          }));

          await backend.upsert(memoryChunks);
          totalChunks += memoryChunks.length;
          console.log(`  ${filePath}: ${memoryChunks.length} chunks`);
        } catch (err) {
          console.warn(`  SKIP ${filePath}: ${err}`);
        }
      }

      status.agents[agentId] = { status: "done", files: mdFiles.length, chunks: totalChunks };
      console.log(`  ${agentId}: ${mdFiles.length} files → ${totalChunks} chunks`);
    } catch (err) {
      status.agents[agentId] = { status: "error", files: 0, chunks: 0, error: String(err) };
      console.error(`  ${agentId} FAILED: ${err}`);
    }
  }

  status.completedAt = new Date().toISOString();
  await fs.mkdir(path.dirname(statusFile), { recursive: true });
  await fs.writeFile(statusFile, JSON.stringify(status, null, 2));

  await backend.close();
  console.log(`Migration complete. Status written to ${statusFile}`);
}

async function walkMdFiles(dir: string): Promise<string[]> {
  const results: string[] = [];
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name.startsWith(".")) continue;
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        results.push(...await walkMdFiles(fullPath));
      } else if (entry.name.endsWith(".md")) {
        results.push(fullPath);
      }
    }
  } catch {
    // Skip inaccessible dirs
  }
  return results;
}

main().catch((err) => {
  console.error("Migration failed:", err);
  process.exit(1);
});
