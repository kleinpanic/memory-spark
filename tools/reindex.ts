/**
 * Full reindex — rebuilds from all workspace + reference paths.
 * Run: npx tsx scripts/reindex.ts
 */
import { resolveConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { EmbedQueue } from "../src/embed/queue.js";
import { ingestFile } from "../src/ingest/pipeline.js";
import { discoverWorkspaceFiles, discoverAllAgents } from "../src/ingest/workspace.js";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";

const expandHome = (p: string) => p.replace(/^~/, os.homedir());

async function main() {
  const cfg = resolveConfig();
  const backend = new LanceDBBackend(cfg);
  await backend.open();

  const embed = await createEmbedProvider(cfg.embed);
  const queue = new EmbedQueue(embed);
  const logger = {
    info: (m: string) => console.log(`  ${m}`),
    warn: (m: string) => console.warn(`  ⚠ ${m}`),
    error: (m: string) => console.error(`  ❌ ${m}`),
  };

  // Discover all agent workspaces
  const agentIds = await discoverAllAgents();
  console.log(`Found ${agentIds.length} agents: ${agentIds.join(", ")}`);

  let totalChunks = 0;
  let totalFiles = 0;
  let errors = 0;

  // Index each agent workspace (memory files only, skip session dumps)
  for (const agentId of agentIds) {
    const discovered = await discoverWorkspaceFiles(agentId);
    console.log(`\n📁 ${agentId}: ${discovered.memoryFiles.length} memory files in ${discovered.workspaceDir}`);

    for (const filePath of discovered.memoryFiles) {
      try {
        const result = await ingestFile({
          filePath,
          agentId,
          workspaceDir: discovered.workspaceDir,
          backend,
          embed: queue,
          cfg,
          source: "memory",
          logger,
        });

        if (result.chunksAdded > 0) {
          totalChunks += result.chunksAdded;
          totalFiles++;
        }
        if (result.error) {
          errors++;
        }
      } catch (err) {
        console.warn(`  ⚠ Failed: ${filePath}: ${err}`);
        errors++;
      }
    }
  }

  // Index reference library
  if (cfg.reference?.enabled && cfg.reference.paths) {
    console.log(`\n📚 Reference library (${cfg.reference.paths.length} paths)`);
    for (const refPath of cfg.reference.paths) {
      const absPath = expandHome(refPath);
      try {
        const stat = await fs.stat(absPath);
        if (!stat.isDirectory()) continue;

        // Walk the directory for supported files
        const files = await walkDir(absPath, cfg.reference.chunkSize ? [".md", ".txt"] : [".md", ".txt", ".pdf"]);
        console.log(`  ${path.basename(absPath)}: ${files.length} files`);

        for (const filePath of files) {
          try {
            const result = await ingestFile({
              filePath,
              agentId: "reference",
              workspaceDir: absPath,
              backend,
              embed: queue,
              cfg,
              source: "ingest",
              contentType: "reference",
              logger,
            });

            if (result.chunksAdded > 0) {
              totalChunks += result.chunksAdded;
              totalFiles++;
            }
          } catch (err) {
            errors++;
          }
        }
      } catch {
        console.warn(`  ⚠ Skipping ${refPath}: not found`);
      }
    }
  }

  const status = await backend.status();
  console.log(`\n✅ Reindex complete`);
  console.log(`   Files indexed: ${totalFiles}`);
  console.log(`   Chunks created: ${totalChunks}`);
  console.log(`   Errors: ${errors}`);
  console.log(`   Backend total: ${status.chunkCount} chunks`);

  await backend.close();
}

async function walkDir(dir: string, exts: string[]): Promise<string[]> {
  const results: string[] = [];
  const extSet = new Set(exts);

  async function walk(d: string) {
    const entries = await fs.readdir(d, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(d, entry.name);
      if (entry.isDirectory()) {
        // Skip hidden dirs, node_modules, .git, zh-CN translations
        if (entry.name.startsWith(".") || entry.name === "node_modules" || 
            entry.name === "zh-CN" || entry.name === "zh-cn" ||
            entry.name === "i18n" || entry.name === "locales") continue;
        await walk(full);
      } else if (extSet.has(path.extname(entry.name).toLowerCase())) {
        results.push(full);
      }
    }
  }

  await walk(dir);
  return results;
}

main().catch((err) => {
  console.error("Reindex failed:", err);
  process.exit(1);
});
