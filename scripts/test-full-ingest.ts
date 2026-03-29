#!/usr/bin/env npx tsx
/**
 * Test full PDF ingest pipeline with embedding.
 * Use this to stress test large PDFs for OOM prevention.
 */

import fs from "node:fs/promises";
import path from "node:path";

import { ingestPdfBatched } from "../src/ingest/pipeline.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { EmbedQueue } from "../src/embed/queue.js";
import { resolveConfig } from "../src/config.js";

async function main() {
  const pdfPath = process.argv[2];
  if (!pdfPath) {
    console.error("Usage: npx tsx scripts/test-full-ingest.ts <pdf-path>");
    process.exit(1);
  }

  console.log(`\n=== Full Ingest Stress Test ===`);
  console.log(`File: ${pdfPath}`);

  const stat = await fs.stat(pdfPath);
  console.log(`Size: ${(stat.size / 1024 / 1024).toFixed(1)} MB`);

  const start = Date.now();

  try {
    const cfg = resolveConfig({});

    // Use test-data for this test
    const testDir = "./test-data";
    await fs.mkdir(testDir, { recursive: true });

    // Create backend
    const backend = new LanceDBBackend(cfg);
    await backend.open();

    // Create embed provider and queue
    const embedProvider = await createEmbedProvider(cfg.embed);
    const embedQueue = new EmbedQueue(embedProvider, {
      concurrency: 1,
      maxRetries: 3,
      timeoutMs: 60000,
    });

    console.log("\nStarting ingest...");
    console.log("(This will process batches: extract → chunk → embed → store per batch)");
    console.log("");

    // Ingest
    const result = await ingestPdfBatched({
      filePath: pdfPath,
      agentId: "test-large-pdf",
      workspaceDir: "/",
      backend,
      embed: embedQueue,
      cfg,
      contentType: "reference",
      logger: {
        info: (m) => console.log(`[INFO] ${m}`),
        warn: (m) => console.log(`[WARN] ${m}`),
        error: (m) => console.log(`[ERROR] ${m}`),
      },
    });

    const totalSeconds = (Date.now() - start) / 1000;

    console.log("\n=== RESULT ===");
    console.log(`Status: ${result.error ? "FAILED" : "SUCCESS"}`);
    console.log(`Chunks added: ${result.chunksAdded}`);
    console.log(`Duration: ${result.durationMs}ms (${totalSeconds.toFixed(1)}s total)`);
    if (result.error) {
      console.log(`Error: ${result.error}`);
    }

    // Performance metrics
    if (result.chunksAdded > 0) {
      const chunksPerSec = result.chunksAdded / totalSeconds;
      console.log(`Throughput: ${chunksPerSec.toFixed(1)} chunks/sec`);
    }

    if (result.error) {
      process.exit(1);
    }
  } catch (err) {
    console.error("\n=== TEST FAILED ===");
    console.error(err);
    process.exit(1);
  }
}

main();
