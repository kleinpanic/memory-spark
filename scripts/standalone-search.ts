#!/usr/bin/env npx tsx
/**
 * Standalone Search — runs the full recall pipeline against the test index.
 *
 * Usage:
 *   MEMORY_SPARK_DATA_DIR=./test-data npx tsx scripts/standalone-search.ts "how to restart gateway"
 *   MEMORY_SPARK_DATA_DIR=./test-data npx tsx scripts/standalone-search.ts --query "spark endpoints" --top 10
 */

import { resolveConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { createReranker } from "../src/rerank/reranker.js";
import type { SearchResult } from "../src/storage/backend.js";
import {
  hybridMerge,
  applyTemporalDecay,
  applySourceWeighting,
  mmrRerank,
} from "../src/auto/recall.js";

async function main() {
  const dataDir = process.env["MEMORY_SPARK_DATA_DIR"];
  if (!dataDir) {
    console.error("ERROR: Set MEMORY_SPARK_DATA_DIR to avoid touching production.");
    process.exit(1);
  }

  // Parse args
  const args = process.argv.slice(2);
  let query = "";
  let topN = 5;
  let verbose = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--query" && args[i + 1]) {
      query = args[++i]!;
    } else if (args[i] === "--top" && args[i + 1]) {
      topN = parseInt(args[++i]!, 10);
    } else if (args[i] === "--verbose" || args[i] === "-v") {
      verbose = true;
    } else if (!args[i]!.startsWith("-")) {
      query = args[i]!;
    }
  }

  if (!query) {
    console.error("Usage: standalone-search.ts <query> [--top N] [--verbose]");
    process.exit(1);
  }

  const cfg = resolveConfig();
  const backend = new LanceDBBackend(cfg);
  await backend.open();

  const status = await backend.status();
  console.log(`LanceDB: ${status.chunkCount} chunks\n`);

  if (status.chunkCount === 0) {
    console.error("Index is empty. Run standalone-index.ts first.");
    await backend.close();
    process.exit(1);
  }

  // Embed query
  const provider = await createEmbedProvider(cfg.embed);
  const queryVector = await provider.embedQuery(query);

  // Fetch candidates
  const fetchN = topN * 4;
  const minScore = 0.2;

  const start = performance.now();

  const [vectorResults, rawFtsResults] = await Promise.all([
    backend.vectorSearch(queryVector, { query, maxResults: fetchN, minScore }).catch(() => [] as SearchResult[]),
    backend.ftsSearch(query, { query, maxResults: fetchN }).catch(() => [] as SearchResult[]),
  ]);

  const ftsResults = rawFtsResults.filter(
    (r) => r.chunk.source !== "sessions" && r.score >= minScore,
  );

  if (verbose) {
    console.log(`Vector results: ${vectorResults.length}, FTS results: ${ftsResults.length}`);
  }

  // Full pipeline
  const merged = hybridMerge(vectorResults, ftsResults, fetchN);
  applySourceWeighting(merged);
  applyTemporalDecay(merged);
  const diverse = mmrRerank(merged, topN * 2, 0.7);

  // Rerank
  let final: SearchResult[];
  try {
    const reranker = await createReranker(cfg.rerank);
    final = await reranker.rerank(query, diverse, topN);
  } catch {
    console.warn("Reranker unavailable, using score-only ranking");
    final = diverse.sort((a, b) => b.score - a.score).slice(0, topN);
  }

  const elapsed = (performance.now() - start).toFixed(1);

  // Print results
  console.log(`Query: "${query}"`);
  console.log(`Results: ${final.length} (${elapsed}ms)\n`);

  for (let i = 0; i < final.length; i++) {
    const r = final[i]!;
    const age = r.chunk.updated_at
      ? `${Math.round((Date.now() - new Date(r.chunk.updated_at).getTime()) / 86400000)}d`
      : "?";
    console.log(`  ${i + 1}. [${r.score.toFixed(3)}] ${r.chunk.agent_id}:${r.chunk.path}:${r.chunk.start_line} (${r.chunk.source}, ${age} old)`);
    const snippet = r.chunk.text.slice(0, 200).replace(/\n/g, " ");
    console.log(`     ${snippet}...`);
    console.log();
  }

  await backend.close();
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
