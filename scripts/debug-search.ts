#!/usr/bin/env npx tsx
/**
 * Debug script: Show what the benchmark pipeline actually retrieves
 * and how it matches against the golden dataset.
 */
import { resolveConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { EmbedQueue } from "../src/embed/queue.js";
import { hybridMerge, applySourceWeighting, applyTemporalDecay, mmrRerank } from "../src/auto/recall.js";
import { createReranker } from "../src/rerank/reranker.js";
import fs from "node:fs/promises";
import path from "node:path";

async function main() {
  const cfg = resolveConfig();
  const backend = new LanceDBBackend(cfg);
  await backend.open();
  const provider = await createEmbedProvider(cfg.embed);
  const embed = new EmbedQueue(provider, { concurrency: 1, maxRetries: 2, timeoutMs: 30000 });
  const reranker = await createReranker(cfg.rerank);

  const status = await backend.status();
  console.log(`Index: ${status.chunkCount} chunks\n`);

  // Run a few diagnostic queries
  const queries = [
    "What machine does OpenClaw run on?",
    "What timezone is Klein in?",
    "How do I safely restart the OpenClaw gateway?",
  ];

  for (const q of queries) {
    console.log(`\n${"=".repeat(70)}`);
    console.log(`QUERY: "${q}"`);
    console.log(`${"=".repeat(70)}`);

    const queryVector = await embed.embedQuery(q);
    
    // Vector search
    const vResults = await backend.vectorSearch(queryVector, { query: q, maxResults: 40, minScore: 0.05 });
    console.log(`\n--- Vector Search (${vResults.length} results) ---`);
    for (const r of vResults.slice(0, 5)) {
      console.log(`  [${r.score.toFixed(4)}] ${r.chunk.agent_id}:${r.chunk.path} (pool=${r.chunk.pool ?? "none"})`);
      console.log(`          "${r.chunk.text.slice(0, 100)}..."`);
    }

    // FTS search  
    const fResults = await backend.ftsSearch(q, { query: q, maxResults: 40 });
    console.log(`\n--- FTS Search (${fResults.length} results) ---`);
    for (const r of fResults.slice(0, 5)) {
      console.log(`  [${r.score.toFixed(4)}] ${r.chunk.agent_id}:${r.chunk.path}`);
    }

    // Hybrid merge
    let merged = hybridMerge(vResults, fResults, 40);
    console.log(`\n--- After Hybrid Merge (${merged.length}) ---`);
    for (const r of merged.slice(0, 5)) {
      console.log(`  [${r.score.toFixed(4)}] ${r.chunk.agent_id}:${r.chunk.path}`);
    }

    // Source weighting
    applySourceWeighting(merged, cfg.autoRecall.weights);
    console.log(`\n--- After Source Weighting ---`);
    merged.sort((a, b) => b.score - a.score);
    for (const r of merged.slice(0, 5)) {
      console.log(`  [${r.score.toFixed(4)}] ${r.chunk.agent_id}:${r.chunk.path}`);
    }

    // Temporal decay
    applyTemporalDecay(merged, cfg.autoRecall.temporalDecay);
    console.log(`\n--- After Temporal Decay ---`);
    merged.sort((a, b) => b.score - a.score);
    for (const r of merged.slice(0, 5)) {
      console.log(`  [${r.score.toFixed(4)}] ${r.chunk.agent_id}:${r.chunk.path}`);
    }

    // MMR
    const mmrResults = mmrRerank(merged, 20, 0.7);
    console.log(`\n--- After MMR (${mmrResults.length}) ---`);
    for (const r of mmrResults.slice(0, 5)) {
      console.log(`  [${r.score.toFixed(4)}] ${r.chunk.agent_id}:${r.chunk.path}`);
    }

    // Reranker
    try {
      const reranked = await reranker.rerank(q, mmrResults.slice(0, 20), 10);
      console.log(`\n--- After Reranker (${reranked.length}) ---`);
      for (const r of reranked.slice(0, 5)) {
        console.log(`  [${r.score.toFixed(4)}] ${r.chunk.agent_id}:${r.chunk.path}`);
        console.log(`          "${r.chunk.text.slice(0, 100)}..."`);
      }
    } catch (e: any) {
      console.log(`\nReranker failed: ${e.message}`);
    }
  }

  await backend.close();
}

main().catch(console.error);
