#!/usr/bin/env npx tsx
/**
 * Quick eval v2: checks content relevance, not just path matching.
 * Also tests the full hybrid pipeline (vector + FTS + rerank).
 */

import { resolveConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { createReranker } from "../src/rerank/reranker.js";
import { applySourceWeighting, applyTemporalDecay } from "../src/auto/recall.js";
import type { SearchResult } from "../src/storage/backend.js";

const TEST_QUERIES = [
  {
    query: "What model does Spark use for embeddings?",
    expectContent: /llama-embed-nemotron|nvidia.*embed|4096/i,
  },
  {
    query: "How to restart the OpenClaw gateway?",
    expectContent: /oc-restart|restart.*gateway|systemctl.*restart|NEVER.*direct/i,
  },
  {
    query: "Klein's timezone and location",
    expectContent: /America\/New_York|Exampleville|Virginia|EST|EDT/i,
  },
  {
    query: "What is the WireGuard IP for Spark?",
    expectContent: /10\.99\.1\.1|10\.88\.88|wireguard|wg/i,
  },
  {
    query: "How does auto-capture decide what to store?",
    expectContent: /classif|heuristic|capture|confidence|zero-shot|minConfidence/i,
  },
  {
    query: "What are the sacred files that only Klein can edit?",
    expectContent: /SOUL\.md|USER\.md|IDENTITY\.md|sacred|Klein-only/i,
  },
  {
    query: "DGX Spark RAM usage and memory pressure",
    expectContent: /90%|111\s*GiB|GH200|memory.*pressure|RAM/i,
  },
  {
    query: "How does temporal decay work in memory recall?",
    expectContent: /0\.8.*0\.2.*exp|temporal.*decay|ageDays|floor/i,
  },
  {
    query: "What school does Klein attend?",
    expectContent: /Virginia\s*Tech|VT|Exampleville|university|college|student/i,
  },
  {
    query: "Nemotron model configuration for vLLM",
    expectContent: /nemotron.*120b|NVFP4|vllm|tensor.*parallel|gpu_memory/i,
  },
];

async function main() {
  const cfg = resolveConfig();
  const backend = new LanceDBBackend(cfg);
  await backend.open();
  const embed = await createEmbedProvider(cfg.embed);
  const reranker = await createReranker(cfg.rerank);

  const status = await backend.status();
  console.log(`Index: ${status.chunkCount} chunks`);
  console.log(`Reranker: ${cfg.rerank.enabled ? "enabled" : "off"}\n`);

  let vectorHits = 0;
  let hybridHits = 0;
  let total = 0;

  for (const tc of TEST_QUERIES) {
    total++;
    const vector = await embed.embedQuery(tc.query);

    // Vector-only search
    const vectorResults = await backend.vectorSearch(vector, {
      query: tc.query,
      maxResults: 10,
      minScore: 0.0,
    });

    // FTS search
    const ftsResults = await backend.ftsSearch(tc.query, {
      query: tc.query,
      maxResults: 10,
    }).catch(() => [] as SearchResult[]);

    // Hybrid merge (simplified)
    const merged = new Map<string, SearchResult>();
    for (const r of vectorResults) merged.set(r.chunk.id, r);
    for (const r of ftsResults) {
      if (!merged.has(r.chunk.id)) {
        r.score *= 0.7; // FTS-only gets lower base
        merged.set(r.chunk.id, r);
      } else {
        // Dual-source boost
        const existing = merged.get(r.chunk.id)!;
        existing.score *= 1.15;
      }
    }

    let hybridResults = Array.from(merged.values());

    // Apply source weighting + temporal decay
    applySourceWeighting(hybridResults);
    applyTemporalDecay(hybridResults);

    // Sort by score descending
    hybridResults.sort((a, b) => b.score - a.score);
    hybridResults = hybridResults.slice(0, 5);

    // Check vector-only top 5
    const vectorTop5Text = vectorResults.slice(0, 5).map((r) => r.chunk.text).join(" ");
    const vMatch = tc.expectContent.test(vectorTop5Text);
    if (vMatch) vectorHits++;

    // Check hybrid top 5
    const hybridTop5Text = hybridResults.map((r) => r.chunk.text).join(" ");
    const hMatch = tc.expectContent.test(hybridTop5Text);
    if (hMatch) hybridHits++;

    console.log(`${hMatch ? "✅" : "❌"} "${tc.query}" [vector:${vMatch ? "✓" : "✗"} hybrid:${hMatch ? "✓" : "✗"}]`);
    console.log(`   Top: ${hybridResults[0]?.score.toFixed(3) ?? "N/A"} ${hybridResults[0]?.chunk.path.slice(0, 50) ?? ""}`);
  }

  console.log(`\n=== Vector-only: ${vectorHits}/${total} (${((vectorHits / total) * 100).toFixed(0)}%) ===`);
  console.log(`=== Hybrid:      ${hybridHits}/${total} (${((hybridHits / total) * 100).toFixed(0)}%) ===`);

  await backend.close();
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
