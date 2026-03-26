#!/usr/bin/env npx tsx
/**
 * Quick evaluation: test search quality on the standalone index.
 * Runs a set of test queries against test-data/ LanceDB and reports relevance.
 */

import { resolveConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider } from "../src/embed/provider.js";

const TEST_QUERIES = [
  { query: "What model does Spark use for embeddings?", expectPath: /memory-spark|embed|spark/i },
  { query: "How to restart the OpenClaw gateway?", expectPath: /restart|oc-restart|AGENTS/i },
  { query: "Klein's timezone and location", expectPath: /USER\.md/i },
  { query: "What is the WireGuard IP for Spark?", expectPath: /spark|wireguard|infrastructure/i },
  { query: "How does auto-capture decide what to store?", expectPath: /capture|auto|quality/i },
  { query: "What are the sacred files that only Klein can edit?", expectPath: /AGENTS|workspace|integrity/i },
  { query: "DGX Spark RAM usage and memory pressure", expectPath: /spark|memory|hardware/i },
  { query: "How does temporal decay work in memory recall?", expectPath: /recall|decay|temporal/i },
  { query: "What school does Klein attend?", expectPath: /USER|school|student/i },
  { query: "Nemotron model configuration for vLLM", expectPath: /nemotron|vllm|spark/i },
];

async function main() {
  const cfg = resolveConfig();
  const backend = new LanceDBBackend(cfg);
  await backend.open();
  const embed = await createEmbedProvider(cfg.embed);

  const status = await backend.status();
  console.log(`Index: ${status.chunkCount} chunks\n`);

  let hits = 0;
  let total = 0;

  for (const tc of TEST_QUERIES) {
    total++;
    const vector = await embed.embedQuery(tc.query);
    const results = await backend.vectorSearch(vector, {
      query: tc.query,
      maxResults: 5,
      minScore: 0.0,
    });

    const topResult = results[0];
    const matched = topResult && tc.expectPath.test(topResult.chunk.path);
    if (matched) hits++;

    const top3Paths = results.slice(0, 3).map((r) =>
      `${r.score.toFixed(3)} ${r.chunk.path.slice(0, 50)}`
    ).join(" | ");

    console.log(`${matched ? "✅" : "❌"} "${tc.query}"`);
    console.log(`   Top 3: ${top3Paths || "(no results)"}`);
  }

  console.log(`\n=== Results: ${hits}/${total} (${((hits / total) * 100).toFixed(0)}%) ===`);
  await backend.close();
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
