#!/usr/bin/env npx tsx
/**
 * Phase 11B Diagnostic: Multi-Query Expansion
 *
 * Runs 10 SciFact queries to verify:
 * 1. LLM generates valid reformulations
 * 2. Expanded results contain docs NOT in original top-40
 * 3. Union size > original result count
 * 4. Latency within acceptable bounds
 *
 * Usage: SPARK_BEARER_TOKEN=xxx npx tsx scripts/diag-multi-query.ts
 */

import fs from "node:fs/promises";
import path from "node:path";

import { resolveConfig } from "../src/config.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { EmbedQueue } from "../src/embed/queue.js";
import { expandQuery, QUERY_EXPANSION_DEFAULTS } from "../src/query/expander.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";

interface BeirQuery {
  _id: string;
  text: string;
}

async function main() {
  // Use the same BEIR-specific DB as the benchmark script
  const defaultDir = `${process.env.HOME}/.openclaw/data/testDbBEIR/lancedb`;
  const lancedbDir = process.env.BEIR_LANCEDB_DIR || defaultDir;
  console.log(`[INFO] Using lancedbDir: ${lancedbDir}`);

  const cfg = resolveConfig({ lancedbDir } as Parameters<typeof resolveConfig>[0]);
  const provider = await createEmbedProvider(cfg.embed);
  const embed = new EmbedQueue(provider, { concurrency: 1, maxRetries: 2, timeoutMs: 30000 });
  const backend = new LanceDBBackend(cfg);
  await backend.open();

  const expansionConfig = {
    ...QUERY_EXPANSION_DEFAULTS,
    apiKey: process.env.SPARK_BEARER_TOKEN,
    timeoutMs: 20000, // generous for diagnostics
  };

  // Load first 10 SciFact queries with relevance judgments
  const queriesFile = path.join(import.meta.dirname!, "../evaluation/beir-datasets/scifact/queries.jsonl");
  const qrelsFile = path.join(import.meta.dirname!, "../evaluation/beir-datasets/scifact/qrels/test.tsv");

  const allQueries: BeirQuery[] = (await fs.readFile(queriesFile, "utf-8"))
    .trim().split("\n").filter(Boolean).map((l) => JSON.parse(l));

  const qrelsLines = (await fs.readFile(qrelsFile, "utf-8")).trim().split("\n");
  const qrelIds = new Set<string>();
  for (const line of qrelsLines) {
    const parts = line.split("\t");
    if (parts.length >= 1 && !line.startsWith("query-id")) qrelIds.add(parts[0]!);
  }

  const evalQueries = allQueries.filter((q) => qrelIds.has(q._id)).slice(0, 10);
  console.log(`\n🔍 Multi-Query Expansion Diagnostic (${evalQueries.length} queries)\n`);

  const results: Array<{
    queryId: string;
    queryText: string;
    reformulations: string[];
    originalHits: number;
    unionHits: number;
    newDocs: number;
    expandMs: number;
    totalMs: number;
  }> = [];

  for (let i = 0; i < evalQueries.length; i++) {
    const q = evalQueries[i]!;
    const t0 = Date.now();
    console.log(`── Query ${i + 1}/10: "${q.text.slice(0, 80)}…"`);

    // 1. Expand query
    const t0expand = Date.now();
    const queries = await expandQuery(q.text, expansionConfig);
    const expandMs = Date.now() - t0expand;
    console.log(`  Expansion: ${queries.length} queries in ${expandMs}ms`);
    for (const mq of queries.slice(1)) {
      console.log(`    → "${mq.slice(0, 100)}"`);
    }

    // 2. Embed all queries (sequentially to be safe with Spark)
    const vectors: number[][] = [];
    for (const mq of queries) {
      const vec = await embed.embedQuery(mq);
      vectors.push(vec);
    }

    // 3. Search original only
    const searchOpts = { query: q.text, maxResults: 40, minScore: 0.0, pathContains: "beir/scifact/" };
    const originalResults = await backend.vectorSearch(vectors[0]!, searchOpts).catch(() => []);
    const originalIds = new Set(originalResults.map((r) => r.chunk.id));

    // 4. Search all reformulations and union
    let unionResults = [...originalResults];
    const seenIds = new Set(originalIds);

    for (let j = 1; j < vectors.length; j++) {
      const reformResults = await backend.vectorSearch(vectors[j]!, searchOpts).catch(() => []);
      for (const r of reformResults) {
        if (!seenIds.has(r.chunk.id)) {
          unionResults.push(r);
          seenIds.add(r.chunk.id);
        }
      }
    }
    unionResults.sort((a, b) => b.score - a.score);

    const newDocs = unionResults.length - originalResults.length;
    const totalMs = Date.now() - t0;

    console.log(`  Original: ${originalResults.length} hits | Union: ${unionResults.length} hits | New docs: +${newDocs}`);
    console.log(`  Total time: ${totalMs}ms\n`);

    results.push({
      queryId: q._id,
      queryText: q.text.slice(0, 120),
      reformulations: queries.slice(1),
      originalHits: originalResults.length,
      unionHits: unionResults.length,
      newDocs,
      expandMs,
      totalMs,
    });
  }

  // Summary
  console.log("\n═══ Summary ═══");
  const avgNewDocs = results.reduce((sum, r) => sum + r.newDocs, 0) / results.length;
  const avgExpandMs = results.reduce((sum, r) => sum + r.expandMs, 0) / results.length;
  const avgTotalMs = results.reduce((sum, r) => sum + r.totalMs, 0) / results.length;
  const maxTotalMs = Math.max(...results.map((r) => r.totalMs));
  const avgReformulations = results.reduce((sum, r) => sum + r.reformulations.length, 0) / results.length;

  console.log(`  Avg reformulations: ${avgReformulations.toFixed(1)}`);
  console.log(`  Avg new docs surfaced: +${avgNewDocs.toFixed(1)}`);
  console.log(`  Avg expansion latency: ${avgExpandMs.toFixed(0)}ms`);
  console.log(`  Avg total latency: ${avgTotalMs.toFixed(0)}ms`);
  console.log(`  Max total latency: ${maxTotalMs}ms`);
  console.log(`  Expansion success rate: ${results.filter((r) => r.reformulations.length > 0).length}/${results.length}`);

  // Save results
  const outFile = path.join(
    import.meta.dirname!,
    `../evaluation/results/diag-multi-query-${Date.now()}.json`,
  );
  await fs.writeFile(outFile, JSON.stringify(results, null, 2));
  console.log(`\n📁 Results saved to: ${outFile}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
