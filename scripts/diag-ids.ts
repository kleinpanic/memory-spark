#!/usr/bin/env npx tsx
/**
 * Diagnostic: compare doc IDs in LanceDB vs qrels to find the mismatch
 * causing 0.0000 NDCG@10 scores.
 */
import fs from "node:fs/promises";
import path from "node:path";
import { resolveConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { EmbedQueue } from "../src/embed/queue.js";

async function main() {
  const dataset = process.argv[2] || "scifact";
  const lancedbDir = process.env.BEIR_LANCEDB_DIR || "/home/node/.openclaw/data/testDbBEIR/lancedb";

  console.log(`\n══ ID Diagnostic: ${dataset} ══`);
  console.log(`DB: ${lancedbDir}\n`);

  // 1. Load qrels
  const qrelsFile = path.join(import.meta.dirname!, "../evaluation/beir-datasets", dataset, "qrels/test.tsv");
  const qrelsContent = await fs.readFile(qrelsFile, "utf-8");
  const qrelDocIds = new Set<string>();
  const qrelQueryIds = new Set<string>();
  for (const line of qrelsContent.trim().split("\n")) {
    if (line.startsWith("query-id")) continue;
    const [qid, did] = line.split("\t");
    if (qid) qrelQueryIds.add(qid);
    if (did) qrelDocIds.add(did);
  }
  console.log(`Qrels: ${qrelQueryIds.size} queries, ${qrelDocIds.size} unique doc IDs`);
  const sampleQrelIds = [...qrelDocIds].slice(0, 10);
  console.log(`  Sample qrel doc IDs: ${sampleQrelIds.join(", ")}`);

  // 2. Check DB row count
  const cfg = resolveConfig({ lancedbDir } as any);
  const backend = new LanceDBBackend(cfg);
  await backend.open();

  // Access internal table for diagnostics
  const table = (backend as any)._table;
  if (table) {
    const count = await table.countRows();
    console.log(`\nDB row count: ${count}`);

    // Sample some raw rows to see ID format
    const sampleRows = await table.query().limit(5).execute();
    console.log(`\nSample DB rows:`);
    for (const row of sampleRows) {
      console.log(`  id=${row.id} | path=${row.path} | source=${row.source}`);
    }

    // Search for beir-specific rows
    const beirRows = await table.query()
      .where(`path LIKE '%beir/${dataset}%'`)
      .limit(5)
      .execute();
    console.log(`\nBEIR ${dataset} rows (path filter):`);
    if (beirRows.length === 0) {
      console.log(`  ⚠️  NO ROWS with path containing 'beir/${dataset}'`);
      // Try without dataset prefix
      const anyBeirRows = await table.query()
        .where(`path LIKE '%beir%'`)
        .limit(5)
        .execute();
      console.log(`\n  Any 'beir' rows:`);
      for (const row of anyBeirRows) {
        console.log(`    id=${row.id} | path=${row.path}`);
      }
    } else {
      for (const row of beirRows) {
        console.log(`  id=${row.id} | path=${row.path}`);
      }
    }

    // Count beir rows
    const beirCount = await table.query()
      .where(`path LIKE '%beir%'`)
      .execute();
    console.log(`\nTotal beir rows in DB: ${beirCount.length}`);
  }

  // 3. Do a vector search to see what the benchmark would get
  const provider = await createEmbedProvider(cfg.embed);
  const embed = new EmbedQueue(provider, { concurrency: 1, maxRetries: 2, timeoutMs: 30000 });

  const queriesFile = path.join(import.meta.dirname!, "../evaluation/beir-datasets", dataset, "queries.jsonl");
  const firstQuery = JSON.parse((await fs.readFile(queriesFile, "utf-8")).split("\n")[0]!);
  console.log(`\nTest query: "${firstQuery.text.slice(0, 80)}..." (ID: ${firstQuery._id})`);

  // Embed and search
  const queryVec = await embed.embedQuery(firstQuery.text);

  const results = await backend.vectorSearch(queryVec, {
    query: firstQuery.text,
    maxResults: 5,
    minScore: 0.0,
    pathContains: `beir/${dataset}/`,
  });
  console.log(`\nVector search results (pathContains='beir/${dataset}/'): ${results.length}`);
  for (const r of results.slice(0, 5)) {
    const stripped = r.chunk.id.replace(/^beir-(scifact|nfcorpus|fiqa)-/, "");
    const inQrels = qrelDocIds.has(stripped) ? "✅ IN QRELS" : "❌ NOT IN QRELS";
    console.log(`  id=${r.chunk.id} stripped=${stripped} score=${r.score.toFixed(4)} ${inQrels}`);
  }

  // Also search without filter
  const resultsNoFilter = await backend.vectorSearch(queryVec, {
    query: firstQuery.text,
    maxResults: 5,
    minScore: 0.0,
  });
  console.log(`\nVector search results (no filter): ${resultsNoFilter.length}`);
  for (const r of resultsNoFilter.slice(0, 5)) {
    const stripped = r.chunk.id.replace(/^beir-(scifact|nfcorpus|fiqa)-/, "");
    const inQrels = qrelDocIds.has(stripped) ? "✅ IN QRELS" : "❌ NOT IN QRELS";
    console.log(`  id=${r.chunk.id} stripped=${stripped} score=${r.score.toFixed(4)} ${inQrels}`);
  }

  // Check what the expected results for this query are
  const expectedDocs = firstQuery._id in Object.fromEntries(
    qrelsContent.trim().split("\n").filter(l => !l.startsWith("query-id")).map(l => {
      const [qid, did, score] = l.split("\t");
      return [qid, { did, score }];
    })
  );
  
  // Get all expected docs for this query
  const qrelEntries: { did: string; score: number }[] = [];
  for (const line of qrelsContent.trim().split("\n")) {
    if (line.startsWith("query-id")) continue;
    const [qid, did, , score] = line.split("\t");
    if (qid === firstQuery._id && did) {
      qrelEntries.push({ did, score: parseInt(score ?? "0", 10) });
    }
  }
  console.log(`\nExpected docs for query ${firstQuery._id}: ${qrelEntries.map(e => `${e.did}(${e.score})`).join(", ") || "NONE"}`);

  await backend.close();
  console.log("\n══ Diagnostic complete ══\n");
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
