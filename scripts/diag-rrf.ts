import { hybridMerge } from "../src/auto/recall.js";
import { resolveConfig } from "../src/config.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { EmbedQueue } from "../src/embed/queue.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";

async function main() {
  const lancedbDir = process.env.BEIR_LANCEDB_DIR || `${process.env.HOME}/.openclaw/data/testDbBEIR/lancedb`;
  const cfg = resolveConfig({ lancedbDir } as any);
  const backend = new LanceDBBackend(cfg);
  await backend.open();
  const provider = await createEmbedProvider(cfg.embed);
  const embed = new EmbedQueue(provider, { concurrency: 1, maxRetries: 2, timeoutMs: 30000 });

  // Use a query where we know vector and FTS disagree
  const queryText = "0-dimensional biomaterials show inductive properties";
  const queryVector = await embed.embedQuery(queryText);

  const vectorResults = await backend.vectorSearch(queryVector, { 
    query: queryText, maxResults: 40, minScore: 0.0, pathContains: "beir/scifact/" 
  });
  const ftsResults = await backend.ftsSearch(queryText, { 
    query: queryText, maxResults: 40, pathContains: "beir/scifact/" 
  });

  // Show raw ranks for both lists
  console.log("=== Vector Top 10 (cosine similarity) ===");
  for (let i = 0; i < 10; i++) {
    const r = vectorResults[i]!;
    const id = r.chunk.id.replace("beir-scifact-", "");
    console.log(`  V-rank ${i}: id=${id} score=${r.score.toFixed(4)}`);
  }

  console.log("\n=== FTS Top 10 (BM25 sigmoid) ===");
  for (let i = 0; i < 10; i++) {
    const r = ftsResults[i]!;
    const id = r.chunk.id.replace("beir-scifact-", "");
    console.log(`  F-rank ${i}: id=${id} score=${r.score.toFixed(4)}`);
  }

  // Find overlap between top-20 of each
  const vSet = new Set(vectorResults.slice(0, 20).map(r => r.chunk.id));
  const fSet = new Set(ftsResults.slice(0, 20).map(r => r.chunk.id));
  const overlap = [...vSet].filter(id => fSet.has(id));
  console.log(`\nOverlap in top-20: ${overlap.length} docs`);
  for (const id of overlap) {
    const vRank = vectorResults.findIndex(r => r.chunk.id === id);
    const fRank = ftsResults.findIndex(r => r.chunk.id === id);
    const shortId = id.replace("beir-scifact-", "");
    console.log(`  ${shortId}: V-rank=${vRank} F-rank=${fRank}`);
  }

  // Show hybrid merge result with provenance
  console.log("\n=== Hybrid (RRF) Top 10 ===");
  const hybrid = hybridMerge(vectorResults, ftsResults, 20);
  for (let i = 0; i < 10; i++) {
    const r = hybrid[i]!;
    const id = r.chunk.id.replace("beir-scifact-", "");
    const vRank = vectorResults.findIndex(v => v.chunk.id === r.chunk.id);
    const fRank = ftsResults.findIndex(f => f.chunk.id === r.chunk.id);
    const inVector = vRank >= 0 ? `V-rank=${vRank}` : "V=MISS";
    const inFts = fRank >= 0 ? `F-rank=${fRank}` : "F=MISS";
    console.log(`  H-rank ${i}: id=${id} rrf=${r.score.toFixed(4)} ${inVector} ${inFts}`);
  }

  // Now show the vector-only top 10 vs hybrid top 10 overlap
  const vectorTop10 = new Set(vectorResults.slice(0, 10).map(r => r.chunk.id));
  const hybridTop10 = new Set(hybrid.slice(0, 10).map(r => r.chunk.id));
  const vectorOnly = [...vectorTop10].filter(id => !hybridTop10.has(id));
  const hybridOnly = [...hybridTop10].filter(id => !vectorTop10.has(id));
  console.log(`\nVector-only top-10 docs lost in hybrid: ${vectorOnly.length}`);
  for (const id of vectorOnly) {
    const vRank = vectorResults.findIndex(r => r.chunk.id === id);
    const fRank = ftsResults.findIndex(r => r.chunk.id === id);
    console.log(`  ${id.replace("beir-scifact-","")} V-rank=${vRank} F-rank=${fRank >= 0 ? fRank : "MISS"}`);
  }
  console.log(`Hybrid-only top-10 docs not in vector top-10: ${hybridOnly.length}`);
  for (const id of hybridOnly) {
    const vRank = vectorResults.findIndex(r => r.chunk.id === id);
    const fRank = ftsResults.findIndex(r => r.chunk.id === id);
    console.log(`  ${id.replace("beir-scifact-","")} V-rank=${vRank >= 0 ? vRank : "MISS"} F-rank=${fRank}`);
  }

  await backend.close();
}

main().catch(e => { console.error(e); process.exit(1); });
