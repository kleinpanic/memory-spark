import { hybridMerge, mmrRerank, cosineSimilarity } from "../src/auto/recall.js";
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

  // Use a single SciFact query
  const queryText = "0-dimensional biomaterials show inductive properties";
  const queryVector = await embed.embedQuery(queryText);

  console.log("=== STAGE 1: Vector Search ===");
  const vectorResults = await backend.vectorSearch(queryVector, { 
    query: queryText, maxResults: 40, minScore: 0.0, pathContains: "beir/scifact/" 
  });
  console.log(`Vector results: ${vectorResults.length}`);
  console.log(`Vector[0] has vector: ${!!vectorResults[0]?.vector}, length: ${vectorResults[0]?.vector?.length}`);
  
  console.log("\n=== STAGE 2: FTS Search ===");
  const ftsResults = await backend.ftsSearch(queryText, { 
    query: queryText, maxResults: 40, pathContains: "beir/scifact/" 
  });
  console.log(`FTS results: ${ftsResults.length}`);
  console.log(`FTS[0] has vector: ${!!ftsResults[0]?.vector}, length: ${ftsResults[0]?.vector?.length}`);

  console.log("\n=== STAGE 3: Hybrid Merge ===");
  const hybrid = hybridMerge(vectorResults, ftsResults, 20);
  console.log(`Hybrid results: ${hybrid.length}`);
  const vectorPresent = hybrid.filter(r => r.vector && r.vector.length > 0).length;
  const vectorMissing = hybrid.filter(r => !r.vector || r.vector.length === 0).length;
  console.log(`With vectors: ${vectorPresent}, Without vectors: ${vectorMissing}`);
  
  // Check if ALL have vectors (needed for cosine MMR)
  const hasAllVectors = hybrid.every(r => r.vector && r.vector.length > 0);
  console.log(`All have vectors (cosine MMR eligible): ${hasAllVectors}`);

  // Compute pairwise cosine similarities for top 10
  console.log("\n=== STAGE 4: Pairwise Cosine Similarity (top 10) ===");
  const top10 = hybrid.slice(0, 10);
  if (hasAllVectors) {
    for (let i = 0; i < Math.min(5, top10.length); i++) {
      for (let j = i+1; j < Math.min(5, top10.length); j++) {
        const sim = cosineSimilarity(top10[i]!.vector!, top10[j]!.vector!);
        console.log(`  sim(${i},${j}) = ${sim.toFixed(4)} | ids: ${top10[i]!.chunk.id.slice(-8)} vs ${top10[j]!.chunk.id.slice(-8)}`);
      }
    }
  }

  // Run MMR and compare
  console.log("\n=== STAGE 5: MMR Comparison ===");
  const mmr09 = mmrRerank([...hybrid], 10, 0.9);
  const mmr07 = mmrRerank([...hybrid], 10, 0.7);
  const mmr05 = mmrRerank([...hybrid], 10, 0.5);
  const noMmr = hybrid.slice(0, 10);

  console.log("No MMR:   ", noMmr.map(r => r.chunk.id.slice(-8)).join(", "));
  console.log("MMR λ=0.9:", mmr09.map(r => r.chunk.id.slice(-8)).join(", "));
  console.log("MMR λ=0.7:", mmr07.map(r => r.chunk.id.slice(-8)).join(", "));
  console.log("MMR λ=0.5:", mmr05.map(r => r.chunk.id.slice(-8)).join(", "));
  
  const same09 = noMmr.every((r, i) => r.chunk.id === mmr09[i]?.chunk.id);
  const same07 = noMmr.every((r, i) => r.chunk.id === mmr07[i]?.chunk.id);
  const same05 = noMmr.every((r, i) => r.chunk.id === mmr05[i]?.chunk.id);
  console.log(`\nλ=0.9 same as no-MMR: ${same09}`);
  console.log(`λ=0.7 same as no-MMR: ${same07}`);
  console.log(`λ=0.5 same as no-MMR: ${same05}`);

  // Score distribution
  console.log("\n=== RRF Score Distribution (top 20) ===");
  for (let i = 0; i < hybrid.length; i++) {
    console.log(`  rank ${i}: score=${hybrid[i]!.score.toFixed(6)} id=${hybrid[i]!.chunk.id.slice(-8)}`);
  }

  await backend.close();
}

main().catch(e => { console.error(e); process.exit(1); });
