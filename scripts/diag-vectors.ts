import { hybridMerge, cosineSimilarity } from "../src/auto/recall.js";
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

  const queryText = "0-dimensional biomaterials show inductive properties";
  const queryVector = await embed.embedQuery(queryText);

  const vectorResults = await backend.vectorSearch(queryVector, { 
    query: queryText, maxResults: 5, minScore: 0.0, pathContains: "beir/scifact/" 
  });
  const ftsResults = await backend.ftsSearch(queryText, { 
    query: queryText, maxResults: 5, pathContains: "beir/scifact/" 
  });

  // Check raw vector data from vector search
  console.log("=== Vector Search Result [0] vector details ===");
  const v0 = vectorResults[0]!;
  console.log(`  .vector exists: ${!!v0.vector}`);
  console.log(`  .vector length: ${v0.vector?.length}`);
  console.log(`  .vector type: ${typeof v0.vector}`);
  console.log(`  .vector[0]: ${v0.vector?.[0]} (type: ${typeof v0.vector?.[0]})`);
  console.log(`  .vector is Array: ${Array.isArray(v0.vector)}`);
  
  // Check if it's a typed array or plain array
  const vec = v0.vector!;
  console.log(`  constructor: ${vec.constructor.name}`);
  console.log(`  First 3 values: ${vec[0]}, ${vec[1]}, ${vec[2]}`);
  
  // Check FTS result vector
  console.log("\n=== FTS Result [0] vector details ===");
  const f0 = ftsResults[0]!;
  console.log(`  .vector exists: ${!!f0.vector}`);
  console.log(`  .vector length: ${f0.vector?.length}`);
  console.log(`  constructor: ${f0.vector?.constructor.name}`);
  console.log(`  First 3 values: ${f0.vector?.[0]}, ${f0.vector?.[1]}, ${f0.vector?.[2]}`);
  
  // Try cosine directly
  console.log("\n=== Direct Cosine Test ===");
  if (v0.vector && f0.vector) {
    const sim = cosineSimilarity(v0.vector, f0.vector);
    console.log(`cosine(vector[0], fts[0]) = ${sim}`);
    
    // Manual dot product check
    let dot = 0, normA = 0, normB = 0;
    const a = v0.vector;
    const b = f0.vector;
    for (let i = 0; i < Math.min(10, a.length); i++) {
      console.log(`  a[${i}]=${a[i]} b[${i}]=${b[i]} a*b=${(a[i]??0)*(b[i]??0)}`);
    }
    for (let i = 0; i < a.length; i++) {
      dot += (a[i] ?? 0) * (b[i] ?? 0);
      normA += (a[i] ?? 0) * (a[i] ?? 0);
      normB += (b[i] ?? 0) * (b[i] ?? 0);
    }
    console.log(`Manual: dot=${dot}, normA=${normA}, normB=${normB}, sim=${dot/(Math.sqrt(normA)*Math.sqrt(normB))}`);
  }

  // After hybridMerge, check the vectors on the merged results
  console.log("\n=== Post-HybridMerge Vector Check ===");
  const hybrid = hybridMerge(vectorResults, ftsResults, 10);
  for (let i = 0; i < Math.min(3, hybrid.length); i++) {
    const r = hybrid[i]!;
    console.log(`  hybrid[${i}]: vector=${!!r.vector}, length=${r.vector?.length}, constructor=${r.vector?.constructor.name}, [0]=${r.vector?.[0]}`);
  }

  await backend.close();
}

main().catch(e => { console.error(e); process.exit(1); });
