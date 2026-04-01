import { LanceDBBackend } from "../src/storage/lancedb.js";
import { resolveConfig } from "../src/config.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { EmbedQueue } from "../src/embed/queue.js";

async function main() {
  const cfg = resolveConfig({ lancedbDir: `${process.env.HOME}/.openclaw/data/testDbBEIR/lancedb` } as any);
  const backend = new LanceDBBackend(cfg);
  await backend.open();
  const provider = await createEmbedProvider(cfg.embed);
  const embed = new EmbedQueue(provider, { concurrency: 1, maxRetries: 2, timeoutMs: 30000 });
  
  const vec = await embed.embedQuery("Aspirin inhibits the production of PGE2.");
  const results = await backend.vectorSearch(vec, { query: "test", maxResults: 5, minScore: 0.0, pathContains: "beir/scifact/" });
  
  for (const r of results) {
    const stripped = r.chunk.id.replace(/^beir-[^-]+-/, "");
    console.log(`id: ${r.chunk.id} | stripped: ${stripped} | score: ${r.score.toFixed(4)}`);
  }
  
  // Also check FTS
  const fts = await backend.ftsSearch("Aspirin inhibits PGE2 production", { query: "test", maxResults: 5, pathContains: "beir/scifact/" });
  console.log("\nFTS results:");
  for (const r of fts) {
    const stripped = r.chunk.id.replace(/^beir-[^-]+-/, "");
    console.log(`id: ${r.chunk.id} | stripped: ${stripped} | score: ${r.score.toFixed(4)}`);
  }
}
main();
