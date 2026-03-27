import { resolveConfig } from "../src/config.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { createReranker } from "../src/rerank/reranker.js";

async function main() {
  const cfg = resolveConfig();
  console.log("Spark config:");
  console.log("  Embed URL:", cfg.embed.baseUrl);
  console.log("  Embed model:", cfg.embed.model);
  console.log("  Token present:", !!cfg.embed.apiKey);
  console.log("  Rerank URL:", cfg.rerank.baseUrl);
  console.log("  Rerank enabled:", cfg.rerank.enabled);
  
  // Test embed
  console.log("\nTesting embed...");
  const provider = await createEmbedProvider(cfg.embed);
  const vec = await provider.embedQuery("What timezone is Klein in?");
  console.log(`  ✅ Embed OK — ${vec.length} dims`);
  
  // Test reranker
  if (cfg.rerank.enabled) {
    console.log("\nTesting reranker...");
    const reranker = await createReranker(cfg.rerank);
    const probeOk = await reranker.probe();
    console.log(`  ✅ Reranker probe: ${probeOk}`);
  }
  
  console.log("\n✅ Spark fully operational!");
}
main().catch(e => { console.error("❌ FAILED:", e.message); process.exit(1); });
