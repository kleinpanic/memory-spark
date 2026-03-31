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

  const queryVector = await embed.embedQuery("test");
  const results = await backend.vectorSearch(queryVector, { 
    query: "test", maxResults: 1, minScore: 0.0, pathContains: "beir/scifact/" 
  });

  const vec = results[0]!.vector!;
  console.log("Type:", typeof vec);
  console.log("Constructor:", vec.constructor.name);
  console.log("Is Array:", Array.isArray(vec));
  console.log("Length:", vec.length);
  console.log("[0]:", vec[0]);
  
  // Try conversion methods
  console.log("\n--- Conversion attempts ---");
  
  // Array.from
  try {
    const arr = Array.from(vec as any);
    console.log("Array.from: length=" + arr.length + " [0]=" + arr[0] + " type=" + typeof arr[0]);
  } catch(e) { console.log("Array.from failed:", e); }
  
  // Spread
  try {
    const arr = [...(vec as any)];
    console.log("Spread: length=" + arr.length + " [0]=" + arr[0]);
  } catch(e) { console.log("Spread failed:", e); }
  
  // toArray()
  try {
    const arr = (vec as any).toArray();
    console.log("toArray(): length=" + arr.length + " [0]=" + arr[0] + " constructor=" + arr.constructor.name);
  } catch(e) { console.log("toArray() failed:", e); }

  // toJSON()
  try {
    const j = (vec as any).toJSON();
    console.log("toJSON(): type=" + typeof j + " isArray=" + Array.isArray(j) + " length=" + j?.length + " [0]=" + j?.[0]);
  } catch(e) { console.log("toJSON() failed:", e); }

  // get()
  try {
    const v = (vec as any).get(0);
    console.log("get(0):", v);
  } catch(e) { console.log("get(0) failed:", e); }

  await backend.close();
}

main().catch(e => { console.error(e); process.exit(1); });
