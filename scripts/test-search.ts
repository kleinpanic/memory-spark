/**
 * Direct search test — bypasses the plugin to isolate the issue.
 * Embeds a query via Spark, then does vector search on LanceDB.
 */
import * as lancedb from "@lancedb/lancedb";
import path from "node:path";
import os from "node:os";
import fs from "node:fs";

const DB = path.join(os.homedir(), ".openclaw/data/memory-spark/lancedb");
const EMBED_URL = "http://10.99.1.1:18091/v1/embeddings";
const MODEL = "nvidia/llama-embed-nemotron-8b";

// Load token from .env
const envFile = path.join(os.homedir(), ".openclaw/.env");
const envContent = fs.readFileSync(envFile, "utf-8");
const tokenMatch = envContent.match(/SPARK_BEARER_TOKEN=["']?([^"'\s\n]+)/);
const TOKEN = tokenMatch?.[1] ?? "none";

async function embed(text: string): Promise<number[]> {
  const resp = await fetch(EMBED_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${TOKEN}`,
    },
    body: JSON.stringify({ input: text, model: MODEL }),
  });
  if (!resp.ok) throw new Error(`Embed failed: ${resp.status} ${await resp.text()}`);
  const data = await resp.json() as any;
  return data.data[0].embedding;
}

async function main() {
  const query = process.argv[2] ?? "voice bridge setup configuration";
  console.log(`Query: "${query}"`);
  console.log(`Token: ${TOKEN.slice(0, 6)}...`);

  // 1. Embed the query
  console.log("\n--- Embedding query ---");
  const vec = await embed(query);
  console.log(`Vector: ${vec.length} dims, first 3: [${vec.slice(0, 3).map(v => v.toFixed(4))}]`);

  // 2. Open table
  const db = await lancedb.connect(DB);
  const t = await db.openTable("memory_chunks");
  const count = await t.countRows();
  console.log(`\nTable rows: ${count}`);

  // 3. Vector search (raw, no filter)
  console.log("\n--- Vector search (top 5, no filter) ---");
  const results = await t.vectorSearch(vec)
    .limit(5)
    .toArray();

  if (results.length === 0) {
    console.log("NO RESULTS — vector search returned empty");
  } else {
    for (const r of results) {
      console.log(`  score=${(r._distance ?? -1).toFixed(4)} path=${r.path} text=${String(r.text).slice(0, 80)}...`);
    }
  }

  // 4. FTS search
  console.log("\n--- FTS search ---");
  try {
    const fts = await t.search(query, "text").limit(5).toArray();
    if (fts.length === 0) {
      console.log("NO FTS RESULTS");
    } else {
      for (const r of fts) {
        console.log(`  score=${(r._score ?? r._relevance_score ?? -1).toFixed?.(4) ?? "N/A"} path=${r.path} text=${String(r.text).slice(0, 80)}...`);
      }
    }
  } catch (e: any) {
    console.log(`FTS error: ${e.message}`);
  }

  t.close();
}

main().catch(console.error);
