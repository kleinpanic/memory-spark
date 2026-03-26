import * as lancedb from "@lancedb/lancedb";
import path from "node:path";
import os from "node:os";

const DB = path.join(os.homedir(), ".openclaw/data/memory-spark/lancedb");

async function main() {
  const db = await lancedb.connect(DB);
  const t = await db.openTable("memory_chunks");
  const s = await t.schema();
  console.log("=== Schema ===");
  for (const f of s.fields) {
    console.log(`  ${f.name}: ${f.type} (nullable=${f.nullable})`);
  }
  const rows = await t.countRows();
  console.log(`\nRows: ${rows}`);
  const indices = await t.listIndices();
  console.log(`Indices: ${indices.length > 0 ? JSON.stringify(indices, null, 2) : "none yet (table may be too small)"}`);

  // Sample a row to verify data
  const sample = await t.query().limit(1).toArray();
  if (sample.length > 0) {
    const r = sample[0];
    console.log("\n=== Sample Row ===");
    console.log(`  id: ${r.id}`);
    console.log(`  path: ${r.path}`);
    console.log(`  source: ${r.source}`);
    console.log(`  content_type: ${r.content_type}`);
    console.log(`  quality_score: ${r.quality_score}`);
    console.log(`  token_count: ${r.token_count}`);
    console.log(`  parent_heading: ${r.parent_heading}`);
    console.log(`  vector dims: ${Array.isArray(r.vector) ? r.vector.length : "N/A"}`);
    console.log(`  text preview: ${String(r.text).slice(0, 120)}...`);
  }
  t.close();
}

main().catch(console.error);
