import * as lancedb from "@lancedb/lancedb";
import * as fs from "node:fs/promises";
import { execSync } from "node:child_process";

const HOME = process.env.HOME;
const DB_PATH = `${HOME}/.openclaw/data/memory-spark/lancedb`;

async function main() {
  const db = await lancedb.connect(DB_PATH);
  const tbl = await db.openTable("memory_chunks");
  
  // Quick FTS search for one of the missing facts to prove it's in the DB
  const results = await tbl.query()
    .fullTextSearch("memory_chunks")
    .limit(5)
    .toArray();
    
  console.log("FTS for 'memory_chunks':");
  results.forEach(r => console.log(" -", r.path, "|", r.text.substring(0, 80).replace(/\n/g, " ")));
  
  const results2 = await tbl.query()
    .fullTextSearch("Blackwell")
    .limit(5)
    .toArray();
    
  console.log("\nFTS for 'Blackwell':");
  results2.forEach(r => console.log(" -", r.path, "|", r.text.substring(0, 80).replace(/\n/g, " ")));
}
main();
