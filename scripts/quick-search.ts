import { LanceDBBackend } from "../src/storage/lancedb.js";
import { resolveConfig } from "../src/config.js";

async function main() {
  const cfg = resolveConfig({ lancedbDir: "./test-data/lancedb" } as Parameters<typeof resolveConfig>[0]);
  const b = new LanceDBBackend(cfg);
  await b.open();

  // Try FTS search (doesn't need Spark)
  const ftsResults = await b.ftsSearch("Klein timezone Blacksburg", { query: "Klein timezone" });
  console.log(`FTS results: ${ftsResults.length}`);
  for (const r of ftsResults.slice(0, 3)) {
    console.log(`  [${r.score.toFixed(3)}] ${r.chunk.agent_id}:${r.chunk.path.slice(0, 50)} — ${r.chunk.text.slice(0, 80)}...`);
  }

  // List some paths
  const paths = await b.listPaths("meta");
  console.log(`\nMeta agent paths: ${paths.length}`);
  for (const p of paths.slice(0, 5)) {
    console.log(`  ${p.path} (${p.chunkCount} chunks)`);
  }

  await b.close();
}
main();
