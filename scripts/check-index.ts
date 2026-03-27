import { LanceDBBackend } from "../src/storage/lancedb.js";
import { resolveConfig } from "../src/config.js";

async function main() {
  const dir = process.argv[2] ?? "./test-data/lancedb";
  const cfg = resolveConfig({ lancedbDir: dir } as Parameters<typeof resolveConfig>[0]);
  const b = new LanceDBBackend(cfg);
  await b.open();
  const s = await b.status();
  console.log("Status:", JSON.stringify(s));
  const agents = await b.discoverAgents();
  console.log("Agents:", agents.join(", "));
  const pools = await b.poolStats();
  console.log("Pool stats:");
  for (const p of pools) console.log(`  ${p.pool}: ${p.chunkCount} chunks`);
  await b.close();
}
main();
