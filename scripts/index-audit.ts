import { connect } from "@lancedb/lancedb";
import path from "path";
import os from "os";

async function main() {
  const dbPath = path.join(os.homedir(), ".openclaw", "data", "memory-spark", "lancedb");
  const db = await connect(dbPath);
  const table = await db.openTable("memory_chunks");

  const rows = await table.query().select(["path", "source", "updated_at"]).limit(200000).toArray();
  console.log("Total chunks:", rows.length);

  // Source distribution
  const sources = new Map<string, number>();
  for (const r of rows as any[]) {
    sources.set(r.source, (sources.get(r.source) ?? 0) + 1);
  }
  console.log("\nSource distribution:");
  for (const [k, v] of [...sources.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`  ${k}: ${v} (${((v / rows.length) * 100).toFixed(1)}%)`);
  }

  // Age distribution
  const now = Date.now();
  let lt1d = 0,
    lt7d = 0,
    lt30d = 0,
    older = 0;
  for (const r of rows as any[]) {
    const age = now - new Date(r.updated_at).getTime();
    if (age < 86400000) lt1d++;
    else if (age < 7 * 86400000) lt7d++;
    else if (age < 30 * 86400000) lt30d++;
    else older++;
  }
  console.log("\nAge distribution:");
  console.log("  <1 day:", lt1d);
  console.log("  1-7 days:", lt7d);
  console.log("  7-30 days:", lt30d);
  console.log("  >30 days:", older);

  // Remaining noise
  const zhCount = (rows as any[]).filter((r) => r.path?.includes("/zh-CN/")).length;
  console.log("\nRemaining zh-CN chunks:", zhCount);

  const sessionDumps = (rows as any[]).filter((r) => r.path?.match(/2026-0[12]-/)).length;
  console.log("Old session dumps (Jan/Feb):", sessionDumps);

  // Disk usage
  const { execSync } = await import("child_process");
  const diskUsage = execSync(`du -sh ${dbPath}`).toString().trim();
  console.log("\nLanceDB disk usage:", diskUsage);

  // Index info
  const indices = await table.listIndices();
  console.log("\nIndices:", JSON.stringify(indices, null, 2));
}

main().catch(console.error);
