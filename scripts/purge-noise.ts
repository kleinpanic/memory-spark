import { connect } from "@lancedb/lancedb";
import path from "node:path";
import os from "node:os";

async function main() {
  const dbPath = path.join(os.homedir(), ".openclaw", "data", "memory-spark", "lancedb");
  const db = await connect(dbPath);
  const table = await db.openTable("memory_chunks");

  const allRows = await table.query().select(["path"]).limit(200000).toArray();
  console.log(`Total chunks before purge: ${allRows.length}`);

  // Categorize noise
  const zhRows = allRows.filter(
    (r: any) => r.path?.includes("/zh-CN/") || r.path?.includes("/zh-TW/"),
  );
  const installedDups = allRows.filter((r: any) => r.path?.includes("installed-v"));
  console.log(`zh-CN/TW chunks: ${zhRows.length}`);
  console.log(`installed-v* duplicate chunks: ${installedDups.length}`);

  // Delete in sequence with retries
  for (const [label, predicate] of [
    ["zh-CN", "path LIKE '%/zh-CN/%'"],
    ["zh-TW", "path LIKE '%/zh-TW/%'"],
    ["installed-v", "path LIKE '%installed-v%'"],
  ] as const) {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        await table.delete(predicate);
        console.log(`✓ Deleted ${label} chunks`);
        break;
      } catch (err: any) {
        if (attempt < 2 && err.message?.includes("Commit conflict")) {
          console.log(`  ⟳ Retry ${label} (conflict)...`);
          await new Promise((r) => setTimeout(r, 2000));
        } else {
          console.error(`✗ Failed to delete ${label}: ${err.message?.slice(0, 100)}`);
        }
      }
    }
  }

  const remaining = await table.query().select(["path"]).limit(200000).toArray();
  console.log(`\nRemaining chunks after purge: ${remaining.length}`);
  console.log(`Removed: ${allRows.length - remaining.length} chunks`);
}

main().catch(console.error);
