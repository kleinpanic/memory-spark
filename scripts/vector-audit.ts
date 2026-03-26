import { connect } from "@lancedb/lancedb";
import path from "path";
import os from "os";

async function main() {
  const dbPath = path.join(os.homedir(), ".openclaw", "data", "memory-spark", "lancedb");
  const db = await connect(dbPath);
  const table = await db.openTable("memory_chunks");

  // Sample a few rows to check vector dims and content quality
  const sample = await table
    .query()
    .select(["vector", "text", "path", "source"])
    .limit(5)
    .toArray();

  for (const row of sample as any[]) {
    const vecLen = row.vector?.length ?? 0;
    const textLen = row.text?.length ?? 0;
    console.log(
      `path=${row.path?.slice(0, 60)} | vec_dims=${vecLen} | text_chars=${textLen} | source=${row.source}`,
    );
  }

  // Check IVF_PQ config
  const schema = await table.schema();
  console.log("\nSchema fields:", schema.fields.map((f: any) => `${f.name}: ${f.type}`).join(", "));

  // Storage per chunk
  const { execSync } = await import("child_process");
  const diskBytes = parseInt(execSync(`du -sb ${dbPath}`).toString().split("\t")[0]);
  const totalChunks = (await table.query().select(["path"]).limit(200000).toArray()).length;
  console.log(`\nStorage per chunk: ${(diskBytes / totalChunks / 1024).toFixed(1)} KB`);
  console.log(`Total disk: ${(diskBytes / 1024 / 1024).toFixed(1)} MB`);
  console.log(
    `Vector storage estimate (4096 dims * 4 bytes * ${totalChunks}): ${((4096 * 4 * totalChunks) / 1024 / 1024).toFixed(1)} MB`,
  );
}

main().catch(console.error);
