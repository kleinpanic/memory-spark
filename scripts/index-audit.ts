/**
 * Audit the LanceDB index — show what's in it, find garbage.
 */
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { resolveConfig } from "../src/config.js";

const cfg = resolveConfig();
const backend = new LanceDBBackend(cfg);

await backend.open();
const paths = await backend.listPaths();

// Group by top-level directory
const byDir: Record<string, { chunks: number; paths: string[] }> = {};
for (const p of paths) {
  const dir = p.path.includes("/") ? p.path.split("/")[0]! : "(root)";
  if (!byDir[dir]) byDir[dir] = { chunks: 0, paths: [] };
  byDir[dir]!.chunks += p.chunkCount;
  byDir[dir]!.paths.push(p.path);
}

console.log("=== Chunks by directory ===");
Object.entries(byDir)
  .sort((a, b) => b[1].chunks - a[1].chunks)
  .forEach(([dir, info]) => {
    console.log(`${String(info.chunks).padStart(6)} chunks | ${String(info.paths.length).padStart(4)} files | ${dir}/`);
  });

// Show source distribution
const bySource: Record<string, number> = {};
for (const p of paths) {
  // We can't get source from listPaths, so infer from path patterns
  if (p.path.startsWith("capture/")) bySource["capture"] = (bySource["capture"] || 0) + p.chunkCount;
  else if (p.path.startsWith("sessions/")) bySource["sessions"] = (bySource["sessions"] || 0) + p.chunkCount;
  else bySource["memory"] = (bySource["memory"] || 0) + p.chunkCount;
}
console.log("\n=== By source (inferred) ===");
Object.entries(bySource).forEach(([s, c]) => console.log(`${s}: ${c}`));

// Identify potential garbage paths
console.log("\n=== Potential garbage (session dumps, archives, default bootstrap) ===");
const garbagePatterns = [
  /^sessions\//,
  /archive\//,
  /knowledge-base\//,
  /\.jsonl$/,
  /learnings\.md$/,
];
let garbageChunks = 0;
for (const p of paths) {
  if (garbagePatterns.some(pat => pat.test(p.path))) {
    console.log(`  ${p.chunkCount} chunks: ${p.path}`);
    garbageChunks += p.chunkCount;
  }
}
console.log(`\nTotal: ${paths.reduce((s, p) => s + p.chunkCount, 0)} chunks in ${paths.length} files`);
console.log(`Potential garbage: ${garbageChunks} chunks`);

const status = await backend.status();
console.log(`\nBackend: ${status.backend}, dims: ${status.vectorDims}, ready: ${status.ready}`);

await backend.close();
