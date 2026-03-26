/**
 * RAG Evaluation Benchmark
 *
 * Tests retrieval quality against known query→path pairs.
 * Computes: Recall@5, Precision@5, MRR (Mean Reciprocal Rank), freshness.
 *
 * Requires live Spark (embed endpoint) and an indexed LanceDB database.
 * Usage: npm run benchmark
 */

import { resolveConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider } from "../src/embed/provider.js";

interface TestQuery {
  query: string;
  expectedPaths: string[]; // Relative paths we expect in top-5
  expectedSnippets?: string[]; // Optional: keywords expected in results
}

const TEST_QUERIES: TestQuery[] = [
  { query: "Spark node IP address", expectedPaths: ["MEMORY.md"] },
  { query: "model for complex coding tasks", expectedPaths: ["MEMORY.md"] },
  { query: "how to restart the gateway", expectedPaths: ["AGENTS.md", "MEMORY.md"] },
  { query: "model alias incident", expectedPaths: ["memory/2026-03-02-model-alias.md"] },
  { query: "never use config.patch for agents.list", expectedPaths: ["MEMORY.md", "mistakes.md"] },
  { query: "embedding dimensions for nemotron", expectedPaths: ["MEMORY.md"] },
  { query: "vLLM port for inference", expectedPaths: ["MEMORY.md"] },
  { query: "LanceDB data directory", expectedPaths: ["MEMORY.md"] },
  { query: "auto-recall hook implementation", expectedPaths: ["MEMORY.md"] },
  { query: "capture source weighting boost", expectedPaths: ["MEMORY.md"] },
  { query: "quality gate minimum score", expectedPaths: ["MEMORY.md"] },
  { query: "rerank model nvidia nemotron", expectedPaths: ["MEMORY.md"] },
  { query: "chokidar file watcher debounce", expectedPaths: ["MEMORY.md"] },
  { query: "RRF reciprocal rank fusion", expectedPaths: ["MEMORY.md"] },
  { query: "session compaction re-index", expectedPaths: ["MEMORY.md"] },
  { query: "LanceDB merge insert upsert", expectedPaths: ["MEMORY.md"] },
  { query: "OpenClaw plugin SDK registerTool", expectedPaths: ["MEMORY.md"] },
  { query: "TypeBox schema for tools", expectedPaths: ["MEMORY.md"] },
  { query: "memory spark plugin architecture", expectedPaths: ["MEMORY.md"] },
  { query: "zero shot classifier categories", expectedPaths: ["MEMORY.md"] },
  { query: "NER entity extraction service", expectedPaths: ["MEMORY.md"] },
  { query: "agent bootstrap spam quality filter", expectedPaths: ["MEMORY.md"] },
  { query: "archive path penalty scoring", expectedPaths: ["MEMORY.md"] },
  { query: "SQLite to LanceDB migration", expectedPaths: ["MEMORY.md"] },
  { query: "vector index IVF PQ partitions", expectedPaths: ["MEMORY.md"] },
  { query: "contextual retrieval prefix embedding", expectedPaths: ["MEMORY.md"] },
  { query: "temporal decay floor 0.8", expectedPaths: ["MEMORY.md"] },
  { query: "MMR maximum marginal relevance diversity", expectedPaths: ["MEMORY.md"] },
  { query: "prompt injection detection security", expectedPaths: ["MEMORY.md"] },
  { query: "MISTAKES.md source weight multiplier", expectedPaths: ["MEMORY.md", "MISTAKES.md"] },
  { query: "EmbedQueue retry backoff concurrency", expectedPaths: ["MEMORY.md"] },
  { query: "FTS full text search BM25", expectedPaths: ["MEMORY.md"] },
];

interface EvalResult {
  query: string;
  expectedPaths: string[];
  retrievedPaths: string[];
  recallAt5: number; // fraction of expected paths found in top 5
  precisionAt5: number; // fraction of top 5 that match expected paths
  reciprocalRank: number; // 1/rank of first match (0 if not found)
  avgAge: number; // average age of top-5 results in days
}

function computeRR(retrievedPaths: string[], expectedPaths: string[]): number {
  for (let i = 0; i < retrievedPaths.length; i++) {
    const rp = retrievedPaths[i]!;
    if (expectedPaths.some((ep) => rp.includes(ep) || ep.includes(rp))) {
      return 1 / (i + 1);
    }
  }
  return 0;
}

function pathMatches(retrieved: string, expected: string): boolean {
  return (
    retrieved.includes(expected) ||
    expected.includes(retrieved) ||
    retrieved.toLowerCase().includes(expected.toLowerCase())
  );
}

async function main() {
  console.log("=== memory-spark RAG Benchmark ===\n");

  const cfg = resolveConfig();

  console.log("Connecting to LanceDB...");
  const backend = new LanceDBBackend(cfg);
  await backend.open();

  const status = await backend.status();
  console.log(`Database: ${status.chunkCount} chunks\n`);

  if (status.chunkCount === 0) {
    console.error("ERROR: No chunks in database. Run 'npm run sync' first.");
    process.exit(1);
  }

  console.log("Loading embedding provider...");
  const embed = await createEmbedProvider(cfg.embed);
  console.log(`Embed: ${embed.id}/${embed.model} (${embed.dims}d)\n`);

  const results: EvalResult[] = [];
  const N = 5; // top-N for evaluation

  console.log(`Running ${TEST_QUERIES.length} queries...\n`);

  for (const testCase of TEST_QUERIES) {
    process.stdout.write(`  Query: "${testCase.query.slice(0, 60)}"... `);

    try {
      const queryVector = await embed.embedQuery(testCase.query);

      const [vectorResults, ftsResults] = await Promise.all([
        backend
          .vectorSearch(queryVector, {
            query: testCase.query,
            maxResults: N * 2,
            minScore: 0.0,
          })
          .catch(() => []),
        backend
          .ftsSearch(testCase.query, {
            query: testCase.query,
            maxResults: N * 2,
          })
          .catch(() => []),
      ]);

      // Simple RRF merge
      const seen = new Map<string, { path: string; score: number; updatedAt: string }>();
      vectorResults.forEach((r, rank) => {
        const existing = seen.get(r.chunk.id);
        const rrfScore = (existing?.score ?? 0) + 1 / (60 + rank);
        seen.set(r.chunk.id, {
          path: r.chunk.path,
          score: rrfScore,
          updatedAt: r.chunk.updated_at,
        });
      });
      ftsResults.forEach((r, rank) => {
        const existing = seen.get(r.chunk.id);
        const rrfScore = (existing?.score ?? 0) + 1 / (60 + rank);
        seen.set(r.chunk.id, {
          path: r.chunk.path,
          score: rrfScore,
          updatedAt: r.chunk.updated_at,
        });
      });

      const sorted = Array.from(seen.values())
        .sort((a, b) => b.score - a.score)
        .slice(0, N);

      const retrievedPaths = sorted.map((r) => r.path);

      // Recall@N: fraction of expected paths found in top N
      const found = testCase.expectedPaths.filter((ep) =>
        retrievedPaths.some((rp) => pathMatches(rp, ep)),
      );
      const recallAt5 =
        testCase.expectedPaths.length > 0 ? found.length / testCase.expectedPaths.length : 0;

      // Precision@N: fraction of top N that match any expected path
      const matchedRetrieved = retrievedPaths.filter((rp) =>
        testCase.expectedPaths.some((ep) => pathMatches(rp, ep)),
      );
      const precisionAt5 =
        retrievedPaths.length > 0 ? matchedRetrieved.length / retrievedPaths.length : 0;

      // Reciprocal Rank
      const rr = computeRR(retrievedPaths, testCase.expectedPaths);

      // Average age of top-N results
      const now = Date.now();
      const avgAge =
        sorted.length > 0
          ? sorted.reduce((sum, r) => {
              const updatedAt = r.updatedAt ? new Date(r.updatedAt).getTime() : now;
              return sum + (now - updatedAt) / (86400 * 1000);
            }, 0) / sorted.length
          : 0;

      results.push({
        query: testCase.query,
        expectedPaths: testCase.expectedPaths,
        retrievedPaths,
        recallAt5,
        precisionAt5,
        reciprocalRank: rr,
        avgAge,
      });

      const symbol = rr > 0 ? "✓" : "✗";
      console.log(
        `${symbol} R@5=${recallAt5.toFixed(2)} P@5=${precisionAt5.toFixed(2)} RR=${rr.toFixed(2)}`,
      );
    } catch (err) {
      console.log(`ERROR: ${err}`);
      results.push({
        query: testCase.query,
        expectedPaths: testCase.expectedPaths,
        retrievedPaths: [],
        recallAt5: 0,
        precisionAt5: 0,
        reciprocalRank: 0,
        avgAge: 0,
      });
    }
  }

  await backend.close();

  // Aggregate metrics
  const validResults = results.filter((r) => r.retrievedPaths.length > 0);
  const mrr = validResults.reduce((sum, r) => sum + r.reciprocalRank, 0) / results.length;
  const avgRecall = results.reduce((sum, r) => sum + r.recallAt5, 0) / results.length;
  const avgPrecision = results.reduce((sum, r) => sum + r.precisionAt5, 0) / results.length;
  const avgAge = validResults.reduce((sum, r) => sum + r.avgAge, 0) / validResults.length;
  const found = results.filter((r) => r.reciprocalRank > 0).length;

  console.log("\n=== Benchmark Results ===");
  console.log(`Total queries:     ${results.length}`);
  console.log(
    `Found in top-5:    ${found}/${results.length} (${((found / results.length) * 100).toFixed(0)}%)`,
  );
  console.log(`MRR:               ${mrr.toFixed(3)}`);
  console.log(`Avg Recall@5:      ${avgRecall.toFixed(3)}`);
  console.log(`Avg Precision@5:   ${avgPrecision.toFixed(3)}`);
  console.log(`Avg result age:    ${avgAge.toFixed(1)} days`);

  // Show failures
  const failures = results.filter((r) => r.reciprocalRank === 0);
  if (failures.length > 0) {
    console.log(`\nMissed queries (${failures.length}):`);
    for (const f of failures.slice(0, 10)) {
      console.log(`  ✗ "${f.query.slice(0, 60)}" — expected: [${f.expectedPaths.join(", ")}]`);
      if (f.retrievedPaths.length > 0) {
        console.log(`      retrieved: [${f.retrievedPaths.slice(0, 3).join(", ")}]`);
      }
    }
  }

  // JSON output
  const jsonOutput = {
    timestamp: new Date().toISOString(),
    metrics: { mrr, avgRecall, avgPrecision, foundInTop5: found, total: results.length, avgAge },
    queries: results,
  };

  const outputPath = "benchmark-results.json";
  const { writeFile } = await import("node:fs/promises");
  await writeFile(outputPath, JSON.stringify(jsonOutput, null, 2), "utf-8");
  console.log(`\nDetailed results written to ${outputPath}`);

  process.exit(mrr >= 0.5 ? 0 : 1); // Exit 1 if MRR below threshold
}

main().catch((err) => {
  console.error("Benchmark failed:", err);
  process.exit(1);
});
