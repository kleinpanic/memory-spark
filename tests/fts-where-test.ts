/**
 * Test: FTS + WHERE clause — the bug that caused the 3x overfetch workaround.
 *
 * On LanceDB 0.14.x, running FTS search with .where() caused Arrow panic:
 *   "ExecNode(Take): could not cast column"
 *
 * LanceDB 0.27.x should fix this. If this test passes, the workaround
 * in lancedb.ts can be replaced with direct FTS + WHERE.
 */

import * as lancedb from "@lancedb/lancedb";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";

const TEST_DIR = path.join(os.tmpdir(), `fts-where-test-${Date.now()}`);

async function main() {
  const { readFileSync } = await import("node:fs");
  const ver = JSON.parse(
    readFileSync("node_modules/@lancedb/lancedb/package.json", "utf8"),
  ).version;
  console.log(`LanceDB version: ${ver}`);
  console.log(`Test dir: ${TEST_DIR}\n`);

  await fs.mkdir(TEST_DIR, { recursive: true });
  const db = await lancedb.connect(TEST_DIR);

  // Create table with metadata columns
  const data = [
    {
      id: "1",
      text: "The quick brown fox jumps over the lazy dog",
      vector: new Array(16).fill(0.1),
      agent_id: "meta",
      pool: "agent_memory",
    },
    {
      id: "2",
      text: "OpenClaw is an AI agent framework for autonomous systems",
      vector: new Array(16).fill(0.2),
      agent_id: "dev",
      pool: "agent_memory",
    },
    {
      id: "3",
      text: "Never use gemini flash for complex coding tasks",
      vector: new Array(16).fill(0.3),
      agent_id: "meta",
      pool: "shared_mistakes",
    },
    {
      id: "4",
      text: "Klein prefers concise responses over verbose explanations",
      vector: new Array(16).fill(0.4),
      agent_id: "meta",
      pool: "shared_rules",
    },
    {
      id: "5",
      text: "LanceDB supports BM25 full text search via FTS index",
      vector: new Array(16).fill(0.5),
      agent_id: "dev",
      pool: "reference_library",
    },
  ];

  const table = await db.createTable("memory_chunks", data, { mode: "overwrite" });

  // Create FTS index
  await table.createIndex("text", { config: lancedb.Index.fts() });

  console.log("=== Test 1: FTS without WHERE (should always work) ===");
  try {
    const results = await table.search("OpenClaw", "fts", "text").limit(5).toArray();
    console.log(`✅ PASS — ${results.length} results`);
    for (const r of results) {
      console.log(`  id=${r.id} agent=${r.agent_id} pool=${r.pool} score=${r._score}`);
    }
  } catch (err) {
    console.log(`❌ FAIL — ${err}`);
  }

  console.log("\n=== Test 2: FTS + WHERE agent_id (THE BUG) ===");
  try {
    const results = await table
      .search("OpenClaw", "fts", "text")
      .where("agent_id = 'dev'")
      .limit(5)
      .toArray();
    console.log(`✅ PASS — ${results.length} results (FTS+WHERE agent_id works!)`);
    for (const r of results) {
      console.log(`  id=${r.id} agent=${r.agent_id} pool=${r.pool} score=${r._score}`);
    }
  } catch (err) {
    console.log(`❌ FAIL — FTS+WHERE still broken: ${err}`);
  }

  console.log("\n=== Test 3: FTS + WHERE pool (metadata filter) ===");
  try {
    const results = await table
      .search("coding", "fts", "text")
      .where("pool = 'shared_mistakes'")
      .limit(5)
      .toArray();
    console.log(`✅ PASS — ${results.length} results (FTS+WHERE pool works!)`);
    for (const r of results) {
      console.log(`  id=${r.id} agent=${r.agent_id} pool=${r.pool} score=${r._score}`);
    }
  } catch (err) {
    console.log(`❌ FAIL — FTS+WHERE pool filter broken: ${err}`);
  }

  console.log("\n=== Test 4: FTS + WHERE combined (agent_id AND pool) ===");
  try {
    const results = await table
      .search("search", "fts", "text")
      .where("agent_id = 'dev' AND pool = 'reference_library'")
      .limit(5)
      .toArray();
    console.log(`✅ PASS — ${results.length} results (compound WHERE works!)`);
    for (const r of results) {
      console.log(`  id=${r.id} agent=${r.agent_id} pool=${r.pool} score=${r._score}`);
    }
  } catch (err) {
    console.log(`❌ FAIL — FTS + compound WHERE broken: ${err}`);
  }

  console.log("\n=== Test 5: Vector search + WHERE pool (control test) ===");
  try {
    const queryVec = new Array(16).fill(0.3);
    const results = await table
      .search(queryVec)
      .where("pool = 'shared_mistakes'")
      .limit(5)
      .toArray();
    console.log(`✅ PASS — ${results.length} results (vector+WHERE works)`);
    for (const r of results) {
      console.log(`  id=${r.id} agent=${r.agent_id} pool=${r.pool} dist=${r._distance}`);
    }
  } catch (err) {
    console.log(`❌ FAIL — ${err}`);
  }

  // Cleanup
  await fs.rm(TEST_DIR, { recursive: true, force: true });
  console.log("\nDone.");
}

main().catch(console.error);
