/**
 * Unit tests for TableManager — multi-table LanceDB management.
 * Tests table naming, lifecycle, agent discovery, and write serialization.
 * No external dependencies (LanceDB is mocked via in-memory dir).
 */

import assert from "node:assert/strict";
import {
  TableManager,
  DEFAULT_TABLE_NAMING,
  type TableNamingConfig,
} from "../src/storage/table-manager.js";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";

const results: Array<{ test: string; status: "PASS" | "FAIL"; error?: string }> = [];

function test(name: string, fn: () => boolean | void | Promise<boolean | void>) {
  const result = fn();
  if (result instanceof Promise) {
    return result
      .then((r) => {
        const passed = r !== false;
        console.log(`[${passed ? "PASS" : "FAIL"}] ${name}`);
        results.push({ test: name, status: passed ? "PASS" : "FAIL" });
      })
      .catch((err) => {
        console.log(`[FAIL] ${name}`);
        console.log(`  Error: ${String(err)}`);
        results.push({ test: name, status: "FAIL", error: String(err) });
      });
  }
  try {
    const passed = result !== false;
    console.log(`[${passed ? "PASS" : "FAIL"}] ${name}`);
    results.push({ test: name, status: passed ? "PASS" : "FAIL" });
  } catch (err) {
    console.log(`[FAIL] ${name}`);
    console.log(`  Error: ${String(err)}`);
    results.push({ test: name, status: "FAIL", error: String(err) });
  }
}

// ── Table Naming Tests ──────────────────────────────────────────────────────

test("Default naming config has all required fields", () => {
  assert.ok(DEFAULT_TABLE_NAMING.agentPrefix);
  assert.ok(DEFAULT_TABLE_NAMING.memorySuffix);
  assert.ok(DEFAULT_TABLE_NAMING.toolsSuffix);
  assert.ok(DEFAULT_TABLE_NAMING.sharedKnowledge);
  assert.ok(DEFAULT_TABLE_NAMING.sharedMistakes);
  assert.ok(DEFAULT_TABLE_NAMING.referenceLibrary);
  assert.ok(DEFAULT_TABLE_NAMING.referenceCode);
});

test("Agent memory table name follows convention", () => {
  const mgr = new TableManager("/tmp/test-tm", DEFAULT_TABLE_NAMING);
  assert.strictEqual(mgr.agentMemoryName("meta"), "agent_meta_memory");
  assert.strictEqual(mgr.agentMemoryName("dev"), "agent_dev_memory");
  assert.strictEqual(mgr.agentMemoryName("school"), "agent_school_memory");
});

test("Agent tools table name follows convention", () => {
  const mgr = new TableManager("/tmp/test-tm", DEFAULT_TABLE_NAMING);
  assert.strictEqual(mgr.agentToolsName("meta"), "agent_meta_tools");
  assert.strictEqual(mgr.agentToolsName("dev"), "agent_dev_tools");
});

test("Shared table names are correct", () => {
  const mgr = new TableManager("/tmp/test-tm");
  assert.strictEqual(mgr.sharedKnowledgeName(), "shared_knowledge");
  assert.strictEqual(mgr.sharedMistakesName(), "shared_mistakes");
  assert.strictEqual(mgr.referenceLibraryName(), "reference_library");
  assert.strictEqual(mgr.referenceCodeName(), "reference_code");
});

test("Agent IDs are sanitized in table names", () => {
  const mgr = new TableManager("/tmp/test-tm");
  // Special characters replaced with underscore, lowercased
  assert.strictEqual(mgr.agentMemoryName("My Agent!"), "agent_my_agent__memory");
  assert.strictEqual(mgr.agentMemoryName("test.agent"), "agent_test_agent_memory");
  assert.strictEqual(mgr.agentMemoryName("UPPER"), "agent_upper_memory");
});

test("Hyphens and underscores preserved in agent IDs", () => {
  const mgr = new TableManager("/tmp/test-tm");
  assert.strictEqual(mgr.agentMemoryName("my-agent"), "agent_my-agent_memory");
  assert.strictEqual(mgr.agentMemoryName("my_agent"), "agent_my_agent_memory");
});

test("Custom naming config overrides defaults", () => {
  const custom: Partial<TableNamingConfig> = {
    agentPrefix: "a_",
    memorySuffix: "_mem",
    sharedKnowledge: "global_facts",
  };
  const mgr = new TableManager("/tmp/test-tm", custom);
  assert.strictEqual(mgr.agentMemoryName("dev"), "a_dev_mem");
  assert.strictEqual(mgr.sharedKnowledgeName(), "global_facts");
  // Non-overridden fields use defaults
  assert.strictEqual(mgr.sharedMistakesName(), "shared_mistakes");
});

// ── Lifecycle Tests (require real LanceDB) ──────────────────────────────────

const TEST_DIR = path.join(os.tmpdir(), `memory-spark-tm-test-${Date.now()}`);

async function withManager(fn: (mgr: TableManager) => Promise<void>): Promise<void> {
  const testDir = path.join(TEST_DIR, `run-${Date.now()}`);
  const mgr = new TableManager(testDir);
  try {
    await mgr.open();
    await fn(mgr);
  } finally {
    await mgr.close();
    await fs.rm(testDir, { recursive: true, force: true }).catch(() => {});
  }
}

await test("TableManager opens and lists empty database", async () => {
  await withManager(async (mgr) => {
    const names = await mgr.listTableNames();
    assert.deepStrictEqual(names, []);
  });
});

await test("discoverAgents returns empty for fresh database", async () => {
  await withManager(async (mgr) => {
    const agents = await mgr.discoverAgents();
    assert.deepStrictEqual(agents, []);
  });
});

await test("createTableWithData creates table and enables FTS", async () => {
  await withManager(async (mgr) => {
    const data = [
      {
        id: "chunk-1",
        text: "The quick brown fox jumps over the lazy dog",
        vector: new Array(16).fill(0.1), // Small dims for testing
        path: "test.md",
        source: "memory",
        agent_id: "test",
        content_type: "knowledge",
        start_line: 1,
        end_line: 1,
        updated_at: new Date().toISOString(),
      },
    ];
    const table = await mgr.createTableWithData(
      mgr.agentMemoryName("test"),
      data,
      "agent_memory",
      "test",
    );
    assert.ok(table, "Table should be created");

    const names = await mgr.listTableNames();
    assert.ok(names.includes("agent_test_memory"), "Table name should appear in list");
  });
});

await test("discoverAgents finds agents with memory tables", async () => {
  await withManager(async (mgr) => {
    const data = [
      {
        id: "chunk-1",
        text: "test content",
        vector: new Array(16).fill(0.1),
        path: "test.md",
        source: "memory",
        agent_id: "meta",
        content_type: "knowledge",
        start_line: 1,
        end_line: 1,
        updated_at: new Date().toISOString(),
      },
    ];
    await mgr.createTableWithData(mgr.agentMemoryName("meta"), data, "agent_memory", "meta");
    await mgr.createTableWithData(mgr.agentMemoryName("dev"), data, "agent_memory", "dev");

    const agents = await mgr.discoverAgents();
    assert.deepStrictEqual(agents, ["dev", "meta"]); // Sorted
  });
});

await test("tableExists returns correct result", async () => {
  await withManager(async (mgr) => {
    assert.strictEqual(await mgr.tableExists("nonexistent"), false);
    const data = [
      {
        id: "chunk-1",
        text: "test",
        vector: new Array(16).fill(0.1),
        path: "test.md",
        source: "memory",
        agent_id: "test",
        content_type: "knowledge",
        start_line: 1,
        end_line: 1,
        updated_at: new Date().toISOString(),
      },
    ];
    await mgr.createTableWithData("test_table", data, "agent_memory", "test");
    assert.strictEqual(await mgr.tableExists("test_table"), true);
  });
});

await test("dropTable removes table from database", async () => {
  await withManager(async (mgr) => {
    const data = [
      {
        id: "chunk-1",
        text: "test",
        vector: new Array(16).fill(0.1),
        path: "test.md",
        source: "memory",
        agent_id: "test",
        content_type: "knowledge",
        start_line: 1,
        end_line: 1,
        updated_at: new Date().toISOString(),
      },
    ];
    await mgr.createTableWithData("to_drop", data, "agent_memory", "test");
    assert.strictEqual(await mgr.tableExists("to_drop"), true);

    await mgr.dropTable("to_drop");
    assert.strictEqual(await mgr.tableExists("to_drop"), false);
  });
});

await test("status reports all open tables", async () => {
  await withManager(async (mgr) => {
    const data = [
      {
        id: "chunk-1",
        text: "test content for status",
        vector: new Array(16).fill(0.1),
        path: "test.md",
        source: "memory",
        agent_id: "meta",
        content_type: "knowledge",
        start_line: 1,
        end_line: 1,
        updated_at: new Date().toISOString(),
      },
    ];
    await mgr.createTableWithData(mgr.agentMemoryName("meta"), data, "agent_memory", "meta");
    await mgr.createTableWithData(mgr.sharedMistakesName(), data, "shared_mistakes");

    const statuses = await mgr.status();
    assert.strictEqual(statuses.length, 2);

    const memStatus = statuses.find((s) => s.category === "agent_memory");
    assert.ok(memStatus);
    assert.strictEqual(memStatus.agentId, "meta");
    assert.strictEqual(memStatus.rowCount, 1);

    const mistakeStatus = statuses.find((s) => s.category === "shared_mistakes");
    assert.ok(mistakeStatus);
    assert.strictEqual(mistakeStatus.rowCount, 1);
  });
});

await test("withWriteLock serializes concurrent writes to same table", async () => {
  await withManager(async (mgr) => {
    const order: number[] = [];
    const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

    // Launch 3 concurrent "writes" — they should execute sequentially
    const p1 = mgr.withWriteLock("test_table", async () => {
      order.push(1);
      await delay(50);
      order.push(11);
    });
    const p2 = mgr.withWriteLock("test_table", async () => {
      order.push(2);
      await delay(30);
      order.push(22);
    });
    const p3 = mgr.withWriteLock("test_table", async () => {
      order.push(3);
      order.push(33);
    });

    await Promise.all([p1, p2, p3]);

    // Should be serialized: 1, 11 (first completes), 2, 22 (second completes), 3, 33
    assert.deepStrictEqual(order, [1, 11, 2, 22, 3, 33]);
  });
});

await test("withWriteLock allows parallel writes to different tables", async () => {
  await withManager(async (mgr) => {
    const order: string[] = [];
    const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

    const p1 = mgr.withWriteLock("table_a", async () => {
      order.push("a-start");
      await delay(50);
      order.push("a-end");
    });
    const p2 = mgr.withWriteLock("table_b", async () => {
      order.push("b-start");
      await delay(30);
      order.push("b-end");
    });

    await Promise.all([p1, p2]);

    // Both should start immediately (different tables)
    assert.ok(
      order.indexOf("a-start") < order.indexOf("a-end"),
      "a completes in order",
    );
    assert.ok(
      order.indexOf("b-start") < order.indexOf("b-end"),
      "b completes in order",
    );
    // b should finish before a (shorter delay)
    assert.ok(
      order.indexOf("b-end") < order.indexOf("a-end"),
      "Different tables run in parallel",
    );
  });
});

await test("Errors in withWriteLock don't block subsequent writes", async () => {
  await withManager(async (mgr) => {
    // First write throws
    try {
      await mgr.withWriteLock("test_table", async () => {
        throw new Error("write failed");
      });
    } catch {
      // Expected
    }

    // Second write should still work (lock chain not permanently blocked)
    let secondCompleted = false;
    await mgr.withWriteLock("test_table", async () => {
      secondCompleted = true;
    });
    assert.strictEqual(secondCompleted, true, "Second write should complete after first error");
  });
});

// ── Summary ─────────────────────────────────────────────────────────────────

// Wait for all async tests to settle
await new Promise((r) => setTimeout(r, 200));

console.log("\n=== Table Manager Tests ===");
const passed = results.filter((r) => r.status === "PASS").length;
const failed = results.filter((r) => r.status === "FAIL").length;
console.log(`Total: ${results.length} | PASS: ${passed} | FAIL: ${failed}`);

if (failed > 0) {
  console.log("\nFailed tests:");
  results
    .filter((r) => r.status === "FAIL")
    .forEach((r) => {
      console.log(`  - ${r.test}`);
      if (r.error) console.log(`    ${r.error}`);
    });
  process.exit(1);
}

// Cleanup
await fs.rm(TEST_DIR, { recursive: true, force: true }).catch(() => {});
