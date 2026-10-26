/**
 * Regression tests for path containment, LanceDB readFile filtering,
 * and .env token parsing.
 */
import { describe, it, expect, beforeAll, afterAll, afterEach } from "vitest";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { resolveConfig } from "../src/config";
import { isPathInside, toRelativePath } from "../src/ingest/workspace";
import { MemorySparkManager } from "../src/manager";
import type { StorageBackend } from "../src/storage/backend";
import { LanceDBBackend } from "../src/storage/lancedb";

describe("isPathInside", () => {
  it("accepts the root and its descendants", () => {
    expect(isPathInside("/ws", "/ws")).toBe(true);
    expect(isPathInside("/ws/memory/a.md", "/ws")).toBe(true);
    expect(isPathInside("/ws/..notes.md", "/ws")).toBe(true);
  });

  it("rejects sibling directories that share a name prefix", () => {
    expect(isPathInside("/home/u/workspace-dev2/a.md", "/home/u/workspace-dev")).toBe(false);
  });

  it("rejects .. escapes", () => {
    expect(isPathInside("/ws/../etc/passwd", "/ws")).toBe(false);
    expect(isPathInside("/etc/passwd", "/ws")).toBe(false);
  });
});

describe("toRelativePath", () => {
  it("does not relativize files in a prefix-sibling workspace", () => {
    const rel = toRelativePath("/srv/workspace-dev2/MEMORY.md", "/srv/workspace-dev");
    expect(rel).toBe("/srv/workspace-dev2/MEMORY.md");
  });

  it("still relativizes files inside the workspace", () => {
    expect(toRelativePath("/srv/workspace-dev/memory/a.md", "/srv/workspace-dev")).toBe(
      path.join("memory", "a.md"),
    );
  });
});

describe("MemorySparkManager.readFile disk fallback", () => {
  let root: string;
  let workspaceDir: string;
  let manager: MemorySparkManager;

  beforeAll(async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), "ms-readfile-"));
    workspaceDir = path.join(root, "workspace");
    await fs.mkdir(workspaceDir);
    await fs.writeFile(path.join(workspaceDir, "MEMORY.md"), "inside workspace");
    await fs.writeFile(path.join(root, "secret.txt"), "TOP SECRET");

    // Storage has nothing, forcing the disk fallback
    const backend = {
      readFile: async (p: { path: string }) => ({ text: "", path: p.path }),
    } as unknown as StorageBackend;
    manager = new MemorySparkManager({
      cfg: resolveConfig(),
      agentId: "test",
      workspaceDir,
      backend,
      embed: {} as never,
      reranker: {} as never,
    });
  });

  afterAll(async () => {
    await fs.rm(root, { recursive: true, force: true });
  });

  it("reads files inside the workspace", async () => {
    const r = await manager.readFile({ relPath: "MEMORY.md" });
    expect(r.text).toBe("inside workspace");
  });

  it("refuses ../ traversal out of the workspace", async () => {
    const r = await manager.readFile({ relPath: "../secret.txt" });
    expect(r.text).toBe("");
  });

  it("refuses absolute paths outside the workspace", async () => {
    const r = await manager.readFile({ relPath: path.join(root, "secret.txt") });
    expect(r.text).toBe("");
  });
});

describe("LanceDBBackend.readFile", () => {
  let dataDir: string;
  let backend: LanceDBBackend;
  const prevDataDir = process.env["MEMORY_SPARK_DATA_DIR"];

  beforeAll(async () => {
    dataDir = await fs.mkdtemp(path.join(os.tmpdir(), "ms-lancedb-readfile-"));
    process.env["MEMORY_SPARK_DATA_DIR"] = dataDir;
    backend = new LanceDBBackend(resolveConfig());
    await backend.open();
    const base = {
      source: "memory" as const,
      agent_id: "agent-a",
      start_line: 1,
      end_line: 1,
      vector: [0.1, 0.2, 0.3, 0.4],
      updated_at: new Date().toISOString(),
    };
    await backend.upsert([
      { ...base, id: "a1", path: "memory/a.md", text: "alpha line" },
      { ...base, id: "b1", path: "memory/b.md", text: "bravo line" },
      { ...base, id: "a2", path: "memory/a.md", text: "other agent", agent_id: "agent-b" },
    ]);
  });

  afterAll(async () => {
    await backend.close();
    if (prevDataDir === undefined) delete process.env["MEMORY_SPARK_DATA_DIR"];
    else process.env["MEMORY_SPARK_DATA_DIR"] = prevDataDir;
    await fs.rm(dataDir, { recursive: true, force: true });
  });

  it("filters by both path and agent", async () => {
    const r = await backend.readFile({ path: "memory/a.md", agentId: "agent-a" });
    expect(r.text).toBe("alpha line");
  });
});

describe("Spark token loading from ~/.openclaw/.env", () => {
  let home: string;
  const prevHome = process.env["HOME"];
  const prevToken = process.env["SPARK_BEARER_TOKEN"];

  afterEach(async () => {
    process.env["HOME"] = prevHome;
    if (prevToken === undefined) delete process.env["SPARK_BEARER_TOKEN"];
    else process.env["SPARK_BEARER_TOKEN"] = prevToken;
    await fs.rm(home, { recursive: true, force: true });
  });

  it("ignores commented-out and prefixed keys", async () => {
    home = await fs.mkdtemp(path.join(os.tmpdir(), "ms-home-"));
    await fs.mkdir(path.join(home, ".openclaw"));
    await fs.writeFile(
      path.join(home, ".openclaw", ".env"),
      "# SPARK_BEARER_TOKEN=revoked\nOLD_SPARK_BEARER_TOKEN=stale\nSPARK_BEARER_TOKEN=current\n",
    );
    process.env["HOME"] = home;
    delete process.env["SPARK_BEARER_TOKEN"];

    expect(resolveConfig().embed.spark?.apiKey).toBe("current");
  });
});
