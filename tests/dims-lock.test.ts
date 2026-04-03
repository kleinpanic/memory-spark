import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import path from "node:path";
import os from "node:os";
import fs from "node:fs/promises";
import { validateDimsLock, type DimsLockInfo } from "../src/embed/dims-lock";

// ─── Temp-directory helpers ──────────────────────────────────────────────────

async function makeTmpDir(): Promise<string> {
  return fs.mkdtemp(path.join(os.tmpdir(), "dims-lock-test-"));
}

async function removeTmpDir(dir: string): Promise<void> {
  await fs.rm(dir, { recursive: true, force: true });
}

async function readLockFile(dir: string): Promise<DimsLockInfo> {
  const raw = await fs.readFile(path.join(dir, "dims-lock.json"), "utf-8");
  return JSON.parse(raw) as DimsLockInfo;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe("validateDimsLock", () => {
  let dataDir: string;

  beforeEach(async () => {
    dataDir = await makeTmpDir();
  });

  afterEach(async () => {
    await removeTmpDir(dataDir);
  });

  // 1. First boot: creates lock file with provider, model, dims
  it("first boot — creates lock file with correct fields", async () => {
    const before = Date.now();
    const result = await validateDimsLock(dataDir, "spark", "nvidia/nv-embedqa-e5-v5", 4096);
    const after = Date.now();

    expect(result.ok).toBe(true);

    const lock = await readLockFile(dataDir);
    expect(lock.provider).toBe("spark");
    expect(lock.model).toBe("nvidia/nv-embedqa-e5-v5");
    expect(lock.dims).toBe(4096);
    expect(typeof lock.lockedAt).toBe("string");

    const lockedMs = new Date(lock.lockedAt).getTime();
    expect(lockedMs).toBeGreaterThanOrEqual(before);
    expect(lockedMs).toBeLessThanOrEqual(after);
  });

  // 7. Lock file format (JSON with provider, model, dims fields)
  it("lock file is valid JSON with provider, model, dims, lockedAt keys", async () => {
    await validateDimsLock(dataDir, "gemini", "text-embedding-004", 3072);

    const raw = await fs.readFile(path.join(dataDir, "dims-lock.json"), "utf-8");
    const parsed = JSON.parse(raw);

    expect(parsed).toHaveProperty("provider");
    expect(parsed).toHaveProperty("model");
    expect(parsed).toHaveProperty("dims");
    expect(parsed).toHaveProperty("lockedAt");
  });

  // 2. Subsequent boot: same dims → ok
  it("subsequent boot with same dims returns ok", async () => {
    // First boot
    await validateDimsLock(dataDir, "spark", "nvidia/nv-embedqa-e5-v5", 4096);

    // Second boot — same provider + dims
    const result = await validateDimsLock(dataDir, "spark", "nvidia/nv-embedqa-e5-v5", 4096);
    expect(result.ok).toBe(true);
  });

  // 3. Subsequent boot: different dims → error with clear message
  it("subsequent boot with different dims returns ok=false and a descriptive error", async () => {
    await validateDimsLock(dataDir, "spark", "nvidia/nv-embedqa-e5-v5", 4096);

    const result = await validateDimsLock(dataDir, "gemini", "text-embedding-004", 3072);

    expect(result.ok).toBe(false);
    if (!result.ok) {
      // Error message must mention both old and new dims
      expect(result.error).toContain("4096");
      expect(result.error).toContain("3072");
      // Should mention the original provider/model
      expect(result.error).toContain("spark");
      // Should hint at resolution
      expect(result.error.toLowerCase()).toMatch(/wipe|re-index|restore/);

      // lock field should reflect what was stored
      expect(result.lock.dims).toBe(4096);
      expect(result.lock.provider).toBe("spark");
    }
  });

  // 4. Subsequent boot: different provider same dims → ok
  it("subsequent boot with different provider but same dims returns ok", async () => {
    await validateDimsLock(dataDir, "openai", "text-embedding-3-large", 3072);

    // Different provider, same 3072 dims
    const result = await validateDimsLock(dataDir, "gemini", "text-embedding-004", 3072);
    expect(result.ok).toBe(true);
  });

  // 5. Corrupt lock file → handles gracefully (treats as first boot)
  it("corrupt lock file is treated as missing — first boot happens cleanly", async () => {
    // Write garbage JSON
    await fs.writeFile(path.join(dataDir, "dims-lock.json"), "{ not valid json %%%");

    const result = await validateDimsLock(dataDir, "spark", "nv-embed", 4096);

    // Should succeed and (re)create the lock
    expect(result.ok).toBe(true);

    const lock = await readLockFile(dataDir);
    expect(lock.dims).toBe(4096);
    expect(lock.provider).toBe("spark");
  });

  // 6. Missing directory → creates it
  it("creates missing data directory on first boot", async () => {
    const nestedDir = path.join(dataDir, "deeply", "nested", "data");

    // Confirm it doesn't exist yet
    await expect(fs.access(nestedDir)).rejects.toThrow();

    const result = await validateDimsLock(nestedDir, "spark", "nv-embed", 4096);
    expect(result.ok).toBe(true);

    // Directory and lock file should now exist
    const lock = await readLockFile(nestedDir);
    expect(lock.dims).toBe(4096);
  });

  // Edge: dims=0 treated as valid (degenerate but shouldn't crash)
  it("handles zero dims without throwing", async () => {
    const result = await validateDimsLock(dataDir, "fake", "fake-model", 0);
    expect(result.ok).toBe(true);

    const lock = await readLockFile(dataDir);
    expect(lock.dims).toBe(0);
  });

  // Edge: lock file written but then changed externally to a different dims
  it("detects externally-modified lock dims correctly", async () => {
    await validateDimsLock(dataDir, "spark", "nv-embed", 4096);

    // External mutation
    const lockPath = path.join(dataDir, "dims-lock.json");
    const lock = await readLockFile(dataDir);
    await fs.writeFile(lockPath, JSON.stringify({ ...lock, dims: 768 }, null, 2) + "\n");

    // Now 4096 dims won't match stored 768
    const result = await validateDimsLock(dataDir, "spark", "nv-embed", 4096);
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.error).toContain("768");
      expect(result.error).toContain("4096");
    }
  });
});
