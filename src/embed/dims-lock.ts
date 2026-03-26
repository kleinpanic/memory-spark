/**
 * Dimension Lock — ensures vector dimensions stay consistent.
 *
 * On first index: writes a lock file with the provider/model/dims.
 * On subsequent boots: refuses to start if the active provider produces
 * different dimensions than what's stored.
 *
 * This prevents the silent corruption scenario where Spark (4096d) indexes
 * data, then Gemini fallback (3072d) tries to search against it.
 */

import fs from "node:fs/promises";
import path from "node:path";

export interface DimsLockInfo {
  provider: string;
  model: string;
  dims: number;
  lockedAt: string;
  totalChunks?: number;
}

const LOCK_FILE = "dims-lock.json";

export async function readDimsLock(dataDir: string): Promise<DimsLockInfo | null> {
  try {
    const raw = await fs.readFile(path.join(dataDir, LOCK_FILE), "utf-8");
    return JSON.parse(raw) as DimsLockInfo;
  } catch {
    return null;
  }
}

export async function writeDimsLock(dataDir: string, info: DimsLockInfo): Promise<void> {
  await fs.mkdir(dataDir, { recursive: true });
  await fs.writeFile(path.join(dataDir, LOCK_FILE), JSON.stringify(info, null, 2) + "\n");
}

/**
 * Validate that the current embed provider matches the locked dimensions.
 * Returns null if OK, or an error message if mismatched.
 */
export async function validateDimsLock(
  dataDir: string,
  currentProvider: string,
  currentModel: string,
  currentDims: number,
): Promise<{ ok: true } | { ok: false; error: string; lock: DimsLockInfo }> {
  const lock = await readDimsLock(dataDir);

  if (!lock) {
    // First boot — create the lock
    await writeDimsLock(dataDir, {
      provider: currentProvider,
      model: currentModel,
      dims: currentDims,
      lockedAt: new Date().toISOString(),
    });
    return { ok: true };
  }

  if (lock.dims !== currentDims) {
    return {
      ok: false,
      error:
        `Dimension mismatch! Stored data uses ${lock.dims}d (${lock.provider}/${lock.model}) but current provider would produce ${currentDims}d (${currentProvider}/${currentModel}). ` +
        `To fix: either restore the original provider, or wipe the data dir and re-index. ` +
        `Delete ${path.join(dataDir, LOCK_FILE)} to force a re-lock (WILL CORRUPT existing vectors).`,
      lock,
    };
  }

  return { ok: true };
}
