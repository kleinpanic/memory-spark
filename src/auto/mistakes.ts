/**
 * MISTAKES.md Enforcement — ensure every workspace has a mistakes log.
 * Also used by the source weighting in recall.ts (1.6x boost).
 */

import fs from "node:fs/promises";
import path from "node:path";

const MISTAKES_TEMPLATE = `# Mistakes Log

Track recurring errors and lessons learned to prevent repeating them.

## Format
- **Date:** YYYY-MM-DD
- **What happened:** Brief description
- **Root cause:** Why it happened
- **Fix:** How to avoid it next time

---
`;

/**
 * For each workspace directory, create MISTAKES.md if neither MISTAKES.md
 * nor mistakes.md exists. Non-fatal — errors are logged and swallowed.
 */
export async function enforceMistakesFiles(
  workspaceDirs: string[],
  logger?: { info: (m: string) => void; warn: (m: string) => void },
): Promise<void> {
  for (const dir of workspaceDirs) {
    try {
      const upper = path.join(dir, "MISTAKES.md");
      const lower = path.join(dir, "mistakes.md");

      const upperExists = await fs.access(upper).then(() => true).catch(() => false);
      const lowerExists = await fs.access(lower).then(() => true).catch(() => false);

      if (!upperExists && !lowerExists) {
        await fs.writeFile(upper, MISTAKES_TEMPLATE, "utf-8");
        logger?.info(`memory-spark: created MISTAKES.md in ${dir}`);
      }
    } catch (err) {
      logger?.warn(`memory-spark: could not create MISTAKES.md in ${dir}: ${err}`);
    }
  }
}
