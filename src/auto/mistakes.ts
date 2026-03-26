/**
 * MISTAKES.md + mistakes/ Enforcement — ensure every workspace has a
 * mistake tracking system with both an index file and a detail directory.
 *
 * Structure:
 *   <workspace>/MISTAKES.md       — Lean index: category, one-liner, link to detail
 *   <workspace>/mistakes/          — Detailed write-ups per incident
 *     └── YYYY-MM-DD-slug.md      — Full post-mortem
 *
 * Also used by the source weighting in recall.ts (1.6x boost for both
 * MISTAKES.md and mistakes/*.md files).
 */

import fs from "node:fs/promises";
import path from "node:path";

const MISTAKES_INDEX_TEMPLATE = `# Mistakes Log

Lean index of recurring errors. Details live in \`mistakes/\` subdirectory.

## How to use
1. When you make a mistake, create \`mistakes/YYYY-MM-DD-slug.md\` with the full write-up
2. Add a one-liner to the table below linking to the detail file
3. Read this file every session — if it's in here, don't repeat it

## Format for detail files (\`mistakes/YYYY-MM-DD-slug.md\`)
\`\`\`markdown
# Short title

**Date:** YYYY-MM-DD
**Count:** N
**Category:** tooling | config-safety | infrastructure | workflow | delegation

## What happened
Brief description of the error.

## Root cause
Why it happened — what assumption was wrong.

## Fix
How to avoid it next time — the specific check or rule.
\`\`\`

## Index

| Date | Category | Mistake | Detail |
|------|----------|---------|--------|
<!-- Add new entries above this line -->
`;

/**
 * For each workspace directory:
 * 1. Create MISTAKES.md index if it doesn't exist
 * 2. Create mistakes/ subdirectory if it doesn't exist
 *
 * Non-fatal — errors are logged and swallowed.
 */
export async function enforceMistakesFiles(
  workspaceDirs: string[],
  logger?: { info: (m: string) => void; warn: (m: string) => void },
): Promise<void> {
  for (const dir of workspaceDirs) {
    try {
      // Ensure mistakes/ directory exists
      const mistakesDir = path.join(dir, "mistakes");
      await fs.mkdir(mistakesDir, { recursive: true });

      // Create MISTAKES.md index if neither case exists
      const upper = path.join(dir, "MISTAKES.md");
      const lower = path.join(dir, "mistakes.md");

      const upperExists = await fs
        .access(upper)
        .then(() => true)
        .catch(() => false);
      const lowerExists = await fs
        .access(lower)
        .then(() => true)
        .catch(() => false);

      if (!upperExists && !lowerExists) {
        await fs.writeFile(upper, MISTAKES_INDEX_TEMPLATE, "utf-8");
        logger?.info(`memory-spark: created MISTAKES.md + mistakes/ in ${dir}`);
      } else {
        // Index exists but maybe no dir yet
        logger?.info(`memory-spark: ensured mistakes/ dir in ${dir}`);
      }
    } catch (err) {
      logger?.warn(`memory-spark: could not enforce mistakes structure in ${dir}: ${err}`);
    }
  }
}
