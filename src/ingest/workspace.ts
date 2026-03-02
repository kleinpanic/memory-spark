/**
 * Workspace Discovery — auto-discovers agent workspace memory files.
 * Replicates memory-core's file discovery behavior.
 *
 * For each agent, indexes:
 *   1. <workspaceDir>/memory/*.md        — daily notes, archives
 *   2. <workspaceDir>/MEMORY.md          — long-term memory
 *   3. <workspaceDir>/SOUL.md            — identity
 *   4. <workspaceDir>/USER.md            — user profile
 *   5. <workspaceDir>/AGENTS.md          — agent config
 *   6. <workspaceDir>/HEARTBEAT.md       — heartbeat config
 *   7. <workspaceDir>/IDENTITY.md        — identity
 *   8. ~/.openclaw/agents/<id>/sessions/ — past session transcripts
 */

import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";

/** Files in workspace root that should always be indexed */
const ROOT_FILES = [
  "MEMORY.md", "SOUL.md", "USER.md", "AGENTS.md",
  "HEARTBEAT.md", "IDENTITY.md", "TOOLS.md",
];

export interface WorkspaceFiles {
  agentId: string;
  workspaceDir: string;
  /** All discoverable .md files (absolute paths) */
  memoryFiles: string[];
  /** Session JSONL files (absolute paths) */
  sessionFiles: string[];
}

/**
 * Discover all indexable files for an agent.
 */
export async function discoverWorkspaceFiles(agentId: string, workspaceDir?: string): Promise<WorkspaceFiles> {
  const ocDir = path.join(os.homedir(), ".openclaw");
  const wsDir = workspaceDir ?? path.join(ocDir, `workspace-${agentId}`);

  const memoryFiles: string[] = [];
  const sessionFiles: string[] = [];

  // 1. Workspace root files
  for (const f of ROOT_FILES) {
    const fp = path.join(wsDir, f);
    try {
      await fs.access(fp);
      memoryFiles.push(fp);
    } catch {
      // File doesn't exist
    }
  }

  // 2. memory/ directory (recursive)
  const memDir = path.join(wsDir, "memory");
  try {
    const mdFiles = await walkMdFiles(memDir);
    memoryFiles.push(...mdFiles);
  } catch {
    // No memory dir
  }

  // 3. Session files
  const sessDir = path.join(ocDir, "agents", agentId, "sessions");
  try {
    const files = await fs.readdir(sessDir);
    for (const f of files) {
      if (f.endsWith(".jsonl") && !f.includes(".reset.") && !f.includes(".deleted.") && !f.includes(".bak")) {
        sessionFiles.push(path.join(sessDir, f));
      }
    }
  } catch {
    // No sessions dir
  }

  return { agentId, workspaceDir: wsDir, memoryFiles, sessionFiles };
}

/**
 * Discover all agents by scanning workspace directories.
 */
export async function discoverAllAgents(): Promise<string[]> {
  const ocDir = path.join(os.homedir(), ".openclaw");
  try {
    const entries = await fs.readdir(ocDir, { withFileTypes: true });
    return entries
      .filter((e) => e.isDirectory() && e.name.startsWith("workspace-"))
      .map((e) => e.name.replace("workspace-", ""));
  } catch {
    return [];
  }
}

/**
 * Convert an absolute path to a relative path for storage.
 * Paths are stored relative to the workspace dir or openclaw root.
 */
export function toRelativePath(absPath: string, workspaceDir: string): string {
  // Try workspace-relative first
  if (absPath.startsWith(workspaceDir)) {
    return path.relative(workspaceDir, absPath);
  }
  // Fall back to openclaw-root-relative
  const ocDir = path.join(os.homedir(), ".openclaw");
  if (absPath.startsWith(ocDir)) {
    return path.relative(ocDir, absPath);
  }
  // External file — store absolute
  return absPath;
}

/**
 * Resolve a relative path back to absolute for reading.
 */
export function toAbsolutePath(relPath: string, workspaceDir: string): string {
  if (path.isAbsolute(relPath)) return relPath;

  // Try workspace-relative
  const wsAbs = path.join(workspaceDir, relPath);
  // Try openclaw-root-relative
  const ocAbs = path.join(os.homedir(), ".openclaw", relPath);

  // Return whichever exists (check workspace first)
  return wsAbs; // Caller should handle existence check
}

async function walkMdFiles(dir: string): Promise<string[]> {
  const results: string[] = [];
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name.startsWith(".")) continue;
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        results.push(...await walkMdFiles(fullPath));
      } else if (entry.name.endsWith(".md")) {
        results.push(fullPath);
      }
    }
  } catch {
    // Skip inaccessible dirs
  }
  return results;
}
