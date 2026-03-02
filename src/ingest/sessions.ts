/**
 * Session JSONL Indexer — extracts conversation text from session transcripts.
 * Replicates memory-core's session indexing behavior.
 */

import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";

export interface SessionEntry {
  path: string;       // relative path: sessions/<filename>
  absPath: string;
  text: string;        // extracted conversation text
  mtimeMs: number;
}

/**
 * List session JSONL files for an agent.
 */
export async function listSessionFiles(agentId: string): Promise<string[]> {
  const sessionsDir = path.join(os.homedir(), ".openclaw", "agents", agentId, "sessions");
  try {
    const files = await fs.readdir(sessionsDir);
    return files
      .filter((f) => f.endsWith(".jsonl") && !f.includes(".reset.") && !f.includes(".deleted.") && !f.includes(".bak"))
      .map((f) => path.join(sessionsDir, f));
  } catch {
    return [];
  }
}

/**
 * Extract readable text from a session JSONL file.
 * Each line is a JSON message with { role, content, ... }.
 * We extract user + assistant text content.
 */
export async function extractSessionText(absPath: string): Promise<SessionEntry | null> {
  try {
    const raw = await fs.readFile(absPath, "utf-8");
    const stat = await fs.stat(absPath);
    const lines = raw.split("\n").filter((l) => l.trim());

    const textParts: string[] = [];

    for (const line of lines) {
      try {
        const msg = JSON.parse(line) as Record<string, unknown>;
        const role = msg.role as string | undefined;
        if (!role || (role !== "user" && role !== "assistant")) continue;

        const content = msg.content;
        if (typeof content === "string") {
          textParts.push(`[${role}] ${content}`);
        } else if (Array.isArray(content)) {
          for (const part of content) {
            if (typeof part === "string") {
              textParts.push(`[${role}] ${part}`);
            } else if (part && typeof part === "object" && "text" in part) {
              textParts.push(`[${role}] ${(part as { text: string }).text}`);
            }
          }
        }
      } catch {
        // Skip unparseable lines
      }
    }

    if (textParts.length === 0) return null;

    const relPath = `sessions/${path.basename(absPath)}`;
    return {
      path: relPath,
      absPath,
      text: textParts.join("\n"),
      mtimeMs: stat.mtimeMs,
    };
  } catch {
    return null;
  }
}
