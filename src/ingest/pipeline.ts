/**
 * Ingest Pipeline — extract → chunk → NER → embed → store.
 * Handles both regular files AND session JSONL transcripts.
 */

import type { MemorySparkConfig } from "../config.js";
import type { StorageBackend, MemoryChunk } from "../storage/backend.js";
import type { EmbedProvider } from "../embed/provider.js";
import type { EmbedQueue } from "../embed/queue.js";
import { chunkDocument } from "../embed/chunker.js";
import { extractText } from "./parsers.js";
import { extractSessionText } from "./sessions.js";
import { toRelativePath } from "./workspace.js";
import { tagEntities } from "../classify/ner.js";
import crypto from "node:crypto";
import path from "node:path";

/** Anything with embedBatch — works with both raw EmbedProvider and EmbedQueue */
export type Embedder = Pick<EmbedProvider, "embedBatch"> | Pick<EmbedQueue, "embedBatch">;

export interface IngestFileOptions {
  filePath: string;
  agentId: string;
  workspaceDir: string;
  backend: StorageBackend;
  embed: Embedder;
  cfg: MemorySparkConfig;
  /** "memory" for workspace files, "sessions" for JSONL transcripts, "ingest" for external */
  source?: "memory" | "sessions" | "ingest";
  logger?: { info: (m: string) => void; warn: (m: string) => void; error: (m: string) => void };
}

export interface IngestResult {
  filePath: string;
  chunksAdded: number;
  chunksRemoved: number;
  durationMs: number;
  error?: string;
}

export async function ingestFile(opts: IngestFileOptions): Promise<IngestResult> {
  const start = Date.now();

  try {
    const ext = path.extname(opts.filePath).replace(".", "").toLowerCase();
    const isSession = ext === "jsonl" || opts.source === "sessions";

    // 1. Extract text
    let rawText: string;
    if (isSession) {
      const entry = await extractSessionText(opts.filePath);
      if (!entry || !entry.text.trim()) {
        return { filePath: opts.filePath, chunksAdded: 0, chunksRemoved: 0, durationMs: Date.now() - start };
      }
      rawText = entry.text;
    } else {
      rawText = await extractText(opts.filePath, ext, opts.cfg);
    }

    if (!rawText.trim()) {
      return { filePath: opts.filePath, chunksAdded: 0, chunksRemoved: 0, durationMs: Date.now() - start };
    }

    // 2. Convert to relative path for storage
    const relPath = toRelativePath(opts.filePath, opts.workspaceDir);
    const source = opts.source ?? (isSession ? "sessions" : "memory");

    // 3. Chunk
    const rawChunks = chunkDocument({
      text: rawText,
      path: relPath,
      source,
      ext: isSession ? "txt" : ext,
    });

    if (rawChunks.length === 0) {
      return { filePath: opts.filePath, chunksAdded: 0, chunksRemoved: 0, durationMs: Date.now() - start };
    }

    // 4. NER tag (best-effort, parallel)
    const entitiesPerChunk = await Promise.all(
      rawChunks.map((c) => tagEntities(c.text, opts.cfg).catch(() => [] as string[]))
    );

    // 5. Embed batch
    const vectors = await opts.embed.embedBatch(rawChunks.map((c) => c.text));

    // 6. Build MemoryChunk objects with RELATIVE paths
    const now = new Date().toISOString();
    const chunks: MemoryChunk[] = rawChunks.map((raw, i) => ({
      id: chunkId(relPath, raw.startLine, opts.agentId),
      path: relPath,
      source,
      agent_id: opts.agentId,
      start_line: raw.startLine,
      end_line: raw.endLine,
      text: raw.text,
      vector: vectors[i]!,
      updated_at: now,
      entities: JSON.stringify(entitiesPerChunk[i] ?? []),
    }));

    // 7. Remove old chunks for this path, then upsert new
    const removed = await opts.backend.deleteByPath(relPath, opts.agentId);
    await opts.backend.upsert(chunks);

    opts.logger?.info(`memory-spark: ${source} ${relPath} → ${chunks.length} chunks`);

    return {
      filePath: opts.filePath,
      chunksAdded: chunks.length,
      chunksRemoved: removed,
      durationMs: Date.now() - start,
    };
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    opts.logger?.error(`memory-spark: ingest failed for ${opts.filePath}: ${error}`);
    return {
      filePath: opts.filePath,
      chunksAdded: 0,
      chunksRemoved: 0,
      durationMs: Date.now() - start,
      error,
    };
  }
}

function chunkId(relPath: string, startLine: number, agentId: string): string {
  return crypto
    .createHash("sha1")
    .update(`${agentId}:${relPath}:${startLine}`)
    .digest("hex")
    .slice(0, 16);
}
