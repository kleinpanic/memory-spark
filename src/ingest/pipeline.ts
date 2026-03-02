/**
 * Ingest Pipeline — extract → chunk → NER → embed → store.
 */

import type { MemorySparkConfig } from "../config.js";
import type { StorageBackend, MemoryChunk } from "../storage/backend.js";
import type { EmbedProvider } from "../embed/provider.js";
import { chunkDocument } from "../embed/chunker.js";
import { extractText } from "./parsers.js";
import { tagEntities } from "../classify/ner.js";
import crypto from "node:crypto";
import path from "node:path";

export interface IngestFileOptions {
  filePath: string;
  agentId?: string;
  backend: StorageBackend;
  embed: EmbedProvider;
  cfg: MemorySparkConfig;
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

    // 1. Extract text
    const rawText = await extractText(opts.filePath, ext, opts.cfg);
    if (!rawText.trim()) {
      return { filePath: opts.filePath, chunksAdded: 0, chunksRemoved: 0, durationMs: Date.now() - start };
    }

    // 2. Chunk
    const rawChunks = chunkDocument({
      text: rawText,
      path: opts.filePath,
      source: "ingest",
      ext,
    });

    if (rawChunks.length === 0) {
      return { filePath: opts.filePath, chunksAdded: 0, chunksRemoved: 0, durationMs: Date.now() - start };
    }

    // 3. NER tag (best-effort, parallel)
    const entitiesPerChunk = await Promise.all(
      rawChunks.map((c) => tagEntities(c.text, opts.cfg).catch(() => [] as string[]))
    );

    // 4. Embed batch
    const vectors = await opts.embed.embedBatch(rawChunks.map((c) => c.text));

    // 5. Build MemoryChunk objects
    const now = new Date().toISOString();
    const agentId = opts.agentId ?? "shared";
    const chunks: MemoryChunk[] = rawChunks.map((raw, i) => ({
      id: chunkId(opts.filePath, raw.startLine, agentId),
      path: opts.filePath,
      source: "ingest" as const,
      agent_id: agentId,
      start_line: raw.startLine,
      end_line: raw.endLine,
      text: raw.text,
      vector: vectors[i]!,
      updated_at: now,
      entities: JSON.stringify(entitiesPerChunk[i] ?? []),
    }));

    // 6. Remove old chunks for this path, then upsert new
    const removed = await opts.backend.deleteByPath(opts.filePath, agentId);
    await opts.backend.upsert(chunks);

    opts.logger?.info(`memory-spark: ingested ${opts.filePath} → ${chunks.length} chunks`);

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

function chunkId(filePath: string, startLine: number, agentId: string): string {
  return crypto
    .createHash("sha1")
    .update(`${agentId}:${filePath}:${startLine}`)
    .digest("hex")
    .slice(0, 16);
}
