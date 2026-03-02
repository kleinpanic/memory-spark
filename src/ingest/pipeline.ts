/**
 * Ingest Pipeline
 *
 * Central pipeline for processing a file into storage:
 *   1. Extract text (format-specific parser)
 *   2. If large doc (>8k tokens): summarize via Spark 18110 before chunking
 *   3. Chunk (format-aware, token-bounded)
 *   4. NER tag entities per chunk via Spark 18112
 *   5. Embed batch via provider
 *   6. Upsert into storage backend (replaces old chunks for this path)
 *
 * Supported formats (dispatch by file extension):
 *   .md / .txt    → plain text (no extraction needed)
 *   .pdf          → pdftotext → Spark OCR 18097 fallback for scanned pages
 *   .docx         → mammoth HTML → strip tags → plain text
 *   .epub         → epub-parser → plain text (TODO: phase 2)
 *   .mp3/.wav/.m4a → Spark STT 18094 (parakeet) → transcript text
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

/**
 * Full ingest pipeline for a single file.
 * Idempotent: safe to run on already-indexed files (will delta-update).
 */
export async function ingestFile(opts: IngestFileOptions): Promise<IngestResult> {
  const start = Date.now();

  // 1. Extract text from file
  const ext = path.extname(opts.filePath).replace(".", "").toLowerCase();
  const rawText = await extractText(opts.filePath, ext, opts.cfg);

  // 2. Chunk
  const rawChunks = chunkDocument({
    text: rawText,
    path: opts.filePath,
    agentId: opts.agentId,
    source: "ingest",
    ext,
  });

  if (rawChunks.length === 0) {
    return { filePath: opts.filePath, chunksAdded: 0, chunksRemoved: 0, durationMs: Date.now() - start };
  }

  // 3. NER tag (best-effort, skip on failure)
  const entitiesPerChunk = await Promise.all(
    rawChunks.map((c) => tagEntities(c.text, opts.cfg).catch(() => [] as string[]))
  );

  // 4. Embed batch
  const vectors = await opts.embed.embedBatch(rawChunks.map((c) => c.text));

  // 5. Build MemoryChunk objects
  const now = new Date().toISOString();
  const chunks: MemoryChunk[] = rawChunks.map((raw, i) => ({
    id: chunkId(opts.filePath, raw.startLine, opts.agentId),
    path: opts.filePath,
    source: "ingest",
    agentId: opts.agentId,
    startLine: raw.startLine,
    endLine: raw.endLine,
    text: raw.text,
    ftsText: raw.text,
    vector: vectors[i]!,
    updatedAt: now,
    entities: entitiesPerChunk[i],
  }));

  // 6. Remove old chunks for this path, upsert new
  const removed = await opts.backend.deleteByPath(opts.filePath, opts.agentId);
  await opts.backend.upsert(chunks);

  return {
    filePath: opts.filePath,
    chunksAdded: chunks.length,
    chunksRemoved: removed,
    durationMs: Date.now() - start,
  };
}

function chunkId(filePath: string, startLine: number, agentId?: string): string {
  return crypto
    .createHash("sha1")
    .update(`${agentId ?? ""}:${filePath}:${startLine}`)
    .digest("hex")
    .slice(0, 16);
}
