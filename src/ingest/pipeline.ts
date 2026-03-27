/**
 * Ingest Pipeline — extract → chunk → NER → embed → store.
 * Handles both regular files AND session JSONL transcripts.
 */

import crypto from "node:crypto";
import path from "node:path";

import { tagEntities } from "../classify/ner.js";
import { scoreChunkQuality } from "../classify/quality.js";
import type { MemorySparkConfig } from "../config.js";
import { chunkDocument, cleanChunkText, estimateTokens } from "../embed/chunker.js";
import type { EmbedProvider } from "../embed/provider.js";
import type { EmbedQueue } from "../embed/queue.js";
import type { StorageBackend, MemoryChunk } from "../storage/backend.js";
import { resolvePool } from "../storage/pool.js";

import { extractText } from "./parsers.js";
import { extractSessionText } from "./sessions.js";
import { toRelativePath } from "./workspace.js";

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
  /** Content type for reference library indexing. Default: "knowledge" */
  contentType?: "knowledge" | "reference" | "tool";
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
        return {
          filePath: opts.filePath,
          chunksAdded: 0,
          chunksRemoved: 0,
          durationMs: Date.now() - start,
        };
      }
      rawText = entry.text;
    } else {
      rawText = await extractText(opts.filePath, ext, opts.cfg);
    }

    if (!rawText.trim()) {
      return {
        filePath: opts.filePath,
        chunksAdded: 0,
        chunksRemoved: 0,
        durationMs: Date.now() - start,
      };
    }

    // 2. Convert to relative path for storage
    const relPath = toRelativePath(opts.filePath, opts.workspaceDir);
    const source = opts.source ?? (isSession ? "sessions" : "memory");
    // Auto-detect content type based on filename
    const basename = path.basename(opts.filePath).toLowerCase();
    let contentType = opts.contentType ?? "knowledge";
    if (basename === "tools.md" || basename.startsWith("tools-")) {
      contentType = "tool" as typeof contentType;
    }

    // 3. Chunk — use reference.chunkSize for reference content, cfg.chunk for others
    const chunkCfg = opts.cfg.chunk;
    const chunkSize =
      contentType === "reference"
        ? (opts.cfg.reference.chunkSize ?? 800)
        : (chunkCfg?.maxTokens ?? 400);
    const rawChunks = chunkDocument(
      {
        text: rawText,
        path: relPath,
        source,
        ext: isSession ? "txt" : ext,
      },
      {
        maxTokens: chunkSize,
        overlapTokens: chunkCfg?.overlapTokens,
        minTokens: chunkCfg?.minTokens,
      },
    );

    if (rawChunks.length === 0) {
      return {
        filePath: opts.filePath,
        chunksAdded: 0,
        chunksRemoved: 0,
        durationMs: Date.now() - start,
      };
    }

    // 3b. Quality gate — score chunks and drop noise before embedding
    const minQuality = opts.cfg.ingest?.minQuality ?? 0.3;
    const qualifiedWithScores = rawChunks
      .map((c) => {
        const quality = scoreChunkQuality(c.text, relPath, source, {
          language: opts.cfg.ingest?.language,
          threshold: opts.cfg.ingest?.languageThreshold,
        });
        return { chunk: c, qualityScore: quality.score };
      })
      .filter((item) => item.qualityScore >= minQuality);

    if (qualifiedWithScores.length === 0) {
      opts.logger?.info(
        `memory-spark: ${source} ${relPath} — all ${rawChunks.length} chunks filtered by quality gate`,
      );
      return {
        filePath: opts.filePath,
        chunksAdded: 0,
        chunksRemoved: 0,
        durationMs: Date.now() - start,
      };
    }

    // 3c. Clean chunk text — strip metadata noise before embedding
    const cleanedWithScores = qualifiedWithScores
      .map((item) => ({
        chunk: { ...item.chunk, text: cleanChunkText(item.chunk.text) },
        qualityScore: item.qualityScore,
      }))
      .filter((item) => item.chunk.text.trim().length > 0);

    if (cleanedWithScores.length === 0) {
      return {
        filePath: opts.filePath,
        chunksAdded: 0,
        chunksRemoved: 0,
        durationMs: Date.now() - start,
      };
    }

    const cleanedChunks = cleanedWithScores.map((item) => item.chunk);
    const qualityScores = cleanedWithScores.map((item) => item.qualityScore);

    // 4. NER tag (best-effort, sequential to avoid overwhelming single-worker service)
    const entitiesPerChunk: string[][] = [];
    for (const c of cleanedChunks) {
      try {
        entitiesPerChunk.push(await tagEntities(c.text, opts.cfg));
      } catch {
        entitiesPerChunk.push([]);
      }
    }

    // 5. Embed raw chunk text.
    //    TODO (Phase 2B): Implement Anthropic "Contextual Retrieval" — prepend
    //    LLM-generated context prefix per chunk before embedding. Currently we
    //    embed raw text only. Contextual metadata (content_type, parent_heading)
    //    is stored as columns for post-retrieval use, not embedded.
    const vectors = await opts.embed.embedBatch(cleanedChunks.map((c) => c.text));

    // 6. Build MemoryChunk objects with RELATIVE paths
    //    Use file mtime as updated_at — NOT current time.
    //    This preserves temporal accuracy: a file from February stays "February"
    //    even when re-indexed on a gateway restart. Without this, every restart
    //    resets all timestamps and breaks temporal decay scoring.
    let fileTime: string;
    try {
      const stat = await import("fs/promises").then((fs) => fs.stat(opts.filePath));
      fileTime = stat.mtime.toISOString();
    } catch {
      fileTime = new Date().toISOString(); // fallback for virtual/missing files
    }
    const chunks: MemoryChunk[] = cleanedChunks.map((raw, i) => {
      const chunk: MemoryChunk = {
        id: chunkId(relPath, raw.startLine, opts.agentId),
        path: relPath,
        source,
        agent_id: opts.agentId,
        start_line: raw.startLine,
        end_line: raw.endLine,
        text: raw.text,
        vector: vectors[i]!,
        updated_at: fileTime,
        entities: JSON.stringify(entitiesPerChunk[i] ?? []),
        content_type: contentType,
        quality_score: qualityScores[i] ?? 0.5,
        token_count: estimateTokens(raw.text),
        parent_heading: raw.parentHeading ?? "",
      };
      // Assign pool based on content_type and path
      chunk.pool = resolvePool(chunk);
      return chunk;
    });

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
