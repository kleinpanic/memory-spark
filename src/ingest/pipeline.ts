/**
 * Ingest Pipeline — extract → chunk → NER → embed → store.
 * Handles both regular files AND session JSONL transcripts.
 */

import crypto from "node:crypto";
import path from "node:path";

import { tagEntities } from "../classify/ner.js";
import { scoreChunkQuality } from "../classify/quality.js";
import type { MemorySparkConfig } from "../config.js";
import {
  chunkDocument,
  chunkDocumentHierarchical,
  cleanChunkText,
  estimateTokens,
} from "../embed/chunker.js";
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
  /** Use hierarchical parent-child chunking for better recall. Default: false */
  hierarchical?: boolean;
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

    // 3. Chunk — hierarchical (parent-child) or flat depending on config
    const chunkCfg = opts.cfg.chunk;
    const useHierarchical = opts.hierarchical ?? chunkCfg?.hierarchical ?? true;
    const chunkInput = {
      text: rawText,
      path: relPath,
      source: source as "memory" | "sessions" | "ingest" | "capture",
      ext: isSession ? "txt" : ext,
    };

    // Quality gate helper
    const minQuality = opts.cfg.ingest?.minQuality ?? 0.3;
    const qualityGate = (text: string) => {
      const quality = scoreChunkQuality(text, relPath, source, {
        language: opts.cfg.ingest?.language,
        threshold: opts.cfg.ingest?.languageThreshold,
      });
      return quality.score >= minQuality ? quality.score : null;
    };

    // Get file mtime for temporal accuracy
    let fileTime: string;
    try {
      const stat = await import("fs/promises").then((fs) => fs.stat(opts.filePath));
      fileTime = stat.mtime.toISOString();
    } catch {
      fileTime = new Date().toISOString();
    }

    let allChunksToStore: MemoryChunk[];

    if (useHierarchical) {
      // ── Hierarchical Parent-Child Chunking ──────────────────────────────
      const hierarchical = chunkDocumentHierarchical(chunkInput, {
        parentMaxTokens: chunkCfg?.parentMaxTokens ?? 2000,
        childMaxTokens: chunkCfg?.childMaxTokens ?? 200,
        childOverlapTokens: chunkCfg?.childOverlapTokens ?? 25,
      });

      if (hierarchical.length === 0) {
        return {
          filePath: opts.filePath,
          chunksAdded: 0,
          chunksRemoved: 0,
          durationMs: Date.now() - start,
        };
      }

      // Collect all texts to embed (parents first, then children)
      const parentTexts: string[] = [];
      const childTexts: string[] = [];
      const parentMeta: Array<{ parentId: string; chunk: typeof hierarchical[0]["parent"]; quality: number }> = [];
      const childMeta: Array<{ parentId: string; chunk: typeof hierarchical[0]["children"][0]; quality: number }> = [];

      for (const group of hierarchical) {
        const cleanedParentText = cleanChunkText(group.parent.text);
        const parentQuality = qualityGate(cleanedParentText);
        if (parentQuality === null || !cleanedParentText.trim()) continue;

        parentTexts.push(cleanedParentText);
        parentMeta.push({ parentId: group.parent.id, chunk: group.parent, quality: parentQuality });

        for (const child of group.children) {
          const cleanedChildText = cleanChunkText(child.text);
          const childQuality = qualityGate(cleanedChildText);
          if (childQuality === null || !cleanedChildText.trim()) continue;

          childTexts.push(cleanedChildText);
          childMeta.push({ parentId: group.parent.id, chunk: child, quality: childQuality });
        }
      }

      if (parentTexts.length === 0) {
        return {
          filePath: opts.filePath,
          chunksAdded: 0,
          chunksRemoved: 0,
          durationMs: Date.now() - start,
        };
      }

      // Embed all in one batch (parents + children)
      const allTexts = [...parentTexts, ...childTexts];
      const allVectors = await opts.embed.embedBatch(allTexts);

      // NER for all chunks (best effort)
      const allEntities: string[][] = [];
      for (const text of allTexts) {
        try {
          allEntities.push(await tagEntities(text, opts.cfg));
        } catch {
          allEntities.push([]);
        }
      }

      allChunksToStore = [];
      const parentCount = parentTexts.length;

      // Build parent MemoryChunks
      for (let i = 0; i < parentCount; i++) {
        const meta = parentMeta[i]!;
        const chunk: MemoryChunk = {
          id: meta.parentId,
          path: relPath,
          source,
          agent_id: opts.agentId,
          start_line: meta.chunk.startLine,
          end_line: meta.chunk.endLine,
          text: parentTexts[i]!,
          vector: allVectors[i]!,
          updated_at: fileTime,
          entities: JSON.stringify(allEntities[i] ?? []),
          content_type: contentType,
          quality_score: meta.quality,
          token_count: estimateTokens(parentTexts[i]!),
          parent_heading: meta.chunk.parentHeading ?? "",
          is_parent: true,
        };
        chunk.pool = resolvePool(chunk);
        allChunksToStore.push(chunk);
      }

      // Build child MemoryChunks
      for (let i = 0; i < childTexts.length; i++) {
        const meta = childMeta[i]!;
        const vecIdx = parentCount + i;
        const chunk: MemoryChunk = {
          id: chunkId(relPath, meta.chunk.startLine, opts.agentId + ":" + meta.parentId),
          path: relPath,
          source,
          agent_id: opts.agentId,
          start_line: meta.chunk.startLine,
          end_line: meta.chunk.endLine,
          text: childTexts[i]!,
          vector: allVectors[vecIdx]!,
          updated_at: fileTime,
          entities: JSON.stringify(allEntities[vecIdx] ?? []),
          content_type: contentType,
          quality_score: meta.quality,
          token_count: estimateTokens(childTexts[i]!),
          parent_heading: meta.chunk.parentHeading ?? "",
          parent_id: meta.parentId,
          is_parent: false,
        };
        chunk.pool = resolvePool(chunk);
        allChunksToStore.push(chunk);
      }
    } else {
      // ── Flat Chunking (legacy) ──────────────────────────────────────────
      const chunkSize =
        contentType === "reference"
          ? (opts.cfg.reference.chunkSize ?? 800)
          : (chunkCfg?.maxTokens ?? 400);
      const rawChunks = chunkDocument(chunkInput, {
        maxTokens: chunkSize,
        overlapTokens: chunkCfg?.overlapTokens,
        minTokens: chunkCfg?.minTokens,
      });

      if (rawChunks.length === 0) {
        return {
          filePath: opts.filePath,
          chunksAdded: 0,
          chunksRemoved: 0,
          durationMs: Date.now() - start,
        };
      }

      // Quality gate
      const qualifiedWithScores = rawChunks
        .map((c) => {
          const score = qualityGate(c.text);
          return score !== null ? { chunk: c, qualityScore: score } : null;
        })
        .filter((item): item is NonNullable<typeof item> => item !== null);

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

      // Clean + filter empty
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

      // NER tag
      const entitiesPerChunk: string[][] = [];
      for (const c of cleanedChunks) {
        try {
          entitiesPerChunk.push(await tagEntities(c.text, opts.cfg));
        } catch {
          entitiesPerChunk.push([]);
        }
      }

      // Embed
      const vectors = await opts.embed.embedBatch(cleanedChunks.map((c) => c.text));

      // Build MemoryChunks
      allChunksToStore = cleanedChunks.map((raw, i) => {
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
        chunk.pool = resolvePool(chunk);
        return chunk;
      });
    }

    // 7. Remove old chunks for this path, then upsert new
    const removed = await opts.backend.deleteByPath(relPath, opts.agentId);
    await opts.backend.upsert(allChunksToStore);

    opts.logger?.info(`memory-spark: ${source} ${relPath} → ${allChunksToStore.length} chunks`);

    return {
      filePath: opts.filePath,
      chunksAdded: allChunksToStore.length,
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
