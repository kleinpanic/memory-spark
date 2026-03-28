/**
 * Smart Chunker — token-aware, markdown-aware document splitting.
 */

import { createHash } from "node:crypto";

export interface ChunkInput {
  text: string;
  path: string;
  source: "memory" | "sessions" | "ingest" | "capture";
  ext?: string;
}

export interface RawChunk {
  text: string;
  startLine: number;
  endLine: number;
  /** Most recent markdown heading (## or ###) above this chunk, if any */
  parentHeading?: string;
}

export interface ChunkerOptions {
  maxTokens?: number; // default: 400
  overlapTokens?: number; // default: 50
  minTokens?: number; // default: 20
}

/** Approximate tokens from character count (4 chars ≈ 1 token for English) */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Split a document into overlapping chunks ready for embedding.
 */
export function chunkDocument(input: ChunkInput, opts: ChunkerOptions = {}): RawChunk[] {
  const maxTokens = opts.maxTokens ?? 400;
  const overlapTokens = opts.overlapTokens ?? 50;
  const minTokens = opts.minTokens ?? 20;
  const maxChars = maxTokens * 4;
  const overlapChars = overlapTokens * 4;
  const minChars = minTokens * 4;

  if (!input.text.trim()) return [];

  const ext = input.ext ?? "txt";
  const isMarkdown = ext === "md" || ext === "rst";

  // Step 1: Split into sections
  const sections = isMarkdown ? splitMarkdownSections(input.text) : splitParagraphs(input.text);

  // Step 2: Split oversized sections + merge tiny ones
  const chunks: RawChunk[] = [];
  let lineOffset = 0;
  let lastHeading: string | undefined;

  for (const section of sections) {
    const sectionLines = section.split("\n");

    // Track the most recent heading for contextual prefix generation
    if (isMarkdown) {
      const heading = extractSectionHeading(section);
      if (heading) lastHeading = heading;
    }

    if (section.length <= maxChars) {
      if (section.trim().length >= minChars) {
        chunks.push({
          text: section.trim(),
          startLine: lineOffset + 1,
          endLine: lineOffset + sectionLines.length,
          parentHeading: lastHeading,
        });
      }
    } else {
      // Hard-split oversized section with overlap
      const subChunks = hardSplitWithOverlap(section, maxChars, overlapChars, lineOffset);
      for (const sc of subChunks) {
        if (sc.text.trim().length >= minChars) {
          chunks.push({ ...sc, parentHeading: lastHeading });
        }
      }
    }

    lineOffset += sectionLines.length;
  }

  return chunks;
}

/**
 * Extract the heading text from the first line of a markdown section.
 * Returns undefined if the section doesn't start with a heading.
 */
function extractSectionHeading(section: string): string | undefined {
  const firstLine = section.split("\n")[0]?.trim() ?? "";
  if (/^#{1,3}\s/.test(firstLine)) {
    return firstLine.replace(/^#+\s*/, "");
  }
  return undefined;
}

/**
 * Split markdown on heading boundaries (## or ###).
 * Keeps code blocks intact.
 */
function splitMarkdownSections(text: string): string[] {
  const lines = text.split("\n");
  const sections: string[] = [];
  let current: string[] = [];
  let inCodeBlock = false;

  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      inCodeBlock = !inCodeBlock;
      current.push(line);
      continue;
    }

    if (!inCodeBlock && /^#{1,3}\s/.test(line) && current.length > 0) {
      sections.push(current.join("\n"));
      current = [line];
    } else {
      current.push(line);
    }
  }

  if (current.length > 0) {
    sections.push(current.join("\n"));
  }

  return sections;
}

/**
 * Split plain text on double newlines (paragraph breaks).
 */
function splitParagraphs(text: string): string[] {
  const paras = text.split(/\n\s*\n/);
  return paras.filter((p) => p.trim().length > 0);
}

/**
 * Clean chunk text before embedding — strip noise patterns so the vector
 * represents the actual content, not metadata wrappers.
 */
export function cleanChunkText(text: string): string {
  // Strip conversation metadata blocks
  text = text.replace(/```json\s*\{[^}]*"message_id"[^}]*\}\s*```/gs, "");
  text = text.replace(/Conversation info \(untrusted metadata\):[^]*?```\s*/gs, "");
  text = text.replace(/Sender \(untrusted metadata\):[^]*?```\s*/gs, "");

  // Strip timestamp headers
  text = text.replace(/\[\w{3} \d{4}-\d{2}-\d{2} \d{2}:\d{2} \w+\]/g, "");

  // Strip exec session IDs
  text = text.replace(/\(session=[a-f0-9-]+,?\s*(?:id=[a-f0-9-]+,?\s*)?code \d+\)/g, "");

  // Collapse excessive whitespace
  text = text.replace(/\n{3,}/g, "\n\n").trim();

  return text;
}

// ── Parent-Child Hierarchical Chunking ────────────────────────────────────
//
// Creates a two-level hierarchy:
//   Parent chunks: ~2000 tokens — large context windows for delivery
//   Child chunks: ~200 tokens — precise search targets
//
// Search is done against child chunks (small = precise embeddings).
// When a child matches, its parent is retrieved for context injection
// (large = full surrounding context).
//
// This is the single biggest recall improvement in modern RAG:
// small chunks embed precisely, large chunks deliver full context.

export interface ParentChildChunk {
  parent: RawChunk & { id: string };
  children: Array<RawChunk & { parentId: string }>;
}

export interface HierarchicalChunkerOptions {
  parentMaxTokens?: number; // default: 2000
  childMaxTokens?: number; // default: 200
  childOverlapTokens?: number; // default: 25
  childMinTokens?: number; // default: 15
}

/**
 * Split a document into hierarchical parent-child chunks.
 * Parents are large context windows; children are precise search targets.
 *
 * Usage: search against children, retrieve parent for context delivery.
 */
export function chunkDocumentHierarchical(
  input: ChunkInput,
  opts: HierarchicalChunkerOptions = {},
): ParentChildChunk[] {
  const parentMaxTokens = opts.parentMaxTokens ?? 2000;
  const childMaxTokens = opts.childMaxTokens ?? 200;
  const childOverlapTokens = opts.childOverlapTokens ?? 25;
  const childMinTokens = opts.childMinTokens ?? 15;

  if (!input.text.trim()) return [];

  // Step 1: Create parent-sized chunks using the standard chunker
  const parentRawChunks = chunkDocument(input, {
    maxTokens: parentMaxTokens,
    overlapTokens: 100, // parents get more overlap for context continuity
    minTokens: 40,
  });

  if (parentRawChunks.length === 0) return [];

  const results: ParentChildChunk[] = [];

  for (const parentChunk of parentRawChunks) {
    // Generate stable parent ID from content hash
    const parentId = `p_${stableHash(input.path + ":" + parentChunk.startLine)}`;

    // Step 2: Split each parent into child-sized chunks
    const childRawChunks = chunkDocument(
      {
        text: parentChunk.text,
        path: input.path,
        source: input.source,
        ext: input.ext,
      },
      {
        maxTokens: childMaxTokens,
        overlapTokens: childOverlapTokens,
        minTokens: childMinTokens,
      },
    );

    // Adjust child line numbers to be relative to document, not parent
    const children = childRawChunks.map((child) => ({
      ...child,
      startLine: parentChunk.startLine + child.startLine - 1,
      endLine: parentChunk.startLine + child.endLine - 1,
      parentHeading: child.parentHeading ?? parentChunk.parentHeading,
      parentId,
    }));

    results.push({
      parent: { ...parentChunk, id: parentId },
      children,
    });
  }

  return results;
}

/**
 * Stable content-based hash for generating parent IDs.
 * Uses SHA-1 truncated to 16 hex chars (64 bits of entropy).
 * Birthday collision probability <0.01% at 1M entries (vs the old 31-bit hash's 50% at 65k).
 */
function stableHash(input: string): string {
  return createHash("sha1").update(input).digest("hex").slice(0, 16);
}

/**
 * Hard-split a string at character boundaries with overlap.
 */
function hardSplitWithOverlap(
  text: string,
  maxChars: number,
  overlapChars: number,
  baseLineOffset: number,
): RawChunk[] {
  const chunks: RawChunk[] = [];
  let offset = 0;

  while (offset < text.length) {
    const end = Math.min(offset + maxChars, text.length);
    // Try to break at a newline or space
    let breakPoint = end;
    if (end < text.length) {
      const lastNewline = text.lastIndexOf("\n", end);
      if (lastNewline > offset + maxChars / 2) {
        breakPoint = lastNewline + 1;
      } else {
        const lastSpace = text.lastIndexOf(" ", end);
        if (lastSpace > offset + maxChars / 2) {
          breakPoint = lastSpace + 1;
        }
      }
    }

    const chunk = text.slice(offset, breakPoint);
    const startLine = baseLineOffset + text.slice(0, offset).split("\n").length;
    const endLine = baseLineOffset + text.slice(0, breakPoint).split("\n").length;

    chunks.push({ text: chunk, startLine, endLine });

    // We've reached the end
    if (breakPoint >= text.length) {
      break;
    }

    // Next chunk starts overlapChars before the breakpoint, but MUST advance forward
    const nextOffset = breakPoint - overlapChars;
    offset = Math.max(nextOffset, offset + 1);
  }

  return chunks;
}
