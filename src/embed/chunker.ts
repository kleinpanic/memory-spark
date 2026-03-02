/**
 * Smart Chunker
 *
 * Splits documents into embedding-ready chunks with:
 *   - Token-aware sizing (tiktoken cl100k_base)
 *   - Overlapping windows for context continuity
 *   - Markdown-aware splitting (respect headers, code blocks)
 *   - Metadata extraction per chunk (line numbers, heading path)
 *
 * Default chunk sizes:
 *   maxTokens:     400   (stays well under 512 token limit)
 *   overlapTokens: 50    (context continuity across chunk boundaries)
 *
 * Markdown splitting priority:
 *   1. Split on ## / ### headers (natural semantic boundary)
 *   2. Split on blank lines between paragraphs
 *   3. Hard split at maxTokens if no boundary found
 *   4. Never split inside a fenced code block
 */

export interface ChunkInput {
  text: string;
  path: string;
  agentId?: string;
  source: "memory" | "sessions" | "ingest" | "capture";
  /** File extension hint for format-aware splitting */
  ext?: string;
}

export interface RawChunk {
  text: string;
  startLine: number;
  endLine: number;
  /** Markdown heading path for this chunk, e.g. "## Setup > ### Config" */
  headingPath?: string;
}

export interface ChunkerOptions {
  maxTokens?: number;      // default: 400
  overlapTokens?: number;  // default: 50
  minTokens?: number;      // default: 20 (discard tiny chunks)
}

/**
 * Split a document into overlapping chunks ready for embedding.
 */
export function chunkDocument(input: ChunkInput, opts: ChunkerOptions = {}): RawChunk[] {
  // TODO:
  // 1. If ext is "md" → markdownAwareChunk()
  // 2. Else → plainTextChunk()
  // 3. Apply overlap sliding window
  // 4. Filter out chunks below minTokens
  throw new Error("chunkDocument() not yet implemented");
}

/**
 * Markdown-aware splitter — respects headers, code fences, tables.
 */
function markdownAwareChunk(_text: string, _opts: ChunkerOptions): RawChunk[] {
  // TODO: parse markdown AST, split at heading boundaries
  throw new Error("markdownAwareChunk() not yet implemented");
}

/**
 * Plain text splitter — splits on paragraph breaks, then hard-splits at token limit.
 */
function plainTextChunk(_text: string, _opts: ChunkerOptions): RawChunk[] {
  // TODO: split on \n\n, then token-window
  throw new Error("plainTextChunk() not yet implemented");
}

/**
 * Count approximate tokens in a string (fast estimate, not exact BPE).
 * For exact counts use tiktoken, but this saves 50ms for small inputs.
 */
export function estimateTokens(text: string): number {
  // ~4 chars per token is a good approximation for English
  return Math.ceil(text.length / 4);
}
