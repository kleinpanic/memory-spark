/**
 * Plain-text chunk line offsets.
 *
 * chunkDocument() must report each chunk's real line range in the source
 * document. Plain-text input is split on blank lines, and the blank separator
 * lines must still count toward the offsets of later chunks.
 */

import { describe, it, expect } from "vitest";
import { chunkDocument } from "../src/embed/chunker.js";

const doc = [
  "Alpha paragraph with enough text to survive the minTokens floor.",
  "",
  "Beta paragraph with enough text to survive the minTokens floor.",
  "",
  "Gamma paragraph with enough text to survive the minTokens floor.",
].join("\n");

describe("chunkDocument line offsets", () => {
  it("reports the real source lines for plain-text paragraphs", () => {
    const chunks = chunkDocument(
      { text: doc, path: "notes.txt", source: "memory", ext: "txt" },
      { maxTokens: 400, overlapTokens: 0, minTokens: 10 },
    );

    expect(chunks.map((c) => c.text)).toEqual([
      "Alpha paragraph with enough text to survive the minTokens floor.",
      "Beta paragraph with enough text to survive the minTokens floor.",
      "Gamma paragraph with enough text to survive the minTokens floor.",
    ]);
    expect(chunks.map((c) => [c.startLine, c.endLine])).toEqual([
      [1, 1],
      [3, 3],
      [5, 5],
    ]);
  });
});
