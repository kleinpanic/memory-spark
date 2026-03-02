/**
 * File parsers — extract raw text from various formats.
 *
 * Dispatch table (by file extension):
 *   md, txt, rst  → identity (return as-is)
 *   pdf           → pdftotext (via exec) → Spark OCR 18097 fallback for scanned pages
 *   docx          → mammoth HTML → strip HTML tags
 *   epub          → TODO: phase 2
 *   mp3,wav,m4a,  → Spark STT 18094 (parakeet) — returns plain transcript text
 *   ogg,flac,opus
 */

import type { MemorySparkConfig } from "../config.js";

/** Supported ingestion extensions */
export const SUPPORTED_EXTS = new Set([
  "md", "txt", "rst",
  "pdf",
  "docx",
  "mp3", "wav", "m4a", "ogg", "flac", "opus",
]);

/**
 * Extract plain text from a file at the given path.
 * Returns the full text content, or throws on unrecoverable error.
 */
export async function extractText(
  filePath: string,
  ext: string,
  cfg: MemorySparkConfig,
): Promise<string> {
  switch (ext) {
    case "md":
    case "txt":
    case "rst":
      return extractPlainText(filePath);

    case "pdf":
      return extractPdf(filePath, cfg);

    case "docx":
      return extractDocx(filePath);

    case "mp3":
    case "wav":
    case "m4a":
    case "ogg":
    case "flac":
    case "opus":
      return extractAudio(filePath, cfg);

    default:
      throw new Error(`memory-spark: unsupported file type ".${ext}" for path ${filePath}`);
  }
}

async function extractPlainText(filePath: string): Promise<string> {
  // TODO: fs.readFile(filePath, "utf-8")
  throw new Error("extractPlainText() not yet implemented");
}

async function extractPdf(filePath: string, cfg: MemorySparkConfig): Promise<string> {
  // TODO:
  // 1. Try pdftotext (exec "pdftotext -layout -q <file> -")
  // 2. If output is empty or mostly whitespace → scanned PDF → Spark OCR fallback
  //    POST to cfg.spark.ocr with the PDF bytes
  // 3. Return extracted text
  throw new Error("extractPdf() not yet implemented");
}

async function extractDocx(filePath: string): Promise<string> {
  // TODO: mammoth.extractRawText({ path: filePath }) → .value (strips all formatting)
  throw new Error("extractDocx() not yet implemented");
}

async function extractAudio(filePath: string, cfg: MemorySparkConfig): Promise<string> {
  // TODO:
  // POST audio bytes to cfg.spark.stt (parakeet HTTP endpoint)
  // Returns: { text: "transcript..." }
  throw new Error("extractAudio() not yet implemented");
}
