/**
 * File parsers — extract raw text from various formats.
 */

import type { MemorySparkConfig } from "../config.js";
import fs from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export const SUPPORTED_EXTS = new Set([
  "md", "txt", "rst",
  "pdf",
  "docx",
  "mp3", "wav", "m4a", "ogg", "flac", "opus",
]);

export async function extractText(
  filePath: string,
  ext: string,
  cfg: MemorySparkConfig,
): Promise<string> {
  switch (ext) {
    case "md":
    case "txt":
    case "rst":
      return fs.readFile(filePath, "utf-8");

    case "pdf":
      return extractPdf(filePath, cfg);

    case "docx":
      return extractDocx(filePath);

    case "mp3": case "wav": case "m4a":
    case "ogg": case "flac": case "opus":
      return extractAudio(filePath, cfg);

    default:
      throw new Error(`memory-spark: unsupported file type ".${ext}"`);
  }
}

async function extractPdf(filePath: string, cfg: MemorySparkConfig): Promise<string> {
  // Try pdftotext first (fast, no JS dep)
  try {
    const { stdout } = await execFileAsync("pdftotext", ["-layout", "-q", filePath, "-"]);
    if (stdout.trim().length > 50) return stdout;
  } catch {
    // pdftotext not available or failed
  }

  // Fallback: pdf-parse library
  try {
    const pdfParse = (await import("pdf-parse")).default;
    const buffer = await fs.readFile(filePath);
    const data = await pdfParse(buffer);
    if (data.text.trim().length > 50) return data.text;
  } catch {
    // pdf-parse not available
  }

  // Fallback: Spark OCR for scanned PDFs
  try {
    const buffer = await fs.readFile(filePath);
    const formData = new FormData();
    formData.append("file", new Blob([buffer]), "document.pdf");

    const resp = await fetch(cfg.spark.ocr, {
      method: "POST",
      body: formData,
      signal: AbortSignal.timeout(60000),
    });
    if (resp.ok) {
      const data = await resp.json() as { text: string };
      return data.text;
    }
  } catch {
    // OCR not available
  }

  throw new Error(`memory-spark: could not extract text from PDF: ${filePath}`);
}

async function extractDocx(filePath: string): Promise<string> {
  try {
    const mammoth = await import("mammoth");
    const result = await mammoth.extractRawText({ path: filePath });
    return result.value;
  } catch {
    throw new Error(`memory-spark: mammoth not available for docx extraction: ${filePath}`);
  }
}

async function extractAudio(filePath: string, cfg: MemorySparkConfig): Promise<string> {
  const buffer = await fs.readFile(filePath);
  const formData = new FormData();
  formData.append("file", new Blob([buffer]), filePath.split("/").pop() ?? "audio");

  const resp = await fetch(`${cfg.spark.stt}/transcribe`, {
    method: "POST",
    body: formData,
    signal: AbortSignal.timeout(120000),
  });

  if (!resp.ok) {
    throw new Error(`memory-spark: STT failed for ${filePath}: ${resp.status}`);
  }

  const data = await resp.json() as { text: string };
  return data.text;
}
