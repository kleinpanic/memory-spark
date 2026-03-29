/**
 * File parsers — extract raw text from various formats.
 */

import { execFile } from "node:child_process";
import fs from "node:fs/promises";
import { promisify } from "node:util";

import type { MemorySparkConfig } from "../config.js";

const execFileAsync = promisify(execFile);

export const SUPPORTED_EXTS = new Set([
  "md",
  "txt",
  "rst",
  "pdf",
  "docx",
  "mp3",
  "wav",
  "m4a",
  "ogg",
  "flac",
  "opus",
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

    case "mp3":
    case "wav":
    case "m4a":
    case "ogg":
    case "flac":
    case "opus":
      return extractAudio(filePath, cfg);

    default:
      throw new Error(`memory-spark: unsupported file type ".${ext}"`);
  }
}

/** Maximum pages to process in a single pdftotext batch */
const PDF_BATCH_PAGES = 25;

/** Timeout for GLM-OCR (per batch) in ms — large PDFs need time */
const GLM_OCR_TIMEOUT_MS = 600_000; // 10 minutes

/** Timeout for EasyOCR (per batch) in ms */
const EASY_OCR_TIMEOUT_MS = 300_000; // 5 minutes

/**
 * Get PDF page count using pdfinfo.
 * Returns null if pdfinfo is unavailable or fails.
 */
async function getPdfPageCount(filePath: string): Promise<number | null> {
  try {
    const { stdout } = await execFileAsync("pdfinfo", [filePath]);
    const match = stdout.match(/Pages:\s*(\d+)/);
    return match ? parseInt(match[1]!, 10) : null;
  } catch {
    return null;
  }
}

/**
 * Extract text from a PDF page range using pdftotext.
 * Returns empty string if the range fails.
 */
async function extractPdfPageRange(
  filePath: string,
  firstPage: number,
  lastPage: number,
): Promise<string> {
  try {
    const { stdout } = await execFileAsync("pdftotext", [
      "-layout",
      "-q",
      "-f",
      String(firstPage),
      "-l",
      String(lastPage),
      filePath,
      "-",
    ]);
    return stdout;
  } catch {
    return "";
  }
}

async function extractPdf(filePath: string, cfg: MemorySparkConfig): Promise<string> {
  const pageCount = await getPdfPageCount(filePath);

  // ── Strategy 1: pdftotext (fast, native) ────────────────────────────────
  // For large PDFs, process in page batches to avoid memory spikes
  if (pageCount !== null && pageCount > PDF_BATCH_PAGES) {
    // Paginated extraction — process in batches of PDF_BATCH_PAGES
    const batchTexts: string[] = [];
    for (let start = 1; start <= pageCount; start += PDF_BATCH_PAGES) {
      const end = Math.min(start + PDF_BATCH_PAGES - 1, pageCount);
      const batchText = await extractPdfPageRange(filePath, start, end);
      if (batchText.trim()) {
        batchTexts.push(batchText);
      }
    }
    const fullText = batchTexts.join("\n\n");
    if (fullText.trim().length > 50) return fullText;
  } else {
    // Small PDF or no pdfinfo — try extracting all at once
    try {
      const { stdout } = await execFileAsync("pdftotext", ["-layout", "-q", filePath, "-"]);
      if (stdout.trim().length > 50) return stdout;
    } catch {
      // pdftotext not available or failed
    }
  }

  // ── Strategy 2: pdf-parse (JS fallback) ─────────────────────────────────
  // Only use for small PDFs (< 2MB) to avoid heap OOM
  try {
    const stat = await fs.stat(filePath);
    if (stat.size < 2 * 1024 * 1024) {
      const pdfParse = (await import("pdf-parse")).default;
      const buffer = await fs.readFile(filePath);
      const data = await pdfParse(buffer);
      if (data.text.trim().length > 50) return data.text;
    }
  } catch {
    // pdf-parse not available or file too large
  }

  // ── Strategy 3: GLM-OCR via vLLM (for scanned PDFs) ────────────────────
  // For large PDFs, split into page-range batches for OCR too
  try {
    const stat = await fs.stat(filePath);
    const fileSizeMB = stat.size / (1024 * 1024);

    if (fileSizeMB <= 10) {
      // Small enough to send as one request
      const buffer = await fs.readFile(filePath);
      const base64 = buffer.toString("base64");
      const text = await glmOcrRequest(base64, cfg, GLM_OCR_TIMEOUT_MS);
      if (text && text.length > 50) return text;
    } else if (pageCount !== null) {
      // Large PDF: use pdftoppm to extract page images in batches, then OCR each
      // For now, skip OCR on very large PDFs — pdftotext should have worked above
      // This is a future enhancement slot
    }
  } catch {
    // GLM-OCR not available — fall through to EasyOCR
  }

  // ── Strategy 4: EasyOCR (legacy fallback) ───────────────────────────────
  try {
    const stat = await fs.stat(filePath);
    if (stat.size < 5 * 1024 * 1024) {
      const buffer = await fs.readFile(filePath);
      const formData = new FormData();
      formData.append("file", new Blob([buffer]), "document.pdf");

      const resp = await fetch(cfg.spark.ocr, {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(EASY_OCR_TIMEOUT_MS),
      });
      if (resp.ok) {
        const data = (await resp.json()) as { text: string };
        if (data.text?.trim().length > 50) return data.text;
      }
    }
  } catch {
    // Legacy OCR not available
  }

  throw new Error(`memory-spark: could not extract text from PDF: ${filePath}`);
}

/**
 * Send a single GLM-OCR request for a base64-encoded PDF.
 * Returns extracted text or null on failure.
 */
async function glmOcrRequest(
  base64: string,
  cfg: MemorySparkConfig,
  timeoutMs: number,
): Promise<string | null> {
  try {
    const resp = await fetch(`${cfg.spark.glmOcr}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(cfg.embed.spark?.apiKey ? { Authorization: `Bearer ${cfg.embed.spark.apiKey}` } : {}),
      },
      body: JSON.stringify({
        model: "zai-org/GLM-OCR",
        messages: [
          {
            role: "user",
            content: [
              {
                type: "image_url",
                image_url: { url: `data:application/pdf;base64,${base64}` },
              },
              {
                type: "text",
                text: "Extract all text from this document. Output only the extracted text in markdown format, preserving structure.",
              },
            ],
          },
        ],
        max_tokens: 8192,
        temperature: 0,
      }),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (resp.ok) {
      const data = (await resp.json()) as { choices: Array<{ message: { content: string } }> };
      return data.choices?.[0]?.message?.content?.trim() ?? null;
    }
  } catch {
    // OCR failed
  }
  return null;
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

  const data = (await resp.json()) as { text: string };
  return data.text;
}
