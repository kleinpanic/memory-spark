/**
 * Zero-Shot Classifier — Spark 18113 (bart-large-mnli).
 * Endpoint: POST /v1/classify with {"text": "...", "labels": [...]}
 * Returns { label: "none", score: 0 } on failure (safe default).
 */

import type { MemorySparkConfig } from "../config.js";

export type CaptureCategory = "fact" | "preference" | "decision" | "code-snippet" | "none";

export interface ClassifyResult {
  label: CaptureCategory;
  score: number;
}

export const CAPTURE_LABELS: CaptureCategory[] = [
  "fact",
  "preference",
  "decision",
  "code-snippet",
  "none",
];

export async function classifyForCapture(
  text: string,
  cfg: MemorySparkConfig,
  minConfidence = 0.75,
): Promise<ClassifyResult> {
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (cfg.embed.spark?.apiKey) {
      headers["Authorization"] = `Bearer ${cfg.embed.spark.apiKey}`;
    }
    const resp = await fetch(`${cfg.spark.zeroShot}/v1/classify`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        text: text.slice(0, 2000),
        labels: CAPTURE_LABELS.filter((l) => l !== "none"),
      }),
      signal: AbortSignal.timeout(10000),
    });
    if (!resp.ok) return { label: "none", score: 0 };

    const data = (await resp.json()) as { labels: string[]; scores: number[] };
    const topLabel = data.labels[0] as CaptureCategory;
    const topScore = data.scores[0]!;

    if (topScore < minConfidence) {
      return { label: "none", score: topScore };
    }
    return { label: topLabel, score: topScore };
  } catch {
    return { label: "none", score: 0 };
  }
}
