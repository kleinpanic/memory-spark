/**
 * Zero-Shot Classifier — Spark 18113 (bart-large-mnli).
 * Returns { label: "none", score: 0 } on failure (safe default).
 */

import type { MemorySparkConfig } from "../config.js";

export type CaptureCategory = "fact" | "preference" | "decision" | "code-snippet" | "none";

export interface ClassifyResult {
  label: CaptureCategory;
  score: number;
}

export const CAPTURE_LABELS: CaptureCategory[] = [
  "fact", "preference", "decision", "code-snippet", "none",
];

export async function classifyForCapture(
  text: string,
  cfg: MemorySparkConfig,
  minConfidence = 0.75,
): Promise<ClassifyResult> {
  try {
    const resp = await fetch(cfg.spark.zeroShot, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        inputs: text.slice(0, 2000),
        parameters: {
          candidate_labels: CAPTURE_LABELS,
          multi_label: false,
        },
      }),
      signal: AbortSignal.timeout(5000),
    });
    if (!resp.ok) return { label: "none", score: 0 };

    const data = await resp.json() as { labels: string[]; scores: number[] };
    const topLabel = data.labels[0] as CaptureCategory;
    const topScore = data.scores[0]!;

    if (topScore < minConfidence || topLabel === "none") {
      return { label: "none", score: topScore };
    }
    return { label: topLabel, score: topScore };
  } catch {
    return { label: "none", score: 0 };
  }
}
