/**
 * Zero-Shot Classifier
 *
 * Used by auto-capture to decide whether a conversation turn contains
 * something worth storing as a persistent memory.
 *
 * Uses Spark 18113 (bart-large-mnli) — a zero-shot classification model
 * that doesn't need training data, just label strings.
 *
 * Labels we classify against:
 *   "fact"          — a factual statement about the user, system, or world
 *   "preference"    — user expressed a preference or taste
 *   "decision"      — a decision was made or committed to
 *   "code-snippet"  — a useful code example or command
 *   "none"          — nothing worth storing
 *
 * Returns the winning label and its confidence score (0–1).
 * If confidence < minConfidence (default 0.75), treat as "none".
 *
 * API: Spark zero-shot is HuggingFace text-classification endpoint.
 * POST { inputs: "text", parameters: { candidate_labels: [...], multi_label: false } }
 * Returns: { sequence: "...", labels: [...], scores: [...] }
 */

import type { MemorySparkConfig } from "../config.js";

export type CaptureCategory = "fact" | "preference" | "decision" | "code-snippet" | "none";

export interface ClassifyResult {
  label: CaptureCategory;
  score: number;
}

/** Default labels for capture classification */
export const CAPTURE_LABELS: CaptureCategory[] = [
  "fact",
  "preference",
  "decision",
  "code-snippet",
  "none",
];

/**
 * Classify text into a capture category.
 * Returns { label: "none", score: 0 } on failure (safe default = don't capture).
 */
export async function classifyForCapture(
  text: string,
  cfg: MemorySparkConfig,
  minConfidence = 0.75,
): Promise<ClassifyResult> {
  // TODO:
  // const resp = await fetch(`${cfg.spark.zeroShot}`, {
  //   method: "POST",
  //   headers: { "Content-Type": "application/json" },
  //   body: JSON.stringify({
  //     inputs: text,
  //     parameters: { candidate_labels: CAPTURE_LABELS, multi_label: false }
  //   }),
  // });
  // const result = await resp.json();
  // const topLabel = result.labels[0] as CaptureCategory;
  // const topScore = result.scores[0] as number;
  // if (topScore < minConfidence) return { label: "none", score: topScore };
  // return { label: topLabel, score: topScore };

  return { label: "none", score: 0 }; // safe default until implemented
}
