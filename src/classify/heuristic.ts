/**
 * Heuristic Classifier — lightweight keyword/regex fallback.
 * Used when Spark zero-shot (/v1/classify) is unavailable or disabled.
 * Returns the same ClassifyResult shape as zero-shot for drop-in compatibility.
 *
 * Trade-off: Lower accuracy than bart-large-mnli, but zero latency and
 * zero network dependency. Useful for resilience when Spark is under
 * resource pressure or the zero-shot container is down.
 */

import type { ClassifyResult } from "./zero-shot.js";

/**
 * Simple pattern-based classifier for user messages.
 * Returns { label, score } where label is a CaptureCategory.
 * Score is a fixed confidence (0.65-0.75) — lower than the ML model
 * to reflect reduced accuracy.
 */
export function heuristicClassify(text: string): ClassifyResult {
  const lower = text.toLowerCase();

  // ── Decision indicators ────────────────────────────────────────────────
  if (
    /\b(decided|decision|chose|going with|switched to|let'?s go with|approved|we'?ll use|moving to|migrated? to)\b/.test(
      lower,
    )
  ) {
    return { label: "decision", score: 0.7 };
  }

  // ── Preference indicators ──────────────────────────────────────────────
  if (
    /\b(prefer|like|want|always use|favorite|don'?t like|hate|avoid|rather|better than|worse than)\b/.test(
      lower,
    )
  ) {
    return { label: "preference", score: 0.7 };
  }

  // ── Code snippet detection ─────────────────────────────────────────────
  // Check the original text (not lowered) for code patterns
  if (
    /```/.test(text) ||
    /^(import|from|const|let|var|function|class|def |export |async |type |interface )\s/m.test(
      text,
    ) ||
    /\b(=>|\.map\(|\.filter\(|\.reduce\(|console\.log|print\(|println!)\b/.test(text)
  ) {
    return { label: "code-snippet", score: 0.7 };
  }

  // ── Fact indicators ────────────────────────────────────────────────────
  // IPs, ports, paths, versions, model names
  if (
    /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/.test(text) || // IP addresses
    /\bport\s+\d+\b/i.test(text) || // port references
    /\/[\w.-]+\/[\w.-]+/.test(text) || // file paths
    /\bv?\d+\.\d+(\.\d+)?\b/.test(text) || // version numbers
    /\b(runs on|located at|hosted at|deployed to|configured as|set to)\b/.test(lower)
  ) {
    return { label: "fact", score: 0.65 };
  }

  return { label: "none", score: 0 };
}
