/**
 * Content Quality Scorer — fast regex/pattern matching to identify noise chunks.
 * Runs on each chunk BEFORE embedding. Zero network calls.
 */

export interface QualityResult {
  score: number;       // 0.0-1.0
  flags: string[];     // Machine-readable noise indicators
}

/** Noise patterns with their penalty weights */
const NOISE_PATTERNS: Array<{ pattern: RegExp; flag: string; penalty: number }> = [
  // Agent bootstrap spam (THE #1 polluter — 23K+ chunks in current index)
  { pattern: /^## \d{4}-\d{2}-\d{2}T[\d:.]+Z — agent bootstrap/m,
    flag: "agent-bootstrap", penalty: 1.0 },

  // Session new entries
  { pattern: /^## \d{4}-\d{2}-\d{2}T[\d:.]+Z — session new/m,
    flag: "session-new", penalty: 1.0 },

  // Discord conversation metadata blocks
  { pattern: /Conversation info \(untrusted metadata\):/,
    flag: "discord-metadata", penalty: 0.8 },
  { pattern: /"message_id":\s*"\d+"/,
    flag: "message-id", penalty: 0.6 },
  { pattern: /Sender \(untrusted metadata\):/,
    flag: "sender-metadata", penalty: 0.6 },

  // Raw exec output
  { pattern: /Exec completed \([^)]+, code \d+\)/,
    flag: "exec-output", penalty: 0.4 },
  { pattern: /session=[a-f0-9-]{8,}/,
    flag: "session-id", penalty: 0.3 },

  // Backfill stubs
  { pattern: /Backfilled by \w+ for continuity/,
    flag: "backfill-stub", penalty: 0.5 },

  // NO_REPLY markers
  { pattern: /^(assistant|user):\s*NO_REPLY\s*$/m,
    flag: "no-reply", penalty: 0.3 },

  // Pure timestamp lines (no content after timestamp)
  { pattern: /^\[\w{3} \d{4}-\d{2}-\d{2} \d{2}:\d{2} \w+\]\s*$/m,
    flag: "timestamp-only", penalty: 0.3 },
];

export function scoreChunkQuality(text: string, filePath: string, source: string): QualityResult {
  const flags: string[] = [];
  let totalPenalty = 0;

  for (const np of NOISE_PATTERNS) {
    if (np.pattern.test(text)) {
      flags.push(np.flag);
      totalPenalty += np.penalty;
    }
  }

  // Information density: ratio of unique meaningful words to total words
  const words = text.match(/\b\w{3,}\b/g) ?? [];
  const uniqueWords = new Set(words.map((w) => w.toLowerCase()));
  const density = words.length > 0 ? uniqueWords.size / words.length : 0;

  // Very short chunks with low density are probably noise
  if (words.length < 10) {
    flags.push("too-short");
    totalPenalty += 0.4;
  }
  if (density < 0.3 && words.length > 5) {
    flags.push("low-density");
    totalPenalty += 0.2;
  }

  // Path-based quality signals
  if (filePath.includes("archive/")) totalPenalty += 0.2;
  if (filePath === "memory/learnings.md") totalPenalty += 0.8;

  // Source-based signals
  if (source === "capture") totalPenalty -= 0.3; // Boost captures

  const score = Math.max(0, Math.min(1, 1.0 - totalPenalty));
  return { score, flags };
}
