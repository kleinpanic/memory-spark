/**
 * Content Quality Scorer — fast regex/pattern matching to identify noise chunks.
 * Runs on each chunk BEFORE embedding. Zero network calls.
 */

export interface QualityResult {
  score: number;       // 0.0-1.0
  flags: string[];     // Machine-readable noise indicators
}

/**
 * Path patterns that should NEVER be indexed — hard exclusion (score = 0).
 * These are checked before any content analysis.
 */
const EXCLUDED_PATH_PATTERNS: RegExp[] = [
  /\/zh-CN\//i,
  /\/zh-TW\//i,
  /\/ja\//i,
  /\/ko\//i,
  /\/fr\//i,
  /\/de\//i,
  /\/es\//i,
  /\/pt-BR\//i,
  /\/ru\//i,
  /(?:^|\/)i18n\//i,
  /(?:^|\/)locales?\//i,
  /(?:^|\/)translations?\//i,
];

/**
 * Detect non-English content by character class ratio.
 * Returns the fraction of characters that are CJK, Cyrillic, Arabic, etc.
 */
export function nonLatinRatio(text: string): number {
  if (!text) return 0;
  // CJK Unified, CJK Extension A/B, Hiragana, Katakana, Hangul, Cyrillic, Arabic, Thai, Devanagari
  const nonLatin = text.match(/[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0e00-\u0e7f\u0900-\u097f]/g);
  const total = text.replace(/\s/g, "").length;
  if (total === 0) return 0;
  return (nonLatin?.length ?? 0) / total;
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

  // Session dump headers (raw conversation logs, not knowledge)
  { pattern: /^# Session: \d{4}-\d{2}-\d{2}/m,
    flag: "session-dump-header", penalty: 0.7 },
  { pattern: /^\*\*Session (?:Key|ID)\*\*:/m,
    flag: "session-dump-metadata", penalty: 0.5 },

  // Raw assistant/user turn prefixes (conversation logs, not knowledge)
  { pattern: /^(assistant|user|system):\s/m,
    flag: "raw-turn-prefix", penalty: 0.3 },

  // Casual chat markers (lol, lmao, lmfao, haha, etc.)
  { pattern: /\b(lol|lmao|lmfao|haha|heh|rofl|bruh|nah|idk|tbh|imo|iirc)\b/i,
    flag: "casual-chat", penalty: 0.3 },

  // External untrusted content wrappers (Klein's raw messages)
  { pattern: /<<<EXTERNAL_UNTRUSTED_CONTENT/,
    flag: "untrusted-content-wrapper", penalty: 0.6 },
  { pattern: /UNTRUSTED Discord message body/,
    flag: "discord-raw-body", penalty: 0.5 },
];

export interface LanguageOpts {
  /** Primary language. "all" disables filtering. Default: "en" */
  language?: string;
  /** Non-Latin ratio threshold for exclusion. Default: 0.3 */
  threshold?: number;
}

export function scoreChunkQuality(text: string, filePath: string, source: string, langOpts?: LanguageOpts): QualityResult {
  const flags: string[] = [];
  let totalPenalty = 0;
  const lang = langOpts?.language ?? "en";
  const langThreshold = langOpts?.threshold ?? 0.3;

  // ── Hard path exclusions (instant zero score) ──────────────────────
  for (const pat of EXCLUDED_PATH_PATTERNS) {
    if (pat.test(filePath)) {
      return { score: 0, flags: ["excluded-path-i18n"] };
    }
  }

  // ── Language filter (configurable) ─────────────────────────────────
  if (lang !== "all") {
    const nlr = nonLatinRatio(text);
    if (nlr > langThreshold) {
      return { score: 0, flags: ["non-english-content"] };
    }
    if (nlr > langThreshold * 0.33) {
      // Mixed content — heavy penalty but don't completely exclude
      flags.push("mixed-language");
      totalPenalty += 0.6;
    }
  }

  // ── Pattern-based noise detection ──────────────────────────────────
  for (const np of NOISE_PATTERNS) {
    if (np.pattern.test(text)) {
      flags.push(np.flag);
      totalPenalty += np.penalty;
    }
  }

  // ── Information density ────────────────────────────────────────────
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

  // ── Path-based quality signals ─────────────────────────────────────
  if (filePath.includes("archive/")) totalPenalty += 0.2;
  if (filePath === "memory/learnings.md") totalPenalty += 0.8;

  // ── Source-based signals ───────────────────────────────────────────
  if (source === "capture") totalPenalty -= 0.3; // Boost captures

  const score = Math.max(0, Math.min(1, 1.0 - totalPenalty));
  return { score, flags };
}
