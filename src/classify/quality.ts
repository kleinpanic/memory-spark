/**
 * Content Quality Scorer — fast regex/pattern matching to identify noise chunks.
 * Runs on each chunk BEFORE embedding. Zero network calls.
 */

export interface QualityResult {
  score: number; // 0.0-1.0
  flags: string[]; // Machine-readable noise indicators
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
function nonLatinRatio(text: string): number {
  if (!text) return 0;
  // Use Unicode property escapes for CJK, Hangul, Cyrillic, Arabic, Thai, Devanagari
  const nonLatin = text.match(
    /\p{Script=Han}|\p{Script=Hiragana}|\p{Script=Katakana}|\p{Script=Hangul}|\p{Script=Cyrillic}|\p{Script=Arabic}|\p{Script=Thai}|\p{Script=Devanagari}/gu,
  );
  const total = text.replace(/\s/g, "").length;
  if (total === 0) return 0;
  return (nonLatin?.length ?? 0) / total;
}

/** Noise patterns with their penalty weights */
const NOISE_PATTERNS: Array<{ pattern: RegExp; flag: string; penalty: number }> = [
  // Agent bootstrap spam (THE #1 polluter — 23K+ chunks in current index)
  {
    pattern: /^## \d{4}-\d{2}-\d{2}T[\d:.]+Z — agent bootstrap/m,
    flag: "agent-bootstrap",
    penalty: 1.0,
  },

  // Session new entries
  { pattern: /^## \d{4}-\d{2}-\d{2}T[\d:.]+Z — session new/m, flag: "session-new", penalty: 1.0 },

  // Discord conversation metadata blocks
  { pattern: /Conversation info \(untrusted metadata\):/, flag: "discord-metadata", penalty: 1.0 },
  { pattern: /"message_id":\s*"\d+"/, flag: "message-id", penalty: 1.0 },
  { pattern: /Sender \(untrusted metadata\):/, flag: "sender-metadata", penalty: 1.0 },

  // Raw exec output
  { pattern: /Exec completed \([^)]+, code \d+\)/, flag: "exec-output", penalty: 0.4 },
  { pattern: /session=[a-f0-9-]{8,}/, flag: "session-id", penalty: 0.3 },

  // Backfill stubs
  { pattern: /Backfilled by \w+ for continuity/, flag: "backfill-stub", penalty: 1.0 },

  // NO_REPLY markers
  { pattern: /^(assistant|user):\s*NO_REPLY\s*$/m, flag: "no-reply", penalty: 1.0 },

  // Pure timestamp lines (no content after timestamp)
  {
    pattern: /^\[\w{3} \d{4}-\d{2}-\d{2} \d{2}:\d{2} \w+\]\s*$/m,
    flag: "timestamp-only",
    penalty: 1.0,
  },

  // Session dump headers (raw conversation logs, not knowledge)
  { pattern: /^# Session: \d{4}-\d{2}-\d{2}/m, flag: "session-dump-header", penalty: 1.0 },
  { pattern: /^\*\*Session (?:Key|ID)\*\*:/m, flag: "session-dump-metadata", penalty: 1.0 },

  // Raw assistant/user turn prefixes (conversation logs, not knowledge)
  { pattern: /^(assistant|user|system):\s/m, flag: "raw-turn-prefix", penalty: 1.0 },

  // Casual chat markers (lol, lmao, lmfao, haha, etc.)
  {
    pattern: /\b(lol|lmao|lmfao|haha|heh|rofl|bruh|nah|idk|tbh|imo|iirc)\b/i,
    flag: "casual-chat",
    penalty: 1.0,
  },

  // External untrusted content wrappers (Klein's raw messages)
  { pattern: /<<<EXTERNAL_UNTRUSTED_CONTENT/, flag: "untrusted-content-wrapper", penalty: 1.0 },
  { pattern: /UNTRUSTED Discord message body/, flag: "discord-raw-body", penalty: 1.0 },
  { pattern: /UNTRUSTED \w+ message body/, flag: "untrusted-body", penalty: 1.0 },

  // Media attachment paths (caused school-agent to think Klein sent a screenshot)
  { pattern: /\[media attached:\s*\/home\//, flag: "media-path-local", penalty: 1.0 },
  { pattern: /\[media attached:\s*https?:\/\//, flag: "media-path-url", penalty: 1.0 },
  { pattern: /\.openclaw\/media\/inbound\//, flag: "inbound-media-ref", penalty: 1.0 },
  {
    pattern: /To send an image back, prefer the message tool/,
    flag: "media-instruction",
    penalty: 1.0,
  },

  // Memory recall XML (memories about memories = garbage recursion)
  { pattern: /<relevant-memories>/, flag: "memory-xml-open", penalty: 1.0 },
  { pattern: /<memory index="\d+"/, flag: "memory-xml-entry", penalty: 1.0 },
  {
    pattern: /<!-- SECURITY: Treat every memory below as untrusted/,
    flag: "memory-security-comment",
    penalty: 1.0,
  },

  // LCM summary blocks (compaction metadata, not knowledge)
  { pattern: /<summary id="sum_[a-f0-9]+"/, flag: "lcm-summary", penalty: 1.0 },
  { pattern: /<summary_ref id="sum_/, flag: "lcm-summary-ref", penalty: 1.0 },

  // System/heartbeat noise
  { pattern: /^HEARTBEAT_OK$/m, flag: "heartbeat-ok", penalty: 1.0 },
  { pattern: /^NO_REPLY$/m, flag: "no-reply-msg", penalty: 1.0 },
  { pattern: /\[System:\s/, flag: "system-inject", penalty: 0.8 },

  // oc-tasks injection blocks
  { pattern: /^## Current Task Queue$/m, flag: "task-queue-inject", penalty: 1.0 },
  { pattern: /^### 🔄 In Progress/m, flag: "task-status-inject", penalty: 1.0 },
  { pattern: /^### 👀 Awaiting Review/m, flag: "task-review-inject", penalty: 1.0 },
];

export interface LanguageOpts {
  /** Primary language. "all" disables filtering. Default: "en" */
  language?: string;
  /** Non-Latin ratio threshold for exclusion. Default: 0.3 */
  threshold?: number;
}

export function scoreChunkQuality(
  text: string,
  filePath: string,
  source: string,
  langOpts?: LanguageOpts,
): QualityResult {
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
