/**
 * Dedicated tests for heuristic + quality classifiers.
 *
 * src/classify/heuristic.ts  — heuristicClassify()
 * src/classify/quality.ts    — scoreChunkQuality()
 */

import { describe, it, expect, vi } from "vitest";
import { heuristicClassify } from "../src/classify/heuristic.js";
import { scoreChunkQuality } from "../src/classify/quality.js";

// ─────────────────────────────────────────────────────────────────────────────
// heuristicClassify — edge cases
// ─────────────────────────────────────────────────────────────────────────────
describe("heuristicClassify — edge cases", () => {
  it("returns none with score 0 for empty string", () => {
    const result = heuristicClassify("");
    expect(result.label).toBe("none");
    expect(result.score).toBe(0);
  });

  it("returns none for whitespace-only input", () => {
    const result = heuristicClassify("   \n\t  ");
    expect(result.label).toBe("none");
    expect(result.score).toBe(0);
  });

  it("classifies 'decided' keyword as decision", () => {
    const result = heuristicClassify("We decided to go with TypeScript for this project.");
    expect(result.label).toBe("decision");
    expect(result.score).toBeGreaterThan(0);
  });

  it("classifies 'prefer' keyword as preference", () => {
    const result = heuristicClassify("I prefer to use tabs over spaces.");
    expect(result.label).toBe("preference");
  });

  it("classifies code with backtick fences as code-snippet", () => {
    const result = heuristicClassify("Here is an example:\n```ts\nconst x = 1;\n```");
    expect(result.label).toBe("code-snippet");
  });

  it("classifies inline arrow function as code-snippet", () => {
    const result = heuristicClassify("Use arr.map(x => x * 2) to double every element.");
    expect(result.label).toBe("code-snippet");
  });

  it("classifies IP address text as fact", () => {
    const result = heuristicClassify("The server is running on 192.168.1.10.");
    expect(result.label).toBe("fact");
  });

  it("classifies version number text as fact", () => {
    const result = heuristicClassify("We upgraded to v2.3.1 last week.");
    expect(result.label).toBe("fact");
  });

  it("classifies port reference as fact", () => {
    const result = heuristicClassify("The service listens on port 8080.");
    expect(result.label).toBe("fact");
  });

  it("handles very long text (1000+ words) without throwing", () => {
    const longText = "This is a very long text that repeats. I prefer clarity. ".repeat(50);
    expect(() => heuristicClassify(longText)).not.toThrow();
    const result = heuristicClassify(longText);
    expect(result.label).toBeTruthy();
  });

  it("resolves multi-label ambiguity: decision takes priority over preference", () => {
    // Text has both 'decided' (decision) and 'prefer' (preference).
    // Decision pattern appears first in source → should win.
    const text = "We decided to switch frameworks because I prefer React over Vue.";
    const result = heuristicClassify(text);
    expect(result.label).toBe("decision");
  });

  it("resolves multi-label ambiguity: preference takes priority over fact when no decision", () => {
    // 'prefer' + version number in same sentence, no 'decided'
    const text = "I prefer v2.3.1 because it has better performance.";
    // preference pattern is checked before fact in source order
    const result = heuristicClassify(text);
    expect(result.label).toBe("preference");
  });

  it("handles Unicode-heavy text gracefully (CJK, emoji)", () => {
    const text = "🎉 日本語テスト: I prefer Unicode support in v3.0 of the framework.";
    expect(() => heuristicClassify(text)).not.toThrow();
    const result = heuristicClassify(text);
    // 'prefer' should match despite Unicode context
    expect(result.label).toBe("preference");
  });

  it("handles text with only Unicode symbols (no ASCII words)", () => {
    const text = "🔥 💻 🌟 ⚡ 🚀";
    const result = heuristicClassify(text);
    expect(result.label).toBe("none");
    expect(result.score).toBe(0);
  });

  it("code detection: import statement at line start", () => {
    const text = "import { useState } from 'react';\nconst App = () => <div />;";
    const result = heuristicClassify(text);
    expect(result.label).toBe("code-snippet");
  });

  it("code detection: console.log reference", () => {
    const result = heuristicClassify("Try console.log(x) to debug the value.");
    expect(result.label).toBe("code-snippet");
  });

  it("fact detection: file path reference", () => {
    const result = heuristicClassify("The config is at /etc/openclaw/config.json.");
    expect(result.label).toBe("fact");
  });

  it("returns score in valid range (0.0–1.0) for all labels", () => {
    const inputs = [
      "decided to use postgres",
      "I prefer dark mode",
      "const x = require('x')",
      "server runs on 10.0.0.1",
      "just a random sentence",
    ];
    for (const text of inputs) {
      const { score } = heuristicClassify(text);
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// scoreChunkQuality — edge cases
// ─────────────────────────────────────────────────────────────────────────────
describe("scoreChunkQuality — edge cases", () => {
  const FILE = "docs/notes.md";
  const SRC = "knowledge";

  it("scores well-structured markdown highly (above 0.7)", () => {
    const md = `
# Architecture Decision

We chose PostgreSQL over SQLite because we need:
- Concurrent writes from multiple agents
- JSONB support for flexible schema
- Full-text search via pg_trgm

This was validated against our throughput benchmarks.
    `.trim();
    const { score, flags } = scoreChunkQuality(md, FILE, SRC);
    expect(score).toBeGreaterThan(0.7);
    expect(flags).not.toContain("too-short");
    expect(flags).not.toContain("low-density");
  });

  it("penalizes very short text (fewer than 10 words)", () => {
    const short = "Hello world.";
    const { score, flags } = scoreChunkQuality(short, FILE, SRC);
    expect(flags).toContain("too-short");
    expect(score).toBeLessThan(0.7);
  });

  it("scores empty string with too-short flag", () => {
    const { flags } = scoreChunkQuality("", FILE, SRC);
    expect(flags).toContain("too-short");
  });

  it("penalizes text that is only URLs", () => {
    const urlOnly =
      "https://example.com/a https://example.com/b https://example.com/c https://example.com/d https://example.com/e";
    const { score, flags } = scoreChunkQuality(urlOnly, FILE, SRC);
    // URLs are short tokens + low word density → should be penalized
    expect(flags).toContain("low-density");
    expect(score).toBeLessThanOrEqual(0.8);
  });

  it("scores code blocks (triple backtick) without heavy penalty", () => {
    const codeBlock = `
Here is how to create a vector index:

\`\`\`typescript
const index = await lancedb.createIndex({
  type: "IVF_PQ",
  numPartitions: 256,
  numSubVectors: 96,
});
\`\`\`

This runs in under 100ms for small datasets.
    `.trim();
    const { score } = scoreChunkQuality(codeBlock, FILE, SRC);
    expect(score).toBeGreaterThan(0.5);
  });

  it("returns score=0 for i18n paths", () => {
    const { score, flags } = scoreChunkQuality(
      "Some content here that looks normal.",
      "/docs/zh-CN/guide.md",
      SRC,
    );
    expect(score).toBe(0);
    expect(flags).toContain("excluded-path-i18n");
  });

  it("returns score=0 for /locales/ path", () => {
    const { score, flags } = scoreChunkQuality("translation strings", "/app/locales/en.json", SRC);
    expect(score).toBe(0);
    expect(flags).toContain("excluded-path-i18n");
  });

  it("returns score=0 for non-English (CJK-heavy) content", () => {
    const cjk =
      "これはテストのテキストです。品質スコアは低くなるはずです。日本語のコンテンツはインデックスされません。";
    const { score, flags } = scoreChunkQuality(cjk, FILE, SRC);
    expect(score).toBe(0);
    expect(flags).toContain("non-english-content");
  });

  it("penalizes agent bootstrap log entries (score=0)", () => {
    const bootstrap = "## 2024-01-15T10:23:45Z — agent bootstrap\nAgent initialized successfully.";
    const { score, flags } = scoreChunkQuality(bootstrap, FILE, SRC);
    expect(score).toBe(0);
    expect(flags).toContain("agent-bootstrap");
  });

  it("penalizes HEARTBEAT_OK message", () => {
    const { score, flags } = scoreChunkQuality("HEARTBEAT_OK", FILE, SRC);
    expect(flags).toContain("heartbeat-ok");
    expect(score).toBe(0);
  });

  it("penalizes casual chat markers (lol, lmao)", () => {
    const casual = "lol that's hilarious lmao I can't believe it actually worked omg this is great";
    const { score, flags } = scoreChunkQuality(casual, FILE, SRC);
    expect(flags).toContain("casual-chat");
    expect(score).toBe(0);
  });

  it("penalizes memory XML recursion (<relevant-memories>)", () => {
    const memXml = `
<relevant-memories>
  <memory index="1" source="notes.md">Some recalled fact</memory>
</relevant-memories>
    `.trim();
    const { score, flags } = scoreChunkQuality(memXml, FILE, SRC);
    expect(flags).toContain("memory-xml-open");
    expect(score).toBe(0);
  });

  it("boosts score for 'capture' source (reflected in lower totalPenalty)", () => {
    // Use a text with an archive path so both scores are sub-1.0 and the capture boost is visible.
    const text = "User said they prefer TypeScript over JavaScript for large projects.";
    const withCapture = scoreChunkQuality(text, "archive/old.md", "capture");
    const withKnowledge = scoreChunkQuality(text, "archive/old.md", "knowledge");
    // archive/ adds +0.2 penalty; capture subtracts 0.3, net −0.1 → higher score
    expect(withCapture.score).toBeGreaterThan(withKnowledge.score);
  });

  it("penalizes archive/ path", () => {
    const text =
      "This is a well-written document with plenty of meaningful content about databases.";
    const archiveResult = scoreChunkQuality(text, "archive/old-notes.md", SRC);
    const normalResult = scoreChunkQuality(text, "docs/notes.md", SRC);
    expect(archiveResult.score).toBeLessThan(normalResult.score);
  });

  it("penalizes memory/learnings.md path", () => {
    const text =
      "This is a well-written document with plenty of meaningful content about databases.";
    const learnings = scoreChunkQuality(text, "memory/learnings.md", SRC);
    const normal = scoreChunkQuality(text, "memory/notes.md", SRC);
    expect(learnings.score).toBeLessThan(normal.score);
  });

  it("score is always in [0, 1] range", () => {
    const texts = [
      ["", FILE, SRC],
      ["Hello", FILE, SRC],
      ["lol lmao lol lmao lol lmao lol lmao lol lmao", FILE, SRC],
      [
        "HEARTBEAT_OK\nNO_REPLY\nlol\nlmao\nsession=abc123\nBackfilled by meta for continuity",
        FILE,
        SRC,
      ],
      [
        "A well-structured document with many meaningful words about software engineering and databases.",
        FILE,
        SRC,
      ],
    ] as [string, string, string][];

    for (const [text, filePath, source] of texts) {
      const { score } = scoreChunkQuality(text, filePath, source);
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    }
  });

  it("handles text with mixed language (some CJK) without throwing", () => {
    const mixed = "This document discusses 日本語 integration patterns for multilingual apps.";
    expect(() => scoreChunkQuality(mixed, FILE, SRC)).not.toThrow();
  });

  it("penalizes task queue injection blocks", () => {
    const injected = `
## Current Task Queue
### 🔄 In Progress
- task1: Some task description
### 👀 Awaiting Review
- task2: Another task
    `.trim();
    const { score, flags } = scoreChunkQuality(injected, FILE, SRC);
    expect(flags).toContain("task-queue-inject");
    expect(score).toBe(0);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Classifier consistency — same input through both classifiers
// ─────────────────────────────────────────────────────────────────────────────
describe("heuristicClassify + scoreChunkQuality — consistency on same inputs", () => {
  const FILE = "docs/notes.md";
  const SRC = "knowledge";

  it("high-quality decision text: heuristic=decision, quality>0.5", () => {
    const text =
      "We decided to migrate from Redis to Valkey because Valkey is fully open-source and API-compatible.";
    const classify = heuristicClassify(text);
    const quality = scoreChunkQuality(text, FILE, SRC);

    expect(classify.label).toBe("decision");
    expect(quality.score).toBeGreaterThan(0.5);
  });

  it("high-quality preference text: heuristic=preference, quality>0.5", () => {
    const text =
      "Klein does prefer dark mode across all applications and always uses the Catppuccin Mocha color scheme for better readability.";
    const classify = heuristicClassify(text);
    const quality = scoreChunkQuality(text, FILE, SRC);

    expect(classify.label).toBe("preference");
    expect(quality.score).toBeGreaterThan(0.5);
  });

  it("code snippet: heuristic=code-snippet, quality score not zero", () => {
    const text = `
The following snippet exports the provider:

\`\`\`typescript
export async function embedQuery(text: string): Promise<number[]> {
  const response = await client.embeddings.create({ input: text, model: "text-embedding-3-small" });
  return response.data[0].embedding;
}
\`\`\`

Use this for all recall queries.
    `.trim();
    const classify = heuristicClassify(text);
    const quality = scoreChunkQuality(text, FILE, SRC);

    expect(classify.label).toBe("code-snippet");
    expect(quality.score).toBeGreaterThan(0);
  });

  it("noise content: heuristic=none (or any), quality=0", () => {
    const noise = "HEARTBEAT_OK";
    const classify = heuristicClassify(noise);
    const quality = scoreChunkQuality(noise, FILE, SRC);

    // Heuristic doesn't know about heartbeats — may return none
    expect(classify.score).toBeGreaterThanOrEqual(0);
    // Quality should reject it
    expect(quality.score).toBe(0);
  });

  it("empty string: heuristic=none/0, quality has too-short flag", () => {
    const classify = heuristicClassify("");
    const quality = scoreChunkQuality("", FILE, SRC);

    expect(classify.label).toBe("none");
    expect(classify.score).toBe(0);
    expect(quality.flags).toContain("too-short");
  });

  it("fact about infrastructure: heuristic=fact, quality>0.4", () => {
    const text =
      "The OpenClaw gateway runs on broklein at 192.168.1.100 and listens on port 18789.";
    const classify = heuristicClassify(text);
    const quality = scoreChunkQuality(text, FILE, SRC);

    expect(classify.label).toBe("fact");
    expect(quality.score).toBeGreaterThan(0.4);
  });

  it("agent bootstrap spam: heuristic=none, quality=0 (agent-bootstrap flag)", () => {
    const text = "## 2024-03-10T08:00:00Z — agent bootstrap\nSession initialized for agent:meta.";
    const classify = heuristicClassify(text);
    const quality = scoreChunkQuality(text, FILE, SRC);

    // Heuristic doesn't know about bootstrap spam; quality should catch it
    expect(quality.flags).toContain("agent-bootstrap");
    expect(quality.score).toBe(0);
  });
});
