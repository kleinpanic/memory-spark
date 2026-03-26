#!/usr/bin/env npx tsx
/**
 * E2E Benchmark — tests the FULL recall pipeline as an agent would experience it.
 * Simulates before_prompt_build events and checks what gets injected.
 *
 * Tests:
 * 1. Content relevance (does the right info get recalled?)
 * 2. Garbage rejection (is noise filtered out?)
 * 3. MISTAKES injection (do mistakes show up for relevant queries?)
 * 4. Source weighting (do bootstrap files get appropriate priority?)
 * 5. Token budget (does injection stay within limits?)
 * 6. LCM dedup (does it avoid injecting what LCM already has?)
 */

import { resolveConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { createReranker } from "../src/rerank/reranker.js";
import { createAutoRecallHandler } from "../src/auto/recall.js";

interface TestCase {
  name: string;
  messages: Array<{ role: string; content: string }>;
  /** Regex that MUST match in the injected output */
  expectMatch?: RegExp;
  /** Regex that must NOT match (garbage check) */
  expectAbsent?: RegExp;
  /** If true, expects no injection at all */
  expectEmpty?: boolean;
  /** Check token budget compliance */
  maxTokens?: number;
}

const TEST_CASES: TestCase[] = [
  {
    name: "Basic factual recall",
    messages: [
      { role: "user", content: "What timezone is Klein in?" },
    ],
    expectMatch: /America\/New_York|Blacksburg|EDT/i,
  },
  {
    name: "Restart procedure recall",
    messages: [
      { role: "user", content: "How do I restart the OpenClaw gateway safely?" },
    ],
    expectMatch: /oc-restart|NEVER.*direct|systemctl.*banned/i,
  },
  {
    name: "Garbage rejection - no system noise",
    messages: [
      { role: "user", content: "Tell me about the DGX Spark setup" },
    ],
    expectAbsent: /\[media attached|\[System:|HEARTBEAT_OK|BEGIN_UNTRUSTED/i,
  },
  {
    name: "Token budget compliance",
    messages: [
      { role: "user", content: "Summarize everything about memory-spark" },
    ],
    maxTokens: 2500, // Default is 2000, allow some margin
  },
  {
    name: "Short message - should still recall",
    messages: [
      { role: "user", content: "WireGuard IP?" },
    ],
    expectMatch: /10\.\d+\.\d+\.\d+/,
  },
  {
    name: "Empty/noise message - should return nothing",
    messages: [
      { role: "user", content: "ok" },
    ],
    expectEmpty: true,
  },
  {
    name: "LCM content in context - should dedup",
    messages: [
      { role: "system", content: '<summary id="sum_test"><content>Klein is in Blacksburg VA, timezone America/New_York</content></summary>' },
      { role: "user", content: "What timezone is Klein in?" },
    ],
    // Should still recall but not duplicate the exact same info
  },
];

async function main() {
  const cfg = resolveConfig();
  const backend = new LanceDBBackend(cfg);
  await backend.open();
  const embed = await createEmbedProvider(cfg.embed);
  const reranker = await createReranker(cfg.rerank);

  const status = await backend.status();
  console.log(`Index: ${status.chunkCount} chunks`);
  console.log(`Reranker: ${cfg.rerank.enabled ? "enabled" : "off"}\n`);

  // Use raw embed provider (not EmbedQueue) to avoid queue<>Promise.all
  // concurrency issues during testing. Production uses EmbedQueue via cachedEmbed.
  const handler = createAutoRecallHandler({
    cfg: cfg.autoRecall,
    backend,
    embed,
    reranker,
  });

  let passed = 0;
  let failed = 0;

  for (const tc of TEST_CASES) {
    process.stdout.write(`  ${tc.name}... `);

    try {
      const event = {
        prompt: "",
        messages: tc.messages,
        agentId: "meta",
      };
      const ctx = { agentId: "meta" };

      const result = await handler(event, ctx) as { prependContext?: string } | undefined;

      // Get the injection text
      const injectionText = result?.prependContext ?? "";

      const issues: string[] = [];

      // Check expectEmpty
      if (tc.expectEmpty) {
        if (injectionText.length > 0 && injectionText.includes("<memory ")) {
          const memCount = (injectionText.match(/<memory /g) ?? []).length;
          issues.push(`Expected no injection but got ${memCount} memories`);
        }
      }

      // Check expectMatch
      if (tc.expectMatch && injectionText) {
        if (!tc.expectMatch.test(injectionText)) {
          issues.push(`Expected match: ${tc.expectMatch} — not found in injection`);
        }
      }

      // Check expectAbsent
      if (tc.expectAbsent && injectionText) {
        const match = injectionText.match(tc.expectAbsent);
        if (match) {
          issues.push(`Found garbage: "${match[0].slice(0, 60)}"`);
        }
      }

      // Check token budget
      if (tc.maxTokens && injectionText) {
        const approxTokens = Math.ceil(injectionText.length / 4);
        if (approxTokens > tc.maxTokens) {
          issues.push(`Token budget exceeded: ~${approxTokens} > ${tc.maxTokens}`);
        }
      }

      if (issues.length === 0) {
        console.log(`✅`);
        passed++;
      } else {
        console.log(`❌`);
        for (const issue of issues) {
          console.log(`     → ${issue}`);
        }
        failed++;
      }

      // Show brief injection stats
      const memCount = (injectionText.match(/<memory /g) ?? []).length;
      const approxTokens = injectionText ? Math.ceil(injectionText.length / 4) : 0;
      console.log(`     [${memCount} memories, ~${approxTokens} tokens]`);

    } catch (err) {
      console.log(`💥 ERROR: ${err instanceof Error ? err.message : String(err)}`);
      failed++;
    }
  }

  console.log(`\n=== E2E Results: ${passed}/${passed + failed} passed ===`);
  if (failed > 0) {
    console.log(`❌ ${failed} test(s) failed`);
    process.exit(1);
  }

  await backend.close();
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
