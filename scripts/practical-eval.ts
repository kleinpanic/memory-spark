/**
 * Practical Agent Utility Eval
 * 
 * Tests whether memory-spark retrieval actually helps agents
 * answer real questions they'd encounter in practice.
 * 
 * Each test case is a real scenario an agent might face,
 * with keywords that MUST appear in retrieved context for
 * the agent to answer correctly without hallucinating.
 */

import { resolveConfig } from "../src/config.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { createEmbedProvider } from "../src/embed/provider.js";

interface PracticalTest {
  scenario: string;           // What the agent is trying to do
  query: string;              // What memory-spark would search for
  mustContain: string[];      // Keywords that MUST be in results for agent to succeed
  niceToHave?: string[];      // Bonus context that would help
  category: "safety" | "infrastructure" | "workflow" | "history" | "reference";
}

const TESTS: PracticalTest[] = [
  // SAFETY — agents must never forget these
  {
    scenario: "Agent wants to restart the gateway after a config change",
    query: "how to restart openclaw gateway",
    mustContain: ["oc-restart", "banned", "approval"],
    niceToHave: ["Discord", "staged", "config-guardian"],
    category: "safety"
  },
  {
    scenario: "Agent is editing openclaw.json during a heartbeat run",
    query: "editing openclaw.json during heartbeat cron",
    mustContain: ["NEVER", "heartbeat", "config"],
    niceToHave: ["2026-02-20", "production break"],
    category: "safety"
  },
  {
    scenario: "Agent wants to use config.patch for agents.list changes",
    query: "config.patch agents.list array mutation",
    mustContain: ["never", "config.patch", "agents.list"],
    category: "safety"
  },
  {
    scenario: "Agent considers using Gemini Flash for a coding task",
    query: "gemini flash for coding tasks",
    mustContain: ["never", "coding", "opus"],
    niceToHave: ["broken work", "2026-03-08"],
    category: "safety"
  },

  // INFRASTRUCTURE — agents need to know the topology
  {
    scenario: "Agent needs to reach the Spark node for inference",
    query: "Spark node IP address connection",
    mustContain: ["10.99.1.1"],
    niceToHave: ["WireGuard", "wg-spark", "10.99.1.2"],
    category: "infrastructure"
  },
  {
    scenario: "Agent needs to SSH to the mt server",
    query: "mt server SSH access PowerEdge",
    mustContain: ["mt", "PowerEdge"],
    niceToHave: ["192.168.1.133", "EXTREME CAUTION", "sacred"],
    category: "infrastructure"
  },
  {
    scenario: "Agent needs to know which machine is broklein vs kernelpanic",
    query: "broklein kernelpanic machine difference",
    mustContain: ["broklein", "kernelpanic"],
    niceToHave: ["Debian", "sid", "laptop"],
    category: "infrastructure"
  },
  {
    scenario: "Agent needs to manage BlueBubbles MCP",
    query: "BlueBubbles MCP tunnel service",
    mustContain: ["bluebubbles", "18800"],
    niceToHave: ["autossh", "collins", "mcporter"],
    category: "infrastructure"
  },

  // WORKFLOW — how things are done
  {
    scenario: "Agent needs to create and track a task",
    query: "oc-tasks create track task workflow",
    mustContain: ["oc-tasks", "add"],
    niceToHave: ["dispatch", "comment", "advance"],
    category: "workflow"
  },
  {
    scenario: "Agent wants to run sudo commands on broklein",
    query: "sudo tmux root commands broklein",
    mustContain: ["tmux", "sudo"],
    niceToHave: ["send-keys", "capture-pane"],
    category: "workflow"
  },
  {
    scenario: "Agent wants to send an iMessage to someone",
    query: "send iMessage via BlueBubbles",
    mustContain: ["bluebubbles", "mcporter"],
    niceToHave: ["send-message", "collins"],
    category: "workflow"
  },

  // HISTORY — learning from past mistakes
  {
    scenario: "Agent is about to reboot Spark and wants to know what to check after",
    query: "spark reboot checklist what to check",
    mustContain: ["WireGuard", "reboot"],
    niceToHave: ["nvidia-persistenced", "wg-spark"],
    category: "history"
  },
  {
    scenario: "Agent sees model aliases missing from config and wants to understand why",
    query: "model alias fields removed from config incident",
    mustContain: ["alias", "removed"],
    niceToHave: ["30", "2026-02-25", "heartbeat"],
    category: "history"
  },
  {
    scenario: "Agent needs to understand the LCM corruption issue",
    query: "LCM database corruption tail messages",
    mustContain: ["lcm", "corruption"],
    niceToHave: ["lcm.db", "tail messages"],
    category: "history"
  },

  // REFERENCE — technical knowledge
  {
    scenario: "Agent needs to know Nemotron-Super token limits",
    query: "Nemotron-Super maxTokens token budget",
    mustContain: ["65536"],
    niceToHave: ["budget_tokens", "reasoning", "think"],
    category: "reference"
  },
  {
    scenario: "Agent needs GPU memory utilization for Spark services",
    query: "GPU_MEMORY_UTIL spark services VRAM",
    mustContain: ["GPU_MEMORY"],
    niceToHave: ["0.65", "VRAM", "boot-loop"],
    category: "reference"
  },
];

async function main() {
  const cfg = await resolveConfig();
  const db = new LanceDBBackend(cfg);
  await db.open();
  const embed = await createEmbedProvider(cfg.embed);

  const status = await db.status();
  console.log("=== Practical Agent Utility Eval ===\n");
  console.log(`Database: ${status.chunkCount} chunks`);
  console.log(`Embed: ${embed.id}/${embed.model} (${embed.dims}d)`);
  console.log(`Testing ${TESTS.length} real agent scenarios...\n`);

  const results: { pass: boolean; scenario: string; category: string; missing: string[]; found: string[]; bonus: string[] }[] = [];

  for (const test of TESTS) {
    // Embed the query
    const queryVec = await embed.embedQuery(test.query);
    
    // Search via both vector and FTS (mimicking what the plugin does)
    const [vectorResults, ftsResults] = await Promise.all([
      db.vectorSearch(queryVec, { query: test.query, maxResults: 10, minScore: 0.0 }).catch(() => []),
      db.ftsSearch(test.query, { query: test.query, maxResults: 10 }).catch(() => []),
    ]);

    // Combine all retrieved text from both result sets
    const allChunks = [...vectorResults, ...ftsResults];
    const allText = allChunks.map((r: any) => `${r.chunk?.text || r.text || ""}`).join("\n").toLowerCase();
    
    // Check must-contain
    const found: string[] = [];
    const missing: string[] = [];
    for (const kw of test.mustContain) {
      if (allText.includes(kw.toLowerCase())) {
        found.push(kw);
      } else {
        missing.push(kw);
      }
    }

    // Check nice-to-have
    const bonus: string[] = [];
    for (const kw of test.niceToHave || []) {
      if (allText.includes(kw.toLowerCase())) {
        bonus.push(kw);
      }
    }

    const pass = missing.length === 0;
    results.push({ pass, scenario: test.scenario, category: test.category, missing, found, bonus });

    const icon = pass ? "✅" : "❌";
    const bonusStr = bonus.length > 0 ? ` (+${bonus.length} bonus)` : "";
    console.log(`${icon} [${test.category}] ${test.scenario}`);
    if (!pass) {
      console.log(`   Missing: ${missing.join(", ")}`);
      console.log(`   Found: ${found.join(", ")}`);
    }
    if (bonus.length > 0) {
      console.log(`   Bonus context: ${bonus.join(", ")}`);
    }
  }

  // Summary
  const total = results.length;
  const passed = results.filter(r => r.pass).length;
  const byCategory = new Map<string, { pass: number; total: number }>();
  for (const r of results) {
    const cat = byCategory.get(r.category) || { pass: 0, total: 0 };
    cat.total++;
    if (r.pass) cat.pass++;
    byCategory.set(r.category, cat);
  }

  console.log("\n=== Results ===");
  console.log(`Overall: ${passed}/${total} scenarios pass (${Math.round(passed/total*100)}%)`);
  console.log("\nBy category:");
  for (const [cat, stats] of [...byCategory].sort()) {
    const pct = Math.round(stats.pass / stats.total * 100);
    const bar = "█".repeat(Math.round(pct / 10)) + "░".repeat(10 - Math.round(pct / 10));
    console.log(`  ${cat.padEnd(16)} ${bar} ${stats.pass}/${stats.total} (${pct}%)`);
  }

  // Compute bonus coverage
  const totalBonus = results.reduce((sum, r) => sum + (r.bonus?.length || 0), 0);
  const maxBonus = TESTS.reduce((sum, t) => sum + (t.niceToHave?.length || 0), 0);
  console.log(`\nBonus context coverage: ${totalBonus}/${maxBonus} (${Math.round(totalBonus/maxBonus*100)}%)`);
  
  console.log("\n=== What This Means ===");
  if (passed / total >= 0.8) {
    console.log("✅ GOOD: Agents can answer most real scenarios using memory context.");
  } else if (passed / total >= 0.5) {
    console.log("⚠️  FAIR: Agents get some help, but significant gaps remain.");
  } else {
    console.log("❌ POOR: Memory isn't providing meaningful help for agent tasks.");
  }
  console.log(`\nFailed scenarios need attention — agents will hallucinate or ask Klein to repeat info.`);

  process.exit(passed === total ? 0 : 1);
}

main().catch(console.error);
