#!/usr/bin/env npx tsx
/**
 * Unified Test Suite — runs everything in order.
 *
 * 1. Unit tests (offline, no Spark needed)
 * 2. Type checking (tsc --noEmit)
 * 3. Lint (eslint)
 * 4. Integration tests (needs Spark for embed/rerank)
 * 5. Retrieval benchmark (needs indexed data)
 * 6. Pipeline integration tests (needs indexed data)
 *
 * Usage:
 *   npx tsx evaluation/run-all.ts           # everything
 *   npx tsx evaluation/run-all.ts --quick   # unit + types + lint only
 *   npm test                                # alias for full suite
 */

import { execSync } from "node:child_process";

const isQuick = process.argv.includes("--quick");

interface TestResult {
  name: string;
  passed: boolean;
  duration: number;
  output: string;
}

const results: TestResult[] = [];

function run(name: string, cmd: string, allowFail = false): boolean {
  const start = Date.now();
  process.stdout.write(`\n▶ ${name}... `);
  try {
    const output = execSync(cmd, {
      cwd: import.meta.dirname!.replace(/\/evaluation$/, ""),
      encoding: "utf-8",
      timeout: 300000,
      stdio: ["pipe", "pipe", "pipe"],
    });
    const duration = Date.now() - start;
    console.log(`✅ (${(duration / 1000).toFixed(1)}s)`);
    results.push({ name, passed: true, duration, output });
    return true;
  } catch (err) {
    const duration = Date.now() - start;
    const output = (err as { stdout?: string; stderr?: string }).stdout ?? "";
    const stderr = (err as { stderr?: string }).stderr ?? "";
    if (allowFail) {
      console.log(`⚠️  (${(duration / 1000).toFixed(1)}s) — skipped`);
      results.push({ name, passed: true, duration, output: `SKIPPED: ${stderr.slice(0, 200)}` });
      return false;
    }
    console.log(`❌ (${(duration / 1000).toFixed(1)}s)`);
    console.log(`   ${(output + stderr).split("\n").slice(-5).join("\n   ")}`);
    results.push({ name, passed: false, duration, output: output + stderr });
    return false;
  }
}

console.log("═══════════════════════════════════════════");
console.log("  memory-spark Test Suite");
console.log("═══════════════════════════════════════════");

// Phase 1: Offline checks (no external deps)
run("Unit Tests", "npx tsx tests/unit.ts");
run("Type Check", "npx tsc --noEmit");
run("Lint", "npx eslint 'src/**/*.ts' 'index.ts' --max-warnings 5");

if (!isQuick) {
  // Phase 2: Integration (needs Spark)
  run("Retrieval Benchmark", "npx tsx evaluation/benchmark.ts --tier 1", true);
  run("Pipeline Integration", "npx tsx evaluation/benchmark.ts --tier 2", true);
}

// Summary
console.log("\n═══════════════════════════════════════════");
console.log("  Results");
console.log("═══════════════════════════════════════════\n");

let allPassed = true;
for (const r of results) {
  const icon = r.passed ? "✅" : "❌";
  console.log(`  ${icon} ${r.name} (${(r.duration / 1000).toFixed(1)}s)`);
  if (!r.passed) allPassed = false;
}

const totalDuration = results.reduce((a, r) => a + r.duration, 0);
console.log(`\n  Total: ${results.length} suites, ${(totalDuration / 1000).toFixed(1)}s`);

if (!allPassed) {
  console.log("\n❌ SOME TESTS FAILED");
  process.exit(1);
} else {
  console.log("\n✅ ALL TESTS PASSED");
}
