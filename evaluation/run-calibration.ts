#!/usr/bin/env npx tsx
/**
 * Run judge calibration only — validates that Nemotron-Super's
 * relevance scoring aligns with human judgment before using it
 * for full evaluation.
 *
 * Pass criteria: ≥80% of 20 calibration pairs within ±1 of expected score.
 * If this fails, the judge is unreliable and we should NOT use it.
 */

import { runCalibration } from "./judge.js";

async function main() {
  console.log("═══════════════════════════════════════════════════════");
  console.log("  memory-spark Judge Calibration");
  console.log("  Model: Nemotron-Super-120B on Spark (127.0.0.1:18080)");
  console.log("═══════════════════════════════════════════════════════");

  const t0 = Date.now();
  const result = await runCalibration();
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

  console.log(`\nCompleted in ${elapsed}s`);

  const passRate = result.passed / result.total;

  if (passRate >= 0.8) {
    console.log("\n✅ CALIBRATION PASSED — Judge is reliable for evaluation");
    console.log("   Safe to proceed with LLM-as-judge scoring in the benchmark suite.");
  } else {
    console.log("\n❌ CALIBRATION FAILED — Judge is NOT reliable");
    console.log("   Do NOT use LLM-as-judge scoring. Stick to NDCG/MRR path-matching only.");
    console.log(`   Pass rate: ${(passRate * 100).toFixed(0)}% (need ≥80%)`);
  }

  // Show score distribution
  const scores = result.details.map((d) => d.score);
  const expected = result.details.map((d) => d.pair.expectedScore);
  const diffs = result.details.map((d) => d.diff);
  console.log(`\n  Score distribution: mean=${(scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2)}`);
  console.log(`  Expected mean: ${(expected.reduce((a, b) => a + b, 0) / expected.length).toFixed(2)}`);
  console.log(`  Mean absolute error: ${(diffs.reduce((a, b) => a + b, 0) / diffs.length).toFixed(2)}`);

  process.exit(passRate >= 0.8 ? 0 : 1);
}

main().catch((e) => {
  console.error("Fatal:", e);
  process.exit(1);
});
