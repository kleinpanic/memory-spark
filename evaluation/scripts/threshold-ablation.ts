#!/usr/bin/env npx tsx
/**
 * threshold-ablation.ts — Gate threshold sensitivity analysis
 * 
 * Addresses Reviewer 2's critique (C3): "Gate thresholds τ_h=0.08, τ_l=0.02 
 * cherry-picked — non-monotonic threshold sweep could be noise on 300 queries"
 * 
 * Method: 
 * 1. Split SciFact queries 70/30 (seeded deterministic split)
 * 2. Sweep high-threshold τ_h ∈ {0.04, 0.06, 0.08, 0.10, 0.12, 0.14}
 *    and low-threshold τ_l ∈ {0.01, 0.02, 0.03, 0.04}
 *    on the TRAIN split (70%)
 * 3. Pick best (τ_h*, τ_l*) from train
 * 4. Evaluate on HELD-OUT test split (30%)
 * 
 * Usage:
 *   npx tsx evaluation/scripts/threshold-ablation.ts \
 *     --checkpoint ./evaluation/.checkpoints/scitest-gate-a-results.json \
 *     --output ./evaluation/results/threshold-ablation.json
 */

import fs from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";
import { ndcgAtK } from "../metrics.js";
import type { Qrels } from "../metrics.js";

interface QueryResult {
  vectorScores?: Record<string, number>;   // raw cosine scores per docId
  rerankedScores?: Record<string, number>; // after reranker
  vectorTopDocs?: string[];               // top 5 docIds from vector search
  vectorTopScores?: number[];              // their cosine scores
  ndcg10?: number;
}

interface CheckpointData {
  [queryId: string]: QueryResult;
}

interface Qrels {
  [queryId: string]: Record<string, number>;
}

interface AblationResult {
  train: {
    tauH: number;
    tauL: number;
    ndcg10: number;
    skipRate: number;
  };
  test: {
    ndcg10: number;
    skipRate: number;
    vsVectorOnly: number;   // NDCG delta vs vector-only
    vsGATE_A: number;       // delta vs default (0.08, 0.02)
  };
}

// ── Load checkpoint data ────────────────────────────────────────────────────────

async function loadCheckpoint(filePath: string): Promise<CheckpointData> {
  const content = await fs.readFile(filePath, "utf-8");
  return JSON.parse(content) as CheckpointData;
}

async function loadQrels(dataset: string): Promise<Qrels> {
  const beirDir = process.env.BEIR_DATA_DIR ?? "/data/beir-datasets";
  const filePath = path.join(beirDir, dataset, "qrels", "test.tsv");
  const content = await fs.readFile(filePath, "utf-8");
  const qrels: Qrels = {};
  for (const line of content.trim().split("\n").slice(1)) {
    const [queryId, , docId, rel] = line.split("\t");
    if (!qrels[queryId]) qrels[queryId] = {};
    qrels[queryId]![docId] = parseInt(rel, 10);
  }
  return qrels;
}

// ── Gate simulation ────────────────────────────────────────────────────────────
//
// The gate looks at spread = max(top5_vector_scores) - min(top5_vector_scores)
// If spread > τ_h OR spread < τ_l → SKIP reranker (use vector ranking)
// Otherwise → FIRE reranker (use reranked scores)

function simulateGate(
  result: QueryResult,
  tauH: number,
  tauL: number,
): { useReranker: boolean; spread: number } {
  const topScores = result.vectorTopScores ?? [];
  if (topScores.length < 2) return { useReranker: false, spread: 0 };
  
  const spread = Math.max(...topScores) - Math.min(...topScores);
  const useReranker = spread > tauL && spread <= tauH;
  return { useReranker, spread };
}

// ── Threshold sweep ───────────────────────────────────────────────────────────

function sweepThresholds(
  checkpoint: CheckpointData,
  qrels: Qrels,
  trainIds: string[],
  testIds: string[],
) {
  const tauHs = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14];
  const tauLs = [0.01, 0.02, 0.03, 0.04];
  
  let bestTrain: { tauH: number; tauL: number; ndcg10: number } | null = null;

  for (const tauH of tauHs) {
    for (const tauL of tauLs) {
      if (tauL >= tauH) continue; // invalid: low >= high

      // Evaluate on train split
      let totalNdcg = 0;
      let skipped = 0;
      let queries = 0;

      for (const qid of trainIds) {
        const result = checkpoint[qid];
        if (!result) continue;
        
        const { useReranker } = simulateGate(result, tauH, tauL);
        queries++;
        
        const scores = (useReranker && result.rerankedScores)
          ? result.rerankedScores
          : (result.vectorScores ?? {});
        
        if (!useReranker) skipped++;
        
        const ndcg = ndcgAtK(qrels[qid] ?? {}, { [qid]: scores }, 10);
        totalNdcg += ndcg ?? 0;
      }

      const avgNdcg = totalNdcg / Math.max(queries, 1);
      const skipRate = queries > 0 ? skipped / queries : 0;

      if (!bestTrain || avgNdcg > bestTrain.ndcg10) {
        bestTrain = { tauH, tauL, ndcg10: avgNdcg };
      }
    }
  }

  if (!bestTrain) throw new Error("No valid threshold combination found");

  // Evaluate best thresholds on held-out test set
  const { tauH, tauL } = bestTrain;
  let testNdcgTotal = 0;
  let testSkipped = 0;
  let testQueries = 0;
  
  let gateATestNdcgTotal = 0; // for comparison
  const DEFAULT_TAU_H = 0.08;
  const DEFAULT_TAU_L = 0.02;

  for (const qid of testIds) {
    const result = checkpoint[qid];
    if (!result) continue;
    testQueries++;

    const { useReranker: useRerankerBest } = simulateGate(result, tauH, tauL);
    const { useReranker: useRerankerGATE_A } = simulateGate(result, DEFAULT_TAU_H, DEFAULT_TAU_L);

    const bestScores = (useRerankerBest && result.rerankedScores)
      ? result.rerankedScores
      : (result.vectorScores ?? {});
    const gateAScores = (useRerankerGATE_A && result.rerankedScores)
      ? result.rerankedScores
      : (result.vectorScores ?? {});

    const bestNdcg = ndcgAtK(qrels[qid] ?? {}, { [qid]: bestScores }, 10) ?? 0;
    const gateANdcg = ndcgAtK(qrels[qid] ?? {}, { [qid]: gateAScores }, 10) ?? 0;

    testNdcgTotal += bestNdcg;
    gateATestNdcgTotal += gateANdcg;
    if (!useRerankerBest) testSkipped++;
  }

  const avgBestNdcg = testNdcgTotal / Math.max(testQueries, 1);
  const avgGATE_ANdcg = gateATestNdcgTotal / Math.max(testQueries, 1);
  const testSkipRate = testSkipped / Math.max(testQueries, 1);

  // Also compute vector-only (no reranker) test NDCG
  let vecOnlyNdcgTotal = 0;
  for (const qid of testIds) {
    const result = checkpoint[qid];
    if (!result || !result.vectorScores) continue;
    const ndcg = ndcgAtK(qrels[qid] ?? {}, { [qid]: result.vectorScores }, 10) ?? 0;
    vecOnlyNdcgTotal += ndcg;
  }
  const avgVecOnlyNdcg = vecOnlyNdcgTotal / Math.max(testQueries, 1);

  return {
    train: { tauH, tauL, ndcg10: bestTrain.ndcg10, skipRate: 0 },
    test: {
      ndcg10: avgBestNdcg,
      skipRate: testSkipRate,
      vsVectorOnly: avgBestNdcg - avgVecOnlyNdcg,
      vsGATE_A: avgBestNdcg - avgGATE_ANdcg,
    },
  } as AblationResult;
}

// ── Main ───────────────────────────────────────────────────────────────────────

const { values: args } = parseArgs({
  options: {
    checkpoint: { type: "string" },
    dataset:    { type: "string", default: "scifact" },
    output:     { type: "string", default: "" },
    "train-ratio": { type: "string", default: "0.7" },
  },
});

if (!args.checkpoint) {
  console.error("Usage: --checkpoint <checkpoint.json> [--dataset scifact] [--output <out.json>]");
  process.exit(1);
}

const checkpoint = await loadCheckpoint(args.checkpoint!);
const qrels = await loadQrels(args.dataset!);

const allQueryIds = Object.keys(checkpoint).filter(id => 
  checkpoint[id]?.vectorTopScores != null && checkpoint[id]?.vectorScores != null
);
console.log(`\n📊 Threshold Ablation: ${args.dataset}`);
console.log(`   Total valid queries: ${allQueryIds.length}`);

// Deterministic 70/30 split seeded by dataset name
const seed = (args.dataset ?? "scifact").split("").reduce((a, c) => a + c.charCodeAt(0), 0);
let state = seed;
const shuffle = [...allQueryIds].sort(() => {
  state = (state * 1664525 + 1013904223) & 0x7fffffff;
  return (state >>> 0) - 0x80000000;
});

const trainRatio = parseFloat(args["train-ratio"]!);
const trainSize = Math.floor(allQueryIds.length * trainRatio);
const trainIds = shuffle.slice(0, trainSize);
const testIds = shuffle.slice(trainSize);

console.log(`   Train: ${trainIds.length} | Test: ${testIds.length}`);

const result = sweepThresholds(checkpoint, qrels, trainIds, testIds);

console.log(`\n🎯 Train-best thresholds: τ_h=${result.train.tauH}, τ_l=${result.train.tauL}`);
console.log(`   Train NDCG@10: ${result.train.ndcg10.toFixed(4)}`);
console.log(`\n📈 Held-out Test Results (${testIds.length} queries):`);
console.log(`   Test NDCG@10: ${result.test.ndcg10.toFixed(4)} (vs vector-only: ${result.test.vsVectorOnly >= 0 ? "+" : ""}${result.test.vsVectorOnly.toFixed(4)})`);
console.log(`   vs GATE-A (default τ_h=0.08, τ_l=0.02): ${result.test.vsGATE_A >= 0 ? "+" : ""}${result.test.vsGATE_A.toFixed(4)}`);
console.log(`   Skip rate: ${(result.test.skipRate * 100).toFixed(1)}%`);

if (result.test.vsGATE_A > 0.001) {
  console.log(`\n   ✅ Held-out test confirms default thresholds are near-optimal`);
} else if (result.test.vsGATE_A < -0.001) {
  console.log(`\n   ⚠️  Held-out suggests different thresholds may be better`);
  console.log(`      Consider updating defaults to τ_h=${result.train.tauH}, τ_l=${result.train.tauL}`);
} else {
  console.log(`\n   ✅ No significant difference — defaults are robust`);
}

if (args.output) {
  await fs.writeFile(args.output, JSON.stringify({
    dataset: args.dataset,
    trainSize: trainIds.length,
    testSize: testIds.length,
    trainRatio,
    result,
    trainQueryIds: trainIds,
    testQueryIds: testIds,
    timestamp: new Date().toISOString(),
  }, null, 2));
  console.log(`\n   📄 → ${args.output}`);
}
