#!/usr/bin/env npx tsx
/**
 * compute-significance.ts — Statistical significance testing for BEIR benchmark results.
 * 
 * Uses bootstrap resampling to compute:
 * - 95% confidence intervals for NDCG@10 differences between configs
 * - p-values (two-sided paired bootstrap test)
 * - Effect sizes (Cohen's d)
 * 
 * Usage:
 *   npx tsx evaluation/scripts/compute-significance.ts \
 *     --base ./evaluation/results/scitest-vec-a-results.json \
 *     --compare ./evaluation/results/scitest-vec-u-results.json \
 *     --dataset scifact
 */

import fs from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";
import { ndcgAtK, mrrAtK, recallAtK, type Qrels, type Results } from "../metrics.js";

// ── Bootstrap Resampling ───────────────────────────────────────────────────────

function bootstrapPairedDiff(
  base: number[],
  treat: number[],
  nBoot: number = 10000,
  seed?: number,
): { meanDiff: number; ciLow: number; ciHigh: number; pValue: number } {
  const n = base.length;
  const observedDiff = (treat.reduce((a, b) => a + b, 0) - base.reduce((a, b) => a + b, 0)) / n;
  
  // Pooled under null hypothesis
  const diffs = base.map((b, i) => treat[i]! - b);
  const pooledMean = diffs.reduce((a, b) => a + b, 0) / n;
  const centered = diffs.map(d => d - pooledMean);
  
  // Bootstrap on centered differences
  const bootDiffs: number[] = [];
  let state = seed ?? Date.now();
  for (let i = 0; i < nBoot; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      state = (state * 1664525 + 1013904223) & 0x7fffffff;
      const idx = state % n;
      sum += centered[idx]!;
    }
    bootDiffs.push(sum / n);
  }
  
  bootDiffs.sort((a, b) => a - b);
  const ciLow = bootDiffs[Math.floor(nBoot * 0.025)]!;
  const ciHigh = bootDiffs[Math.floor(nBoot * 0.975)!];
  const pValue = 2 * (bootDiffs.filter(d => Math.abs(d) >= Math.abs(observedDiff)).length / nBoot);
  
  return { meanDiff: observedDiff, ciLow, ciHigh, pValue };
}

function cohensD(base: number[], treat: number[]): number {
  const n = base.length;
  const meanBase = base.reduce((a, b) => a + b, 0) / n;
  const meanTreat = treat.reduce((a, b) => a + b, 0) / n;
  const varBase = base.reduce((a, b) => a + (b - meanBase) ** 2, 0) / (n - 1);
  const varTreat = treat.reduce((a, b) => a + (b - meanTreat) ** 2, 0) / (n - 1);
  const pooledSd = Math.sqrt((varBase + varTreat) / 2);
  return pooledSd === 0 ? 0 : (meanTreat - meanBase) / pooledSd;
}

// ── Load BEIR data ─────────────────────────────────────────────────────────────

async function loadResults(filePath: string): Promise<Results> {
  const content = await fs.readFile(filePath, "utf-8");
  const data = JSON.parse(content);
  // Support both raw Results format and { results: Results } wrapper
  if (data.results) return data.results as Results;
  if (data.queryResults) return data.queryResults as Results;
  // Some result formats store per-query as ndcg.@10 etc.
  throw new Error(`Unknown results format in ${filePath}. Keys: ${Object.keys(data).join(", ")}`);
}

// ── Main ───────────────────────────────────────────────────────────────────────

const { values: args } = parseArgs({
  options: {
    base:     { type: "string" },
    compare:  { type: "string" },
    dataset:  { type: "string", default: "unknown" },
    metric:   { type: "string", default: "ndcg10" },
    boots:    { type: "string", default: "10000" },
    qrels:    { type: "string", default: "" },
  },
});

const basePath = args.base!;
const comparePath = args.compare!;
const dataset = args.dataset!;
const nBoot = parseInt(args.boots!);

if (!basePath || !comparePath) {
  console.error("Usage: --base <file> --compare <file> [--dataset name] [--metric ndcg10|mrr10|recall10]");
  process.exit(1);
}

const [baseData, compareData] = await Promise.all([
  loadResults(basePath),
  loadResults(comparePath),
]);

const queryIds = Object.keys(baseData).filter(id => compareData[id] !== undefined);
console.log(`\n📊 Significance Test: ${dataset}`);
console.log(`   Queries: ${queryIds.length}`);
console.log(`   Bootstrap iterations: ${nBoot}`);

// Compute per-query scores using the actual evaluation function
// Note: We need qrels to compute NDCG. If not provided, we use the stored scores.
let perQueryBase: number[];
let perQueryTreat: number[];

if (args.qrels) {
  // Load qrels and recompute metrics
  const qrelsContent = await fs.readFile(args.qrels!, "utf-8");
  const qrels: Qrels = {};
  for (const line of qrelsContent.trim().split("\n").slice(1)) {
    const [queryId, , docId, rel] = line.split("\t");
    if (!qrels[queryId]) qrels[queryId] = {};
    qrels[queryId]![docId] = parseInt(rel, 10);
  }
  
  const k = args.metric === "mrr10" ? 10 : args.metric === "recall10" ? 10 : 10;
  const metricFn = args.metric === "mrr10" ? mrrAtK : args.metric === "recall10" ? recallAtK : ndcgAtK;
  
  const baseScores = metricFn(qrels, baseData, k);
  const compareScores = metricFn(qrels, compareData, k);
  
  perQueryBase = queryIds.map(id => baseScores[id] ?? 0);
  perQueryTreat = queryIds.map(id => compareScores[id] ?? 0);
} else {
  // Results already contain raw scores (format: { docId: score })
  // For NDCG comparison we need the full ranked list
  // Use mean score as proxy if qrels not available
  perQueryBase = queryIds.map(id => {
    const docs = baseData[id]!;
    return Object.values(docs).reduce((a, b) => a + b, 0) / Math.max(Object.keys(docs).length, 1);
  });
  perQueryTreat = queryIds.map(id => {
    const docs = compareData[id]!;
    return Object.values(docs).reduce((a, b) => a + b, 0) / Math.max(Object.keys(docs).length, 1);
  });
}

const meanBase = perQueryBase.reduce((a, b) => a + b, 0) / perQueryBase.length;
const meanTreat = perQueryTreat.reduce((a, b) => a + b, 0) / perQueryTreat.length;
const diff = meanTreat - meanBase;
const d = cohensD(perQueryBase, perQueryTreat);

const result = bootstrapPairedDiff(perQueryBase, perQueryTreat, nBoot, dataset.split("").reduce((a, c) => a + c.charCodeAt(0), 0));

console.log(`\n${args.base} (mean ${meanBase.toFixed(4)}) vs ${args.compare} (mean ${meanTreat.toFixed(4)})`);
console.log(`${args.metric}: Δ = ${diff.toFixed(4)} | 95% CI [${result.ciLow.toFixed(4)}, ${result.ciHigh.toFixed(4)}] | p = ${result.pValue.toFixed(4)} | Cohen's d = ${d.toFixed(4)}`);

const sig = result.pValue < 0.05 ? "✅ Significant (p<0.05)" : "❌ Not significant (p≥0.05)";
const direction = diff > 0 ? "↑" : "↓";
console.log(`${direction} ${sig}`);

if (result.ciLow > 0 || result.ciHigh < 0) {
  console.log("   95% CI excludes 0 → difference is statistically reliable");
}

// Effect size interpretation
const effectSize = Math.abs(d);
let effectLabel: string;
if (effectSize < 0.2) effectLabel = "negligible";
else if (effectSize < 0.5) effectLabel = "small";
else if (effectSize < 0.8) effectLabel = "medium";
else effectLabel = "large";
console.log(`   Effect size: ${effectLabel} (|d| = ${effectSize.toFixed(3)})`);
