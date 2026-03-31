#!/usr/bin/env npx tsx
/**
 * FORENSIC DIAGNOSTIC — traces every pipeline stage per-query
 * to definitively identify what hurts performance.
 * 
 * Outputs:
 * 1. Arrow Vector bug verification
 * 2. RRF overlap rates and dual-evidence promotion analysis
 * 3. Reranker score distribution and rank changes  
 * 4. MMR behavior with fixed vs broken vectors
 * 5. HyDE activation status
 * 6. Per-query NDCG impact of each stage
 *
 * Usage:
 *   npx tsx scripts/diag-full.ts scifact 50
 *   npx tsx scripts/diag-full.ts nfcorpus 100
 */

import fs from "node:fs/promises";
import path from "node:path";

import { hybridMerge, mmrRerank, cosineSimilarity } from "../src/auto/recall.js";
import { resolveConfig, type HydeConfig } from "../src/config.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { EmbedQueue } from "../src/embed/queue.js";
import { generateHypotheticalDocument } from "../src/hyde/generator.js";
import { createReranker } from "../src/rerank/reranker.js";
import type { SearchResult } from "../src/storage/backend.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { ndcgAtK, type Qrels, type Results } from "../evaluation/metrics.js";

// ── Types ─────────────────────────────────────────────────────────────────

interface BeirQuery { _id: string; text: string; }

async function loadQueries(dataset: string): Promise<BeirQuery[]> {
  const file = path.join(import.meta.dirname!, "../evaluation/beir-datasets", dataset, "queries.jsonl");
  const content = await fs.readFile(file, "utf-8");
  return content.trim().split("\n").filter(l => l.trim()).map(l => JSON.parse(l));
}

async function loadQrels(dataset: string): Promise<Qrels> {
  const file = path.join(import.meta.dirname!, "../evaluation/beir-datasets", dataset, "qrels", "test.tsv");
  const content = await fs.readFile(file, "utf-8");
  const qrels: Qrels = {};
  for (const line of content.trim().split("\n")) {
    if (line.startsWith("query-id")) continue;
    const [qid, did, score] = line.split("\t");
    if (!qrels[qid!]) qrels[qid!] = {};
    qrels[qid!]![did!] = parseInt(score!, 10);
  }
  return qrels;
}

function stripBeirPrefix(id: string): string {
  return id.replace(/^beir-(scifact|nfcorpus|fiqa)-/, "");
}

/** Convert Arrow Vector to JS array for cosine computation */
function toJsArray(vec: unknown): number[] {
  if (Array.isArray(vec)) return vec;
  if (vec && typeof (vec as any).toArray === "function") return Array.from((vec as any).toArray());
  if (vec && typeof (vec as any)[Symbol.iterator] === "function") return Array.from(vec as Iterable<number>);
  return [];
}

function resultsToMap(results: SearchResult[], k: number): Record<string, number> {
  const map: Record<string, number> = {};
  for (const r of results.slice(0, k)) {
    map[stripBeirPrefix(r.chunk.id)] = r.score;
  }
  return map;
}

function meanOfValues(scores: Record<string, number>): number {
  const vals = Object.values(scores);
  if (vals.length === 0) return 0;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

async function main() {
  const dataset = process.argv[2] || "scifact";
  const sampleSize = parseInt(process.argv[3] || "50");

  console.log(`\n${"═".repeat(70)}`);
  console.log(`  FORENSIC DIAGNOSTIC — ${dataset} (${sampleSize} queries)`);
  console.log(`${"═".repeat(70)}\n`);

  const lancedbDir = process.env.BEIR_LANCEDB_DIR || `${process.env.HOME}/.openclaw/data/testDbBEIR/lancedb`;
  const cfg = resolveConfig({ lancedbDir } as Parameters<typeof resolveConfig>[0]);
  const backend = new LanceDBBackend(cfg);
  await backend.open();
  const provider = await createEmbedProvider(cfg.embed);
  const embed = new EmbedQueue(provider, { concurrency: 1, maxRetries: 2, timeoutMs: 30000 });
  const reranker = cfg.rerank.enabled ? await createReranker(cfg.rerank) : null;

  const queries = await loadQueries(dataset);
  const qrels = await loadQrels(dataset);
  const evalQueries = queries.filter(q => qrels[q._id] && Object.keys(qrels[q._id]!).length > 0);
  const sample = evalQueries.slice(0, sampleSize);

  console.log(`[INFO] ${evalQueries.length} total eval queries, using ${sample.length}`);
  console.log(`[INFO] Embed provider: ${provider.id} / ${provider.model} (${provider.dims}d)`);
  console.log(`[INFO] Reranker: ${reranker ? "enabled" : "disabled"}`);

  // ── BUG 1: Arrow Vector Verification ──────────────────────────────────
  console.log("\n" + "─".repeat(70));
  console.log("  BUG 1: Arrow Vector Type Check");
  console.log("─".repeat(70));
  
  const testVec = await embed.embedQuery("test");
  const testResults = await backend.vectorSearch(testVec, { 
    query: "test", maxResults: 1, minScore: 0.0, pathContains: `beir/${dataset}/` 
  });
  if (testResults.length > 0) {
    const v = testResults[0]!.vector!;
    const isJsArray = Array.isArray(v);
    const bracketWorks = v[0] !== undefined;
    const hasToArray = typeof (v as any).toArray === "function";
    console.log(`  vector constructor: ${v.constructor.name}`);
    console.log(`  is JS Array: ${isJsArray}`);
    console.log(`  bracket indexing works: ${bracketWorks}`);
    console.log(`  has .toArray(): ${hasToArray}`);
    
    if (!bracketWorks && hasToArray) {
      console.log(`  ⚠️  CONFIRMED: Vectors are Arrow Vector objects, not JS arrays`);
      console.log(`  ⚠️  cosineSimilarity() returns NaN → MMR is completely broken`);
      const jsArr = toJsArray(v);
      console.log(`  ✓  toJsArray() fix: [0]=${jsArr[0]?.toFixed(6)}`);
    } else if (bracketWorks) {
      console.log(`  ✓  Vectors are proper JS arrays — cosine should work`);
    }
  }

  // ── Collect per-stage results for each query ──────────────────────────
  const k = 10;

  const perQueryNdcg: {
    vectorOnly: Results;
    hybrid: Results;
    hybridReranked: Results;
    hybridMmrBroken: Results;
    hybridMmrFixed: Results;
    vectorReranked: Results;
  } = {
    vectorOnly: {},
    hybrid: {},
    hybridReranked: {},
    hybridMmrBroken: {},
    hybridMmrFixed: {},
    vectorReranked: {},
  };

  // Aggregate stats
  let overlapSum = 0;
  let dualEvidenceWinsCorrect = 0;
  let dualEvidenceWinsWrong = 0;
  let rerankerPromotesRelevant = 0;
  let rerankerDemotesRelevant = 0;
  let rerankerNoChange = 0;
  let rerankerScoreSaturation = 0;
  let mmrFixedChangesOrder = 0;
  let hydeActivated = 0;
  let hydeFailed = 0;

  for (let qi = 0; qi < sample.length; qi++) {
    const q = sample[qi]!;
    if ((qi + 1) % 10 === 0 || qi === 0) {
      process.stdout.write(`\r  Processing query ${qi + 1}/${sample.length}...`);
    }

    const queryQrels = qrels[q._id]!;
    const relevantDocs = new Set(
      Object.entries(queryQrels).filter(([, r]) => r > 0).map(([id]) => id)
    );

    const queryVector = await embed.embedQuery(q.text);

    // Stage 1: Vector search
    const vectorResults = await backend.vectorSearch(queryVector, {
      query: q.text, maxResults: k * 4, minScore: 0.0, pathContains: `beir/${dataset}/`
    }).catch(() => [] as SearchResult[]);

    // Stage 2: FTS search
    const ftsResults = await backend.ftsSearch(q.text, {
      query: q.text, maxResults: k * 4, pathContains: `beir/${dataset}/`
    }).catch(() => [] as SearchResult[]);

    perQueryNdcg.vectorOnly[q._id] = resultsToMap(vectorResults, k);

    // Stage 3: Hybrid merge (RRF)
    const hybridResults = (vectorResults.length > 0 && ftsResults.length > 0)
      ? hybridMerge(vectorResults, ftsResults, k * 2)
      : [...vectorResults, ...ftsResults].slice(0, k * 2);
    perQueryNdcg.hybrid[q._id] = resultsToMap(hybridResults, k);

    // RRF overlap analysis
    const vSet = new Set(vectorResults.slice(0, k * 4).map(r => r.chunk.id));
    const fSet = new Set(ftsResults.slice(0, k * 4).map(r => r.chunk.id));
    const overlap = [...vSet].filter(id => fSet.has(id));
    overlapSum += overlap.length;

    if (hybridResults.length > 0 && vectorResults.length > 0) {
      const topHybridId = stripBeirPrefix(hybridResults[0]!.chunk.id);
      const topVectorId = stripBeirPrefix(vectorResults[0]!.chunk.id);
      if (topHybridId !== topVectorId) {
        if (relevantDocs.has(topHybridId)) dualEvidenceWinsCorrect++;
        else dualEvidenceWinsWrong++;
      }
    }

    // Stage 4a: Reranker on hybrid results
    let rerankedHybrid = hybridResults.slice(0, k);
    if (reranker && hybridResults.length > 0) {
      rerankedHybrid = await reranker.rerank(q.text, hybridResults, k);
      
      if (rerankedHybrid.length > 0 && rerankedHybrid[0]!.score >= 0.999) {
        rerankerScoreSaturation++;
      }

      const hybridTop10 = hybridResults.slice(0, k);
      const hybridRelevantRanks = hybridTop10
        .map((r, i) => relevantDocs.has(stripBeirPrefix(r.chunk.id)) ? i : -1)
        .filter(i => i >= 0);
      const rerankedRelevantRanks = rerankedHybrid
        .map((r, i) => relevantDocs.has(stripBeirPrefix(r.chunk.id)) ? i : -1)
        .filter(i => i >= 0);
      
      if (rerankedRelevantRanks.length > 0 && hybridRelevantRanks.length > 0) {
        const bestBefore = Math.min(...hybridRelevantRanks);
        const bestAfter = Math.min(...rerankedRelevantRanks);
        if (bestAfter < bestBefore) rerankerPromotesRelevant++;
        else if (bestAfter > bestBefore) rerankerDemotesRelevant++;
        else rerankerNoChange++;
      }
    }
    perQueryNdcg.hybridReranked[q._id] = resultsToMap(rerankedHybrid, k);

    // Stage 4b: Reranker on vector-only results (no FTS noise)
    let rerankedVector = vectorResults.slice(0, k);
    if (reranker && vectorResults.length > 0) {
      rerankedVector = await reranker.rerank(q.text, vectorResults, k);
    }
    perQueryNdcg.vectorReranked[q._id] = resultsToMap(rerankedVector, k);

    // Stage 5a: MMR with BROKEN vectors (current behavior — Arrow Vector objects)
    const mmrBroken = mmrRerank([...hybridResults], k, 0.7);
    perQueryNdcg.hybridMmrBroken[q._id] = resultsToMap(mmrBroken, k);

    // Stage 5b: MMR with FIXED vectors (converted to JS arrays)
    const fixedResults = hybridResults.map(r => ({
      ...r,
      vector: toJsArray(r.vector),
    }));
    const mmrFixed = mmrRerank(fixedResults, k, 0.7);
    perQueryNdcg.hybridMmrFixed[q._id] = resultsToMap(mmrFixed, k);

    const brokenOrder = mmrBroken.slice(0, k).map(r => r.chunk.id).join(",");
    const fixedOrder = mmrFixed.slice(0, k).map(r => r.chunk.id).join(",");
    if (brokenOrder !== fixedOrder) mmrFixedChangesOrder++;

    // Stage 6: HyDE check (first 5 queries only — slow)
    if (qi < 5 && cfg.hyde?.enabled) {
      try {
        const hyp = await generateHypotheticalDocument(q.text, cfg.hyde);
        if (hyp) hydeActivated++;
        else hydeFailed++;
      } catch {
        hydeFailed++;
      }
    }
  }

  console.log("\n");

  // ── Compute aggregate NDCG@10 for each stage ──────────────────────────
  console.log("─".repeat(70));
  console.log("  AGGREGATE RESULTS — NDCG@10");
  console.log("─".repeat(70));

  const stages = [
    { name: "A: Vector-Only", results: perQueryNdcg.vectorOnly },
    { name: "C: Hybrid (RRF)", results: perQueryNdcg.hybrid },
    { name: "D: Hybrid + Reranker", results: perQueryNdcg.hybridReranked },
    { name: "E: Hybrid + MMR (broken)", results: perQueryNdcg.hybridMmrBroken },
    { name: "E*: Hybrid + MMR (fixed)", results: perQueryNdcg.hybridMmrFixed },
    { name: "H: Vector + Reranker", results: perQueryNdcg.vectorReranked },
  ];

  const vectorNdcg = ndcgAtK(qrels, perQueryNdcg.vectorOnly, k);
  const vectorMean = meanOfValues(vectorNdcg);

  for (const stage of stages) {
    const scores = ndcgAtK(qrels, stage.results, k);
    const mean = meanOfValues(scores);
    const delta = ((mean - vectorMean) / vectorMean * 100).toFixed(1);
    const arrow = mean > vectorMean ? "↑" : mean < vectorMean ? "↓" : "=";
    console.log(`  ${stage.name.padEnd(30)} NDCG@10 = ${mean.toFixed(4)}  (${arrow}${delta}% vs Vector-Only)`);
  }

  // ── Detailed diagnostic summaries ─────────────────────────────────────
  console.log("\n" + "─".repeat(70));
  console.log("  BUG 2: RRF Overlap & Dual-Evidence Analysis");
  console.log("─".repeat(70));
  console.log(`  Avg overlap between Vector top-40 and FTS top-40: ${(overlapSum / sample.length).toFixed(1)} docs`);
  console.log(`  When RRF promotes a different #1 than Vector:`);
  console.log(`    Correct (promoted doc IS relevant): ${dualEvidenceWinsCorrect}`);
  console.log(`    Wrong (promoted doc NOT relevant):  ${dualEvidenceWinsWrong}`);

  console.log("\n" + "─".repeat(70));
  console.log("  BUG 3: Reranker Impact on Relevant Document Ranking");
  console.log("─".repeat(70));
  console.log(`  Score saturation (top-1 ≥ 0.999): ${rerankerScoreSaturation}/${sample.length} (${(rerankerScoreSaturation/sample.length*100).toFixed(0)}%)`);
  console.log(`  Reranker PROMOTES relevant docs to higher rank: ${rerankerPromotesRelevant}`);
  console.log(`  Reranker DEMOTES relevant docs to lower rank:   ${rerankerDemotesRelevant}`);
  console.log(`  Reranker no change to relevant doc rank:        ${rerankerNoChange}`);

  console.log("\n" + "─".repeat(70));
  console.log("  BUG 4: MMR Arrow Vector Bug Impact");
  console.log("─".repeat(70));
  console.log(`  Queries where fixing vectors CHANGES MMR output: ${mmrFixedChangesOrder}/${sample.length}`);
  console.log(`  (If 0, MMR was a complete no-op due to NaN cosine similarity)`);

  if (cfg.hyde?.enabled) {
    console.log("\n" + "─".repeat(70));
    console.log("  BUG 5: HyDE Activation (first 5 queries)");
    console.log("─".repeat(70));
    console.log(`  Activated successfully: ${hydeActivated}`);
    console.log(`  Failed (timeout/error): ${hydeFailed}`);
  }

  // Save full results as JSON
  const reportFile = path.join(
    import.meta.dirname!, 
    "../evaluation/results", 
    `diag-${dataset}-${Date.now()}.json`
  );
  await fs.writeFile(reportFile, JSON.stringify({
    dataset,
    sampleSize: sample.length,
    stages: stages.map(s => {
      const scores = ndcgAtK(qrels, s.results, k);
      return { name: s.name, ndcg: meanOfValues(scores) };
    }),
    overlapAvg: overlapSum / sample.length,
    dualEvidence: { correct: dualEvidenceWinsCorrect, wrong: dualEvidenceWinsWrong },
    reranker: { 
      saturation: rerankerScoreSaturation, 
      promotes: rerankerPromotesRelevant, 
      demotes: rerankerDemotesRelevant,
      noChange: rerankerNoChange,
    },
    mmrFixedChanges: mmrFixedChangesOrder,
    hyde: { activated: hydeActivated, failed: hydeFailed },
  }, null, 2));
  console.log(`\n[INFO] Full report saved to: ${reportFile}`);

  await backend.close();
}

main().catch(e => { console.error(e); process.exit(1); });
