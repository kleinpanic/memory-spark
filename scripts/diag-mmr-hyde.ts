/**
 * Diagnostic: Investigate MMR and HyDE behavior on SciFact
 * 
 * 1. MMR: Does diversity pressure hurt precision on narrow scientific queries?
 * 2. HyDE: Is the hypothetical doc vector better or worse than the raw query?
 */

import { resolveConfig } from "../src/config.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { cosineSimilarity, mmrRerank } from "../src/auto/recall.js";
import { generateHypotheticalDocument } from "../src/hyde/generator.js";
import type { SearchResult } from "../src/storage/backend.js";
import fs from "node:fs";

const cfg = resolveConfig();
cfg.lancedbDir = `${process.env.HOME}/.openclaw/data/testDbBEIR/lancedb`;
const embed = await createEmbedProvider(cfg.embed);
const backend = new LanceDBBackend(cfg);
await backend.open();

// Load SciFact qrels from TSV
const qrelsPath = `${process.env.HOME}/codeWS/TypeScript/memory-spark/evaluation/beir-datasets/scifact/qrels/test.tsv`;
const qrels: Record<string, Record<string, number>> = {};
for (const line of fs.readFileSync(qrelsPath, "utf-8").split("\n").slice(1)) {
  const [qid, docId, score] = line.split("\t");
  if (!qid || !docId) continue;
  if (!qrels[qid]) qrels[qid] = {};
  qrels[qid]![docId!] = parseInt(score!, 10);
}

// Load telemetry to get query texts
const telPath = fs.readdirSync("evaluation/results/")
  .filter(f => f.includes("telemetry"))
  .sort()
  .pop();
const telData = JSON.parse(fs.readFileSync(`evaluation/results/${telPath}`, "utf-8"));

// Build query map from config A entries
const queryMap = new Map<string, string>();
for (const entry of telData) {
  if (entry.config === "A" && entry.queryText) {
    queryMap.set(entry.queryId, entry.queryText);
  }
}

const queryIds = Object.keys(qrels).filter(qid => queryMap.has(qid)).slice(0, 20);

console.log("=== MMR DIAGNOSTIC (20 queries) ===\n");

let mmr09_helped = 0, mmr09_hurt = 0, mmr09_neutral = 0;
let adaptive_helped = 0, adaptive_hurt = 0, adaptive_neutral = 0;
const pairwiseSims: number[] = [];

for (const qid of queryIds) {
  const queryText = queryMap.get(qid)!;
  const queryVec = await embed.embedQuery(queryText);
  
  // Vector search with pathContains filter for SciFact
  const results = await backend.vectorSearch(queryVec, {
    query: queryText,
    maxResults: 40,
    minScore: 0.05,
    pathContains: "scifact",
  });
  
  if (results.length < 10) {
    console.log(`Q${qid}: Only ${results.length} results — skipping`);
    continue;
  }
  
  // Get relevant doc IDs
  const relevantIds = new Set(
    Object.keys(qrels[qid]!).filter(id => qrels[qid]![id]! > 0).map(id => `beir-scifact-${id}`)
  );
  
  // Top-10 WITHOUT MMR (pure vector ranking)
  const noMmrTop10 = results.slice(0, 10);
  const noMmrRelevant = noMmrTop10.filter(r => relevantIds.has(r.chunk.id)).length;
  
  // MMR with λ=0.9 (from pool of 20)
  const mmr09Top10 = mmrRerank(results.slice(0, 20), 10, 0.9);
  const mmr09Relevant = mmr09Top10.filter(r => relevantIds.has(r.chunk.id)).length;
  
  // Adaptive MMR (from pool of 20)
  const adaptiveTop10 = mmrRerank(results.slice(0, 20), 10, "adaptive");
  const adaptiveRelevant = adaptiveTop10.filter(r => relevantIds.has(r.chunk.id)).length;
  
  // Pairwise cosine similarity of top-5 results
  const sims: number[] = [];
  for (let i = 0; i < Math.min(5, results.length); i++) {
    for (let j = i + 1; j < Math.min(5, results.length); j++) {
      if (results[i]!.vector && results[j]!.vector) {
        const s = cosineSimilarity(results[i]!.vector!, results[j]!.vector!);
        sims.push(s);
        pairwiseSims.push(s);
      }
    }
  }
  const avgSim = sims.length > 0 ? sims.reduce((a, b) => a + b, 0) / sims.length : 0;
  
  const delta09 = mmr09Relevant - noMmrRelevant;
  const deltaAdaptive = adaptiveRelevant - noMmrRelevant;
  
  console.log(`Q${qid}: "${queryText.slice(0, 55)}…" | relevant=${relevantIds.size}`);
  console.log(`  Scores: top1=${results[0]!.score.toFixed(3)}, top10=${results[9]!.score.toFixed(3)}, spread=${(results[0]!.score - results[9]!.score).toFixed(3)}`);
  console.log(`  NoMMR: ${noMmrRelevant}/${relevantIds.size} | MMR(0.9): ${mmr09Relevant} (${delta09 >= 0 ? "+" : ""}${delta09}) | Adaptive: ${adaptiveRelevant} (${deltaAdaptive >= 0 ? "+" : ""}${deltaAdaptive})`);
  console.log(`  Avg pairwise sim (top-5): ${avgSim.toFixed(4)}`);
  
  if (delta09 > 0) mmr09_helped++;
  else if (delta09 < 0) mmr09_hurt++;
  else mmr09_neutral++;
  
  if (deltaAdaptive > 0) adaptive_helped++;
  else if (deltaAdaptive < 0) adaptive_hurt++;
  else adaptive_neutral++;
}

const avgPairwise = pairwiseSims.length > 0 ? pairwiseSims.reduce((a, b) => a + b, 0) / pairwiseSims.length : 0;
console.log(`\n--- MMR(0.9) Impact: helped=${mmr09_helped}, hurt=${mmr09_hurt}, neutral=${mmr09_neutral}`);
console.log(`--- Adaptive Impact: helped=${adaptive_helped}, hurt=${adaptive_hurt}, neutral=${adaptive_neutral}`);
console.log(`--- Global avg pairwise cosine sim (top-5): ${avgPairwise.toFixed(4)}`);

// ═══════════════════════════════════════════════════════════════════
console.log("\n\n=== HyDE DIAGNOSTIC (5 queries) ===\n");

const hydeConfig = cfg.hyde ? { ...cfg.hyde, timeoutMs: 30000 } : undefined;
if (!hydeConfig) {
  console.log("HyDE config not found — skipping");
} else {
  let hydeHelped = 0, hydeHurt = 0, hydeNeutral = 0;
  
  for (const qid of queryIds.slice(0, 5)) {
    const queryText = queryMap.get(qid)!;
    
    // Raw query vector
    const rawVec = await embed.embedQuery(queryText);
    
    // Generate hypothetical document
    let hypoDoc: string | null = null;
    try {
      hypoDoc = await generateHypotheticalDocument(queryText, hydeConfig);
    } catch (e) {
      console.log(`Q${qid}: HyDE generation error: ${e}`);
      continue;
    }
    if (!hypoDoc) {
      console.log(`Q${qid}: HyDE returned null`);
      continue;
    }
    
    // HyDE vector (document embedding — no instruction prefix)
    const hydeVec = await embed.embedDocument(hypoDoc);
    
    // Search with both vectors
    const rawResults = await backend.vectorSearch(rawVec, {
      query: queryText, maxResults: 10, minScore: 0.05, pathContains: "scifact",
    });
    const hydeResults = await backend.vectorSearch(hydeVec, {
      query: queryText, maxResults: 10, minScore: 0.05, pathContains: "scifact",
    });
    
    const relevantIds = new Set(
      Object.keys(qrels[qid]!).filter(id => qrels[qid]![id]! > 0).map(id => `beir-scifact-${id}`)
    );
    
    const rawRelevant = rawResults.filter(r => relevantIds.has(r.chunk.id)).length;
    const hydeRelevant = hydeResults.filter(r => relevantIds.has(r.chunk.id)).length;
    
    // Cosine sim between raw and hyde vectors
    const queryHydeSim = cosineSimilarity(rawVec, hydeVec);
    
    // Check if HyDE retrieved any NEW relevant docs
    const rawIds = new Set(rawResults.map(r => r.chunk.id));
    const hydeExclusive = hydeResults.filter(r => !rawIds.has(r.chunk.id));
    const hydeExclusiveRelevant = hydeExclusive.filter(r => relevantIds.has(r.chunk.id)).length;
    
    const delta = hydeRelevant - rawRelevant;
    
    console.log(`Q${qid}: "${queryText.slice(0, 55)}…"`);
    console.log(`  HyDE: "${hypoDoc.slice(0, 80)}…"`);
    console.log(`  Query↔HyDE cosine: ${queryHydeSim.toFixed(4)}`);
    console.log(`  Raw top-1: ${rawResults[0]?.score.toFixed(3) ?? "N/A"} | HyDE top-1: ${hydeResults[0]?.score.toFixed(3) ?? "N/A"}`);
    console.log(`  Raw rel@10: ${rawRelevant} | HyDE rel@10: ${hydeRelevant} (${delta >= 0 ? "+" : ""}${delta})`);
    console.log(`  HyDE-exclusive results: ${hydeExclusive.length} (${hydeExclusiveRelevant} relevant)`);
    
    if (delta > 0) hydeHelped++;
    else if (delta < 0) hydeHurt++;
    else hydeNeutral++;
    console.log();
  }
  
  console.log(`--- HyDE Impact: helped=${hydeHelped}, hurt=${hydeHurt}, neutral=${hydeNeutral}`);
}

await backend.close();
console.log("\nDone.");
