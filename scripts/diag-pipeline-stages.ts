/**
 * Pipeline Stage Diagnostic — traces exactly where and how each stage
 * changes the result ranking vs the vector-only baseline.
 * 
 * For each query:
 *   1. Vector-only top-10 → baseline
 *   2. FTS top-10 → overlap analysis
 *   3. Hybrid (RRF) top-10 → what changed? what got displaced?
 *   4. Reranker on hybrid → did it fix or hurt?
 *   5. MMR on hybrid → did diversity help?
 * 
 * Output: per-stage damage report with concrete examples.
 */

import { LanceDBBackend } from "../src/storage/lancedb.js";
import { resolveConfig } from "../src/config.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { EmbedQueue } from "../src/embed/queue.js";
import { hybridMerge, mmrRerank } from "../src/auto/recall.js";
import { createReranker } from "../src/rerank/reranker.js";
import type { SearchResult } from "../src/storage/backend.js";

// BEIR data loading
import { readFileSync } from "fs";
import { join } from "path";

const BEIR_DATA = join(process.env.HOME!, "codeWS/TypeScript/memory-spark/evaluation/beir-datasets");

function loadQueries(dataset: string): Array<{ _id: string; text: string }> {
  const lines = readFileSync(join(BEIR_DATA, dataset, "queries.jsonl"), "utf-8").split("\n").filter(Boolean);
  return lines.map((l) => JSON.parse(l));
}

function loadQrels(dataset: string): Record<string, Record<string, number>> {
  const lines = readFileSync(join(BEIR_DATA, dataset, "qrels", "test.tsv"), "utf-8").split("\n").filter(Boolean);
  const qrels: Record<string, Record<string, number>> = {};
  for (const line of lines.slice(1)) {
    const parts = line.replace(/\r/g, "").split("\t");
    const qid = parts[0], docid = parts[1], score = parts[2];
    if (!qid || !docid || !score) continue;
    qrels[qid] ??= {};
    qrels[qid]![docid] = parseInt(score, 10);
  }
  return qrels;
}

function stripBeirPrefix(id: string): string {
  return id.replace(/^beir-[^-]+-/, "");
}

function isRelevant(docId: string, qrels: Record<string, number>): boolean {
  const stripped = stripBeirPrefix(docId);
  return (qrels[stripped] ?? 0) > 0;
}

function ndcg10(rankedIds: string[], qrels: Record<string, number>): number {
  const k = 10;
  let dcg = 0;
  for (let i = 0; i < Math.min(rankedIds.length, k); i++) {
    const stripped = stripBeirPrefix(rankedIds[i]!);
    const rel = qrels[stripped] ?? 0;
    dcg += rel / Math.log2(i + 2);
  }
  // IDCG: sort all relevant docs by their relevance score
  const idealRels = Object.values(qrels).filter((r) => r > 0).sort((a, b) => b - a);
  let idcg = 0;
  for (let i = 0; i < Math.min(idealRels.length, k); i++) {
    idcg += idealRels[i]! / Math.log2(i + 2);
  }
  return idcg > 0 ? dcg / idcg : 0;
}

async function main() {
  const dataset = "scifact";
  const lancedbDir = `${process.env.HOME}/.openclaw/data/testDbBEIR/lancedb`;

  const cfg = resolveConfig({ lancedbDir } as any);
  const backend = new LanceDBBackend(cfg);
  await backend.open();

  const provider = await createEmbedProvider(cfg.embed);
  console.log(`Embed: ${provider.id}/${provider.model} (${provider.dims}d)`);
  const embed = new EmbedQueue(provider, { concurrency: 1, maxRetries: 2, timeoutMs: 30000 });
  
  const reranker = cfg.rerank.enabled ? await createReranker(cfg.rerank) : null;
  console.log(`Reranker: ${reranker ? "active" : "none"}`);

  const queries = loadQueries(dataset);
  const qrels = loadQrels(dataset);
  const judgedQueries = queries.filter((q) => qrels[String(q._id)]);
  
  // Sample 50 queries for detailed diagnostics
  const sample = judgedQueries.slice(0, 50);
  console.log(`\nAnalyzing ${sample.length} queries...\n`);

  // Accumulators
  const stats = {
    vectorOnlyNDCG: [] as number[],
    hybridNDCG: [] as number[],
    hybridRerankerNDCG: [] as number[],
    hybridMmrNDCG: [] as number[],
    
    // Overlap analysis
    vectorFtsOverlap: [] as number[], // how many of top-10 vector are also in top-40 FTS
    hybridDisplacedRelevant: 0,       // relevant docs in vector top-10 NOT in hybrid top-10
    hybridPromotedIrrelevant: 0,      // irrelevant docs in hybrid top-10 NOT in vector top-10
    
    rerankerImproved: 0,
    rerankerHurt: 0,
    rerankerNeutral: 0,
    
    mmrImproved: 0,
    mmrHurt: 0,
    mmrNeutral: 0,
    
    queriesWithDamage: [] as string[],
  };

  const k = 10;
  for (let i = 0; i < sample.length; i++) {
    const q = sample[i]!;
    const qr = qrels[String(q._id)]!;
    
    process.stdout.write(`  [${i + 1}/${sample.length}] `);

    // 1. Embed query
    const queryVector = await embed.embedQuery(q.text);

    // 2. Vector search
    const vectorResults = await backend
      .vectorSearch(queryVector, { query: q.text, maxResults: k * 4, minScore: 0.0, pathContains: `beir/${dataset}/` })
      .catch(() => []);

    // 3. FTS search  
    const ftsResults = await backend
      .ftsSearch(q.text, { query: q.text, maxResults: k * 4, pathContains: `beir/${dataset}/` })
      .catch(() => []);

    // 4. Hybrid merge
    const hybridResults = hybridMerge(vectorResults, ftsResults, k * 2);

    // 5. Reranker on hybrid
    let rerankerResults: SearchResult[] = [];
    if (reranker) {
      rerankerResults = await reranker.rerank(q.text, hybridResults, k);
    }

    // 6. MMR on hybrid
    const mmrResults = mmrRerank(hybridResults, k, 0.9);

    // --- Analysis ---
    const vecIds = vectorResults.slice(0, k).map((r) => r.chunk.id);
    const ftsIds = ftsResults.slice(0, k * 4).map((r) => r.chunk.id);
    const hybIds = hybridResults.slice(0, k).map((r) => r.chunk.id);
    const rerIds = rerankerResults.slice(0, k).map((r) => r.chunk.id);
    const mmrIds = mmrResults.slice(0, k).map((r) => r.chunk.id);

    // NDCG@10 for each
    const vecNDCG = ndcg10(vecIds, qr);
    const hybNDCG = ndcg10(hybIds, qr);
    const rerNDCG = reranker ? ndcg10(rerIds, qr) : 0;
    const mmrNDCG = ndcg10(mmrIds, qr);

    stats.vectorOnlyNDCG.push(vecNDCG);
    stats.hybridNDCG.push(hybNDCG);
    stats.hybridRerankerNDCG.push(rerNDCG);
    stats.hybridMmrNDCG.push(mmrNDCG);

    // Overlap: how many vector top-10 appear in FTS top-40?
    const vecInFts = vecIds.filter((id) => ftsIds.includes(id)).length;
    stats.vectorFtsOverlap.push(vecInFts);

    // Damage: relevant vector docs displaced by hybrid
    const vecRelevantInTop10 = vecIds.filter((id) => isRelevant(id, qr));
    const hybRelevantInTop10 = hybIds.filter((id) => isRelevant(id, qr));
    const displaced = vecRelevantInTop10.filter((id) => !hybIds.includes(id));
    const promoted = hybIds.filter((id) => !vecIds.includes(id) && !isRelevant(id, qr));
    stats.hybridDisplacedRelevant += displaced.length;
    stats.hybridPromotedIrrelevant += promoted.length;

    // Reranker delta
    if (reranker) {
      if (rerNDCG > hybNDCG + 0.001) stats.rerankerImproved++;
      else if (rerNDCG < hybNDCG - 0.001) stats.rerankerHurt++;
      else stats.rerankerNeutral++;
    }

    // MMR delta
    if (mmrNDCG > hybNDCG + 0.001) stats.mmrImproved++;
    else if (mmrNDCG < hybNDCG - 0.001) stats.mmrHurt++;
    else stats.mmrNeutral++;

    // Track worst damage cases
    if (hybNDCG < vecNDCG - 0.1) {
      stats.queriesWithDamage.push(
        `Q${q._id}: vec=${vecNDCG.toFixed(3)} → hyb=${hybNDCG.toFixed(3)} (Δ=${(hybNDCG - vecNDCG).toFixed(3)}) | ` +
        `overlap=${vecInFts}/10 | displaced=${displaced.length} relevant, promoted=${promoted.length} irrelevant`
      );
    }
  }

  // --- Report ---
  const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;

  console.log("\n\n═══════════════════════════════════════════");
  console.log("  PIPELINE STAGE DAMAGE REPORT");
  console.log("═══════════════════════════════════════════\n");

  console.log("NDCG@10 Averages (50 queries):");
  console.log(`  Vector-Only:       ${avg(stats.vectorOnlyNDCG).toFixed(4)}`);
  console.log(`  Hybrid (RRF):      ${avg(stats.hybridNDCG).toFixed(4)} (Δ ${(avg(stats.hybridNDCG) - avg(stats.vectorOnlyNDCG)).toFixed(4)})`);
  console.log(`  Hybrid + Reranker: ${avg(stats.hybridRerankerNDCG).toFixed(4)} (Δ ${(avg(stats.hybridRerankerNDCG) - avg(stats.vectorOnlyNDCG)).toFixed(4)})`);
  console.log(`  Hybrid + MMR:      ${avg(stats.hybridMmrNDCG).toFixed(4)} (Δ ${(avg(stats.hybridMmrNDCG) - avg(stats.vectorOnlyNDCG)).toFixed(4)})`);

  console.log(`\nVector ↔ FTS Overlap (in top-40 FTS):`);
  console.log(`  Mean overlap: ${avg(stats.vectorFtsOverlap).toFixed(1)}/10 vector docs found in FTS`);
  console.log(`  Min: ${Math.min(...stats.vectorFtsOverlap)}, Max: ${Math.max(...stats.vectorFtsOverlap)}`);

  console.log(`\nHybrid Damage vs Vector-Only:`);
  console.log(`  Relevant docs displaced: ${stats.hybridDisplacedRelevant} total across ${sample.length} queries`);
  console.log(`  Irrelevant docs promoted: ${stats.hybridPromotedIrrelevant} total`);

  console.log(`\nReranker Effect (on hybrid pool):`);
  console.log(`  Improved: ${stats.rerankerImproved}/${sample.length}`);
  console.log(`  Hurt:     ${stats.rerankerHurt}/${sample.length}`);
  console.log(`  Neutral:  ${stats.rerankerNeutral}/${sample.length}`);

  console.log(`\nMMR Effect (on hybrid pool):`);
  console.log(`  Improved: ${stats.mmrImproved}/${sample.length}`);
  console.log(`  Hurt:     ${stats.mmrHurt}/${sample.length}`);
  console.log(`  Neutral:  ${stats.mmrNeutral}/${sample.length}`);

  if (stats.queriesWithDamage.length > 0) {
    console.log(`\nWorst Damage Cases (hybrid NDCG > 0.1 below vector):`);
    for (const d of stats.queriesWithDamage.slice(0, 15)) {
      console.log(`  ${d}`);
    }
  }
}

main().catch(console.error);
