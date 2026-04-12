#!/usr/bin/env npx tsx
/**
 * BEIR Benchmark Runner — Test A/B/C/D/E/F/G configurations against BEIR.
 *
 * Configurations:
 *   A: Vector-Only        — Pure semantic search baseline
 *   B: FTS-Only           — Pure keyword search baseline
 *   C: Hybrid             — Vector + FTS combined (no reranker, no MMR)
 *   D: Hybrid + Reranker  — With Nemotron cross-encoder reranking
 *   E: Hybrid + MMR       — With Maximal Marginal Relevance diversity
 *   F: Hybrid + HyDE      — With Hypothetical Document Embeddings
 *   G: Full Pipeline      — All features: Vector + FTS + Reranker + MMR
 *
 * Usage:
 *   npx tsx evaluation/run-beir-bench.ts                    # All configs
 *   npx tsx evaluation/run-beir-bench.ts --dataset scifact  # Specific dataset
 *   npx tsx evaluation/run-beir-bench.ts --config A         # Specific config
 *
 * Output:
 *   - JSON results in evaluation/results/
 *   - Telemetry logs for each query showing stage contributions
 *   - Summary table with NDCG@10, MAP@10, Recall@10
 */

import fs from "node:fs/promises";
import path from "node:path";

import {
  hybridMerge,
  mmrRerank,
  prepareRerankerFusion,
} from "../src/auto/recall.js";
import { resolveConfig, type HydeConfig } from "../src/config.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { EmbedQueue } from "../src/embed/queue.js";
import { generateHypotheticalDocument } from "../src/hyde/generator.js";
import { createReranker } from "../src/rerank/reranker.js";
import type { SearchResult } from "../src/storage/backend.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import { expandQuery, QUERY_EXPANSION_DEFAULTS, type QueryExpansionConfig } from "../src/query/expander.js";

import { evaluateBEIR, type Qrels, type Results } from "../evaluation/metrics.js";

// ── Types ───────────────────────────────────────────────────────────────────

interface BeirQuery {
  _id: string;
  text: string;
  metadata?: Record<string, unknown>;
}

interface RetrievalConfig {
  id: string;
  label: string;
  useVector: boolean;
  useFts: boolean;
  useReranker: boolean;
  useMmr: boolean;
  useHyde: boolean;
  mmrLambda: number | "adaptive";
  maxResults: number;
  /** Phase 8: Use overlap-aware adaptive RRF weights instead of static */
  adaptiveRrf?: boolean;
  /** Phase 8: Use reranker-as-fusioner (pass union to reranker, skip RRF) */
  rerankerFusion?: boolean;
  /** Phase 9A: Score interpolation alpha (0 = pure reranker, 0.3 = recommended blend) */
  scoreBlendAlpha?: number;
  /** Phase 9B: Conditional routing — skip reranker when vector is confident */
  conditionalRerank?: boolean;
  /** Phase 9B: Spread threshold for confident vector results (skip reranker if spread > this) */
  confidenceThreshold?: number;
  /** Phase 11B: Use multi-query expansion (generate LLM reformulations) */
  useMultiQuery?: boolean;
  /** Phase 11B: Number of reformulations to generate (default: 3) */
  multiQueryN?: number;
  /** Phase 12: Blend mode for reranker fusion ("score" = legacy alpha, "rrf" = rank fusion) */
  blendMode?: "score" | "rrf";
  /** Phase 12: RRF k constant (default 60). Lower = top ranks dominate more */
  rrfK?: number;
  /** Phase 12: RRF vector weight (default 1.0) */
  rrfVectorWeight?: number;
  /** Phase 12: RRF reranker weight (default 1.0) */
  rrfRerankerWeight?: number;
  /** Phase 12: Reranker gate mode ("off" | "hard" | "soft") */
  rerankerGate?: "off" | "hard" | "soft";
  /** Phase 12: Gate threshold (skip if spread > this) */
  rerankerGateThreshold?: number;
  /** Phase 12: Gate low threshold (skip if spread < this) */
  rerankerGateLowThreshold?: number;
}

interface QueryTelemetry {
  queryId: string;
  queryText: string;
  config: string;
  stages: {
    vector?: { count: number; top1Id: string; top1Score: number };
    fts?: { count: number; top1Id: string; top1Score: number };
    hybrid?: { count: number; top1Id: string; top1Score: number };
    reranker?: { top1Id: string; top1Score: number; reorderCount: number };
    mmr?: { removedNearDuplicates: number };
    hyde?: { hypotheticalDoc: string };
    multiQuery?: {
      numQueries: number;
      reformulations: string[];
      perQueryHits: number[];
      unionSize: number;
      expandMs: number;
      totalMs: number;
    };
  };
  finalResults: Array<{ id: string; score: number; text: string }>;
  latencyMs: number;
}

interface ConfigResult {
  config: string;
  label: string;
  ndcg: number;
  mrr: number;
  recall: number;
  map: number;
  latencyP50: number;
  latencyP95: number;
  latencyMean: number;
}

// ── A/B/C/D/E/F/G Configurations ─────────────────────────────────────────────

const CONFIGS: RetrievalConfig[] = [
  {
    id: "A",
    label: "Vector-Only",
    useVector: true,
    useFts: false,
    useReranker: false,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
  },
  {
    id: "B",
    label: "FTS-Only",
    useVector: false,
    useFts: true,
    useReranker: false,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
  },
  {
    id: "C",
    label: "Hybrid",
    useVector: true,
    useFts: true,
    useReranker: false,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
  },
  {
    id: "D",
    label: "Hybrid + Reranker",
    useVector: true,
    useFts: true,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
  },
  {
    id: "E",
    label: "Hybrid + MMR",
    useVector: true,
    useFts: true,
    useReranker: false,
    useMmr: true,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
  },
  {
    id: "F",
    label: "Hybrid + HyDE",
    useVector: true,
    useFts: true,
    useReranker: false,
    useMmr: false,
    useHyde: true,
    mmrLambda: 0.9,
    maxResults: 10,
  },
  {
    id: "G",
    label: "Full Pipeline",
    useVector: true,
    useFts: true,
    useReranker: true,
    useMmr: true,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
  },
  // ── Phase 8: Adaptive Pipeline Configs ──────────────────────────────────
  {
    id: "H",
    label: "Vector → Reranker (no RRF)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
  },
  {
    id: "I",
    label: "Adaptive Hybrid (overlap-aware RRF)",
    useVector: true,
    useFts: true,
    useReranker: false,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    adaptiveRrf: true,
  },
  {
    id: "J",
    label: "Reranker-as-Fusioner (union → rerank)",
    useVector: true,
    useFts: true,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    rerankerFusion: true,
  },
  {
    id: "K",
    label: "Vector → Adaptive MMR",
    useVector: true,
    useFts: false,
    useReranker: false,
    useMmr: true,
    useHyde: false,
    mmrLambda: "adaptive",
    maxResults: 10,
  },
  {
    id: "L",
    label: "Full Adaptive (overlap RRF → Reranker → Adaptive MMR)",
    useVector: true,
    useFts: true,
    useReranker: true,
    useMmr: true,
    useHyde: false,
    mmrLambda: "adaptive",
    maxResults: 10,
    adaptiveRrf: true,
  },
  // ── Phase 9A: Score Interpolation configs ──────────────────────────
  {
    id: "M",
    label: "Vector → Blended Reranker (α=0.3)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    scoreBlendAlpha: 0.3,
  },
  {
    id: "N",
    label: "Vector → Blended Reranker (α=0.5)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    scoreBlendAlpha: 0.5,
  },
  // ── Phase 9B: Conditional Routing configs ──────────────────────────
  {
    id: "O",
    label: "Conditional Rerank (skip when confident, α=0.3)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    scoreBlendAlpha: 0.3,
    conditionalRerank: true,
    confidenceThreshold: 0.15,
  },
  {
    id: "P",
    label: "Full 9A+9B (Adaptive RRF → Conditional Blended Reranker → Adaptive MMR)",
    useVector: true,
    useFts: true,
    useReranker: true,
    useMmr: true,
    useHyde: false,
    mmrLambda: "adaptive",
    maxResults: 10,
    adaptiveRrf: true,
    scoreBlendAlpha: 0.3,
    conditionalRerank: true,
    confidenceThreshold: 0.15,
  },
  // ── Phase 10A: Logit Recovery configs ─────────────────────────────────
  // These use the same pipeline as Phase 9, but blendScores() now applies
  // inverse-sigmoid (logit recovery) before normalizing reranker scores.
  // The reranker's discrimination signal is recovered from sigmoid compression.
  {
    id: "Q",
    label: "Vector → Logit-Recovered Blend (α=0.5)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    scoreBlendAlpha: 0.5,
  },
  {
    id: "R",
    label: "Vector → Pure Logit Reranker (α=0)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    scoreBlendAlpha: 0,
  },
  {
    id: "S",
    label: "Vector → Logit Blend + Conditional (α=0.3)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    scoreBlendAlpha: 0.3,
    conditionalRerank: true,
    confidenceThreshold: 0.15,
  },
  {
    id: "T",
    label: "Vector → Logit Blend (α=0.3)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    scoreBlendAlpha: 0.3,
  },
  // ── Phase 11A: Alpha Sweep ────────────────────────────────────────────────
  // Find optimal blending weight between vector scores and logit-recovered
  // reranker scores. Q (α=0.5) is current best at 0.7863 NDCG@10.
  {
    id: "U",
    label: "Vector → Logit Blend (α=0.4)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    scoreBlendAlpha: 0.4,
  },
  {
    id: "V",
    label: "Vector → Logit Blend (α=0.6)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    scoreBlendAlpha: 0.6,
  },
  {
    id: "W",
    label: "Vector → Logit Blend (α=0.7)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    scoreBlendAlpha: 0.7,
  },
  {
    id: "X",
    label: "Vector → Logit Blend (α=0.8)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    scoreBlendAlpha: 0.8,
  },
  // ── Phase 11B: Multi-Query Expansion ──────────────────────────────────────
  // Generate LLM reformulations → parallel embed + search → union → rerank.
  // Attacks the 11% retrieval ceiling where relevant docs are invisible to single queries.
  {
    id: "MQ-A",
    label: "Multi-Query (3) → Vector-Only",
    useVector: true,
    useFts: false,
    useReranker: false,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    useMultiQuery: true,
    multiQueryN: 3,
  },
  {
    id: "MQ-B",
    label: "Multi-Query (3) → Logit Blend (α=0.4)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    useMultiQuery: true,
    multiQueryN: 3,
    scoreBlendAlpha: 0.4,
  },
  {
    id: "MQ-C",
    label: "Multi-Query (3) → Logit Blend (α=0.5)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    useMultiQuery: true,
    multiQueryN: 3,
    scoreBlendAlpha: 0.5,
  },
  {
    id: "MQ-D",
    label: "Multi-Query (3) → Conditional Logit Blend (α=0.4)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    useMultiQuery: true,
    multiQueryN: 3,
    scoreBlendAlpha: 0.4,
    conditionalRerank: true,
    confidenceThreshold: 0.15,
  },

  // ── Phase 12: RRF Rank Fusion Configs ──────────────────────────────────
  {
    id: "RRF-A",
    label: "Vector ⊕ Reranker via RRF (k=60, equal weight)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    blendMode: "rrf",
    rrfK: 60,
    rrfVectorWeight: 1.0,
    rrfRerankerWeight: 1.0,
  },
  {
    id: "RRF-B",
    label: "Vector ⊕ Reranker via RRF (k=60, vector=1.5)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    blendMode: "rrf",
    rrfK: 60,
    rrfVectorWeight: 1.5,
    rrfRerankerWeight: 1.0,
  },
  {
    id: "RRF-C",
    label: "Vector ⊕ Reranker via RRF (k=60, reranker=1.5)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    blendMode: "rrf",
    rrfK: 60,
    rrfVectorWeight: 1.0,
    rrfRerankerWeight: 1.5,
  },
  {
    id: "RRF-D",
    label: "Vector ⊕ Reranker via RRF (k=20, equal weight — sharper top focus)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    blendMode: "rrf",
    rrfK: 20,
    rrfVectorWeight: 1.0,
    rrfRerankerWeight: 1.0,
  },

  // ── Phase 12: Reranker Gate Configs ────────────────────────────────────
  {
    id: "GATE-A",
    label: "RRF + Hard Gate (skip if spread > 0.08 or < 0.02)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    blendMode: "rrf",
    rrfK: 60,
    rrfVectorWeight: 1.0,
    rrfRerankerWeight: 1.0,
    rerankerGate: "hard",
    rerankerGateThreshold: 0.08,
    rerankerGateLowThreshold: 0.02,
  },
  {
    id: "GATE-B",
    label: "RRF + Soft Gate (dynamic vector weight from spread)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    blendMode: "rrf",
    rrfK: 60,
    rrfVectorWeight: 1.0,
    rrfRerankerWeight: 1.0,
    rerankerGate: "soft",
    rerankerGateThreshold: 0.08,
    rerankerGateLowThreshold: 0.02,
  },
  {
    id: "GATE-C",
    label: "RRF + Soft Gate (threshold=0.12, wider confidence window)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    blendMode: "rrf",
    rrfK: 60,
    rrfVectorWeight: 1.0,
    rrfRerankerWeight: 1.0,
    rerankerGate: "soft",
    rerankerGateThreshold: 0.12,
    rerankerGateLowThreshold: 0.02,
  },
  {
    id: "GATE-D",
    label: "RRF (vector=1.5) + Soft Gate (best of Fix 1 + Fix 2)",
    useVector: true,
    useFts: false,
    useReranker: true,
    useMmr: false,
    useHyde: false,
    mmrLambda: 0.9,
    maxResults: 10,
    blendMode: "rrf",
    rrfK: 60,
    rrfVectorWeight: 1.5,
    rrfRerankerWeight: 1.0,
    rerankerGate: "soft",
    rerankerGateThreshold: 0.08,
    rerankerGateLowThreshold: 0.02,
  },
];

// ── Functions ───────────────────────────────────────────────────────────────

async function loadQueries(dataset: string): Promise<BeirQuery[]> {
  const file = path.join(import.meta.dirname!, "../evaluation/beir-datasets", dataset, "queries.jsonl");
  const content = await fs.readFile(file, "utf-8");
  return content
    .trim()
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as BeirQuery);
}

async function loadQrels(dataset: string): Promise<Qrels> {
  const file = path.join(import.meta.dirname!, "../evaluation/beir-datasets", dataset, "qrels", "test.tsv");
  const content = await fs.readFile(file, "utf-8");
  const lines = content.trim().split("\n");
  const qrels: Qrels = {};

  for (const line of lines) {
    if (!line.trim() || line.startsWith("query-id")) continue;
    const parts = line.split("\t");
    if (parts.length < 3) continue;
    const [queryId, corpusId, score] = parts;
    if (!qrels[queryId!]) qrels[queryId!] = {};
    qrels[queryId!]![corpusId!] = parseInt(score!, 10);
  }
  return qrels;
}

function stripBeirPrefix(id: string): string {
  return id.replace(/^beir-[^-]+-/, "");
}

async function runRetrieval(
  queries: BeirQuery[],
  qrels: Qrels,
  backend: LanceDBBackend,
  embed: EmbedQueue,
  reranker: Awaited<ReturnType<typeof createReranker>> | null,
  hydeConfig: HydeConfig | undefined,
  expansionConfig: QueryExpansionConfig | undefined,
  config: RetrievalConfig,
  dataset: string,
): Promise<{ results: Results; telemetry: QueryTelemetry[] }> {
  const results: Results = {};
  const telemetry: QueryTelemetry[] = [];
  const k = config.maxResults;
  const evalQueries = queries.filter((q) => qrels[q._id] && Object.keys(qrels[q._id]!).length > 0);

  for (let i = 0; i < evalQueries.length; i++) {
    const q = evalQueries[i]!;
    const startTime = Date.now();
    const tel: QueryTelemetry = {
      queryId: q._id,
      queryText: q.text,
      config: config.id,
      stages: {},
      finalResults: [],
      latencyMs: 0,
    };

    const verbose = process.env.VERBOSE || process.env.DEBUG_PIPELINE;
    if ((i + 1) % 25 === 0 || i === 0) {
      process.stdout.write(`\r    [${i + 1}/${evalQueries.length}]`);
    }
    if (verbose) {
      console.log(`\n[bench] ── query ${i + 1}/${evalQueries.length}: id=${q._id} config=${config.id}`);
      console.log(`[bench]   text: "${q.text.slice(0, 100)}"`);
    }

    // Get query vector
    let queryVector: number[];
    try {
      const t0embed = Date.now();
      queryVector = await embed.embedQuery(q.text);
      if (verbose) {
        console.log(`[bench]   embed: dims=${queryVector.length} latency=${Date.now() - t0embed}ms`);
      }
    } catch (e) {
      if (i < 3) console.error(`\n[WARN] embedQuery error on query ${q._id}: ${e}`);
      tel.latencyMs = Date.now() - startTime;
      telemetry.push(tel);
      results[q._id] = {};
      continue;
    }

    // HyDE: generate hypothetical document and REPLACE query vector (Gao et al. 2022).
    // The hypothetical is embedded as a document (no instruction prefix) to project
    // it into the same vector space as the indexed corpus.
    if (config.useHyde && hydeConfig?.enabled) {
      if (verbose) console.log(`[bench]   HyDE: generating hypothetical document…`);
      try {
        const t0hyde = Date.now();
        const hypothetical = await generateHypotheticalDocument(q.text, hydeConfig);
        if (hypothetical) {
          tel.stages.hyde = { hypotheticalDoc: hypothetical.slice(0, 200) + "..." };
          if (verbose) {
            console.log(`[bench]   HyDE: success in ${Date.now() - t0hyde}ms, doc=${hypothetical.length} chars`);
            console.log(`[bench]   HyDE preview: "${hypothetical.slice(0, 100).replace(/\n/g, " ")}…"`);
          }
          // embedDocument() = no instruction prefix → document space
          queryVector = await embed.embedDocument(hypothetical);
        } else {
          if (verbose) console.log(`[bench]   HyDE: generation returned null — using raw query vector`);
        }
      } catch (hydeErr) {
        if (verbose) console.log(`[bench]   HyDE: error (${hydeErr instanceof Error ? hydeErr.message : String(hydeErr)}) — using raw query vector`);
        // Fall back to raw query (queryVector already set above)
      }
    }

    // Keep vector and FTS results separate until explicit fusion
    let vectorResults: SearchResult[] = [];
    let ftsResults: SearchResult[] = [];

    // Vector search (with optional multi-query expansion)
    if (config.useVector) {
      const t0vec = Date.now();

      if (config.useMultiQuery && expansionConfig) {
        // ── Phase 11B: Multi-Query Expansion ──────────────────────────────
        // Generate reformulations → embed each → search each → union results.
        // Parallel embeds + searches, but SEQUENTIAL LLM call (avoid OOM on Spark).
        const t0expand = Date.now();
        const queries = await expandQuery(q.text, expansionConfig);
        const expandMs = Date.now() - t0expand;

        if (verbose) {
          console.log(`[bench]   multi-query: ${queries.length} queries (${expandMs}ms for expansion)`);
          for (const mq of queries) console.log(`[bench]     → "${mq.slice(0, 100)}"`);
        }

        // Embed all queries in parallel (embeds are cheap, ~200ms each)
        const t0emb = Date.now();
        const queryVectors = await Promise.all(
          queries.map((mq) => embed.embedQuery(mq).catch(() => null)),
        );
        const validVectors = queryVectors.filter((v): v is number[] => v !== null);
        if (verbose) {
          console.log(`[bench]   multi-query embed: ${validVectors.length}/${queries.length} vectors in ${Date.now() - t0emb}ms`);
        }

        // Search each vector in parallel (searches are independent)
        const searchOpts = { query: q.text, maxResults: k * 4, minScore: 0.0, pathContains: `beir/${dataset}/` };
        const allResults = await Promise.all(
          validVectors.map((vec) =>
            backend.vectorSearch(vec, searchOpts).catch(() => [] as SearchResult[]),
          ),
        );

        // Union: dedupe by chunk ID, keep highest score
        const best = new Map<string, SearchResult>();
        for (const resultSet of allResults) {
          for (const r of resultSet) {
            const existing = best.get(r.chunk.id);
            if (!existing || r.score > existing.score) {
              best.set(r.chunk.id, r);
            }
          }
        }
        vectorResults = [...best.values()].sort((a, b) => b.score - a.score);

        const perQueryHits = allResults.map((r) => r.length);
        tel.stages.multiQuery = {
          numQueries: queries.length,
          reformulations: queries.slice(1).map((r) => r.slice(0, 80)),
          perQueryHits,
          unionSize: vectorResults.length,
          expandMs,
          totalMs: Date.now() - t0vec,
        };

        if (verbose) {
          console.log(
            `[bench]   multi-query union: ${vectorResults.length} unique docs ` +
            `(per-query: [${perQueryHits.join(", ")}]) in ${Date.now() - t0vec}ms total`,
          );
        }
      } else {
        // Standard single-query vector search
        vectorResults = await backend
          .vectorSearch(queryVector, { query: q.text, maxResults: k * 4, minScore: 0.0, pathContains: `beir/${dataset}/` })
          .catch((e) => { if (i === 0) console.error(`\n[WARN] vectorSearch error on query ${q._id}: ${e}`); return []; });
      }

      if (vectorResults.length > 0) {
        tel.stages.vector = {
          count: vectorResults.length,
          top1Id: vectorResults[0]!.chunk.id,
          top1Score: vectorResults[0]!.score,
        };
      }
      if (verbose) {
        const top3 = vectorResults.slice(0, 3).map((r) => `${r.chunk.id.slice(-8)}:${r.score.toFixed(3)}`).join(", ");
        console.log(`[bench]   retrieve→vector: ${vectorResults.length} results in ${Date.now() - t0vec}ms | top3: [${top3}]`);
      }
    }

    // FTS search
    if (config.useFts) {
      const t0fts = Date.now();
      ftsResults = await backend
        .ftsSearch(q.text, { query: q.text, maxResults: k * 4, pathContains: `beir/${dataset}/` })
        .catch(() => []);
      if (ftsResults.length > 0) {
        tel.stages.fts = {
          count: ftsResults.length,
          top1Id: ftsResults[0]!.chunk.id,
          top1Score: ftsResults[0]!.score,
        };
      }
      if (verbose) {
        const top3 = ftsResults.slice(0, 3).map((r) => `${r.chunk.id.slice(-8)}:${r.score.toFixed(3)}`).join(", ");
        console.log(`[bench]   retrieve→fts: ${ftsResults.length} results in ${Date.now() - t0fts}ms | top3: [${top3}]`);
      }
    }

    // ── Fusion stage: Hybrid merge or Reranker-as-Fusioner ────────────────
    let candidates: SearchResult[];

    if (config.rerankerFusion && reranker && vectorResults.length > 0 && ftsResults.length > 0) {
      // Phase 8 Fix 2: Reranker-as-Fusioner — skip RRF entirely,
      // pass raw union to cross-encoder and let it score independently
      const fusionPool = prepareRerankerFusion(vectorResults, ftsResults, k * 4);
      candidates = await reranker.rerank(q.text, fusionPool, k);
      tel.stages.hybrid = {
        count: fusionPool.length,
        top1Id: candidates[0]?.chunk.id ?? "",
        top1Score: candidates[0]?.score ?? 0,
      };
      tel.stages.reranker = {
        top1Id: candidates[0]?.chunk.id ?? "",
        top1Score: candidates[0]?.score ?? 0,
        reorderCount: -1, // N/A for fusion mode
      };
    } else if (config.useVector && config.useFts && vectorResults.length > 0 && ftsResults.length > 0) {
      // Hybrid merge — adaptive or static RRF
      const mode = config.adaptiveRrf ? "adaptive" : "static";
      candidates = hybridMerge(vectorResults, ftsResults, k * 2, 60, 1.0, 1.0, mode);
      if (candidates.length > 0) {
        tel.stages.hybrid = {
          count: candidates.length,
          top1Id: candidates[0]!.chunk.id,
          top1Score: candidates[0]!.score,
        };
      }
      if (verbose) {
        const top3 = candidates.slice(0, 3).map((r) => `${r.chunk.id.slice(-8)}:${r.score.toFixed(3)}`).join(", ");
        console.log(`[bench]   hybrid (mode=${mode}): ${candidates.length} merged | top3: [${top3}]`);
      }
    } else {
      // Single-source — dedupe by ID
      const combined = [...vectorResults, ...ftsResults];
      const seen = new Set<string>();
      candidates = combined.filter((r) => {
        if (seen.has(r.chunk.id)) return false;
        seen.add(r.chunk.id);
        return true;
      });
    }

    // ── Reranker (standard path, skip if rerankerFusion already handled it) ─
    if (config.useReranker && reranker && candidates.length > 0 && !config.rerankerFusion) {
      // Phase 9B: Conditional routing — skip reranker when vector is already confident
      let skipReranker = false;
      if (config.conditionalRerank) {
        const topScores = candidates.slice(0, 5).map((c) => c.score);
        const spread = topScores.length >= 2 ? Math.max(...topScores) - Math.min(...topScores) : 0;
        const threshold = config.confidenceThreshold ?? 0.15;
        if (spread > threshold) {
          skipReranker = true;
          if (process.env.MEMORY_SPARK_DEBUG) {
            console.debug(
              `[conditional-rerank] spread=${spread.toFixed(4)} > threshold=${threshold} — skipping reranker`,
            );
          }
        }
      }

      if (!skipReranker) {
        const beforeOrder = candidates.slice(0, 5).map((c) => c.chunk.id);

        // Phase 10B/12: Unified reranker path — ALL configs go through reranker.rerank().
        // Options let the benchmark control blend weight/mode per-config without
        // duplicating normalization, telemetry, or error handling.
        const rerankOpts: import("../src/rerank/reranker.js").RerankOptions = {};
        if (config.scoreBlendAlpha != null) rerankOpts.alphaOverride = config.scoreBlendAlpha;
        if (config.blendMode) rerankOpts.blendModeOverride = config.blendMode;
        if (config.rrfK != null) rerankOpts.rrfKOverride = config.rrfK;
        if (config.rrfVectorWeight != null) rerankOpts.vectorWeightOverride = config.rrfVectorWeight;
        if (config.rrfRerankerWeight != null) rerankOpts.rerankerWeightOverride = config.rrfRerankerWeight;
        if (config.rerankerGate) rerankOpts.gateOverride = config.rerankerGate;
        if (config.rerankerGateThreshold != null) rerankOpts.gateThresholdOverride = config.rerankerGateThreshold;
        if (config.rerankerGateLowThreshold != null) rerankOpts.gateLowThresholdOverride = config.rerankerGateLowThreshold;
        candidates = await reranker.rerank(
          q.text,
          candidates,
          k,
          Object.keys(rerankOpts).length > 0 ? rerankOpts : undefined,
        );

        const afterOrder = candidates.slice(0, 5).map((c) => c.chunk.id);
        const reorderCount = beforeOrder.filter((id, idx) => id !== afterOrder[idx]).length;
        tel.stages.reranker = {
          top1Id: candidates[0]?.chunk.id ?? "",
          top1Score: candidates[0]?.score ?? 0,
          reorderCount,
        };
        if (verbose) {
          const top3 = candidates.slice(0, 3).map((r) => `${r.chunk.id.slice(-8)}:${r.score.toFixed(3)}`).join(", ");
          console.log(`[bench]   rerank: ${candidates.length} results, reordered=${reorderCount}/5 | top3: [${top3}]`);
        }
      }
    }

    // ── MMR diversity (supports adaptive lambda) ─────────────────────────
    if (config.useMmr && candidates.length > 0) {
      const beforeCount = candidates.length;
      candidates = mmrRerank(candidates, k, config.mmrLambda);
      const removed = beforeCount - candidates.length;
      tel.stages.mmr = { removedNearDuplicates: removed };
      if (verbose) {
        const top3 = candidates.slice(0, 3).map((r) => `${r.chunk.id.slice(-8)}:${r.score.toFixed(3)}`).join(", ");
        console.log(`[bench]   MMR (λ=${config.mmrLambda}): ${candidates.length} results, removed=${removed} near-dupes | top3: [${top3}]`);
      }
    }

    // Record final results
    tel.finalResults = candidates.slice(0, k).map((r) => ({
      id: stripBeirPrefix(r.chunk.id),
      score: r.score,
      text: r.chunk.text.slice(0, 100) + "...",
    }));
    tel.latencyMs = Date.now() - startTime;
    telemetry.push(tel);

    if (verbose) {
      console.log(`[bench]   ── final: ${tel.finalResults.length} results in ${tel.latencyMs}ms`);
      for (let ri = 0; ri < Math.min(tel.finalResults.length, 5); ri++) {
        const fr = tel.finalResults[ri]!;
        const isRelevant = qrels[q._id]?.[fr.id] ? "✓" : "✗";
        console.log(`[bench]     [${ri + 1}] ${isRelevant} id=${fr.id.slice(0, 24)} score=${fr.score.toFixed(4)} | "${fr.text.slice(0, 60).replace(/\n/g, " ")}…"`);
      }
    }

    // Convert results for BEIR metrics - Results is Record<queryId, Record<docId, score>>
    results[q._id] = {};
    for (const r of candidates.slice(0, k)) {
      results[q._id]![stripBeirPrefix(r.chunk.id)] = r.score;
    }
  }

  console.log("");
  return { results, telemetry };
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const datasetArg = args.includes("--dataset") ? args[args.indexOf("--dataset") + 1] : "scifact";
  const configArg = args.includes("--config") ? args[args.indexOf("--config") + 1] : null;
  const cliHyde = args.includes("--hyde");

  console.log("═══════════════════════════════════════════");
  console.log(`  BEIR Benchmark: ${datasetArg}`);
  console.log("═══════════════════════════════════════════\n");

  // Config for testDbBEIR
  // Default: detect if running in Docker (/home/node) or on host
  const defaultDir = process.env.HOME === "/home/node"
    ? "/home/node/.openclaw/data/testDbBEIR/lancedb"
    : `${process.env.HOME}/.openclaw/data/testDbBEIR/lancedb`;
  const lancedbDir = process.env.BEIR_LANCEDB_DIR || defaultDir;
  console.log(`[INFO] Using lancedbDir: ${lancedbDir}`);

  const cfg = resolveConfig({ lancedbDir } as Parameters<typeof resolveConfig>[0]);

  // Initialize
  const backend = new LanceDBBackend(cfg);
  await backend.open();

  const provider = await createEmbedProvider(cfg.embed);
  console.log(`[INFO] Embed provider: ${provider.id} / ${provider.model} / dims=${provider.dims ?? "unknown"}`);
  const embed = new EmbedQueue(provider, { concurrency: 1, maxRetries: 2, timeoutMs: 30000 });

  // Sanity check: verify embedding dimensions match DB
  const testVec = await embed.embedQuery("test query");
  console.log(`[INFO] Test embed dimensions: ${testVec.length}`);

  const reranker = cfg.rerank.enabled ? await createReranker(cfg.rerank) : null;
  console.log(`[INFO] Reranker: ${reranker ? "enabled" : "disabled"}`);

  // HyDE config — always available; per-config useHyde controls activation.
  // CLI --hyde forces HyDE on for ALL configs (useful for quick testing).
  const hydeConfig = (cliHyde || cfg.hyde?.enabled) ? cfg.hyde : undefined;
  console.log(`[INFO] HyDE: ${hydeConfig?.enabled ? "enabled" : "disabled"} (cli-forced: ${cliHyde})`);

  // Multi-query expansion config — uses same Spark vLLM as HyDE.
  const expansionConfig: QueryExpansionConfig = {
    ...QUERY_EXPANSION_DEFAULTS,
    apiKey: process.env.SPARK_BEARER_TOKEN,
  };
  console.log(`[INFO] Multi-Query Expansion: available (per-config useMultiQuery controls activation)`);

  // Load dataset
  console.log(`[INFO] Loading ${datasetArg} queries and qrels...`);
  const queries = await loadQueries(datasetArg);
  const qrels = await loadQrels(datasetArg);
  console.log(`[INFO] ${queries.length} queries, ${Object.keys(qrels).length} with judgments`);

  // Filter configs
  const configs = configArg
    ? CONFIGS.filter((c) => configArg.split(",").map((s) => s.trim().toUpperCase()).includes(c.id))
    : CONFIGS;

  // Results directory
  const resultsDir = path.join(import.meta.dirname!, "../evaluation/results");
  await fs.mkdir(resultsDir, { recursive: true });

  // Run benchmarks
  const allResults: ConfigResult[] = [];
  const allTelemetry: QueryTelemetry[] = [];

  for (const config of configs) {
    console.log(`\n▶ Config ${config.id}: ${config.label}`);

    const { results, telemetry } = await runRetrieval(
      queries,
      qrels,
      backend,
      embed,
      reranker,
      hydeConfig,
      config.useMultiQuery ? expansionConfig : undefined,
      config,
      datasetArg,
    );

    const metrics = evaluateBEIR(qrels, results, [10]);

    // Compute latency stats from telemetry
    const latencies = telemetry.map((t) => t.latencyMs).sort((a, b) => a - b);
    const p50 = latencies[Math.floor(latencies.length * 0.5)] ?? 0;
    const p95 = latencies[Math.floor(latencies.length * 0.95)] ?? 0;
    const mean = latencies.reduce((a, b) => a + b, 0) / latencies.length;

    console.log(`  NDCG@10:    ${metrics.ndcg["@10"].toFixed(4)}`);
    console.log(`  MRR@10:     ${metrics.mrr["@10"].toFixed(4)}`);
    console.log(`  Recall@10:  ${metrics.recall["@10"].toFixed(4)}`);
    console.log(`  MAP@10:     ${metrics.map["@10"].toFixed(4)}`);
    console.log(`  Latency:    p50=${p50}ms, p95=${p95}ms, mean=${mean.toFixed(0)}ms`);

    const configResult: ConfigResult = {
      config: config.id,
      label: config.label,
      ndcg: metrics.ndcg["@10"],
      mrr: metrics.mrr["@10"],
      recall: metrics.recall["@10"],
      map: metrics.map["@10"],
      latencyP50: p50,
      latencyP95: p95,
      latencyMean: mean,
    };

    allResults.push(configResult);
    allTelemetry.push(...telemetry);

    // Save individual config results
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const resultsFile = path.join(resultsDir, `beir-${datasetArg}-${config.id}-${timestamp}.json`);
    await fs.writeFile(resultsFile, JSON.stringify({ config, metrics, results }, null, 2));
  }

  await backend.close();

  // Summary table
  console.log("\n═══════════════════════════════════════════");
  console.log("  Summary: BEIR " + datasetArg);
  console.log("═══════════════════════════════════════════\n");

  console.log("ID | Config               | NDCG@10 | MRR@10  | Recall@10 | MAP@10  | p95(ms)");
  console.log("---|----------------------|---------|---------|-----------|---------|--------");
  for (const r of allResults) {
    console.log(
      `${r.config}  | ${r.label.padEnd(20)} | ${r.ndcg.toFixed(4)}  | ${r.mrr.toFixed(4)}  | ${r.recall.toFixed(4)}   | ${r.map.toFixed(4)}  | ${r.latencyP95}`,
    );
  }

  // Save telemetry for audit
  const telemetryFile = path.join(resultsDir, `beir-${datasetArg}-telemetry-${Date.now()}.json`);
  await fs.writeFile(telemetryFile, JSON.stringify(allTelemetry, null, 2));
  console.log(`\n[INFO] Telemetry saved to: ${telemetryFile}`);

  // Summary file
  const summaryFile = path.join(resultsDir, `beir-${datasetArg}-summary-${Date.now()}.json`);
  await fs.writeFile(summaryFile, JSON.stringify({ dataset: datasetArg, results: allResults }, null, 2));

  console.log("\n✅ BEIR benchmark complete");
}

main().catch((err) => {
  console.error("\n❌ FATAL:", err);
  process.exit(1);
});
