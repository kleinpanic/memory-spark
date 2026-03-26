import fs from "node:fs/promises";
import path from "node:path";

import { resolveConfig } from "../src/config.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { createReranker } from "../src/rerank/reranker.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import type { SearchResult } from "../src/storage/backend.js";
import {
  hybridMerge,
  applyTemporalDecay,
  applySourceWeighting,
  mmrRerank,
} from "../src/auto/recall.js";
import {
  compileResults,
  type EvalResults,
  type QueryResult,
  type RelevanceJudgment,
  type RetrievedDoc,
} from "./metrics.js";

interface GroundTruthQuery {
  id: string;
  category: string;
  query: string;
  relevant: RelevanceJudgment[];
}

interface GroundTruthSet {
  categories: string[];
  queries: GroundTruthQuery[];
}

interface EvalConfig {
  rerank: boolean;
  decay: boolean;
  fts: boolean;
  quality: boolean;
  context: boolean;
  mistakes: boolean;
}

interface EvalRun {
  name: string;
  label: string;
  config: EvalConfig;
  results: EvalResults;
}

interface EvaluationSuite {
  schemaVersion: string;
  generatedAt: string;
  mode: "live";
  dataset: {
    path: string;
    queryCount: number;
    categories: string[];
  };
  runs: EvalRun[];
  summary: {
    bestByNdcgAt10: string;
    baselineNdcgAt10: number;
    vanillaNdcgAt10: number;
    absoluteGainVsVanilla: number;
  };
}

const RESULTS_DIR = path.join("evaluation", "results");
const GROUND_TRUTH_PATH = path.join("evaluation", "ground-truth.json");

const ABLATIONS: Array<{ name: string; label: string; config: EvalConfig }> = [
  {
    name: "full",
    label: "Full Pipeline",
    config: { rerank: true, decay: true, fts: true, quality: true, context: true, mistakes: true },
  },
  {
    name: "no-rerank",
    label: "- Rerank",
    config: { rerank: false, decay: true, fts: true, quality: true, context: true, mistakes: true },
  },
  {
    name: "no-decay",
    label: "- Temporal Decay",
    config: { rerank: true, decay: false, fts: true, quality: true, context: true, mistakes: true },
  },
  {
    name: "no-fts",
    label: "- Hybrid FTS",
    config: { rerank: true, decay: true, fts: false, quality: true, context: true, mistakes: true },
  },
  {
    name: "no-quality",
    label: "- Quality Filter",
    config: { rerank: true, decay: true, fts: true, quality: false, context: true, mistakes: true },
  },
  {
    name: "no-context",
    label: "- Contextual Prefix",
    config: { rerank: true, decay: true, fts: true, quality: true, context: false, mistakes: true },
  },
  {
    name: "no-mistakes",
    label: "- Mistake Weighting",
    config: { rerank: true, decay: true, fts: true, quality: true, context: true, mistakes: false },
  },
  {
    name: "vanilla",
    label: "Vanilla Retrieval",
    config: {
      rerank: false,
      decay: false,
      fts: false,
      quality: false,
      context: false,
      mistakes: false,
    },
  },
];

function parseArgs(argv: string[]) {
  const args = new Set(argv);
  const selectedConfig: EvalConfig = {
    rerank: !args.has("--no-rerank"),
    decay: !args.has("--no-decay"),
    fts: !args.has("--no-fts"),
    quality: !args.has("--no-quality"),
    context: !args.has("--no-context"),
    mistakes: !args.has("--no-mistakes"),
  };

  const hasAnyAblationFlag =
    args.has("--no-rerank") ||
    args.has("--no-decay") ||
    args.has("--no-fts") ||
    args.has("--no-quality") ||
    args.has("--no-context") ||
    args.has("--no-mistakes");

  return {
    suiteMode: !hasAnyAblationFlag,
    selectedConfig,
  };
}

function configKey(cfg: EvalConfig): string {
  if (!cfg.rerank && !cfg.decay && !cfg.fts && !cfg.quality && !cfg.context && !cfg.mistakes) {
    return "vanilla";
  }
  const parts: string[] = [];
  if (!cfg.rerank) parts.push("no-rerank");
  if (!cfg.decay) parts.push("no-decay");
  if (!cfg.fts) parts.push("no-fts");
  if (!cfg.quality) parts.push("no-quality");
  if (!cfg.context) parts.push("no-context");
  if (!cfg.mistakes) parts.push("no-mistakes");
  return parts.length === 0 ? "full" : parts.join("+");
}

function asRetrievedDoc(result: SearchResult, score: number): RetrievedDoc {
  return {
    path: result.chunk.path,
    text: result.snippet || result.chunk.text,
    score,
  };
}

/**
 * Live query using the ACTUAL production pipeline from recall.ts.
 * Previous version used mergeCandidates() + scoreCandidate() which was a
 * broken simulation that preserved cosine scores (production doesn't) and
 * simulated reranking with lexical multipliers (production uses Spark cross-encoder).
 *
 * This version uses the real: hybridMerge → sourceWeighting → temporalDecay → MMR → reranker
 */
async function liveQuery(
  backend: LanceDBBackend,
  q: GroundTruthQuery,
  cfg: EvalConfig,
  vectorEnabled: boolean,
  embedQuery: ((text: string) => Promise<number[]>) | null,
  reranker?: { rerank: (query: string, results: SearchResult[], topN: number) => Promise<SearchResult[]> },
): Promise<QueryResult> {
  const start = performance.now();

  const fetchLimit = 50;
  const minScore = 0.2; // Match production config

  // Fetch from both retrievers (matching production recall.ts)
  const vectorPromise =
    vectorEnabled && embedQuery
      ? embedQuery(q.query)
          .then((vec) =>
            backend.vectorSearch(vec, { query: q.query, maxResults: fetchLimit, minScore }),
          )
          .catch(() => [] as SearchResult[])
      : Promise.resolve([] as SearchResult[]);

  const ftsPromise = cfg.fts
    ? backend
        .ftsSearch(q.query, { query: q.query, maxResults: fetchLimit })
        .catch(() => [] as SearchResult[])
    : Promise.resolve([] as SearchResult[]);

  const [vector, rawFts] = await Promise.all([vectorPromise, ftsPromise]);

  // Filter FTS: exclude sessions, apply minScore (matches production recall.ts)
  const fts = rawFts.filter(
    (r) => r.chunk.source !== "sessions" && r.score >= minScore,
  );

  // Use real production pipeline functions
  let candidates = cfg.fts ? hybridMerge(vector, fts, fetchLimit) : [...vector];

  if (candidates.length === 0 && vector.length === 0) {
    // Fallback so ablations still run if vector retrieval is unavailable
    const fallbackFts = await backend
      .ftsSearch(q.query, { query: q.query, maxResults: fetchLimit })
      .catch(() => [] as SearchResult[]);
    candidates = fallbackFts.filter((r) => r.chunk.source !== "sessions");
  }

  // Source weighting EARLY (matches production order)
  if (cfg.mistakes || cfg.quality) {
    applySourceWeighting(candidates);
  }

  // Temporal decay
  if (cfg.decay) {
    applyTemporalDecay(candidates);
  }

  // MMR diversity
  const diverse = mmrRerank(candidates, 20, 0.7);

  // Cross-encoder reranking (use real Spark reranker if available and enabled)
  let final: SearchResult[];
  if (cfg.rerank && reranker) {
    final = await reranker.rerank(q.query, diverse, 10);
  } else {
    final = diverse.sort((a, b) => b.score - a.score).slice(0, 10);
  }

  const scored = final.map((r) => asRetrievedDoc(r, r.score));

  const latencyMs = Number((performance.now() - start).toFixed(2));

  return {
    queryId: q.id,
    category: q.category,
    query: q.query,
    retrieved: scored,
    relevant: q.relevant,
    latencyMs,
  };
}

async function runLiveEvaluation(
  queries: GroundTruthQuery[],
  cfg: EvalConfig,
): Promise<EvalResults> {
  const runtimeCfg = resolveConfig();
  const backend = new LanceDBBackend(runtimeCfg);
  await backend.open();

  let embedQuery: ((text: string) => Promise<number[]>) | null = null;
  let vectorEnabled = true;
  try {
    const provider = await createEmbedProvider(runtimeCfg.embed);
    embedQuery = (text: string) => provider.embedQuery(text);
  } catch {
    vectorEnabled = false;
  }

  // Create the real Spark cross-encoder reranker (same as production)
  let reranker: { rerank: (query: string, results: SearchResult[], topN: number) => Promise<SearchResult[]> } | undefined;
  try {
    reranker = await createReranker(runtimeCfg.rerank);
  } catch {
    console.warn("⚠ Spark reranker unavailable — rerank ablations will use score-only ranking");
  }

  try {
    const perQuery: QueryResult[] = [];
    for (const q of queries) {
      perQuery.push(await liveQuery(backend, q, cfg, vectorEnabled, embedQuery, reranker));
    }

    return compileResults(perQuery, {
      mode: "live",
      vector: vectorEnabled,
      ...cfg,
    });
  } finally {
    await backend.close().catch(() => undefined);
  }
}

async function loadGroundTruth(): Promise<GroundTruthSet> {
  const raw = await fs.readFile(GROUND_TRUTH_PATH, "utf8");
  return JSON.parse(raw) as GroundTruthSet;
}

function prettyConfigLabel(name: string, cfg: EvalConfig): string {
  const lookup = ABLATIONS.find((x) => x.name === name);
  if (lookup) return lookup.label;
  return configKey(cfg);
}

function buildSuite(
  mode: "live",
  dataset: GroundTruthSet,
  runs: EvalRun[],
): EvaluationSuite {
  const sorted = [...runs].sort(
    (a, b) => b.results.metrics.ndcg_at_10.mean - a.results.metrics.ndcg_at_10.mean,
  );
  const full = runs.find((r) => r.name === "full") ?? sorted[0]!;
  const vanilla = runs.find((r) => r.name === "vanilla") ?? sorted[sorted.length - 1]!;

  return {
    schemaVersion: "1.0.0",
    generatedAt: new Date().toISOString(),
    mode,
    dataset: {
      path: GROUND_TRUTH_PATH,
      queryCount: dataset.queries.length,
      categories: dataset.categories,
    },
    runs,
    summary: {
      bestByNdcgAt10: sorted[0]!.name,
      baselineNdcgAt10: Number(full.results.metrics.ndcg_at_10.mean.toFixed(4)),
      vanillaNdcgAt10: Number(vanilla.results.metrics.ndcg_at_10.mean.toFixed(4)),
      absoluteGainVsVanilla: Number(
        (full.results.metrics.ndcg_at_10.mean - vanilla.results.metrics.ndcg_at_10.mean).toFixed(4),
      ),
    },
  };
}

async function writeResults(suite: EvaluationSuite): Promise<string> {
  await fs.mkdir(RESULTS_DIR, { recursive: true });
  const ts = suite.generatedAt.replace(/[:.]/g, "-");
  const suiteName = `${suite.mode}-${ts}.json`;
  const suitePath = path.join(RESULTS_DIR, suiteName);
  const latestPath = path.join(RESULTS_DIR, "latest.json");

  const payload = JSON.stringify(suite, null, 2);
  await fs.writeFile(suitePath, payload, "utf8");
  await fs.writeFile(latestPath, payload, "utf8");

  for (const run of suite.runs) {
    const runPath = path.join(RESULTS_DIR, `${run.name}.json`);
    await fs.writeFile(runPath, JSON.stringify(run.results, null, 2), "utf8");
  }

  return suitePath;
}

function printRunSummary(run: EvalRun) {
  const m = run.results.metrics;
  console.log(
    `${run.label.padEnd(20)} NDCG@10=${m.ndcg_at_10.mean.toFixed(3)} ` +
      `MRR=${m.mrr.mean.toFixed(3)} Recall@5=${m.recall_at_5.mean.toFixed(3)} ` +
      `p95=${m.p95_latency_ms.toFixed(1)}ms`,
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const gt = await loadGroundTruth();

  const configs = args.suiteMode
    ? ABLATIONS
    : [
        {
          name: configKey(args.selectedConfig),
          label: prettyConfigLabel(configKey(args.selectedConfig), args.selectedConfig),
          config: args.selectedConfig,
        },
      ];

  const runs: EvalRun[] = [];
  for (const entry of configs) {
    const results = await runLiveEvaluation(gt.queries, entry.config);

    runs.push({
      name: entry.name,
      label: entry.label,
      config: entry.config,
      results,
    });
  }

  const suite = buildSuite("live", gt, runs);
  const suitePath = await writeResults(suite);

  console.log(`\\nEvaluation mode: ${suite.mode}`);
  console.log(`Queries: ${gt.queries.length} | Categories: ${gt.categories.length}`);
  runs.forEach(printRunSummary);
  console.log(`\\nSaved: ${suitePath}`);
  console.log(`Updated: ${path.join(RESULTS_DIR, "latest.json")}`);
}

main().catch((err) => {
  console.error("Evaluation failed:", err);
  process.exit(1);
});
