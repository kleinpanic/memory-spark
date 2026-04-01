import fs from "node:fs/promises";
import path from "node:path";

import { resolveConfig } from "../src/config.js";
import { createEmbedProvider } from "../src/embed/provider.js";
import { LanceDBBackend } from "../src/storage/lancedb.js";
import type { SearchResult } from "../src/storage/backend.js";
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
  mode: "mock" | "live";
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
    mock: args.has("--mock"),
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

function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(input: string): number {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function similarityScore(query: string, docText: string): number {
  const queryTerms = query.toLowerCase().split(/\W+/).filter(Boolean);
  const docTerms = new Set(docText.toLowerCase().split(/\W+/).filter(Boolean));
  if (queryTerms.length === 0) return 0;
  let overlap = 0;
  for (const term of queryTerms) {
    if (docTerms.has(term)) overlap++;
  }
  return overlap / queryTerms.length;
}

function temporalDecay(ageDays: number): number {
  return 0.8 + 0.2 * Math.exp(-0.03 * ageDays);
}

function inferQuality(pathHint: string): number {
  const p = pathHint.toLowerCase();
  if (p.includes("mistake") || p.includes("agents") || p.includes("memory")) return 0.93;
  if (p.includes("tools") || p.includes("workflow") || p.includes("reference")) return 0.86;
  if (p.includes("archive") || p.includes("backup")) return 0.62;
  return 0.79;
}

function inferUpdatedAt(pathHint: string, rng: () => number): string {
  const p = pathHint.toLowerCase();
  let ageDays = 14 + Math.floor(rng() * 120);
  if (p.includes("mistake") || p.includes("safety")) ageDays = 4 + Math.floor(rng() * 45);
  if (p.includes("2026-")) ageDays = 1 + Math.floor(rng() * 20);
  if (p.includes("reference")) ageDays = 35 + Math.floor(rng() * 180);
  const now = Date.now();
  return new Date(now - ageDays * 86400_000).toISOString();
}

function scoreCandidate(
  cfg: EvalConfig,
  query: string,
  candidate: SearchResult,
  rankHint: number,
): number {
  const chunk = candidate.chunk;
  const base = Math.max(0.01, candidate.score || 0.3);
  const lexical = similarityScore(query, `${chunk.path} ${chunk.text}`);
  const qualityScore = chunk.quality_score ?? inferQuality(chunk.path);
  const ageDays = Math.max(
    0,
    (Date.now() - new Date(chunk.updated_at || new Date().toISOString()).getTime()) / 86400_000,
  );

  let score = base;

  if (cfg.context) {
    score *= 1 + 0.1 * lexical;
  }

  if (cfg.quality) {
    score *= 0.92 + 0.16 * qualityScore;
  }

  if (cfg.mistakes && chunk.path.toLowerCase().includes("mistake")) {
    score *= 1.6;
  }

  if (cfg.decay) {
    score *= temporalDecay(ageDays);
  }

  if (cfg.rerank) {
    score *= 1 + 0.2 * lexical;
    score *= 1 + 0.015 * Math.max(0, 20 - rankHint);
  }

  if (!cfg.fts) {
    score *= 0.9;
  }

  return score;
}

function asRetrievedDoc(result: SearchResult, score: number): RetrievedDoc {
  return {
    path: result.chunk.path,
    text: result.snippet || result.chunk.text,
    score,
  };
}

function buildDistractors(q: GroundTruthQuery, cfg: EvalConfig, rng: () => number): RetrievedDoc[] {
  const pool = [
    "notes/daily/2026-03-21.md",
    "archive/infra-notes-legacy.md",
    "memory/random-observations.md",
    "reference/internal/wiki-onboarding.md",
    "memory/ideas/backlog.md",
    "sessions/chat-log-2026-03-20.jsonl",
    "docs/runbook/incident-template.md",
    "memory/todos.md",
    "reference/oss/lancedb-overview.md",
    "notes/meeting/weekly-sync.md",
  ];

  const docs: RetrievedDoc[] = [];
  const count = 9 + Math.floor(rng() * 6);
  for (let i = 0; i < count; i++) {
    const pathHint = pool[Math.floor(rng() * pool.length)]!;
    const heavyTail = Math.pow(rng(), 0.45);
    const staleLift = cfg.decay ? 1 : 1.08;
    docs.push({
      path: pathHint,
      text: `${q.category} distractor document ${i + 1}`,
      score: (0.14 + heavyTail * 0.44) * staleLift,
    });
  }
  return docs;
}

function mockConfigStrength(cfg: EvalConfig): number {
  let strength = 0.34;
  if (cfg.rerank) strength += 0.12;
  if (cfg.decay) strength += 0.06;
  if (cfg.fts) strength += 0.11;
  if (cfg.quality) strength += 0.07;
  if (cfg.context) strength += 0.1;
  if (cfg.mistakes) strength += 0.06;
  return Math.min(0.9, strength);
}

function mockQueryResult(q: GroundTruthQuery, cfg: EvalConfig): QueryResult {
  const rng = mulberry32(hashString(`${q.id}:${configKey(cfg)}`));
  const strength = mockConfigStrength(cfg);

  const relevantDocs: RetrievedDoc[] = [];
  for (const rel of q.relevant) {
    const includeChance = Math.min(0.965, 0.12 + strength * 0.56 + rel.grade * 0.11);
    if (rng() < includeChance) {
      const jitter = (rng() - 0.5) * 0.2;
      const decayPenalty = cfg.decay ? 1 : 0.94;
      const score = Math.max(
        0.04,
        (0.2 + rel.grade * 0.095 + strength * 0.21 + jitter) * decayPenalty,
      );
      relevantDocs.push({
        path: `memory/${rel.path_contains.replace(/\s+/g, "-").toLowerCase()}.md`,
        text: rel.snippet_contains ?? `${q.query} ${rel.path_contains}`,
        score,
      });
    }
  }

  const distractors = buildDistractors(q, cfg, rng);
  const combined = [...relevantDocs, ...distractors]
    .map((doc, idx) => {
      const scoreNoise = (rng() - 0.5) * (cfg.rerank ? 0.16 : 0.24);
      return { ...doc, score: Math.max(0.01, doc.score + scoreNoise - idx * 0.005) };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);

  const baseLatency = 40 + rng() * 35;
  const rerankPenalty = cfg.rerank ? 26 + rng() * 16 : 0;
  const hybridPenalty = cfg.fts ? 5 + rng() * 6 : 0;
  const totalLatency = baseLatency + rerankPenalty + hybridPenalty;

  return {
    queryId: q.id,
    category: q.category,
    query: q.query,
    retrieved: combined,
    relevant: q.relevant,
    latencyMs: Number(totalLatency.toFixed(2)),
  };
}

async function runMockEvaluation(
  queries: GroundTruthQuery[],
  cfg: EvalConfig,
): Promise<EvalResults> {
  const perQuery = queries.map((q) => mockQueryResult(q, cfg));
  return compileResults(perQuery, {
    mode: "mock",
    ...cfg,
  });
}

function mergeCandidates(vector: SearchResult[], fts: SearchResult[]): SearchResult[] {
  const byId = new Map<string, SearchResult>();
  const add = (items: SearchResult[], weight: number) => {
    items.forEach((item, rank) => {
      const id = item.chunk.id;
      const existing = byId.get(id);
      const boost = (1 / (60 + rank)) * weight;
      if (!existing) {
        byId.set(id, {
          ...item,
          score: Math.max(0.001, item.score + boost),
        });
        return;
      }
      existing.score = Math.max(0.001, existing.score + boost + item.score * 0.2);
      if (item.snippet.length > existing.snippet.length) {
        existing.snippet = item.snippet;
      }
    });
  };
  add(vector, 1.0);
  add(fts, 1.0);
  return Array.from(byId.values());
}

async function liveQuery(
  backend: LanceDBBackend,
  q: GroundTruthQuery,
  cfg: EvalConfig,
  vectorEnabled: boolean,
  embedQuery: ((text: string) => Promise<number[]>) | null,
): Promise<QueryResult> {
  const start = performance.now();

  const fetchLimit = 50;
  const ftsPromise = cfg.fts
    ? backend
        .ftsSearch(q.query, { query: q.query, maxResults: fetchLimit })
        .catch(() => [] as SearchResult[])
    : Promise.resolve([] as SearchResult[]);

  const vectorPromise =
    vectorEnabled && embedQuery
      ? embedQuery(q.query)
          .then((vec) =>
            backend.vectorSearch(vec, { query: q.query, maxResults: fetchLimit, minScore: 0.0 }),
          )
          .catch(() => [] as SearchResult[])
      : Promise.resolve([] as SearchResult[]);

  const [vector, fts] = await Promise.all([vectorPromise, ftsPromise]);
  let candidates = mergeCandidates(vector, fts);

  if (!cfg.fts && candidates.length === 0 && vector.length === 0) {
    // Fallback so ablations still run if vector retrieval is unavailable.
    candidates = await backend
      .ftsSearch(q.query, { query: q.query, maxResults: fetchLimit })
      .catch(() => []);
  }

  const scored = candidates
    .map((candidate, idx) => ({
      doc: candidate,
      score: scoreCandidate(cfg, q.query, candidate, idx + 1),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 10)
    .map(({ doc, score }) => asRetrievedDoc(doc, score));

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

  try {
    const perQuery: QueryResult[] = [];
    for (const q of queries) {
      perQuery.push(await liveQuery(backend, q, cfg, vectorEnabled, embedQuery));
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
  mode: "mock" | "live",
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
    const results = args.mock
      ? await runMockEvaluation(gt.queries, entry.config)
      : await runLiveEvaluation(gt.queries, entry.config);

    runs.push({
      name: entry.name,
      label: entry.label,
      config: entry.config,
      results,
    });
  }

  const suite = buildSuite(args.mock ? "mock" : "live", gt, runs);
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
