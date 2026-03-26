import fs from "node:fs/promises";
import path from "node:path";

import type { EvalResults } from "./metrics.js";

interface EvalRun {
  name: string;
  label: string;
  results: EvalResults;
}

interface EvaluationSuite {
  mode: "mock" | "live";
  runs: EvalRun[];
}

const FIGURES_DIR = path.join("docs", "figures");
const DEFAULT_RESULTS_PATH = path.join("evaluation", "results", "latest.json");

const C = {
  primary: "#58a6ff",
  accent: "#3fb950",
  warning: "#d29922",
  danger: "#f85149",
  bg: "#0d1117",
  card: "#161b22",
  text: "#c9d1d9",
  muted: "#8b949e",
  grid: "#30363d",
};

function parseArgs(argv: string[]): { resultsPath: string } {
  const idx = argv.indexOf("--results");
  if (idx >= 0 && argv[idx + 1]) {
    return { resultsPath: argv[idx + 1]! };
  }
  return { resultsPath: DEFAULT_RESULTS_PATH };
}

function esc(s: string): string {
  return s.replace(
    /[&<>"']/g,
    (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch] ?? ch,
  );
}

function frame(width: number, height: number, inner: string): string {
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="memory-spark chart">
  <defs>
    <style>
      text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    </style>
  </defs>
  <rect x="0" y="0" width="${width}" height="${height}" fill="${C.bg}"/>
  <rect x="12" y="12" width="${width - 24}" height="${height - 24}" rx="12" fill="${C.card}" stroke="${C.grid}"/>
  ${inner}
</svg>`;
}

function findRun(runs: EvalRun[], name: string): EvalRun | undefined {
  return runs.find((r) => r.name === name);
}

function linePath(points: Array<{ x: number; y: number }>): string {
  if (points.length === 0) return "";
  return points
    .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)} ${p.y.toFixed(2)}`)
    .join(" ");
}

function ablationChart(runs: EvalRun[]): string {
  const width = 1200;
  const height = 640;
  const margin = { top: 90, right: 50, bottom: 40, left: 270 };
  const plotW = width - margin.left - margin.right;
  const barH = 42;
  const gap = 18;

  const order = [
    "full",
    "no-rerank",
    "no-decay",
    "no-fts",
    "no-quality",
    "no-context",
    "no-mistakes",
    "vanilla",
  ];
  const rows = order
    .map((name) => findRun(runs, name))
    .filter((r): r is EvalRun => !!r)
    .map((r) => ({ label: r.label, value: r.results.metrics.ndcg_at_10.mean, name: r.name }));

  const x = (v: number) => margin.left + Math.max(0, Math.min(1, v)) * plotW;

  const bars = rows
    .map((row, i) => {
      const y = margin.top + i * (barH + gap);
      const fill = row.name === "full" ? C.accent : row.name === "vanilla" ? C.danger : C.primary;
      const w = x(row.value) - margin.left;
      return `
        <text x="${margin.left - 14}" y="${y + barH * 0.66}" fill="${C.text}" text-anchor="end" font-size="20">${esc(row.label)}</text>
        <rect x="${margin.left}" y="${y}" width="${w.toFixed(2)}" height="${barH}" rx="6" fill="${fill}" opacity="0.9"/>
        <text x="${(margin.left + w + 10).toFixed(2)}" y="${y + barH * 0.66}" fill="${C.text}" font-size="18">${row.value.toFixed(3)}</text>
      `;
    })
    .join("\n");

  const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    .map((t) => {
      const tx = x(t);
      return `
        <line x1="${tx}" x2="${tx}" y1="${margin.top - 20}" y2="${height - margin.bottom}" stroke="${C.grid}" stroke-width="1"/>
        <text x="${tx}" y="${height - margin.bottom + 24}" text-anchor="middle" fill="${C.muted}" font-size="14">${t.toFixed(1)}</text>
      `;
    })
    .join("\n");

  return frame(
    width,
    height,
    `
    <text x="40" y="58" fill="${C.text}" font-size="30" font-weight="700">Ablation Study: NDCG@10 by Component</text>
    <text x="40" y="84" fill="${C.muted}" font-size="16">Full pipeline vs component removals</text>
    ${ticks}
    ${bars}
    `,
  );
}

function recallCurve(runs: EvalRun[]): string {
  const width = 1200;
  const height = 580;
  const margin = { top: 90, right: 80, bottom: 70, left: 90 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const full = findRun(runs, "full") ?? runs[0]!;
  const vanilla = findRun(runs, "vanilla") ?? runs[runs.length - 1]!;

  const kVals = [1, 3, 5, 10];
  const getRecall = (r: EvalRun, k: number) => {
    if (k === 1) return r.results.metrics.recall_at_1.mean;
    if (k === 3) return r.results.metrics.recall_at_3.mean;
    if (k === 5) return r.results.metrics.recall_at_5.mean;
    return r.results.metrics.recall_at_10.mean;
  };

  const x = (i: number) => margin.left + (i / (kVals.length - 1)) * plotW;
  const y = (v: number) => margin.top + (1 - v) * plotH;

  const makeSeries = (run: EvalRun, color: string) => {
    const points = kVals.map((k, i) => ({ x: x(i), y: y(getRecall(run, k)) }));
    const circles = points
      .map(
        (p, i) =>
          `<circle cx="${p.x}" cy="${p.y}" r="5" fill="${color}"><title>K=${kVals[i]} ${getRecall(run, kVals[i]!).toFixed(3)}</title></circle>`,
      )
      .join("\n");
    return `<path d="${linePath(points)}" fill="none" stroke="${color}" stroke-width="3"/>${circles}`;
  };

  const hTicks = [0, 0.25, 0.5, 0.75, 1.0]
    .map(
      (t) => `
      <line x1="${margin.left}" x2="${width - margin.right}" y1="${y(t)}" y2="${y(t)}" stroke="${C.grid}"/>
      <text x="${margin.left - 10}" y="${y(t) + 5}" fill="${C.muted}" text-anchor="end" font-size="13">${t.toFixed(2)}</text>
    `,
    )
    .join("\n");

  const xTicks = kVals
    .map(
      (k, i) =>
        `<text x="${x(i)}" y="${height - margin.bottom + 28}" fill="${C.muted}" text-anchor="middle" font-size="14">${k}</text>`,
    )
    .join("\n");

  return frame(
    width,
    height,
    `
    <text x="40" y="58" fill="${C.text}" font-size="30" font-weight="700">Recall@K: Full Pipeline vs Vanilla</text>
    <text x="40" y="84" fill="${C.muted}" font-size="16">Coverage improves at every cutoff with hybrid reranked retrieval</text>
    ${hTicks}
    ${xTicks}
    ${makeSeries(full, C.accent)}
    ${makeSeries(vanilla, C.danger)}
    <text x="${width - 220}" y="${margin.top + 16}" fill="${C.accent}" font-size="16">● ${esc(full.label)}</text>
    <text x="${width - 220}" y="${margin.top + 40}" fill="${C.danger}" font-size="16">● ${esc(vanilla.label)}</text>
    <text x="${margin.left + plotW / 2}" y="${height - 16}" fill="${C.muted}" text-anchor="middle" font-size="14">K</text>
    <text transform="translate(24 ${margin.top + plotH / 2}) rotate(-90)" fill="${C.muted}" text-anchor="middle" font-size="14">Recall</text>
    `,
  );
}

function radarChart(runs: EvalRun[]): string {
  const width = 760;
  const height = 760;
  const cx = width / 2;
  const cy = height / 2 + 20;
  const radius = 250;

  const full = findRun(runs, "full") ?? runs[0]!;
  const entries = Object.entries(full.results.perCategory).sort(([a], [b]) => a.localeCompare(b));
  const n = entries.length;

  const rings = [0.2, 0.4, 0.6, 0.8, 1.0]
    .map(
      (r) =>
        `<circle cx="${cx}" cy="${cy}" r="${(radius * r).toFixed(1)}" fill="none" stroke="${C.grid}"/>`,
    )
    .join("\n");

  const axes = entries
    .map(([cat], i) => {
      const a = (Math.PI * 2 * i) / n - Math.PI / 2;
      const x = cx + Math.cos(a) * (radius + 28);
      const y = cy + Math.sin(a) * (radius + 28);
      const ax = cx + Math.cos(a) * radius;
      const ay = cy + Math.sin(a) * radius;
      return `
        <line x1="${cx}" y1="${cy}" x2="${ax.toFixed(1)}" y2="${ay.toFixed(1)}" stroke="${C.grid}"/>
        <text x="${x.toFixed(1)}" y="${y.toFixed(1)}" fill="${C.text}" font-size="14" text-anchor="middle">${esc(cat)}</text>
      `;
    })
    .join("\n");

  const points = entries.map(([, m], i) => {
    const a = (Math.PI * 2 * i) / n - Math.PI / 2;
    const r = Math.max(0, Math.min(1, m.ndcg_at_10)) * radius;
    const x = cx + Math.cos(a) * r;
    const y = cy + Math.sin(a) * r;
    return { x, y };
  });

  const poly = points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const markers = points
    .map((p) => `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4" fill="${C.primary}"/>`)
    .join("\n");

  return frame(
    width,
    height,
    `
    <text x="32" y="58" fill="${C.text}" font-size="30" font-weight="700">Per-Category NDCG@10 (Full Pipeline)</text>
    <text x="32" y="84" fill="${C.muted}" font-size="16">Eight-category retrieval profile</text>
    ${rings}
    ${axes}
    <polygon points="${poly}" fill="${C.primary}" fill-opacity="0.22" stroke="${C.primary}" stroke-width="2"/>
    ${markers}
    `,
  );
}

function decayChart(): string {
  const width = 1100;
  const height = 520;
  const margin = { top: 80, right: 60, bottom: 60, left: 90 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const maxDays = 365;
  const x = (d: number) => margin.left + (d / maxDays) * plotW;
  const y = (v: number) => margin.top + (1 - v) * plotH;

  const pts = Array.from({ length: 366 }, (_, d) => {
    const value = 0.8 + 0.2 * Math.exp(-0.03 * d);
    return { x: x(d), y: y(value) };
  });

  const keyDays = [0, 7, 30, 90, 180, 365];
  const markers = keyDays
    .map((d) => {
      const v = 0.8 + 0.2 * Math.exp(-0.03 * d);
      return `
        <circle cx="${x(d)}" cy="${y(v)}" r="4" fill="${C.warning}"/>
        <text x="${x(d) + 8}" y="${y(v) - 8}" fill="${C.text}" font-size="13">d${d}: ${v.toFixed(3)}</text>
      `;
    })
    .join("\n");

  const yTicks = [0.8, 0.85, 0.9, 0.95, 1.0]
    .map(
      (v) => `
      <line x1="${margin.left}" x2="${width - margin.right}" y1="${y(v)}" y2="${y(v)}" stroke="${C.grid}"/>
      <text x="${margin.left - 10}" y="${y(v) + 5}" fill="${C.muted}" text-anchor="end" font-size="13">${v.toFixed(2)}</text>
    `,
    )
    .join("\n");

  return frame(
    width,
    height,
    `
    <text x="36" y="54" fill="${C.text}" font-size="30" font-weight="700">Temporal Decay Curve</text>
    <text x="36" y="80" fill="${C.muted}" font-size="16">score factor = 0.8 + 0.2 * exp(-0.03 * ageDays)</text>
    ${yTicks}
    <path d="${linePath(pts)}" fill="none" stroke="${C.warning}" stroke-width="3"/>
    ${markers}
    <text x="${margin.left + plotW / 2}" y="${height - 16}" fill="${C.muted}" text-anchor="middle" font-size="14">Age (days)</text>
    <text transform="translate(26 ${margin.top + plotH / 2}) rotate(-90)" fill="${C.muted}" text-anchor="middle" font-size="14">Decay Factor</text>
    `,
  );
}

function latencyChart(runs: EvalRun[]): string {
  const width = 1160;
  const height = 560;
  const margin = { top: 90, right: 50, bottom: 60, left: 90 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const full = findRun(runs, "full") ?? runs[0]!;
  const latencies = full.results.perQuery.map((q) => q.latencyMs).sort((a, b) => a - b);
  const max = Math.max(...latencies, full.results.metrics.p99_latency_ms);
  const bins = 12;
  const binSize = max / bins;

  const hist = Array.from({ length: bins }, () => 0);
  for (const v of latencies) {
    const idx = Math.min(bins - 1, Math.floor(v / binSize));
    hist[idx]++;
  }

  const maxCount = Math.max(...hist, 1);
  const x = (i: number) => margin.left + (i / bins) * plotW;
  const y = (count: number) => margin.top + (1 - count / maxCount) * plotH;

  const bars = hist
    .map((count, i) => {
      const x0 = x(i) + 2;
      const x1 = x(i + 1) - 2;
      const yy = y(count);
      return `<rect x="${x0}" y="${yy}" width="${Math.max(2, x1 - x0)}" height="${margin.top + plotH - yy}" fill="${C.primary}" opacity="0.75"/>`;
    })
    .join("\n");

  const percentileLines = [
    { name: "p50", value: full.results.metrics.p50_latency_ms, color: C.accent },
    { name: "p95", value: full.results.metrics.p95_latency_ms, color: C.warning },
    { name: "p99", value: full.results.metrics.p99_latency_ms, color: C.danger },
  ]
    .map((p, idx) => {
      const px = margin.left + (p.value / max) * plotW;
      return `
        <line x1="${px}" x2="${px}" y1="${margin.top}" y2="${margin.top + plotH}" stroke="${p.color}" stroke-width="2" stroke-dasharray="5 4"/>
        <text x="${px + 6}" y="${margin.top + 20 + idx * 18}" fill="${p.color}" font-size="13">${p.name}: ${p.value.toFixed(1)}ms</text>
      `;
    })
    .join("\n");

  return frame(
    width,
    height,
    `
    <text x="40" y="58" fill="${C.text}" font-size="30" font-weight="700">Latency Distribution (Full Pipeline)</text>
    <text x="40" y="84" fill="${C.muted}" font-size="16">Histogram over per-query retrieval latency</text>
    ${bars}
    ${percentileLines}
    <line x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${margin.top + plotH}" stroke="${C.grid}"/>
    <line x1="${margin.left}" x2="${margin.left + plotW}" y1="${margin.top + plotH}" y2="${margin.top + plotH}" stroke="${C.grid}"/>
    <text x="${margin.left + plotW / 2}" y="${height - 16}" fill="${C.muted}" text-anchor="middle" font-size="14">Latency (ms)</text>
    <text transform="translate(24 ${margin.top + plotH / 2}) rotate(-90)" fill="${C.muted}" text-anchor="middle" font-size="14">Query Count</text>
    `,
  );
}

async function main() {
  const { resultsPath } = parseArgs(process.argv.slice(2));
  const payload = await fs.readFile(resultsPath, "utf8");
  const suite = JSON.parse(payload) as EvaluationSuite;

  await fs.mkdir(FIGURES_DIR, { recursive: true });

  const files: Array<[string, string]> = [
    ["ablation-ndcg.svg", ablationChart(suite.runs)],
    ["recall-curve.svg", recallCurve(suite.runs)],
    ["category-radar.svg", radarChart(suite.runs)],
    ["temporal-decay.svg", decayChart()],
    ["latency-distribution.svg", latencyChart(suite.runs)],
  ];

  await Promise.all(
    files.map(([file, svg]) => fs.writeFile(path.join(FIGURES_DIR, file), svg, "utf8")),
  );

  console.log(`Generated ${files.length} charts in ${FIGURES_DIR}`);
}

main().catch((err) => {
  console.error("Chart generation failed:", err);
  process.exit(1);
});
