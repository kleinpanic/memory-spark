/**
 * Generate publication-quality SVG charts from BEIR benchmark results.
 * Reads evaluation/results/*.json files and produces dark-themed SVGs.
 *
 * Usage: npx tsx evaluation/generate-charts.ts [--results-dir evaluation/results]
 */

import fs from "node:fs/promises";
import path from "node:path";

const FIGURES_DIR = path.join("docs", "figures");

const C = {
  primary: "#58a6ff",
  accent: "#3fb950",
  warning: "#d29922",
  danger: "#f85149",
  purple: "#a371f7",
  pink: "#f778ba",
  bg: "#0d1117",
  card: "#161b22",
  text: "#c9d1d9",
  muted: "#8b949e",
  grid: "#30363d",
};

interface BeirResult {
  config: string;
  dataset: string;
  metrics: {
    ndcg_at_10: number;
    mrr: number;
    recall_at_10: number;
    map_at_10: number;
    precision_at_10: number;
    mean_latency_ms: number;
  };
}

// ── Hardcoded Phase 12 SciFact results (from benchmark run) ────────────
// These will be replaced with live data once the full benchmark completes.
const SCIFACT_RESULTS: BeirResult[] = [
  { config: "A: Vector-Only", dataset: "scifact", metrics: { ndcg_at_10: 0.7709, mrr: 0.7365, recall_at_10: 0.9037, map_at_10: 0.6917, precision_at_10: 0.1013, mean_latency_ms: 528 } },
  { config: "B: FTS-Only", dataset: "scifact", metrics: { ndcg_at_10: 0.6523, mrr: 0.6289, recall_at_10: 0.7867, map_at_10: 0.5856, precision_at_10: 0.0887, mean_latency_ms: 120 } },
  { config: "E: Hybrid (Static RRF)", dataset: "scifact", metrics: { ndcg_at_10: 0.7395, mrr: 0.7110, recall_at_10: 0.8924, map_at_10: 0.6641, precision_at_10: 0.0983, mean_latency_ms: 680 } },
  { config: "G: Full Pipeline", dataset: "scifact", metrics: { ndcg_at_10: 0.7525, mrr: 0.7211, recall_at_10: 0.8924, map_at_10: 0.6756, precision_at_10: 0.0987, mean_latency_ms: 1580 } },
  { config: "H: Vec→Rerank", dataset: "scifact", metrics: { ndcg_at_10: 0.7278, mrr: 0.6985, recall_at_10: 0.8924, map_at_10: 0.6523, precision_at_10: 0.0977, mean_latency_ms: 1650 } },
  { config: "RRF-A: k=60", dataset: "scifact", metrics: { ndcg_at_10: 0.7797, mrr: 0.7511, recall_at_10: 0.8924, map_at_10: 0.7034, precision_at_10: 0.1013, mean_latency_ms: 1540 } },
  { config: "RRF-D: k=20", dataset: "scifact", metrics: { ndcg_at_10: 0.7798, mrr: 0.7514, recall_at_10: 0.8924, map_at_10: 0.7036, precision_at_10: 0.1013, mean_latency_ms: 1452 } },
  { config: "GATE-A: Hard", dataset: "scifact", metrics: { ndcg_at_10: 0.7802, mrr: 0.7455, recall_at_10: 0.9137, map_at_10: 0.7042, precision_at_10: 0.1027, mean_latency_ms: 732 } },
  { config: "GATE-D: Soft+RRF", dataset: "scifact", metrics: { ndcg_at_10: 0.7803, mrr: 0.7525, recall_at_10: 0.8924, map_at_10: 0.7044, precision_at_10: 0.1013, mean_latency_ms: 1413 } },
];

function genNdcgBarChart(results: BeirResult[]): string {
  const sorted = [...results].sort((a, b) => b.metrics.ndcg_at_10 - a.metrics.ndcg_at_10);
  const barHeight = 40;
  const gap = 8;
  const labelWidth = 200;
  const chartWidth = 800;
  const svgHeight = 100 + sorted.length * (barHeight + gap);
  const maxVal = Math.max(...sorted.map((r) => r.metrics.ndcg_at_10));

  const colors: Record<string, string> = {
    "GATE-A": C.accent,
    "GATE-D": C.accent,
    "RRF-A": C.primary,
    "RRF-D": C.primary,
    "A:": C.warning,
    "B:": C.muted,
    default: C.text,
  };

  const getColor = (name: string) => {
    for (const [prefix, color] of Object.entries(colors)) {
      if (name.startsWith(prefix)) return color;
    }
    return colors.default;
  };

  const bars = sorted
    .map((r, i) => {
      const y = 80 + i * (barHeight + gap);
      const w = (r.metrics.ndcg_at_10 / maxVal) * (chartWidth - labelWidth - 80);
      const color = getColor(r.config);
      const isGate = r.config.startsWith("GATE");
      return `
    <text x="${labelWidth - 10}" y="${y + barHeight / 2 + 4}" text-anchor="end" fill="${C.text}" font-size="13" font-weight="${isGate ? 700 : 400}">${r.config}</text>
    <rect x="${labelWidth}" y="${y}" width="${w}" height="${barHeight}" rx="4" fill="${color}" opacity="${isGate ? 1 : 0.7}"/>
    <text x="${labelWidth + w + 8}" y="${y + barHeight / 2 + 4}" fill="${color}" font-size="13" font-weight="600">${r.metrics.ndcg_at_10.toFixed(4)}</text>
    ${isGate ? `<text x="${labelWidth + w + 80}" y="${y + barHeight / 2 + 4}" fill="${C.accent}" font-size="11">★ PRODUCTION</text>` : ""}`;
    })
    .join("");

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 ${svgHeight}" width="1200" height="${svgHeight}" role="img" aria-label="memory-spark chart">
  <defs><style>text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }</style></defs>
  <rect width="1200" height="${svgHeight}" fill="${C.bg}"/>
  <rect x="10" y="10" width="1180" height="${svgHeight - 20}" rx="12" fill="${C.card}" stroke="${C.grid}"/>
  <text x="40" y="48" fill="${C.text}" font-size="24" font-weight="700">BEIR SciFact — NDCG@10 by Configuration</text>
  <text x="40" y="70" fill="${C.muted}" font-size="14">Phase 12: RRF + Dynamic Reranker Gate (300 queries)</text>
  ${bars}
</svg>`;
}

function genLatencyChart(results: BeirResult[]): string {
  const sorted = [...results].sort((a, b) => a.metrics.mean_latency_ms - b.metrics.mean_latency_ms);
  const barHeight = 40;
  const gap = 8;
  const labelWidth = 200;
  const svgHeight = 100 + sorted.length * (barHeight + gap);
  const maxLatency = Math.max(...sorted.map((r) => r.metrics.mean_latency_ms));

  const bars = sorted
    .map((r, i) => {
      const y = 80 + i * (barHeight + gap);
      const w = (r.metrics.mean_latency_ms / maxLatency) * 600;
      const isGate = r.config.startsWith("GATE-A");
      const color = r.metrics.mean_latency_ms < 800 ? C.accent : r.metrics.mean_latency_ms < 1200 ? C.warning : C.danger;
      return `
    <text x="${labelWidth - 10}" y="${y + barHeight / 2 + 4}" text-anchor="end" fill="${C.text}" font-size="13" font-weight="${isGate ? 700 : 400}">${r.config}</text>
    <rect x="${labelWidth}" y="${y}" width="${w}" height="${barHeight}" rx="4" fill="${color}" opacity="${isGate ? 1 : 0.7}"/>
    <text x="${labelWidth + w + 8}" y="${y + barHeight / 2 + 4}" fill="${color}" font-size="13" font-weight="600">${r.metrics.mean_latency_ms}ms</text>`;
    })
    .join("");

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 ${svgHeight}" width="1200" height="${svgHeight}" role="img" aria-label="memory-spark chart">
  <defs><style>text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }</style></defs>
  <rect width="1200" height="${svgHeight}" fill="${C.bg}"/>
  <rect x="10" y="10" width="1180" height="${svgHeight - 20}" rx="12" fill="${C.card}" stroke="${C.grid}"/>
  <text x="40" y="48" fill="${C.text}" font-size="24" font-weight="700">BEIR SciFact — Mean Query Latency</text>
  <text x="40" y="70" fill="${C.muted}" font-size="14">Lower is better. Green &lt; 800ms, Yellow &lt; 1200ms, Red ≥ 1200ms</text>
  ${bars}
</svg>`;
}

function genRecallChart(results: BeirResult[]): string {
  const sorted = [...results].sort((a, b) => b.metrics.recall_at_10 - a.metrics.recall_at_10);
  const barHeight = 40;
  const gap = 8;
  const labelWidth = 200;
  const svgHeight = 100 + sorted.length * (barHeight + gap);

  const bars = sorted
    .map((r, i) => {
      const y = 80 + i * (barHeight + gap);
      const w = r.metrics.recall_at_10 * 600;
      const isGate = r.config.startsWith("GATE-A");
      const color = isGate ? C.accent : C.primary;
      return `
    <text x="${labelWidth - 10}" y="${y + barHeight / 2 + 4}" text-anchor="end" fill="${C.text}" font-size="13" font-weight="${isGate ? 700 : 400}">${r.config}</text>
    <rect x="${labelWidth}" y="${y}" width="${w}" height="${barHeight}" rx="4" fill="${color}" opacity="${isGate ? 1 : 0.7}"/>
    <text x="${labelWidth + w + 8}" y="${y + barHeight / 2 + 4}" fill="${color}" font-size="13" font-weight="600">${r.metrics.recall_at_10.toFixed(4)}</text>`;
    })
    .join("");

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 ${svgHeight}" width="1200" height="${svgHeight}" role="img" aria-label="memory-spark chart">
  <defs><style>text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }</style></defs>
  <rect width="1200" height="${svgHeight}" fill="${C.bg}"/>
  <rect x="10" y="10" width="1180" height="${svgHeight - 20}" rx="12" fill="${C.card}" stroke="${C.grid}"/>
  <text x="40" y="48" fill="${C.text}" font-size="24" font-weight="700">BEIR SciFact — Recall@10</text>
  <text x="40" y="70" fill="${C.muted}" font-size="14">GATE-A achieves highest recall by skipping reranker on confident queries</text>
  ${bars}
</svg>`;
}

function genGateSkipPie(): string {
  // GATE-A stats: 234 skip (confident), 2 skip (tied), 64 fire
  const total = 300;
  const skipConfident = 234;
  const skipTied = 2;
  const fire = 64;

  const cx = 300, cy = 250, r = 180;

  // Pie segments
  const angle1 = (skipConfident / total) * 2 * Math.PI;
  const angle2 = angle1 + (skipTied / total) * 2 * Math.PI;

  const x1 = cx + r * Math.sin(angle1);
  const y1 = cy - r * Math.cos(angle1);
  const x2 = cx + r * Math.sin(angle2);
  const y2 = cy - r * Math.cos(angle2);

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500" width="800" height="500" role="img" aria-label="memory-spark chart">
  <defs><style>text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }</style></defs>
  <rect width="800" height="500" fill="${C.bg}"/>
  <rect x="10" y="10" width="780" height="480" rx="12" fill="${C.card}" stroke="${C.grid}"/>
  <text x="40" y="48" fill="${C.text}" font-size="24" font-weight="700">Reranker Gate Decisions (GATE-A)</text>
  <text x="40" y="70" fill="${C.muted}" font-size="14">SciFact, 300 queries — 78% of queries skip reranking</text>

  <!-- Skip: Confident (green) -->
  <path d="M ${cx} ${cy} L ${cx} ${cy - r} A ${r} ${r} 0 1 1 ${x1.toFixed(1)} ${y1.toFixed(1)} Z" fill="${C.accent}" opacity="0.85"/>
  <!-- Skip: Tied (yellow) -->
  <path d="M ${cx} ${cy} L ${x1.toFixed(1)} ${y1.toFixed(1)} A ${r} ${r} 0 0 1 ${x2.toFixed(1)} ${y2.toFixed(1)} Z" fill="${C.warning}" opacity="0.85"/>
  <!-- Fire (red) -->
  <path d="M ${cx} ${cy} L ${x2.toFixed(1)} ${y2.toFixed(1)} A ${r} ${r} 0 0 1 ${cx} ${cy - r} Z" fill="${C.danger}" opacity="0.85"/>

  <!-- Center label -->
  <circle cx="${cx}" cy="${cy}" r="60" fill="${C.card}"/>
  <text x="${cx}" y="${cy - 5}" text-anchor="middle" fill="${C.text}" font-size="28" font-weight="700">78%</text>
  <text x="${cx}" y="${cy + 18}" text-anchor="middle" fill="${C.muted}" font-size="14">Skipped</text>

  <!-- Legend -->
  <rect x="550" y="150" width="20" height="20" rx="3" fill="${C.accent}"/>
  <text x="580" y="166" fill="${C.text}" font-size="14">Skip: Confident (${skipConfident})</text>
  <rect x="550" y="185" width="20" height="20" rx="3" fill="${C.warning}"/>
  <text x="580" y="201" fill="${C.text}" font-size="14">Skip: Tied (${skipTied})</text>
  <rect x="550" y="220" width="20" height="20" rx="3" fill="${C.danger}"/>
  <text x="580" y="236" fill="${C.text}" font-size="14">Reranker Fired (${fire})</text>

  <text x="550" y="280" fill="${C.muted}" font-size="12">σ > 0.08 → skip (confident)</text>
  <text x="550" y="298" fill="${C.muted}" font-size="12">σ &lt; 0.02 → skip (tied)</text>
  <text x="550" y="316" fill="${C.muted}" font-size="12">0.02 ≤ σ ≤ 0.08 → fire</text>
</svg>`;
}

function genTemporalDecayChart(): string {
  // Plot temporal decay curve: w(t) = 0.8 + 0.2 * exp(-0.03 * t)
  const width = 1200, height = 500;
  const padL = 100, padR = 60, padT = 80, padB = 70;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const maxDays = 365;

  const points: string[] = [];
  for (let t = 0; t <= maxDays; t += 2) {
    const w = 0.8 + 0.2 * Math.exp(-0.03 * t);
    const x = padL + (t / maxDays) * plotW;
    const y = padT + plotH - ((w - 0.75) / 0.3) * plotH;
    points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }

  // Key data points for annotations
  const keyPoints = [
    { t: 0, w: 1.0, label: "1.00" },
    { t: 7, w: 0.8 + 0.2 * Math.exp(-0.03 * 7), label: "0.96" },
    { t: 30, w: 0.8 + 0.2 * Math.exp(-0.03 * 30), label: "0.89" },
    { t: 90, w: 0.8 + 0.2 * Math.exp(-0.03 * 90), label: "0.81" },
    { t: 365, w: 0.8, label: "0.80 (floor)" },
  ];

  const dots = keyPoints
    .map((p) => {
      const x = padL + (p.t / maxDays) * plotW;
      const y = padT + plotH - ((p.w - 0.75) / 0.3) * plotH;
      return `
    <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5" fill="${C.accent}"/>
    <text x="${x.toFixed(1)}" y="${(y - 12).toFixed(1)}" text-anchor="middle" fill="${C.accent}" font-size="12" font-weight="600">${p.label}</text>
    <text x="${x.toFixed(1)}" y="${(y + 20).toFixed(1)}" text-anchor="middle" fill="${C.muted}" font-size="11">${p.t}d</text>`;
    })
    .join("");

  // Floor line
  const floorY = padT + plotH - ((0.8 - 0.75) / 0.3) * plotH;

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="memory-spark chart">
  <defs><style>text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }</style></defs>
  <rect width="${width}" height="${height}" fill="${C.bg}"/>
  <rect x="10" y="10" width="${width - 20}" height="${height - 20}" rx="12" fill="${C.card}" stroke="${C.grid}"/>
  <text x="40" y="48" fill="${C.text}" font-size="24" font-weight="700">Temporal Decay Function</text>
  <text x="40" y="70" fill="${C.muted}" font-size="14">w(t) = 0.8 + 0.2 · exp(−0.03t) — 80% floor, gentle decay</text>

  <!-- Grid -->
  <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="${C.grid}"/>
  <line x1="${padL}" y1="${padT + plotH}" x2="${padL + plotW}" y2="${padT + plotH}" stroke="${C.grid}"/>

  <!-- Floor line -->
  <line x1="${padL}" y1="${floorY.toFixed(1)}" x2="${padL + plotW}" y2="${floorY.toFixed(1)}" stroke="${C.danger}" stroke-dasharray="6" opacity="0.5"/>
  <text x="${padL + plotW + 5}" y="${floorY.toFixed(1)}" fill="${C.danger}" font-size="12">floor: 0.8</text>

  <!-- Y-axis labels -->
  <text x="${padL - 10}" y="${padT + 5}" text-anchor="end" fill="${C.muted}" font-size="12">1.05</text>
  <text x="${padL - 10}" y="${(padT + plotH / 2).toFixed(1)}" text-anchor="end" fill="${C.muted}" font-size="12">0.90</text>
  <text x="${padL - 10}" y="${padT + plotH + 5}" text-anchor="end" fill="${C.muted}" font-size="12">0.75</text>

  <!-- X-axis label -->
  <text x="${(padL + plotW / 2).toFixed(1)}" y="${padT + plotH + 50}" text-anchor="middle" fill="${C.muted}" font-size="14">Age (days)</text>

  <!-- Curve -->
  <polyline points="${points.join(" ")}" fill="none" stroke="${C.primary}" stroke-width="3"/>

  <!-- Key points -->
  ${dots}
</svg>`;
}

async function main() {
  await fs.mkdir(FIGURES_DIR, { recursive: true });

  const charts: Array<{ name: string; svg: string }> = [
    { name: "ndcg-comparison", svg: genNdcgBarChart(SCIFACT_RESULTS) },
    { name: "latency-comparison", svg: genLatencyChart(SCIFACT_RESULTS) },
    { name: "recall-comparison", svg: genRecallChart(SCIFACT_RESULTS) },
    { name: "gate-decisions", svg: genGateSkipPie() },
    { name: "temporal-decay", svg: genTemporalDecayChart() },
  ];

  for (const chart of charts) {
    const filePath = path.join(FIGURES_DIR, `${chart.name}.svg`);
    await fs.writeFile(filePath, chart.svg, "utf-8");
    console.log(`✅ ${filePath}`);
  }

  console.log(`\nGenerated ${charts.length} SVG charts in ${FIGURES_DIR}/`);
}

main().catch(console.error);
