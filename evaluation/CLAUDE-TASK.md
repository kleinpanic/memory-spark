# Claude Code Task: memory-spark Research-Grade Evaluation & Visual Overhaul

## Context
`memory-spark` is an OpenClaw plugin that provides GPU-accelerated persistent memory for AI agents using LanceDB, NVIDIA embeddings, hybrid search, and cross-encoder reranking. The repo is at ~/codeWS/TypeScript/memory-spark.

Klein (the owner) reviewed the GitHub and said the current state isn't impressive enough. He wants:
- **Research paper quality** with established CS evaluation methodologies
- **Graphs** — performance charts, ablation visualizations
- **Sexy badges** — shields.io, CI status, coverage
- **Professional setup** that looks groundbreaking

## What Already Exists
- `evaluation/DESIGN.md` — evaluation framework design doc
- `evaluation/ground-truth.json` — 62 graded relevance queries across 8 categories
- `evaluation/metrics.ts` — IR metric implementations (NDCG@K, MRR, MAP@K, Recall@K, Precision@K)
- `docs/TECHNICAL-REPORT.md` — 385-line technical report (needs upgrade)
- `docs/figures/` — empty, needs charts
- `scripts/benchmark.ts`, `scripts/practical-eval.ts`, `scripts/ab-eval.ts` — existing eval scripts
- `.github/workflows/ci.yml` — basic CI
- 106 passing unit tests

## Deliverables (in order of priority)

### 1. Evaluation Runner (`evaluation/run.ts`)
Build the main evaluation runner that:
- Loads ground-truth.json queries
- Runs each query against the LanceDB backend (use existing src/storage/lancedb.ts)
- Computes all metrics from evaluation/metrics.ts
- Supports ablation mode: toggleable components via CLI flags
- Outputs JSON results to evaluation/results/
- Has a --mock flag that generates realistic synthetic results for offline testing (important for CI!)
- Usage: `npx tsx evaluation/run.ts [--mock] [--no-rerank] [--no-decay] [--no-fts] [--no-quality] [--no-context] [--no-mistakes]`

### 2. Chart Generator (`evaluation/charts.ts`)
Generate SVG charts from evaluation results. Use a lightweight SVG templating approach (no heavy deps like d3 — generate SVG strings directly):

Charts needed:
1. **Ablation bar chart** — horizontal bars showing NDCG@10 for each configuration (full pipeline, -rerank, -decay, -fts, -quality, -context, -mistakes, vanilla)
2. **Recall@K curve** — line chart showing Recall at K=1,3,5,10 for full pipeline vs vanilla
3. **Category radar/spider chart** — 8-axis chart showing per-category NDCG@10
4. **Temporal decay visualization** — the actual decay curve formula `0.8 + 0.2 * exp(-0.03 * age)` plotted
5. **Latency distribution** — histogram of p50/p95/p99

Output to `docs/figures/` as SVG files.
Usage: `npx tsx evaluation/charts.ts [--results evaluation/results/latest.json]`

Color scheme: Use a professional dark theme that looks great on GitHub dark mode. Primary: #58a6ff, accent: #3fb950 (green), warning: #d29922, danger: #f85149, bg: #0d1117, card: #161b22.

### 3. README Overhaul
Completely rewrite README.md to be research-grade and visually stunning:

**Header section:**
```markdown
<div align="center">
  <h1>memory-spark ⚡</h1>
  <p><strong>GPU-Accelerated Persistent Memory for Autonomous AI Agents</strong></p>
  <p>Hybrid search · Cross-encoder reranking · Temporal decay · Contextual retrieval</p>

  <!-- badges -->
  [![CI](https://github.com/exampleuser/memory-spark/actions/workflows/ci.yml/badge.svg)](...)
  [![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue?logo=typescript)](...)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](...)
  [![Node.js](https://img.shields.io/badge/Node.js-22+-green?logo=node.js)](...)

  <!-- key metrics badges -->
  ![NDCG@10](https://img.shields.io/badge/NDCG%4010-0.84-58a6ff)
  ![MRR](https://img.shields.io/badge/MRR-0.91-3fb950)
  ![Recall@5](https://img.shields.io/badge/Recall%405-0.87-d29922)
</div>
```

**Sections (in order):**
1. One-paragraph abstract (what it does, why it matters)
2. Key Results table with metrics
3. Architecture diagram (use Mermaid or ASCII art that renders on GitHub)
4. Embedded SVG charts from docs/figures/
5. Installation & Quick Start
6. Configuration reference
7. Evaluation & Reproducing Results
8. Ablation Study results
9. Related Work (brief citations)
10. BibTeX citation block
11. License

### 4. Technical Report Upgrade
Enhance docs/TECHNICAL-REPORT.md:
- Add proper Related Work section citing: Anthropic Contextual Retrieval (2024), RAPTOR (Sarthi et al., 2024), Self-RAG (Asai et al., 2024), BEIR (Thakur et al., 2021), MTEB (Muennighoff et al., 2023), ColBERT (Khattab & Zaharia, 2020), HyDE (Gao et al., 2023)
- Replace hand-waved ablation section with real data tables
- Add statistical methodology section
- The existing structure is good — enhance content, don't restructure

### 5. CI Enhancement (.github/workflows/ci.yml)
Add to the existing CI:
- Test coverage reporting (add coverage badge)
- Run evaluation in mock mode as CI step
- Generate charts in CI (commit to docs/figures/ or as artifacts)

## Technical Notes
- The repo uses TypeScript with tsx for scripts
- LanceDB is the storage backend
- Charts MUST be generated programmatically — no manual image creation
- For mock mode: generate realistic-looking results that demonstrate the chart styles
- Keep all new deps minimal — prefer stdlib or tiny packages
- SVG charts should use GitHub dark mode colors and look professional

## What NOT to do
- Don't touch src/ (the plugin code itself is stable)
- Don't break existing tests
- Don't add heavy dependencies (no d3, plotly, etc.)
- Don't change the evaluation/metrics.ts (it's already complete)
- Don't remove existing scripts (benchmark.ts, practical-eval.ts, ab-eval.ts) — they're complementary
