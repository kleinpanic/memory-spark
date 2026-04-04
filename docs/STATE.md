# memory-spark: Current State (2026-04-04)

## Overview

`memory-spark` is a semantic memory system for OpenClaw agents. It provides automatic context recall via vector search (LanceDB), full-text search (BM25), cross-encoder reranking, and diversity filtering (MMR). The system is being benchmarked against the BEIR (Benchmarking Entity Retrieval) standard, specifically the SciFact dataset.

## 2026-04-04 Addendum (Latest Session Handoff)

### Retrieval / reranker gate state (code-aligned)
- Gate logic in `src/rerank/reranker.ts` (`computeRerankerGate`) uses **top-5 vector scores** and computes spread as:
  - `σ = max(top5) - min(top5)`
- Hard-gate thresholds:
  - `σ > 0.08` → skip reranker (`hard-gate-high`, vector confident)
  - `σ < 0.02` → skip reranker (`hard-gate-low`, tied set)
  - `0.02 ≤ σ ≤ 0.08` → pass reranker (`hard-gate-pass`)
- RRF default constant remains `k=60`; MMR remains relevance-heavy (λ near 0.9 in tuned configs).

### Docs-site educational state (current)
- `docs-site/index.html` now includes:
  - long-form educational structure (What/Why/How + 15-stage centerpiece + deep dives)
  - code-aligned Dynamic Reranker Gate explainer/animation
  - pool architecture visualization with merge-node spacing adjustments
  - mobile responsiveness improvements (nav + canvas sizing guards)
- Recent docs-site stabilization commits include:
  - `bde5118`, `4d8c565`, `4e59bc7`

### Known in-repo working state notes
- `paper/memory-spark.pdf` has local modifications during docs/report iteration.
- Agent-only helper files should not be tracked as project deliverables.
- `.gitignore` now explicitly ignores agent-context doc patterns (AGENTS/SOUL/USER/etc) and `.git-rewrite/`.

### Next session priority checklist
1. Verify Gate deep-dive visuals against live GH Pages after build propagation.
2. Run one final layout pass for Gate/Pool text spacing on smallest mobile viewport.
3. Decide whether to keep `docs-site/colors.css` + `docs-site/intro-sections.html` as product assets or remove if agent-temp.
4. Reconcile/commit or discard `paper/memory-spark.pdf` local delta intentionally.

## Architecture

```
User Query
    │
    ├── [Phase 11B] Multi-Query Expansion (LLM reformulations)
    │   └── 3 alternative phrasings via Nemotron-Super
    │
    ├── Asymmetric Embedding (Instruct: prefix for queries, raw for docs)
    │   └── llama-embed-nemotron-8b (4096 dims)
    │
    ├── Vector Search (LanceDB, cosine similarity)
    │   └── Per-pool: agent_memory, agent_tools, agent_mistakes, shared_*
    │
    ├── [Optional] FTS Search (BM25, sigmoid-normalized)
    │   └── Hybrid merge via Adaptive RRF
    │
    ├── Source Weighting + Temporal Decay
    │
    ├── Cross-Encoder Reranking
    │   └── llama-nemotron-rerank-1b-v2
    │   └── Logit-space blending (α configurable)
    │   └── Query normalization (declarative → interrogative)
    │
    └── MMR Diversity (adaptive λ)
        └── Final top-K results
```

## Models Used

| Model | Role | Hosted On |
|-------|------|-----------|
| `nvidia/llama-embed-nemotron-8b` | Embedding (4096 dims) | DGX Spark (:18081) |
| `nvidia/llama-nemotron-rerank-1b-v2` | Cross-encoder reranking | DGX Spark (:18082) |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | HyDE / Multi-Query Expansion | DGX Spark (:18080) |

## BEIR Benchmark Evolution

### Phase 1-6: Foundation (2026-03-30 – 2026-03-31)
- Built BEIR evaluation pipeline (`scripts/run-beir-bench.ts`)
- Identified BM25 sigmoid saturation bug, metrics denominator bugs
- Implemented RRF, instruction-aware embedding, pipeline reordering
- **Baseline established: Config A (Vector-Only) = 0.7709 NDCG@10**

### Phase 7: Forensic Diagnostics (2026-03-31)
- Discovered Arrow Vector type mismatch → MMR was a complete no-op (NaN scores)
- RRF rank-washout: 90% of reordered docs were irrelevant
- Reranker score compression (0.83–1.0 range)
- Created diagnostic suite (`scripts/diag-*.ts`)
- Fixed Arrow `.toArray()`, weighted RRF, reranker spread guard

### Phase 8: Adaptive Remediation (2026-04-01)
- Adaptive RRF (overlap-aware weighting)
- Union Reranking (late fusion)
- Adaptive MMR (distribution-aware diversity)
- Config K (Vector + Adaptive MMR) = 0.7622

### Phase 9: Score Interpolation (2026-04-01)
- Score blending (vector + reranker scores)
- Conditional routing (skip reranker on high-confidence results)
- Logit recovery (`recoverLogit()` for sigmoid-compressed scores)
- Query normalization (declarative → interrogative for reranker)

### Phase 10A: Logit-Space Calibration (2026-04-01)
- Logit-space blending to prevent high-confidence vector matches from being smothered
- Spread guard in logit space
- Results: Still trailing baseline (best: Q at 0.7445)

### Phase 10B: Unified Reranker (2026-04-01) — BREAKTHROUGH
- **Root cause found:** Blended configs were bypassing `normalizeQueryForReranker()`
- Unified all reranker paths through a single `rerank()` call with `alphaOverride`
- **First time reranker beat baseline!**

| Config | NDCG@10 | vs Baseline |
|--------|---------|-------------|
| Q (α=0.5) | 0.7863 | +2.0% |
| S (α=0.3) | 0.7792 | +1.1% |
| T (α=0.3) | 0.7756 | +0.6% |
| A (baseline) | 0.7709 | — |

### Phase 11A: Alpha Sweep (2026-04-01)
- Swept α ∈ {0.4, 0.6, 0.7, 0.8}

| α | Config | NDCG@10 | Recall@10 |
|---|--------|---------|-----------|
| 0.4 | U | **0.7889** 🏆 | 0.9099 |
| 0.6 | V | 0.7885 | **0.9243** |
| 0.7 | W | 0.7883 | 0.9177 |
| 0.8 | X | 0.7847 | 0.9143 |

Sweet spot: α ∈ [0.4, 0.6]. Plateau is remarkably flat.

### Phase 11B: Multi-Query Expansion (2026-04-01) — IN PROGRESS
- **Problem:** 11% of relevant docs never appear in top-40 (hard retrieval ceiling)
- **Solution:** Generate 3 LLM reformulations → embed all 4 in parallel → search all 4 → union → rerank
- **Implementation:** Complete (5 git commits, 33 new tests, 436 total passing)
- **Diagnostic verified:**
  - +14.7 new docs/query average (37% candidate pool increase)
  - 10/10 expansion success rate
  - 3.0 avg reformulations, 4.2s avg expansion latency
- **Full benchmark:** Running unattended (MQ-A through MQ-D, ~2 hours)

## Current Best Results (SciFact 300)

| Metric | Value | Config |
|--------|-------|--------|
| **NDCG@10** | **0.7889** | U (Vector → Logit Blend α=0.4) |
| **Recall@10** | **0.9243** | V (Vector → Logit Blend α=0.6) |
| **Latency p95** | ~1500ms | U/V (vector + reranker) |

## Key Bugs Fixed

1. **Arrow Vector Type Mismatch (Phase 7):** LanceDB returns Apache Arrow `Vector` objects. Bracket indexing returns `undefined`, causing NaN cosine similarity. Fix: `.toArray()` in `lancedb.ts`.

2. **RRF Rank-Washout (Phase 7-8):** FTS and Vector results had <5% overlap. RRF interleaved irrelevant keyword matches. Fix: Adaptive RRF with overlap-aware weighting, eventually superseded by vector-primary architecture.

3. **Reranker Score Compression (Phase 9-10):** `llama-nemotron-rerank-1b-v2` trained on Q&A pairs but fed declarative claims. Scores compressed to 0.83-1.0 range. Fix: `normalizeQueryForReranker()` + logit recovery.

4. **Normalization Bypass (Phase 10B):** Blended configs used direct `fetch()` to reranker, skipping query normalization. Fix: Unified all paths through `reranker.rerank()` with `alphaOverride`.

5. **Metrics Bugs (Phase 1-6):** `Precision@k` used result set length instead of k as denominator. `MAP@k` had similar issues. Both fixed in `src/eval/metrics.ts`.

## File Structure

```
src/
├── auto/recall.ts        — Main recall pipeline (hook: before_prompt_build)
├── query/expander.ts     — Phase 11B: Multi-query expansion module
├── embed/provider.ts     — Asymmetric embedding (query vs document)
├── embed/queue.ts        — Embedding queue with concurrency control
├── rerank/reranker.ts    — Cross-encoder reranking + logit blending
├── storage/lancedb.ts    — LanceDB backend (vector + FTS)
├── hyde/generator.ts     — Hypothetical Document Embeddings
├── eval/metrics.ts       — NDCG, MAP, Recall, Precision calculators
├── config.ts             — Configuration schema + defaults
└── security.ts           — Prompt injection filtering

scripts/
├── run-beir-bench.ts     — BEIR benchmark runner (Configs A-X, MQ-A-D)
├── diag-multi-query.ts   — Phase 11B diagnostic (10-query verification)
├── diag-full.ts          — Forensic diagnostic (per-query pipeline traces)
├── diag-arrow.ts         — Arrow vector type verification
├── diag-mmr.ts           — MMR stage diagnostic
├── diag-rrf.ts           — RRF stage diagnostic
└── diag-vectors.ts       — Vector search diagnostic

tests/
├── expander.test.ts      — Multi-query expansion (33 tests)
├── integration.test.ts   — End-to-end pipeline tests
├── metrics.test.ts       — Metric calculation tests
└── ... (7 test files, 436 total tests)

evaluation/
├── beir-datasets/        — SciFact, FiQA, NFCorpus datasets
└── results/              — Benchmark result logs + JSON telemetry

docs/
├── STATE.md              — This file
├── PLAN-phase7-remediation.md
├── PLAN-phase11-next-improvements.md
├── PLAN-phase11b-multi-query-expansion.md
└── CHANGELOG.md          — Detailed change log
```

## Git Branches

| Branch | Status | Description |
|--------|--------|-------------|
| `main` | Stable | Last known good before BEIR work |
| `fix/phase7-pipeline-bugs` | Merged → phase10b | Phase 7 fixes |
| `fix/phase10b-unified-reranker` | Active parent | Phase 10B + 11A |
| `feat/phase11b-multi-query` | **Active (HEAD)** | Multi-query expansion |

## What's Left

### Immediate (Phase 11B completion)
- [ ] Full benchmark results (MQ-A through MQ-D) — running unattended
- [ ] Analyze results, pick winning config
- [ ] Merge to main if results are positive
- [ ] Update production config with optimal α and expansion settings

### Phase 11C: HyDE with Fast Model (task `8999599f`)
- HyDE timed out in all previous benchmarks (Nemotron-Super too slow at ~10.5s)
- Plan: Use HyDE as ADDITIVE recall (not replacement) with faster model
- Options: Aggressive Nemotron settings, lighter Ollama model, or combine with multi-query

### Phase 12: Multi-Dataset Validation
- Current results are SciFact-only (300 queries, scientific claims)
- Need to validate on FiQA (financial Q&A) and NFCorpus (medical)
- Different datasets stress different pipeline components

### Phase 13: Production Hardening
- Cache reformulations for repeated/similar queries
- Latency optimization (current MQ adds ~4s per query)
- Graceful degradation under Spark load
- Integration tests with OpenClaw's live recall hook

### Phase 14: Advanced Strategies
- Learned sparse retrieval (SPLADE-like)
- Document expansion (enrich stored docs with LLM-generated queries)
- Adaptive pipeline routing (easy queries skip expensive stages)
