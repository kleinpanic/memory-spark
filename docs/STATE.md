# memory-spark: Current State (2026-04-04)

## TL;DR
`memory-spark` is in production (v0.4.0 codebase) with a retrieval pipeline centered on:
- vector retrieval + optional FTS
- rank-based fusion (RRF)
- dynamic reranker gating
- late MMR diversity
- parent-child context expansion

This file is the **operational handoff** for the next session (not a historical log).

---

## 1) Repository Snapshot

- **Branch:** `main`
- **Package version:** `0.4.0` (`package.json`)
- **Working-tree note:** `paper/memory-spark.pdf` is locally modified during docs iteration (intentional review needed)
- **Untracked docs-site artifacts currently present:** `docs-site/colors.css`, `docs-site/intro-sections.html` (decide keep vs remove)

---

## 2) Verified Build / Test State

Latest local validation run during this audit:
- **Typecheck:** pass
- **Lint:** pass (warnings only, no lint errors)
- **Tests:** **697 passing** across current test suite

Interpretation:
- Core code is stable at compile + test level.
- Remaining risk is mainly **docs drift / architectural clarity / benchmark protocol completeness**, not obvious compile failures.

---

## 3) Runtime Defaults (Source of Truth: `src/config.ts`)

## Core model/service endpoints

| Service | Default endpoint | Purpose |
|---|---|---|
| Embedding | `http://<sparkHost>:18091/v1` | `llama-embed-nemotron-8b` embeddings |
| Reranker | `http://<sparkHost>:18096/v1` | `llama-nemotron-rerank-1b-v2` cross-encoder |
| OCR | `http://<sparkHost>:18097` | OCR service endpoint |
| GLM-OCR | `http://<sparkHost>:18080/v1` | vLLM/OpenAI-compatible OCR model endpoint |
| NER | `http://<sparkHost>:18112` | entity tagging |
| Zero-shot | `http://<sparkHost>:18113` | classification |
| Summarizer | `http://<sparkHost>:18110` | summarization endpoint |
| STT | `http://<sparkHost>:18094` | speech-to-text |

### Important guardrail
If any document conflicts with these ports, **`src/config.ts` wins**.

---

## 4) Retrieval Pipeline (Current Behavior)

High-level active order in `src/auto/recall.ts` + `src/rerank/reranker.ts`:

1. Query cleaning
2. Optional HyDE generation/embedding path
3. Optional multi-query expansion
4. Embedding
5. Pool searches (agent + shared pools)
6. Hybrid merge (`hybridMerge`, RRF-based)
7. Dedup + source weighting + temporal decay + normalization
8. Dynamic reranker gate
9. Reranking (when gate allows)
10. MMR diversity
11. Parent-child expansion
12. LCM overlap filtering + security filtering + token budgeting

### Gate behavior (hard mode)
`computeRerankerGate()` uses top-5 vector candidate spread:
- `σ = max(top5) - min(top5)`
- `σ > 0.08` → skip reranker (`hard-gate-high`)
- `σ < 0.02` → skip reranker (`hard-gate-low`)
- `0.02 ≤ σ ≤ 0.08` → run reranker (`hard-gate-pass`)

### Fusion / diversity constants
- **RRF k default:** `60`
- **MMR λ default (config):** `0.9`
- **Rerank candidate cap:** driven by `topN` defaults in config/reranker flow

---

## 5) Known Documentation Drift (Now corrected here)

Prior `STATE.md` had stale info (fixed by this rewrite), including:
- wrong embed/rerank ports (`18081/18082`)
- stale branch/HEAD references
- stale test counts
- references to plan docs that moved to `docs/archive/`
- outdated “running unattended” language treated as present-tense state

If similar drift appears elsewhere (`ARCHITECTURE.md`, `CONFIGURATION.md`, figures), treat this as the canonical current-state baseline and reconcile docs to code.

---

## 6) Known Code-Level Risk to Address Next

### Soft-gate boundary discontinuity
In `computeRerankerGate()` soft mode, multiplier behavior around `lowThreshold` can jump unexpectedly just below 0.02 (piecewise branch behavior). Hard gate logic is correct; soft gate continuity should be smoothed.

This is not blocking basic operation, but it should be fixed before final benchmark policy lock.

---

## 7) What’s Next (Execution Sequence)

## Phase A — Fix Loop (Docs + Logic)
1. Reconcile all docs (`CONFIGURATION.md`, `ARCHITECTURE.md`, README figures/text) to current source.
2. Fix soft-gate continuity edge case.
3. Re-run full audit (typecheck/lint/tests + docs-vs-code checks).
4. Repeat until no mismatches remain (**clean pass**).

## Phase B — True OpenClaw Evaluation
After clean pass:
1. Build/finalize **golden dataset** for OpenClaw-realistic retrieval tasks.
2. Add **custom benchmark harness** focused on agent memory behaviors (not only generic BEIR retrieval).
3. Run full benchmark matrix:
   - golden dataset
   - custom benchmark
   - **all BEIR datasets** used by this project (duration-unbounded run policy)
4. Produce final report with:
   - quality metrics (NDCG/MRR/Recall/MAP/Precision where relevant)
   - latency distributions
   - per-dataset win/loss analysis
   - gate/fusion ablation summary

## Exit criterion
No release promotion unless:
- docs and source align,
- fix-loop audit is clean,
- golden/custom + all-BEIR runs are complete and reported.

---

## 8) Session Handoff Notes

Before ending the next session, always leave:
- exact commit SHA evaluated,
- benchmark command set used,
- dataset manifests,
- unresolved blockers,
- explicit “next command to run” for continuation.
