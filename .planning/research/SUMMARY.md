# Project Research Summary

**Project:** memory-spark v1.0
**Domain:** Production-grade agentic RAG memory plugin (brownfield TypeScript/Node, LanceDB + GPU Spark services, OpenClaw plugin SDK)
**Researched:** 2026-04-09
**Confidence:** HIGH

---

## CRITICAL FINDINGS — Read Before Anything Else

Five research findings that materially change the v1.0 scope vs. what PROJECT.md originally described. The roadmapper must incorporate these as hard constraints.

### 1. License Blocker — Stack Swap is v1.0 REQUIRED, Not Optional

Both current GPU models are non-commercial and **cannot ship in a public v1.0**:

- `nvidia/llama-embed-nemotron-8b` — `customized-nscl-v1` + Llama 3.1 community license. Non-commercial.
- `nvidia/llama-nemotron-rerank-1b-v2` — same NVIDIA non-commercial license class.

**Required replacements (Apache 2.0, verified from HuggingFace model cards):**
- Embedder: `Qwen/Qwen3-VL-Embedding-8B` — Apache 2.0, MMEB-V2 77.9 (SOTA open-source), MRL 64-4096 dims, multimodal text+image, vLLM `runner=pooling` drop-in.
- Reranker: `Qwen/Qwen3-VL-Reranker-2B` — Apache 2.0, designed as a two-stage pipeline with the embedder (arXiv:2601.04720), fixes the documented 58%-at-0.999 score saturation bug.

This elevates the Spark v2 migration (PLAN-spark-v2-architecture.md M3/M4) from "Phase B nice-to-have" to **v1.0 critical path**. The stack swap must precede benchmarks — you cannot publish benchmark numbers on models you cannot legally ship.

The HyDE LLM swap (Nemotron-Super-120B → small fast model) is separately required for correctness (100% timeout rate in BEIR runs). Recommended: `Qwen/Qwen3-4B-Instruct` (Apache 2.0) or `microsoft/Phi-4-mini-instruct` (MIT). The model is config-only; the technique (HyDE) stays.

### 2. Stale Plan Discovery — 18 Tools Already Shipped

`docs/PLAN-v040-release.md` Phase C lists "expand 9 → 14 tools" as TODO. It is wrong. `src/index.ts` already registers **18 tools** as of 2026-04-09, including all 5 Phase C tools plus additional ones not in the plan:

```
memory_search, memory_get, memory_store, memory_forget, memory_reference_search,
memory_index_status, memory_forget_by_path, memory_inspect, memory_reindex,
memory_mistakes_search, memory_mistakes_store, memory_rules_store, memory_rules_search,
memory_recall_debug, memory_bulk_ingest, memory_temporal, memory_related, memory_gate_status
```

**Roadmap reframe:** v1.0 tool-surface scope is **validation + testing + documentation + benchmarks** for the 18 existing tools, not implementing new ones. PLAN-v040-release.md must be updated to reflect reality.

### 3. PII Leak Already Occurred — Incident Response Required

`evaluation/golden-dataset.json` is already on public GitHub containing: real emails (`user1@example.com`, `user2@example.com`, `user3@example.com`, `noreply@example.com`), WireGuard subnet `10.x.x.x`, LAN `10.0.0.x`, real family names ("Alice Example"), "Exampleville, CA" in ~13 docs, plus full production AGENTS.md / MEMORY.md / HEARTBEAT.md / TOOLS.md / cron schedules.

The privacy phase is not just "prevent future leaks." It requires incident response:

1. **Immediate decision:** `git filter-repo` to rewrite history, or treat the tokens as burned and document the disclosure.
2. **Pre-commit hook** that blocks staging files matching known PII patterns (email regex, `10.x.x`, `10.x.x`, `user`, `example.edu`, `Exampleville`).
3. **Canary test:** seed 20 synthetic PII tokens into the input corpus, run the scrubber, assert zero canaries survive. This proves the scrubber catches the classes we care about.
4. **`gitleaks` + custom pattern CI scan** on every push to main.
5. **Double-scrub pipeline for golden dataset generation:** scrub corpus BEFORE Nemotron-Super sees it (LLMs paraphrase source PII into generated questions), then scrub Nemotron's output again.

This must be the first code-touching phase. Nothing else runs on real data until the privacy foundation is in place and the incident is resolved.

### 4. LongMemEval Replaces "OCMemory" as the Agent-Memory Benchmark

The 2026 standard benchmark for agent memory systems is **LongMemEval** (arXiv:2410.10813), not the ambiguous "OCMemory" label used in PROJECT.md. Published scores from the 2026 ecosystem:

| System | LongMemEval Score |
|--------|------------------|
| Mem0 | 49.0% |
| Zep | 63.8% |
| TiMem | 76.88% |
| EverMemOS | 83.0% |
| Supermemory | 85.4% |
| OMEGA | 95.4% |

Running LongMemEval puts memory-spark on this scoreboard. This is the single highest-value benchmark addition for paper positioning. The golden dataset generated via Nemotron-Super-3-122B should be treated as the OpenClaw-specific supplement to LongMemEval, not the primary benchmark.

### 5. Anthropic Contextual Retrieval — Highest-ROI Chunking Upgrade

Anthropic's contextual retrieval approach (2024, validated in 2026) delivers **-49% retrieval errors** by prepending a short LLM-generated context summary before embedding each chunk. Apply this to **workspace document ingestion only** (not agent memories, which are already atomic <=200-token facts). The same small HyDE LLM (Qwen3-4B-Instruct or Phi-4-mini) handles the context generation offline at ingest time — one LLM call per chunk, cacheable.

---

## Executive Summary

memory-spark is a production-grade RAG memory plugin for OpenClaw agents with a substantially complete 18-tool surface, a 13-stage recall pipeline (vector+FTS hybrid, HyDE, reranking with dynamic gate, MMR), and an in-repo BEIR evaluation harness. This milestone takes it from a drifted v0.4.0-in-progress to a publicly shippable v1.0 with validated benchmark claims, a research paper, and an educational website. The core differentiator — scientific rigor and in-repo reproducibility — is the exact thing that makes the ordering of phases non-negotiable.

Research revealed that the v1.0 scope is substantially different from what PROJECT.md describes. The work is not "implement Phase C tools + Spark v2 migration + benchmarks." The 18 tools are already shipped; the Spark v2 model swap is required for the public license (not optional); and the benchmark infrastructure has correctness bugs that would invalidate every published number. The right framing for v1.0 is: **fix what is broken, clear the legal and privacy blockers, run scientifically valid benchmarks on a correct system, and document it all**. That is a substantial and meaningful v1.0 — it just does not involve the feature-building work the stale plans describe.

The three hardest constraints that shape everything else: (1) privacy incident response must precede any data operations; (2) all critical bugs must be fixed before any benchmark number is generated for the paper; (3) the license-blocked models must be swapped before the paper or website can cite any benchmark results on the "shipping" system. These form a hard-ordered critical path. Everything else — LongMemEval, contextual retrieval, documentation overhaul, website — happens downstream of that sequence.

---

## Key Findings

### Stack — Keep / Replace / Upgrade

| Component | Action | Reason |
|-----------|--------|--------|
| LanceDB `^0.27.1` | KEEP, migrate to native FTS | Embedded disk-native, hybrid search, native FTS shipped 2025; only local-first option with this feature set |
| Nemotron-8B embedder | REPLACE with `Qwen/Qwen3-VL-Embedding-8B` | Non-commercial license blocks public release; Qwen is Apache 2.0 + MMEB-V2 SOTA + MRL + multimodal |
| Nemotron-1B reranker | REPLACE with `Qwen/Qwen3-VL-Reranker-2B` | Non-commercial license + documented 58%-at-0.999 score saturation; Qwen fixes both |
| Nemotron-Super-120B (HyDE) | DOWNGRADE to Qwen3-4B-Instruct or Phi-4-mini | 100% timeout rate in BEIR runs; HyDE needs fluency, not reasoning depth |
| HyDE technique | KEEP + FIX (default off, fix averaging bug) | Not replaced in 2026; the LLM was wrong, not the technique |
| Hybrid vector+FTS+RRF | KEEP | Anthropic 2024 validates: hybrid + reranker = -49% errors; still SOTA |
| MMR diversity | KEEP + TUNE (default lambda >= 0.7) | No 2026 replacement; fix lambda default (SciFact lost 28% at aggressive lambda) |
| EasyOCR | RETIRE | Superseded by VL embedding (embed image pages directly, no OCR step) |
| GLM-OCR | KEEP, dedicate to port 18081 | Currently shares port 18080 with HyDE LLM |
| Contextual retrieval | ADD for workspace docs | -49% retrieval errors (Anthropic 2024); offline, one LLM call per chunk |
| Tool semantic retrieval | WIRE existing partial | `content_type="tool"` exists; boost logic needs wiring |
| BEIR eval framework | FIX runner bugs | Already present; MAP denominator wrong, inputs corrupted, dead non-compiling eval code |
| LongMemEval | ADD | 2026 SOTA agent-memory benchmark; positions on same scoreboard as Zep/Mem0/Supermemory |
| RAGAS | DEFER to v1.1 | Stretch goal per PROJECT.md; golden dataset + BEIR sufficient for v1.0 |

No npm dependency changes required for the stack swap. All changes are service-side (vLLM model IDs) and config-side. Code changes limited to: `src/config.ts`, `src/embed/provider.ts`, `src/rerank/client.ts`, `src/storage/lancedb.ts` (config-driven dims), `src/ingest/parsers.ts` (remove EasyOCR branch).

### Features — Current State vs. v1.0 Scope

All table-stakes features are shipped. memory-spark meets or exceeds the feature bar of Mem0, Zep, Letta, and LangMem on every core dimension: semantic search, explicit store/delete, auto-capture, auto-recall, health endpoints, content typing, cross-agent isolation, pipeline introspection, reindex, temporal filtering.

v1.0 tool surface scope: audit, test, document, and benchmark the 18 tools that exist. Not implement new ones.

**Genuine differentiators to highlight in paper and website:**
- 13-stage recall trace (`memory_recall_debug`) — retrieval-layer observability that LangSmith/Langfuse/Arize do not provide
- Reranker gate telemetry (`memory_gate_status`) — first-class gate decision exposure; no peer has this
- Reproducible in-repo BEIR harness — no peer publishes comparable benchmarks
- Local-first with zero data egress — Mem0/Zep/Supermemory all require cloud API
- Classifier-gated capture — cheaper and more auditable than Mem0's LLM-extraction approach
- `contentType=mistake` and `contentType=rule` — coding-agent-native memory taxonomy

**Honest gaps vs. peers (document in paper, do not build for v1.0):**
- No LLM-arbitrated UPDATE/DELETE on capture (Mem0 signature) — defer to v1.1 as offline consolidation worker
- No bitemporal modeling (Zep/Graphiti) — 18-month head start, solo dev, uncatchable; cite as principled alternative
- No reflections/hierarchical consolidation (Generative Agents) — v1.1

**Anti-features — explicitly do not build:**
- Full Graphiti-style temporal knowledge graph (H-complexity, invalidates LanceDB-centric architecture)
- Cloud-hosted managed service (kills local-first differentiator)
- LLM-driven UPDATE on every capture (same latency objection that killed unthrottled HyDE)
- Agentic chunking (non-reproducible, fails scientific rigor requirement)
- ARES evaluation (requires training pipeline; scope risk)

### Architecture — Canonical Boundaries

The codebase already has the right shape. Key invariants to preserve and enforce:

- `index.ts` is the only file that imports from `openclaw/plugin-sdk` — enforce via ESLint `no-restricted-imports` on `src/**`
- Capture path and recall path are strictly independent — share only `StorageBackend`, `EmbedProvider`, `config`, `security`
- All embed calls go through `EmbedQueue` — direct `provider.embed()` bypasses circuit breaker (existing audit violation sites)
- All storage queries must pass `pools: string[]` and `userId` — missing pool filter caused C1 cross-agent data leakage
- `src/storage/pool.ts` is the single source of truth for pool routing

**One boundary that is wrong today:** `src/manager.ts` `search()` reimplements a simplified recall pipeline, duplicating `recall.ts` orchestration. Fix: extract `runRecallPipeline(query, ctx, deps)` from `recall.ts`; both callers invoke it. Without this fix, Phase 7 benchmarks measure a fiction — they would test the manager's simplified path, not the production hook-based path.

**Two physically separate LanceDB directories for benchmarks.** Mixing corpora is scientifically invalid (IVF_PQ training, BM25 IDF statistics, reranker score distributions are all corpus-sensitive). Enforce via `corpus-lock.json` alongside `dims-lock.json`.

### Critical Pitfalls by Phase

1. **Publishing benchmark numbers before fixing critical bugs** — MAP@k denominator bug inflates NFCorpus MAP ~3.8x; BM25 sigmoid saturation made reranker gate misfire; benchmark runner fed corrupt hybrid merge inputs. Hard rule: no benchmark results for the paper until P3+P4 are merged and tagged. Embed commit SHA in every result file.

2. **PII leak recurrence during golden dataset generation** — LLMs paraphrase source PII into generated questions. Double-scrub required: scrub the input corpus before Nemotron-Super sees it, then scrub the output. Canary test proves the scrubber works.

3. **Score fusion failing silently** — any `Math.min(1.0, score * weight)` without subsequent normalize can collapse the gradient and make hybrid merge return random noise while looking correct. Add `tests/score-invariants.test.ts` with per-stage assertions: scores in [0,1], monotone, no NaN/Infinity.

4. **Reranker destroying precision on agent memory domain** — cross-encoders trained on MS MARCO do not generalize to distilled agent facts. Existing audit found 41s p50 reranker latency. Report NDCG with and without reranker separately on BEIR AND on agent-memory corpora. If reranker hurts on agent memory, say so in the paper.

5. **HyDE averaging bug** — current code averages the hypothetical vector with the original query vector, destroying both signals. Correct pattern: replace the query vector with the HyDE vector, OR issue two parallel searches and RRF-merge. Fix in P3 Criticals. Default HyDE to off even after LLM swap.

6. **Cross-agent data leakage via fluent API** — LanceDB's `.where()` returns a new query object; discarding the return value silently drops the filter. C1 already fixed, but every new storage query path is a fresh regression risk. Agent isolation integration test required.

7. **Docs drift killing public credibility** — wrong port numbers in README/paper/CONFIGURATION.md, 91% coverage badge vs. 35% real, CHANGELOG PENDING markers on shipped features. Generate docs snippets, paper table numbers, and coverage badge from code at CI time.

---

## Implications for Roadmap

The dependency structure is hard. Three rules constrain all phase ordering:

- Cannot benchmark what is broken. Critical fixes precede any benchmark run.
- Cannot publish what leaks PII. Privacy foundation precedes any data operation.
- Cannot ship a non-commercial model. License-blocked models must be replaced before benchmark results can be cited in the paper as describing the shipping system.

### Phase 1: Read-Only Recon

**Rationale:** Truth map of codebase vs. docs vs. plans before any changes. The stale plan discovery (18 tools already shipped) is exactly the kind of finding that saves weeks of wasted implementation work.
**Delivers:** Authoritative state document; list of drifted docs; confirmed tool surface; confirmed port numbers; confirmed architecture boundaries; native FTS TS SDK verification.
**No code changes. No data operations.**

### Phase 2: Privacy Foundation + Incident Response

**Rationale:** PII is already on public GitHub. Every phase that follows touches data. Cannot be deferred.
**Delivers:** Decision on `git filter-repo` for history rewrite; pre-commit hook blocking known PII patterns; `evaluation/scrub.ts` with canary test passing; `gitleaks` CI on push; `.gitignore` audit covering `testDbOCMemory/`, `golden-dataset-raw.json`, all raw benchmark data dirs.
**Avoids:** Pitfall 2 (PII leak recurrence).

### Phase 3: Critical Bug Fixes

**Rationale:** 11 criticals from `docs/ISSUES.md` + `docs/AUDIT-2026-04-02.md` undermine every benchmark result. Also where the manager.search() pipeline duplication gets fixed.
**Delivers:** All C-series criticals resolved; `manager.search()` refactored to share `runRecallPipeline()` with `recall.ts`; HyDE default off + averaging bug fixed + timeout 3-5s + fallback path tested; dead eval code deleted; commit SHA stamped into benchmark result format.
**Avoids:** Pitfall 1 (bug-invalidated numbers), Pitfall 5 (HyDE), Pitfall 8 (benchmark runner corruption).

### Phase 4: License Swap + Stack Migration (Spark v2 Phase A)

**Rationale:** Cannot legally benchmark or publish the system with non-commercial models. Stack swap requires full re-index (dims may change with MRL configuration). Doing this before benchmarks avoids running them twice.
**Delivers:** `Qwen/Qwen3-VL-Embedding-8B` on port 18091; `Qwen/Qwen3-VL-Reranker-2B` on port 18096; HyDE LLM `Qwen/Qwen3-4B-Instruct` on port 18080; GLM-OCR dedicated to port 18081; EasyOCR branch removed; `dims-lock.json` + `corpus-lock.json` enforcement wired; embedder dims now config-driven; re-index run on all test DBs.
**Research flag:** Confirm Qwen3-VL-Reranker-2B latency on DGX Spark hardware before committing (on-hardware measurement needed). Fallback: `BAAI/bge-reranker-v2-m3` (Apache 2.0, text-only, ~240ms). Confirm LanceDB TS SDK native FTS API.

### Phase 5: Test Harness Restoration

**Rationale:** Safety net before benchmarks. Docker environment has config mismatch (C3). Plugin boundary lint rules prevent regression during later phases.
**Delivers:** Docker test environment working with correct config; T1 unit tests fixed (`return bool` → `expect()` in 20 tests); T2 integration tests with probe-then-skip; agent isolation integration test (agent A + agent B, assert no leakage); corpus isolation enforcement (`corpus-lock.json`); ESLint rules blocking SDK leak + fluent API filter discard.
**Avoids:** Pitfall 7 (cross-agent leakage regression).

### Phase 6: High-Severity Audit Fixes

**Rationale:** Remaining audit items affecting retrieval quality and test coverage.
**Delivers:** Score clamping removed (replace `Math.min(1.0, score * weight)` with rank-space bonus or normalized boost); `tests/score-invariants.test.ts` with per-stage assertions; `tests/capture.test.ts` (zero coverage currently); capture error logging (replace silent `catch {}`); unguarded embed calls routed through `EmbedQueue`.
**Avoids:** Pitfall 3 (score fusion), Pitfall 11 (capture garbage, silent errors).

### Phase 7: Benchmarks

**Rationale:** System is now correct, license-clear, and tested. Time to generate the numbers the paper will cite.
**Delivers:** LongMemEval run (primary SOTA positioning); BEIR SciFact/FiQA/NFCorpus run against `testDbBEIR`; golden dataset generation via Nemotron-Super-3-122B (scrubbed input + scrubbed output + canary verified); OCMemory supplement against `testDbOCMemory`; results stamped with commit SHA; bootstrap CI on every metric (95% CIs); paired tests between configs; MMR lambda sweep; HyDE-on vs HyDE-off per dataset; reranker on vs off per dataset; pytrec_eval cross-check on BEIR metrics.
**Avoids:** Pitfall 1 (stale numbers), Pitfall 4 (reranker domain mismatch documented honestly), Pitfall 6 (MMR lambda), Pitfall 10 (no statistical significance).
**Research flag:** LongMemEval requires 115k-1.5M token conversations x 500 questions; confirm DGX Spark can handle this before scheduling. LoCoMo is the fallback.

### Phase 8: SOTA Validation (parallel with Phase 7)

**Rationale:** Every claim in the existing research docs must be verified against 2026 literature before the paper cites them.
**Delivers:** Stale claims killed or corrected; `docs/RESEARCH-SOTA-2026-VALIDATED.md` updated; PLAN-spark-v2-architecture.md M3 pivot from NVIDIA VL to Qwen3-VL documented; PLAN-v040-release.md updated to reflect 18 tools shipped.

### Phase 9: Documentation Overhaul

**Rationale:** Docs depend on benchmark numbers (Phase 7) and SOTA validation (Phase 8) to be accurate.
**Delivers:** All docs synced with reality; port numbers corrected throughout; version mismatch resolved (package.json + plugin manifest + VERSION constant all equal); `scripts/gen-docs.ts` generating snippets from `src/config.ts` as source of truth; coverage badge CI-driven; CHANGELOG cleaned.
**Avoids:** Pitfall 9 (docs drift).

### Phase 10: Website

**Rationale:** Docs must be stable before website content is finalized.
**Delivers:** docs-site evolved into full educational site; 13-stage pipeline interactive visualization (top differentiator demo); benchmark results injected from JSON at build time; reranker gate skip-rate chart; paper embedded; every claim has a source link.

### Phase 11: Paper Expansion

**Rationale:** Depends on Phases 7 (benchmarks), 8 (SOTA validation), and 9 (docs stable).
**Delivers:** Golden dataset methodology section; full BEIR results with bootstrap CIs; LongMemEval results + comparison table vs Zep/Mem0/Supermemory/OMEGA/TiMem; honest limitations section (no bitemporal, no LLM UPDATE, flat temporal filter, reranker domain mismatch if found); sensitivity analysis for key hyperparameters; supplementary: golden dataset generation pipeline (not the data).
**Avoids:** Pitfall 10 (statistical significance).

### Phase 12: v1.0 Release

**Delivers:** GitHub tag; release notes; green CI; privacy audit sign-off (one more canary-seeded scan); port number grep confirms zero old ports in docs; version pins verified across all three version fields.

### Phase Ordering Rationale

The ordering is entirely dependency-driven:

- Phases 1-2 are prerequisites for everything. Cannot know what to fix without recon; cannot touch data without privacy.
- Phase 3 must precede Phase 7: bug-invalidated results would require re-running benchmarks.
- Phase 4 must precede Phase 7: non-commercial model benchmarks cannot be cited as describing the shipping system.
- Phase 5 must precede Phase 6: changing logic without a safety net is dangerous.
- Phases 7 and 8 can run in parallel.
- Phases 9-11 form the publication sequence and depend on Phases 7+8.
- Phase 12 depends on everything.

### Research Flags

**Phases needing on-hardware validation before committing:**
- **Phase 4 (stack migration):** Qwen3-VL-Reranker-2B latency on DGX Spark not yet measured. If p95 > 500ms at realistic batch sizes, fall back to `BAAI/bge-reranker-v2-m3`. Also confirm LanceDB TS SDK native FTS API.
- **Phase 7 (LongMemEval):** Compute requirements are large. Pre-estimate token budget before scheduling.

**Phases with well-documented patterns (no additional research needed):**
- Phase 1 (recon): read-only, no decisions.
- Phase 2 (privacy): PITFALLS.md documents concrete implementation steps.
- Phase 3 (critical bug fixes): all 11 criticals documented in existing audit files.
- Phase 9 (docs): mechanical sync from code as source of truth.

---

## v1.0 Scope Reframe

**What PROJECT.md originally described:**
- Read-only recon + privacy + critical fixes + high-sev fixes + tool-calling injection (new) + expand tool surface 9 to 14 tools (new) + Spark v2 migration + Docker env + golden dataset + BEIR benchmarks + OCMemory benchmark + SOTA validation + docs + website + paper + release

**What research revealed the actual scope is:**

1. Tool surface expansion is already done. 18 tools shipped. Scope is validation + docs, not implementation.
2. Spark v2 stack swap is required, not optional. License blocker promotes it to critical path before benchmarks.
3. Benchmarks must run on the post-swap, post-bugfix system. This tightens the dependency chain.
4. LongMemEval replaces the ambiguous "OCMemory" benchmark as the primary agent-memory positioning vehicle.
5. Privacy work is incident response, not just policy. The leak already happened.

The v1.0 milestone is still the same ambition: a publicly shippable, scientifically credible, well-documented v1.0. Research clarifies which work is already done, which order is non-negotiable, and which new finding (LongMemEval) makes the paper significantly stronger.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | License decisions sourced directly from HuggingFace model cards. LanceDB native FTS confirmed via official blog (TS SDK API pending Phase 1 recon verify). |
| Features | HIGH | Tool surface verified by grep against `src/index.ts` on 2026-04-09. LongMemEval scores from published benchmark paper. |
| Architecture | HIGH | Derived from existing code. Boundary violations sourced from actual audit findings. |
| Pitfalls | HIGH | Every critical pitfall is either a confirmed existing bug from the audit or a documented failure mode in published 2026 literature. |

**Overall confidence: HIGH**

### Gaps to Address During Planning

- **Qwen3-VL-Reranker-2B on-hardware latency** — MEDIUM confidence. Benchmark before committing. Fallback (bge-reranker-v2-m3) is clearly defined.
- **LanceDB TS SDK native FTS API** — Python SDK confirmed, TS SDK unverified. Flag for Phase 1 recon.
- **LongMemEval compute budget on DGX Spark** — verify feasibility before scheduling. LoCoMo is the fallback.
- **Nemotron-Mini-4B-Instruct license** — PLAN-spark-v2-architecture.md M1 references it; recommend defaulting to Qwen3-4B-Instruct (Apache 2.0) to avoid uncertainty.
- **`git filter-repo` decision** — must be made before Phase 2 can be scoped. Forks/mirrors keep old history; affected tokens should be treated as burned regardless.

---

## Sources

### Primary (HIGH confidence)

- `src/index.ts` — verified 18-tool surface by grep, 2026-04-09
- `src/manager.ts`, `src/auto/recall.ts`, `src/auto/capture.ts`, `src/storage/pool.ts`, `src/storage/backend.ts` — architecture boundaries
- `docs/ISSUES.md` + `docs/AUDIT-2026-04-02.md` — 11 criticals + high-severity findings
- `docs/ARCHITECTURE.md` — authoritative current-state architecture
- `docs/PLAN-spark-v2-architecture.md` — service migration plan (M1-M4)
- `evaluation/golden-dataset.json` — PII leak confirmed present
- [Qwen/Qwen3-VL-Embedding-8B HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) — dims, license, MMEB-V2 77.9, Apache 2.0, vLLM runner=pooling
- [Qwen/Qwen3-VL-Reranker-2B HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) — license, MMEB-v2 75.1, Apache 2.0
- [nvidia/llama-embed-nemotron-8b HuggingFace](https://huggingface.co/nvidia/llama-embed-nemotron-8b) — customized-nscl-v1 license confirmed non-commercial
- arXiv:2601.04720 — Qwen3-VL-Embedding + Qwen3-VL-Reranker unified framework
- arXiv:2603.20313 — semantic tool retrieval (97.1% hit@3, 99.6% token reduction)
- arXiv:2410.10813 — LongMemEval; Zep 63.8%, Mem0 49.0%, Supermemory 85.4%, OMEGA 95.4%, TiMem 76.88%
- Anthropic contextual retrieval cookbook — -49% retrieval errors

### Secondary (MEDIUM confidence)

- agentset.ai 2026 reranker benchmarks — latency figures (Jina 188ms, Nemotron 243ms, BGE ~240ms)
- LanceDB blog "No more Tantivy!" + WikiSearch demo — native FTS API (Python SDK confirmed)
- arXiv:2311.09476 — ARES vs RAGAS comparison (+59.3pts context relevance)
- arXiv:2309.15217 — RAGAS framework paper

### Tertiary (LOW confidence — validate before citing)

- Specific LongMemEval score for memory-spark (not yet run)
- Qwen3-VL-Reranker-2B latency on DGX Spark hardware (needs on-hardware benchmark)
- LangMem P95 search latency 59.82s (web-search sourced, not verified from primary source)

---
*Research completed: 2026-04-09*
*Ready for roadmap: yes*
