# memory-spark

## What This Is

memory-spark is a Spark-powered, scientifically-rigorous RAG memory plugin for OpenClaw agents. It provides auto-capture, auto-recall, vector+FTS hybrid retrieval, reranking, and HyDE over LanceDB — designed to replace memory-core with a measurable, benchmarked, 2026-current retrieval stack. This milestone takes it from a partially-drifted v0.4.0-in-progress to a publicly shippable v1.0 with validated claims, extensive docs, an educational website, and an expanded research paper.

## Core Value

Every claim this plugin, its docs, its website, and its research paper make must be **true, measured, and current** — no drift, no vibes, no stale SOTA citations. Scientific credibility is the product.

## Requirements

### Validated

<!-- Shipped in prior work — confirmed present in the codebase. -->

- ✓ Memory auto-capture (facts/preferences after each agent turn) — existing
- ✓ Memory auto-recall (inject memories before each agent turn) — existing
- ✓ `memory_search` + `memory_get` plugin tools — existing
- ✓ LanceDB vector storage backend with connection pooling — existing
- ✓ Hybrid vector + FTS retrieval with RRF merge — existing
- ✓ Nemotron embedding provider with dims-lock validation — existing
- ✓ Reranker stage (llama-nemotron-rerank-1b-v2) — existing
- ✓ HyDE query expansion (configurable) — existing
- ✓ MMR diversity stage — existing
- ✓ Embed cache + queue with circuit breaker — existing
- ✓ Chunker with PDF/DOCX/Markdown parsers — existing
- ✓ Classification: zero-shot + NER + quality + heuristic — existing
- ✓ Session JSONL ingestion + workspace file watcher — existing
- ✓ Security module + migration from memory-core — existing
- ✓ BEIR benchmark harness (partial — runner bugs present) — existing
- ✓ Initial research paper + technical report — existing
- ✓ Initial docs-site (HTML + colors.css + diagrams + paper.pdf) — existing

### Active

<!-- Current milestone scope. Hypotheses until shipped and validated. -->

- [ ] **Read-only recon** — truth map of codebase vs. docs vs. plans; no code changes; produces authoritative state document before anything is touched. Must verify Phase C tools already shipped, confirm docs drift, map actual audit status.
- [ ] **Privacy incident response + prevention** — **immediate fix first**: `git filter-repo` the leaked `evaluation/golden-dataset.json` from history, force-push to remove from public GitHub. Then prevention: pre-commit hook for PII patterns, gitleaks CI, canary PII seed test, `.gitignore` audit, scrub policy for all future benchmark/fixture data.
- [ ] **Critical bug fixes** — all criticals from `docs/ISSUES.md` + `docs/AUDIT-2026-04-02.md`: BM25 sigmoid saturation, Nemotron query instruction prefix, FTS WHERE filter data leakage (C1), getState recovery path (C2), MAP@k denominator (C4), dead eval code (C5), HyDE timeout/averaging, MMR recall destruction, reranker latency. Must include `manager.search()` refactor to share `runRecallPipeline` with `auto/recall.ts` — otherwise benchmarks measure a fiction.
- [ ] **High-severity audit fixes** — score clamping (H1), unguarded embed calls (H2), silent capture errors (H3), test coverage gaps on capture/queue/dims-lock/classifiers (H4-H8), ~20 `return bool` → `expect()` test conversions.
- [ ] **Test harness restoration** — revive `<external>/openclaw-plugin-test/` as reproducible test harness, fix C3 config mismatch, probe-then-skip integration suite, `corpus-lock.json` enforcement, agent-isolation integration tests on every read path, ESLint `no-restricted-imports` boundary rule for `openclaw/plugin-sdk` in `src/**`.
- [ ] **Logic upgrades** — swap HyDE LLM Nemotron-Super-120B → Nemotron-Mini-4B (frees 52GB VRAM, kills 100% HyDE timeout rate), retire EasyOCR, dedicate GLM-OCR to port 18081, parallelize per-pool search in recall.ts, implement tool-calling injection feature (content_type="tool" pool per `docs/RESEARCH-TOOLS-INJECTION-2026.md`). *Optional:* Qwen3-VL-Embedding-8B / Qwen3-VL-Reranker-2B swap if VL capabilities become wanted — not required for v1.0.
- [ ] **Golden dataset generation** — use Nemotron-Super-3-122B on NVIDIA DGX Spark to generate QA golden dataset from scrubbed/synthetic OpenClaw agent data; commit only scrubbed output; canary PII test must pass before commit.
- [ ] **LongMemEval benchmark** — run the 2026 standard agent-memory benchmark against `testDbLongMemEval`; the paper's headline scientific claim (puts memory-spark on the Zep / Mem0 / Supermemory / OMEGA / TiMem scoreboard).
- [ ] **Full BEIR benchmarks** — SciFact, FiQA, NFCorpus against `testDbBEIR`; separate LanceDB dir; `corpus-lock.json` enforced.
- [ ] **Custom OCMemory golden dataset benchmark** — supplementary eval on user's specific workload; proves plugin works for agent memory, not just text retrieval.
- [ ] **2026 SOTA research validation** — verify every claim in `docs/RESEARCH-SOTA-2026.md`, `RESEARCH-SOTA-2026-VALIDATED.md`, `RESEARCH-TOOLS-INJECTION-2026.md` against current 2026 literature; kill stale citations (e.g., "voyage-3-large leads MTEB by 9.74%" is 2024 framing); correct the HyDE conclusion (was wrongly attributed).
- [ ] **Documentation overhaul** — bring every doc in sync with reality (README, ARCHITECTURE, CONFIGURATION, TOOLS, DEPLOYMENT, EVALUATION, TECHNICAL-REPORT, KNOWN-ISSUES); fix port numbers (docs say 18081/18098, code uses 18091/18096), version mismatch (0.4.0/0.1.0), coverage badge (claims 91% real ~35%), drifted CONFIGURATION.md (missing Phase 8-12 features); add model attribution section crediting Nemotron embed/rerank/HyDE, GLM-OCR with their licenses; generate numerical claims from `src/config.ts` and `evaluation/results/*.json` via `scripts/gen-docs.ts` rather than hand-editing (prevents future drift).
- [ ] **Sexy educational website** — evolve `docs-site/` into a full site that explains RAG internals, shows benchmark results with charts, embeds the paper, teaches readers the architecture; every public numerical claim must link to source (benchmark JSON / paper section / code); CLAIMS.md audit before ship.
- [ ] **Research paper expansion** — add LongMemEval methodology + results, BEIR results, custom golden set results, validated 2026 SOTA comparison, honest limitations section (no bitemporal modeling, no LLM consolidation, no hierarchical reflection — cite Zep/Mem0/Generative Agents for alternatives), bootstrap confidence intervals on all headline numbers. arXiv target quality bar.
- [ ] **v1.0 release** — GitHub tag, release notes, green CI, privacy re-audit gate (one more gitleaks + canary scan + CLAIMS.md walkthrough) before cutting the tag.

### Out of Scope

<!-- Explicit exclusions for this milestone, with reasoning. -->

- **Hardware partition / multi-GPU deployment** (Spark v2 Phase D) — deferred; requires new hardware planning, not needed for v1.0 public release
- **Federated memory** — future direction noted in TECHNICAL-REPORT; out of scope for a single-user RAG plugin
- **Multi-language memory** — deferred; current embedder is English-optimized
- **Active learning loop** — deferred; no user signal infrastructure in place yet
- **Video / audio memory** — multimodal embeddings land, but only for text+diagrams+tables; no A/V
- **RAGAS integration** — stretch goal only; drop if it threatens timeline; the golden dataset + BEIR already provides solid evaluation
- **Real-time collaborative memory across agents** — out of scope; memory is per-agent
- **Fine-tuning the embedder or reranker** — use off-the-shelf 2026-current models; fine-tuning is future work

## Context

**Project nature:** Brownfield. Substantial existing code (v0.4.0 in package.json, v0.1.0 in plugin manifest — a version mismatch we'll fix), multiple detailed internal plans, a partial audit, and an initial research paper + website. The code has drifted ahead of the docs in some areas and behind the plans in others.

**Existing planning documents (all in `docs/`):**
- `PLAN-spark-v2-architecture.md` — the LLM/embedder/reranker migration
- `PLAN-v040-release.md` — the 7-phase release plan (A-G)
- `PLAN-phase13.md` — pipeline hardening items
- `ISSUES.md` + `AUDIT-2026-04-02.md` — critical + high-severity findings
- `RESEARCH-SOTA-2026.md` + `RESEARCH-SOTA-2026-VALIDATED.md` — SOTA analysis
- `RESEARCH-TOOLS-INJECTION-2026.md` — tool retrieval research
- `TECHNICAL-REPORT.md` — implementation details
- `memory/2026-03-29-benchmark-plan.md` — benchmark execution plan

**Infrastructure:**
- **LanceDB** for vector + FTS storage
- **NVIDIA DGX Spark** with Nemotron-Super-3-122B (for golden dataset generation and research validation)
- **Docker test environment** at `<external>/openclaw-plugin-test/` (still present as of 2026-04-04; structural config mismatch to fix)
- **GPU-backed LLM/embedder/reranker services** on dedicated ports

**Known drift (pre-recon):**
- Version mismatch: package.json says 0.4.0, plugin manifest says 0.1.0
- Port numbers wrong in README, paper, CONFIGURATION.md (docs say 18081/18098, code uses 18091/18096)
- CONFIGURATION.md missing Phase 8-12 features
- Coverage badge claims 91%, actual ~35%
- ~20 unit tests use `return bool` instead of `expect()` — silent passes
- Benchmark runner has corrupt hybrid merge inputs

**User context:** Solo developer (Klein). This project is personally important and headed for public release. Quality bar is high across all dimensions — no shortcuts on rigor, privacy, or docs.

## Constraints

- **Privacy**: No personal OpenClaw data may enter the repo, `.planning/`, fixtures, benchmark corpora, or any public artifact — *ever*. Golden dataset generation must operate on scrubbed/synthetic inputs; outputs must be scrubbed before commit. This is load-bearing.
- **Scientific soundness**: Every quantitative claim in docs/paper/website/README must be backed by a measurement or citation. No hand-waving.
- **2026-current**: SOTA claims must be verified against 2026 literature (not pre-2024 training data). Use Context7 / web search for verification.
- **Git tracking**: Every step committed. Atomic commits per phase and plan. Planning docs live in git.
- **Existing tech stack**: TypeScript + Node, LanceDB, vitest, Nemotron embedder, llama-nemotron-rerank, HyDE via LLM, chokidar watcher. Don't rewrite what works.
- **Existing infrastructure**: Use the DGX Spark for LLM work; use the existing Docker test env (fix, don't replace).
- **Read-only first**: Phase 1 must be strictly read-only. No code changes until we have a verified truth map.

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Start with read-only recon before any code changes | Codebase has drifted from docs + plans; we need a verified truth map before committing to fixes | — Pending |
| Privacy foundation phase early (before real data touches anything) | Project is heading to public release; personal OpenClaw data must never leak | — Pending |
| Fix critical bugs BEFORE building new features | 11 critical audit items undermine every benchmark result; building on broken foundations wastes effort | — Pending |
| Generate golden dataset with Nemotron-Super-3-122B on DGX Spark | Highest-quality LLM available locally for QA pair generation; avoids cloud API costs and data egress | — Pending |
| Three separate benchmark databases (testDbLongMemEval, testDbBEIR, testDbOCMemory) | Per architecture research: corpus mixing contaminates BM25 IDF, IVF_PQ tuning, and reranker score distributions. Each benchmark gets its own LanceDB dir with `corpus-lock.json` enforcement. | — Pending |
| LongMemEval is the headline agent-memory benchmark | 2026 standard (Zep/Mem0/Supermemory/OMEGA/TiMem published scores). Custom OCMemory golden set is supplementary showing user's specific workload. | — Pending |
| Keep current Nemotron embedder + reranker stack | Non-commercial licenses only bind weight redistribution; memory-spark calls models as HTTP services and ships no weights. Attribution added in README. Qwen3-VL stack swap deferred to optional future upgrade. | — Pending |
| Privacy: `git filter-repo` the existing leak immediately | `evaluation/golden-dataset.json` already contains real PII on public GitHub. Rewrite history only (no contact rotation — treat as already scraped/forked). | — Pending |
| Evolve existing `docs-site/` rather than rebuild | Assets (colors.css, diagrams, paper.pdf) already exist; rebuild would waste work | — Pending |
| Defer RAGAS integration to stretch goal | LongMemEval + BEIR + custom golden set provides sufficient evaluation rigor for v1.0 arXiv paper; RAGAS adds scope risk | — Pending |
| Target venue: arXiv | Independent researcher, fast publication, no peer-review gate. Rigor bar: bootstrap CI on headline numbers, honest limitations section, not full SIGIR-grade statistical apparatus. | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-09 after initialization*
