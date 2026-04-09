# Requirements: memory-spark v1.0

**Defined:** 2026-04-09
**Core Value:** A scientifically-credible, 2026-current RAG memory plugin for OpenClaw agents where every claim in docs, paper, and website is true, measured, and current.

## v1 Requirements

Requirements for the v1.0 public release. Each maps to a roadmap phase. All checkboxes must be verified before tagging `v1.0.0`.

### Recon (RECON)

- [ ] **RECON-01**: Produce `.planning/intel/TRUTH-MAP.md` documenting actual vs. documented state of every runtime component (versions, ports, config keys, tool surface, recall pipeline, audit status)
- [ ] **RECON-02**: Enumerate every pre-existing planning doc in `docs/` and mark which items are shipped, partial, or not started (replaces guesswork in later phases)
- [ ] **RECON-03**: Produce a drift-delta table (docs say X, code does Y) covering README, ARCHITECTURE, CONFIGURATION, TOOLS, DEPLOYMENT, EVALUATION, TECHNICAL-REPORT, KNOWN-ISSUES
- [ ] **RECON-04**: No files mutated during this phase — recon is strictly read-only; commit only `.planning/intel/*` and planning docs

### Privacy (PRIV)

- [ ] **PRIV-01**: Rewrite git history with `git filter-repo` to remove `evaluation/golden-dataset.json` (and any other PII-bearing files) from every reachable commit on `main`
- [ ] **PRIV-02**: Force-push the rewritten history to `origin/main` (one-time, gated by explicit user approval)
- [ ] **PRIV-03**: Install a pre-commit hook that blocks commits containing email addresses, internal IPs (10.x/192.168.x), and a configurable PII pattern list
- [ ] **PRIV-04**: Add gitleaks CI workflow that fails PRs on detected secrets or PII
- [ ] **PRIV-05**: Add a canary PII seed test that plants a known-fake PII string and asserts the pre-commit hook + gitleaks both catch it
- [ ] **PRIV-06**: Audit `.gitignore` for benchmark/fixture/LanceDB/session-log paths and commit any missing entries
- [ ] **PRIV-07**: Document a scrub policy in `docs/PRIVACY.md` for all future benchmark/golden/fixture data

### Critical Bug Fixes (BUG)

- [ ] **BUG-01**: Fix BM25 sigmoid saturation bug (scores collapse to ~1.0 at high BM25 values)
- [ ] **BUG-02**: Add Nemotron query instruction prefix to query-side embeddings (asymmetric query/document encoding)
- [ ] **BUG-03**: Fix FTS WHERE-filter data leakage (C1) — filters applied post-rank leak cross-agent docs
- [ ] **BUG-04**: Fix getState recovery path (C2) — silent swallow on state decode failure
- [ ] **BUG-05**: Fix MAP@k denominator (C4) — currently divides by retrieved count, should divide by min(|relevant|, k)
- [ ] **BUG-06**: Remove dead eval code (C5) in `evaluation/` that shadows the live harness
- [ ] **BUG-07**: Fix HyDE timeout/averaging — current code hangs per-query on Nemotron-120B and never averages hypothetical embeddings
- [ ] **BUG-08**: Fix MMR recall destruction — current λ default eliminates relevant docs
- [ ] **BUG-09**: Fix reranker latency spike — batch or stream reranker requests instead of per-doc calls
- [ ] **BUG-10**: Refactor `manager.search()` to share a single `runRecallPipeline(query, ctx, deps)` with `auto/recall.ts` — benchmarks must measure the same pipeline the agent uses
- [ ] **BUG-11**: All eleven critical items from `docs/ISSUES.md` + `docs/AUDIT-2026-04-02.md` have a regression test before they're marked fixed

### High-Severity Audit (HIGH)

- [ ] **HIGH-01**: Clamp all retrieval scores to a documented, asserted range (H1) and add invariant test
- [ ] **HIGH-02**: Guard every embed call with circuit breaker + timeout (H2) — no unguarded network calls on the hot path
- [ ] **HIGH-03**: Capture path logs errors loudly instead of swallowing them (H3)
- [ ] **HIGH-04**: Add unit tests for capture pipeline (H4)
- [ ] **HIGH-05**: Add unit tests for embed queue (H5)
- [ ] **HIGH-06**: Add unit tests for dims-lock validation (H6)
- [ ] **HIGH-07**: Add unit tests for classifier stages (zero-shot + NER + quality + heuristic) (H7)
- [ ] **HIGH-08**: Add unit tests for session JSONL ingestion (H8)
- [ ] **HIGH-09**: Convert ~20 `return bool` tests to `expect(...)` assertions so silent passes become failures

### Test Harness (HARNESS)

- [ ] **HARNESS-01**: Revive `<external>/openclaw-plugin-test/` as a reproducible integration environment (Docker build, compose, documented bring-up)
- [ ] **HARNESS-02**: Fix C3 structural config mismatch between harness and plugin runtime
- [ ] **HARNESS-03**: Integration suite probes required services and skips (not fails) when unavailable
- [ ] **HARNESS-04**: Enforce `corpus-lock.json` — benchmark dirs refuse to mutate once locked
- [ ] **HARNESS-05**: Add agent-isolation integration tests on every read path (search, recall, get, list) asserting zero cross-tenant leakage
- [ ] **HARNESS-06**: Add ESLint `no-restricted-imports` rule forbidding `openclaw/plugin-sdk` imports anywhere under `src/**` except `src/index.ts`

### Logic Upgrades (LOGIC)

- [ ] **LOGIC-01**: Swap HyDE LLM from Nemotron-Super-120B to Nemotron-Mini-4B (or equivalent small model), eliminating the 100% HyDE timeout rate and freeing ~52 GB VRAM
- [ ] **LOGIC-02**: Retire EasyOCR and route all OCR through GLM-OCR on a dedicated port
- [ ] **LOGIC-03**: Parallelize per-pool search in `auto/recall.ts` so multi-pool queries run concurrently
- [ ] **LOGIC-04**: Implement tool-calling injection feature (`content_type="tool"` pool) per `docs/RESEARCH-TOOLS-INJECTION-2026.md`
- [ ] **LOGIC-05**: All LOGIC changes land behind config flags defaulted to the new behavior, with rollback paths documented in CONFIGURATION.md

### Golden Dataset (GOLDEN)

- [ ] **GOLDEN-01**: Generate QA golden dataset using Nemotron-Super-3-122B on NVIDIA DGX Spark, running only over scrubbed/synthetic agent data
- [ ] **GOLDEN-02**: Canary PII test passes on the generated dataset before commit
- [ ] **GOLDEN-03**: Dataset committed under `evaluation/golden/ocmemory/` with generation script, seed, model version, and provenance manifest
- [ ] **GOLDEN-04**: Dataset has train/eval split and documented question-type taxonomy

### LongMemEval Benchmark (LONGMEM)

- [ ] **LONGMEM-01**: LongMemEval corpus ingested into a dedicated `testDbLongMemEval` LanceDB directory with `corpus-lock.json`
- [ ] **LONGMEM-02**: Runner produces JSON results file with per-task scores and aggregate metric
- [ ] **LONGMEM-03**: Bootstrap 95% confidence interval computed on the aggregate score
- [ ] **LONGMEM-04**: Results reproducible from a single `npm run bench:longmem` command
- [ ] **LONGMEM-05**: Results published in paper + website with source JSON linked

### BEIR Benchmarks (BEIR)

- [ ] **BEIR-01**: SciFact benchmark runs against `testDbBEIR` and emits nDCG@10 + Recall@100
- [ ] **BEIR-02**: FiQA benchmark runs against `testDbBEIR` and emits nDCG@10 + Recall@100
- [ ] **BEIR-03**: NFCorpus benchmark runs against `testDbBEIR` and emits nDCG@10 + Recall@100
- [ ] **BEIR-04**: Each benchmark has its own `corpus-lock.json` preventing cross-contamination
- [ ] **BEIR-05**: Hybrid-merge input validation — benchmark runner asserts RRF inputs are well-formed (closes the corrupt-hybrid-merge bug)
- [ ] **BEIR-06**: Results reproducible from a single `npm run bench:beir` command with seed pinning

### OCMemory Custom Benchmark (OCMEM)

- [ ] **OCMEM-01**: Custom OCMemory golden dataset benchmark runs against `testDbOCMemory` with its own `corpus-lock.json`
- [ ] **OCMEM-02**: Evaluates recall@k, precision@k, MRR on agent-memory workload
- [ ] **OCMEM-03**: Results committed as supplementary eval showing real-world agent-memory performance

### 2026 SOTA Validation (SOTA)

- [ ] **SOTA-01**: Every claim in `docs/RESEARCH-SOTA-2026.md` verified against current (2026) literature with citation
- [ ] **SOTA-02**: Every claim in `docs/RESEARCH-SOTA-2026-VALIDATED.md` re-verified, stale 2024-era framings removed
- [ ] **SOTA-03**: Every claim in `docs/RESEARCH-TOOLS-INJECTION-2026.md` verified
- [ ] **SOTA-04**: HyDE conclusion corrected (previously mis-attributed)
- [ ] **SOTA-05**: Produce `docs/CLAIMS.md` — a flat list of every quantitative claim memory-spark makes publicly, each linked to its evidence source (benchmark JSON, paper section, or code)

### Documentation Overhaul (DOCS)

- [ ] **DOCS-01**: README reflects actual runtime: correct version, correct ports, correct coverage, model attribution section crediting Nemotron embed/rerank/HyDE and GLM-OCR with licenses
- [ ] **DOCS-02**: ARCHITECTURE.md matches the actual component layout
- [ ] **DOCS-03**: CONFIGURATION.md documents Phase 8-12 features that currently exist in code but aren't in docs
- [ ] **DOCS-04**: TOOLS.md lists every shipped tool with schema and example
- [ ] **DOCS-05**: DEPLOYMENT.md matches actual deploy flow
- [ ] **DOCS-06**: EVALUATION.md matches the benchmark harness in code
- [ ] **DOCS-07**: TECHNICAL-REPORT.md updated with current numbers
- [ ] **DOCS-08**: KNOWN-ISSUES.md reflects current issue state (after BUG + HIGH phases close)
- [ ] **DOCS-09**: Version mismatch fixed — `package.json` and plugin manifest agree on `1.0.0`
- [ ] **DOCS-10**: Coverage badge reflects actual measured coverage (not the claimed 91%)
- [ ] **DOCS-11**: `scripts/gen-docs.ts` generates numerical claims from `src/config.ts` + `evaluation/results/*.json` so docs can't drift back

### Website (WEB)

- [ ] **WEB-01**: Evolve existing `docs-site/` (keep `colors.css`, diagrams, `paper.pdf`) into a full educational site
- [ ] **WEB-02**: Site explains RAG internals (capture, embed, recall, rerank, HyDE, MMR) with diagrams
- [ ] **WEB-03**: Benchmark results shown with charts rendered from committed JSON source
- [ ] **WEB-04**: Paper embedded + downloadable
- [ ] **WEB-05**: Every public numerical claim on the site links to its source (benchmark JSON / paper section / code)
- [ ] **WEB-06**: CLAIMS.md walkthrough completes successfully before the site is considered shippable
- [ ] **WEB-07**: Site deploys from `docs-site/` via the existing GitHub Pages workflow

### Research Paper (PAPER)

- [ ] **PAPER-01**: LongMemEval methodology + results section added
- [ ] **PAPER-02**: BEIR (SciFact/FiQA/NFCorpus) results section added
- [ ] **PAPER-03**: Custom OCMemory golden set results section added
- [ ] **PAPER-04**: 2026 SOTA comparison section with validated citations
- [ ] **PAPER-05**: Honest Limitations section — no bitemporal modeling, no LLM consolidation, no hierarchical reflection; cite Zep / Mem0 / Generative Agents as alternatives
- [ ] **PAPER-06**: Bootstrap 95% confidence intervals on every headline number
- [ ] **PAPER-07**: Paper compiled (`paper.pdf`) and committed to `docs-site/`

### v1.0 Release (REL)

- [ ] **REL-01**: Green CI — every suite (unit, integration, benchmark smoke, gitleaks, canary PII) passes on `main`
- [ ] **REL-02**: Privacy re-audit gate — one more gitleaks + canary PII + CLAIMS.md walkthrough before cutting the tag
- [ ] **REL-03**: GitHub release `v1.0.0` tag created with release notes linking paper, website, and benchmark results
- [ ] **REL-04**: Website + `paper.pdf` live and accessible
- [ ] **REL-05**: Post-release verification: fresh clone + `npm install` + `npm run bench:*` reproduces committed results within CI tolerance

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Hardware and Deployment

- **DEPLOY-01**: Hardware partition / multi-GPU deployment (Spark v2 Phase D)
- **DEPLOY-02**: Federated memory across agents

### Advanced Retrieval

- **RET-01**: Multi-language memory (current embedder is English-optimized)
- **RET-02**: Active learning loop using user signal
- **RET-03**: Video / audio memory
- **RET-04**: Qwen3-VL-Embedding-8B / Qwen3-VL-Reranker-2B swap (optional upgrade if VL features become desired)

### Evaluation Depth

- **EVAL-01**: RAGAS integration (stretch goal only — dropped if it threatens timeline)
- **EVAL-02**: Fine-tuning embedder or reranker on OpenClaw-style data

### Collaboration

- **COLLAB-01**: Real-time collaborative memory across agents

## Out of Scope

| Feature | Reason |
|---------|--------|
| Hardware partition / multi-GPU deployment | Requires new hardware planning; not needed for v1.0 public release |
| Federated memory | Future direction noted in TECHNICAL-REPORT; out of scope for a single-user RAG plugin |
| Multi-language memory | Current embedder is English-optimized; swap is future work |
| Active learning loop | No user signal infrastructure in place yet |
| Video / audio memory | Only text + diagrams + tables in scope |
| RAGAS integration | LongMemEval + BEIR + custom golden set already provides sufficient rigor for arXiv v1.0 |
| Real-time collaborative memory | Memory is per-agent by design |
| Fine-tuning embedder or reranker | Off-the-shelf 2026-current models are sufficient; fine-tuning is future work |
| Commercial licensing audit | Klein is independent, not shipping model weights; attribution in README covers obligations |
| Peer-reviewed venue submission (SIGIR/EMNLP) | Target is arXiv; statistical rigor bar matched to arXiv, not full peer-review apparatus |

## Traceability

Empty until roadmap creation. Populated by `gsd-roadmapper`.

| Requirement | Phase | Status |
|-------------|-------|--------|
| RECON-01 | Phase [N] | Pending |
| RECON-02 | Phase [N] | Pending |
| RECON-03 | Phase [N] | Pending |
| RECON-04 | Phase [N] | Pending |
| PRIV-01 | Phase [N] | Pending |
| PRIV-02 | Phase [N] | Pending |
| PRIV-03 | Phase [N] | Pending |
| PRIV-04 | Phase [N] | Pending |
| PRIV-05 | Phase [N] | Pending |
| PRIV-06 | Phase [N] | Pending |
| PRIV-07 | Phase [N] | Pending |
| BUG-01 | Phase [N] | Pending |
| BUG-02 | Phase [N] | Pending |
| BUG-03 | Phase [N] | Pending |
| BUG-04 | Phase [N] | Pending |
| BUG-05 | Phase [N] | Pending |
| BUG-06 | Phase [N] | Pending |
| BUG-07 | Phase [N] | Pending |
| BUG-08 | Phase [N] | Pending |
| BUG-09 | Phase [N] | Pending |
| BUG-10 | Phase [N] | Pending |
| BUG-11 | Phase [N] | Pending |
| HIGH-01 | Phase [N] | Pending |
| HIGH-02 | Phase [N] | Pending |
| HIGH-03 | Phase [N] | Pending |
| HIGH-04 | Phase [N] | Pending |
| HIGH-05 | Phase [N] | Pending |
| HIGH-06 | Phase [N] | Pending |
| HIGH-07 | Phase [N] | Pending |
| HIGH-08 | Phase [N] | Pending |
| HIGH-09 | Phase [N] | Pending |
| HARNESS-01 | Phase [N] | Pending |
| HARNESS-02 | Phase [N] | Pending |
| HARNESS-03 | Phase [N] | Pending |
| HARNESS-04 | Phase [N] | Pending |
| HARNESS-05 | Phase [N] | Pending |
| HARNESS-06 | Phase [N] | Pending |
| LOGIC-01 | Phase [N] | Pending |
| LOGIC-02 | Phase [N] | Pending |
| LOGIC-03 | Phase [N] | Pending |
| LOGIC-04 | Phase [N] | Pending |
| LOGIC-05 | Phase [N] | Pending |
| GOLDEN-01 | Phase [N] | Pending |
| GOLDEN-02 | Phase [N] | Pending |
| GOLDEN-03 | Phase [N] | Pending |
| GOLDEN-04 | Phase [N] | Pending |
| LONGMEM-01 | Phase [N] | Pending |
| LONGMEM-02 | Phase [N] | Pending |
| LONGMEM-03 | Phase [N] | Pending |
| LONGMEM-04 | Phase [N] | Pending |
| LONGMEM-05 | Phase [N] | Pending |
| BEIR-01 | Phase [N] | Pending |
| BEIR-02 | Phase [N] | Pending |
| BEIR-03 | Phase [N] | Pending |
| BEIR-04 | Phase [N] | Pending |
| BEIR-05 | Phase [N] | Pending |
| BEIR-06 | Phase [N] | Pending |
| OCMEM-01 | Phase [N] | Pending |
| OCMEM-02 | Phase [N] | Pending |
| OCMEM-03 | Phase [N] | Pending |
| SOTA-01 | Phase [N] | Pending |
| SOTA-02 | Phase [N] | Pending |
| SOTA-03 | Phase [N] | Pending |
| SOTA-04 | Phase [N] | Pending |
| SOTA-05 | Phase [N] | Pending |
| DOCS-01 | Phase [N] | Pending |
| DOCS-02 | Phase [N] | Pending |
| DOCS-03 | Phase [N] | Pending |
| DOCS-04 | Phase [N] | Pending |
| DOCS-05 | Phase [N] | Pending |
| DOCS-06 | Phase [N] | Pending |
| DOCS-07 | Phase [N] | Pending |
| DOCS-08 | Phase [N] | Pending |
| DOCS-09 | Phase [N] | Pending |
| DOCS-10 | Phase [N] | Pending |
| DOCS-11 | Phase [N] | Pending |
| WEB-01 | Phase [N] | Pending |
| WEB-02 | Phase [N] | Pending |
| WEB-03 | Phase [N] | Pending |
| WEB-04 | Phase [N] | Pending |
| WEB-05 | Phase [N] | Pending |
| WEB-06 | Phase [N] | Pending |
| WEB-07 | Phase [N] | Pending |
| PAPER-01 | Phase [N] | Pending |
| PAPER-02 | Phase [N] | Pending |
| PAPER-03 | Phase [N] | Pending |
| PAPER-04 | Phase [N] | Pending |
| PAPER-05 | Phase [N] | Pending |
| PAPER-06 | Phase [N] | Pending |
| PAPER-07 | Phase [N] | Pending |
| REL-01 | Phase [N] | Pending |
| REL-02 | Phase [N] | Pending |
| REL-03 | Phase [N] | Pending |
| REL-04 | Phase [N] | Pending |
| REL-05 | Phase [N] | Pending |

**Coverage:**
- v1 requirements: 91 total
- Mapped to phases: 0 (populated by roadmapper)
- Unmapped: 91 (expected until roadmap creation)

---
*Requirements defined: 2026-04-09*
*Last updated: 2026-04-09 after initial definition (post-research)*
