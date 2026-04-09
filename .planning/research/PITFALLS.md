# Pitfalls Research

**Domain:** Scientifically-rigorous RAG memory plugin for LLM agents (brownfield, public release + research paper + website)
**Researched:** 2026-04-09
**Confidence:** HIGH (grounded in the project's own audit + 2025–2026 ecosystem research)

This document is deliberately specific to memory-spark. Generic "test your code" pitfalls are excluded. Every pitfall is either (a) already confirmed on this codebase or (b) a known failure mode in the agent-memory / hybrid-RAG / research-publishing ecosystems that this project has a structural risk of hitting. Each one maps to a concrete phase of the current milestone (see the Active list in `.planning/PROJECT.md`).

The phase names used here are:

| Short | Active item in PROJECT.md |
|-------|---------------------------|
| **P1 Recon** | Read-only recon |
| **P2 Privacy** | Privacy foundation |
| **P3 Criticals** | Critical bug fixes (11) |
| **P4 High-sev** | High-severity audit fixes |
| **P5 Tool-inject** | Tool-calling injection feature |
| **P6 Tool-surface** | Expanded plugin tool surface (9 → 14 tools) |
| **P7 Spark-v2** | Spark v2 architecture migration |
| **P8 Docker-test** | Docker test environment restoration |
| **P9 Golden-set** | Golden dataset generation |
| **P10 BEIR** | Full BEIR benchmarks |
| **P11 OCMemory-bench** | OCMemory benchmark |
| **P12 SOTA-valid** | 2026 SOTA research validation |
| **P13 Docs** | Documentation overhaul |
| **P14 Website** | Sexy educational website |
| **P15 Paper** | Research paper expansion |
| **P16 Release** | v1.0 release |

---

## Critical Pitfalls

### Pitfall 1: Shipping a research paper whose numbers are invalidated by bugs fixed after generation

**What goes wrong:**
Benchmark runs are executed, charts are baked into the paper/website, then a critical bug is fixed in the retrieval pipeline. The old numbers no longer describe the shipped artifact. The paper's main table describes a broken system; the repo describes a fixed one. Reviewers and readers find this within a day of release.

**Why it happens on this project specifically:**
There are already concrete examples where this would have happened if the paper had shipped before the audit: MAP@10 was computed with `min(R, k)` instead of `R`, inflating NFCorpus MAP ~3.8× (C4 in AUDIT-2026-04-02); the BM25 sigmoid saturation bug (H1 / score-clamping) made the reranker gate misfire so the "with reranker" configs in the paper weren't actually reranking the intended set; the benchmark runner was feeding corrupt hybrid-merge inputs; the FTS `.where()` bug meant cross-agent bleed into "filtered" result sets. Every one of these bugs changes headline numbers.

**How to avoid:**
1. **Hard freeze order.** Do not regenerate the paper/website numbers until after **P3 Criticals + P4 High-sev are merged and tagged**. Any benchmark that will appear in a public artifact must be run against a tagged commit SHA that is itself post-audit.
2. **Embed the commit SHA in every result file.** `run-beir-bench.ts` should stamp `git rev-parse HEAD` into `results/*.json` at the top. Charts must read it. Paper tables must include "results generated from commit `abc1234`".
3. **Regeneration assertion in CI.** Add a script that, for every figure in the paper and every chart on the website, asserts the referenced commit SHA matches the current repo HEAD (or an explicitly allowed frozen tag). Fail CI if drift detected.
4. **`evaluation/generate-charts.ts` must not hardcode numbers.** Audit flagged it already (M8): the SCIFACT_RESULTS array is hardcoded. Make it read JSON, or delete it.

**Warning signs:**
- Benchmark result JSON files have no commit SHA metadata
- Chart generator is a TypeScript file with numeric literals
- Paper LaTeX has numbers that don't match any file in `evaluation/results/`
- Any `git log --oneline` line between the benchmark run and the release touches `src/rerank/`, `src/auto/recall.ts`, `src/storage/lancedb.ts`, `evaluation/metrics.ts`, or `src/embed/`

**Phase to address:** P3 + P4 gate this. Benchmarks in **P10 BEIR / P11 OCMemory-bench** only legal to run after those phases are green. **P15 Paper** numbers come from P10/P11 results JSON, never hand-typed.

---

### Pitfall 2: Golden dataset leaks Klein's personal data into the public repo (again)

**What goes wrong:**
The prior `evaluation/golden-dataset.json` leaked real PII into public git history: personal emails, private WireGuard and LAN subnets, a family name, a hometown repeated across ~13 docs, and full contents of production AGENTS.md / MEMORY.md / HEARTBEAT.md / TOOLS.md plus a cron audit with agent names and schedules. That file has since been removed from all git history via `git filter-repo`, but the same category of leak will happen again if new QA pairs are generated with Nemotron-Super-3-122B against OpenClaw agent data unless the input corpus is scrubbed first *and* the output is scrubbed again — the LLM will happily copy identifiers from the source text into generated questions and synthesized answers.

**Why it happens on this project specifically:**
- Nemotron-Super runs locally on the DGX Spark, so there is no cloud egress guardrail to catch PII before it's written to disk.
- LLM-generated QA pairs are a known PII amplifier — the model can memorize and paraphrase identifiers, and even "synthetic" questions often embed spans of source text verbatim (see PII-Scope, ProPILE — a single epoch of fine-tuning on sensitive data leaks ~19% of PII tokens; generation without fine-tuning still lifts identifiers from prompt context).
- The source corpus (`testDbOCMemory` = 3807 files from Klein's agent workspaces) is the most PII-dense input this project will ever process.
- Once committed, `git filter-repo` is the only remediation, and any fork/mirror keeps the history.

**How to avoid (this is load-bearing; multiple redundant layers required):**
1. **Scrub the input corpus, not just the output.** Build `evaluation/scrub.ts` that runs BEFORE Nemotron sees any chunk. Required replacements:
   - Emails → `user@example.com` (preserve only whether one existed)
   - IPv4s in `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` → `10.x.x.x`
   - Private WireGuard + LAN subnets (maintained in an internal block-list; not reproduced here) → `10.x.x.x`
   - Real person names (maintain an internal block-list of names that appear in agent docs; not reproduced here) → `Alice`, `Bob`, `Carol`
   - Hometown / state / university / university email domain (maintained in an internal block-list; not reproduced here) → `Exampleville`
   - Full file paths containing real home directory, workspace prefixes, agent workspace paths → relative anonymized paths
   - `openclaw`, `workspace-dev`, `workspace-meta`, cron job names → generic `agent-a`, `agent-b`
   - Bearer tokens, API keys (any 32+ char hex, any string starting `nvapi-`, `sk-`, `ghp_`) → `<REDACTED>`
2. **Gold-standard canary test.** Before declaring the scrubber ready, seed the input corpus with 20 synthetic canary PII tokens (fake emails, fake names, fake IPs generated with a seeded RNG). Run scrubber. Assert zero canaries survive. This is the only way to prove the scrubber actually caught the classes you care about.
3. **Scrub the output again.** Run the same scrubber over Nemotron-generated questions AND over any retrieved chunk text that lands in the golden dataset. Assume the LLM smuggles things through.
4. **`.gitignore` the raw pipeline.** Add to `.gitignore`: `testDbOCMemory/`, `evaluation/golden-dataset-raw.json`, `evaluation/ocmemory-source-chunks.json`, `bench-data/ocmemory-raw/`. Only the scrubbed `evaluation/golden-ocmemory-scrubbed.json` may be committed.
5. **Pre-commit hook.** Add a `pre-commit` hook that greps staged files against an internal block-list of known-bad patterns (email regex, private subnets, personal names, location strings, username) and blocks the commit. Block-list is stored outside the repo and referenced by the hook — never inlined into public docs.
6. **Sanitize the currently-committed golden dataset.** Don't just stop adding PII — the existing leak is on public GitHub. Either `git filter-repo` the history, or treat the tokens as burned and continue but with mitigation (the emails and IPs should be considered publicly disclosed).
7. **CI privacy scan.** Run a scheduled `gitleaks` + custom pattern scan on `main` every commit. Fail the build on any match.

**Warning signs:**
- A diff to `evaluation/` contains a string matching `\b[\w.+-]+@[\w-]+\.[\w.-]+\b` where the email isn't `user@example.com`
- A diff matches any entry in the internal PII block-list (private subnet prefixes, hometown, personal usernames, university email domain) — block-list is maintained outside the repo
- Any committed JSON file has `doc` names in the 000-350 range (the existing leaked golden-dataset.json structure)
- Scrubber exists but isn't invoked in the benchmark pipeline
- Golden dataset contains any file path under `workspace-` or `.claude/` or `.openclaw/`

**Phase to address:** **P2 Privacy** (scrubber + canary test + pre-commit hook + gitleaks CI MUST ship before anything else touches real data). **P9 Golden-set** runs the scrubber in-pipeline. **P16 Release** has an explicit "privacy audit sign-off" gate — do not cut v1.0 without one more canary-seeded scan.

---

### Pitfall 3: Hybrid merge silently invalidated by score distribution mismatch (the BM25 sigmoid pattern, in four more places)

**What goes wrong:**
The BM25-sigmoid-saturation bug is a symptom of a general class: score fusion fails silently whenever the score distributions coming in don't match the assumption of the fusion function. Score clamping, wrong normalizer, incompatible score scales, or "helpful" boosts that push everything past a saturation point — all produce the same failure mode: the ranker looks like it's working, top-k is populated, every test passes, but the effective ordering is random noise. Known instances on this codebase already:
- **Clamping** (`Math.min(1.0, r.score * weight)` in applySourceWeighting) → tied-score set → reranker gate skips → "with reranker" config silently bypasses reranker (H1/P1-A).
- **Sigmoid saturation on BM25 scores** → high-BM25 documents all clamp to ~1.0 → RRF-weighted blend has no gradient to sort by.
- **Post-weighting dedup** (P2-B) → Jaccard tiebreaker runs on inflated scores.
- **Mixed vector and rerank logit scales** → blendByRank collapses any weakly-ranked doc below the top-10 cutoff (P1-B GUARD-A/B/C).
- **Pooled scores without per-pool normalization** → one pool's top chunk has cosine 0.95, another's has 0.3, ranked together they all appear "worse" than the first pool even if the 0.3 is the correct answer.

**Why it happens on this project specifically:**
The pipeline has 5 pools × (vector + FTS) × (HyDE + non-HyDE) × source weighting × temporal decay × MMR × reranker × blend. That's 11 places a score can be transformed. Every transformation is a chance to break fusion assumptions. The codebase has no single place that asserts "scores in [0, 1] and monotone with relevance."

**How to avoid:**
1. **Score-invariant tests.** Add `tests/score-invariants.test.ts`. For each stage (weight, decay, dedup, normalize, rerank, blend), assert: (a) output length ≥ input length × 0 (no dropping without explicit reason), (b) output scores are in `[0, 1]`, (c) monotonicity — if input A > input B, and neither is dropped, output A ≥ output B unless explicitly re-ranked, (d) no NaN, no Infinity, no negative, (e) if `n` inputs tie, `n` outputs also tie (no arbitrary tiebreaking from a hash).
2. **Spread assertion at gate entry.** When the reranker gate computes spread, log (to a debug channel) if spread < 0.02 AND >2 chunks have score == 1.0 exactly. That's the fingerprint of clamping. Catch it before production.
3. **Per-stage score histograms in benchmark runner.** `scripts/run-beir-bench.ts` should emit the 10/50/90 percentiles of scores entering and leaving each stage, per query, to `results/*.stats.jsonl`. Diff these across configs. If adding a stage changes the distribution but NDCG doesn't move, the stage is broken.
4. **No more "apply boost by multiply."** When you want to bias a pool, use rank-space or use an additive rank bonus (`score += (K - rank) * ε` bounded to [0, 1]). Multiplying cosine similarities by >1 is the root pattern that caused H1 and is still idiomatic in the codebase.
5. **Type the score.** Define `type Score01 = number & { __brand: "Score01" }` and a `clampToScore01()` constructor that asserts `[0,1]` at runtime. Use it everywhere. TypeScript won't catch the bug alone but the branding forces awareness.

**Warning signs:**
- Any stage computes `r.score * k` without a subsequent normalize
- Any stage uses `Math.min(1.0, ...)` without a corresponding `Math.max`
- Histograms of reranker input scores show a spike at 1.0
- Reranker gate skip rate is >50% on configs that should re-rank
- NDCG@10 changes by <0.5% when a stage is toggled on/off (stage is a no-op)
- Distribution of final scores is bimodal with a gap

**Phase to address:** **P3 Criticals** for the known bugs. **P4 High-sev** for the score-invariant test file and score-type branding. Re-verify in **P10 BEIR** by running histograms on every config.

---

### Pitfall 4: Cross-encoder reranker destroys precision on this domain (already happened once)

**What goes wrong:**
Cross-encoder rerankers are trained on MS MARCO / BEIR / TriviaQA. They assume query-document pairs that look like "short natural-language question" + "paragraph of general-domain text." Agent memory chunks are distilled facts, config snippets, code diffs, and shell outputs. On out-of-domain input the reranker's logits stop being useful; it re-sorts noise. The TREC TOT 2025 result (RR 0.2838 → 0.0601 when reranking was added) and the LanceDB April 2025 blog (FTS reranking showed up to 20.65% *degradation* on some models) are not anomalies — they're the expected behavior when you apply a BEIR-trained reranker to a novel domain. Additionally, on memory-spark's own benchmarks: reranker p50 was 41 seconds — 60× worse than vector-only — making it unusable even when it *would* have helped.

**Why it happens on this project specifically:**
- The current reranker (`llama-nemotron-rerank-1b-v2`) was trained on MS MARCO / BEIR. Agent memory is explicitly out-of-domain.
- The codebase has `MAX_RERANK_CANDIDATES = 30` (now 40). With K=5-10 post-dedup, the reranker is shuffling a tiny pool where BEIR-style relevance signals don't generalize.
- The reranker gate thresholds (τ_high=0.08, τ_low=0.02) were picked on SciFact, not on OCMemory data. Nothing says they transfer.
- The audit paper admits "in-domain gap" in Limitations but the main results are already BEIR-only.
- Upgrading to bge-reranker-v2-m3 in **P7 Spark-v2** will repeat this failure unless the domain mismatch is tested.

**How to avoid:**
1. **Test the reranker on OCMemory data explicitly, separately from SciFact.** In **P11 OCMemory-bench**, report NDCG@10 with and without reranker. If reranker helps SciFact but hurts OCMemory, the paper must say so. That's scientific honesty, not a bug.
2. **Implement rerankerGate benchmarks properly.** The gate is supposed to skip reranking on easy queries (high spread). Verify it's actually skipping: log `gate.skipped` per query, assert distribution matches expectation. Post-fix of H1, re-verify.
3. **Position preservation guard (P1-B already on the plan).** GUARD-A / B / C must land before the reranker ships as a default. Without them, any weakly-ranked-by-reranker doc that was strongly ranked by vector falls out of top-10 and NDCG collapses to 0 on that query.
4. **Latency budget assertion.** No reranker config enabled by default whose p95 > 500ms. The current 41s p50 is a ship-stopper for auto-recall. Add `rerank.maxLatencyMs` and abort the reranker stage (return pre-rerank order) if exceeded.
5. **Document the domain-mismatch result honestly in the paper.** Do not hide the OCMemory result if the reranker hurts on it. "Reranker helps BEIR (+0.9% NDCG) but hurts OCMemory (−4.2% NDCG)" is a far more valuable contribution than "reranker helps."
6. **For the reranker upgrade in P7 (bge-reranker-v2-m3):** benchmark against SciFact AND OCMemory AND FiQA before cutting the migration. Add to `evaluation/results/` and diff.

**Warning signs:**
- p50 reranker latency > 1s (instant red flag; something is wrong)
- Reranker NDCG delta on domain-local data (OCMemory) is negative or <0.5%
- Gate skip rate is ~0% (gate never fires; reranker runs on everything)
- Reranker score distribution per query is flat (no spread; noise sort)
- Queries that worked fine without reranker now score 0 with reranker (P1-B position preservation failure)

**Phase to address:** **P3 Criticals** (fix latency + P1-B position preservation), **P7 Spark-v2** (bge-reranker-v2-m3 migration must re-verify on OCMemory), **P11 OCMemory-bench** (honest domain benchmark), **P15 Paper** (reported as a finding, not hidden).

---

### Pitfall 5: HyDE burns the latency budget and hallucinates on well-specified queries

**What goes wrong:**
HyDE generates a hypothetical document by asking an LLM to answer the query, then embeds the hypothetical doc and searches with that vector. It works well when the query-document vocabulary gap is large (question-style query against paragraph-style corpus). It fails when:
- The query is fact-bound (memory-spark agent queries: "what was the docker restart command", "which port does embed use"). HyDE hallucinates a plausible-sounding but wrong command, retrieves on the wrong semantic region, and degrades recall. This is the single largest failure mode documented for HyDE in 2025 literature.
- The LLM's cold-start latency dominates. The benchmark runs showed 100% HyDE failure at 15s timeout. The fix was to raise timeout to 30s — which means every agent turn can now hang for 30s. On a system that targets <2s agent turns, this is user-destroying.
- Vector averaging vs. replacement: averaging the hypothetical vector with the original query vector destroys both signals. The existing code does the wrong thing (averaging); the known-correct pattern is to replace the query vector with the HyDE vector, or issue two parallel searches and merge.

**Why it happens on this project specifically:**
- The `HEY-OPENCLAW-STOP-BEING-RETARDED.md` and SOTA-VALIDATED notes say "HyDE — NOT for us" on latency grounds. But HyDE is now re-enabled with default `timeoutMs: 30000`.
- Agent memories are already "answer-shaped" — the query-document gap HyDE exists to bridge is small.
- The HyDE model is currently Nemotron-120B (slow cold start). **P7 Spark-v2** plans to swap to Nemotron-Mini-4B, but that doesn't make HyDE *correct*, only faster.
- Benchmark corruption: if HyDE fails (timeout), the benchmark runner's fallback path may feed the wrong vector into the merge, producing the "corrupted hybrid merge inputs" bug observed in the audit.

**How to avoid:**
1. **Default HyDE off.** `buildDefaults().hyde.enabled = false`. Opt-in only. The 2025 literature consensus (Medium 2026, machinelearningplus, Haystack docs) is that HyDE is situational.
2. **Fix the averaging bug.** When HyDE is enabled, REPLACE the query vector OR issue two searches and RRF-merge. Do not average.
3. **Reduce timeout to 3-5s and fall back on timeout.** On timeout, use the original query vector, do NOT block, do NOT error the whole pipeline, and log the fallback.
4. **Measure HyDE-negative queries.** Add to the golden dataset (M17 in audit already flagged this) a category of fact-bound queries where HyDE should *hurt*. If HyDE doesn't hurt on those, the test is wrong.
5. **Benchmark HyDE separately.** Report NDCG@10 with HyDE on and off, per dataset, per query category. Expect mixed results. Report them.
6. **Latency budget assertion.** If `hyde.enabled == true` and the LLM endpoint is unreachable or p95 > `hyde.timeoutMs`, auto-disable HyDE and warn. Better UX than hanging.
7. **Document when to enable.** Ship with a README table: "Enable HyDE when: queries are short and ambiguous. Disable when: queries are code-like or contain identifiers."

**Warning signs:**
- Agent turn p95 > 5 seconds (HyDE almost certainly the cause)
- HyDE failure rate > 10% (either the LLM is dying or timeout is wrong)
- HyDE-enabled NDCG is worse than HyDE-off on any dataset
- HyDE fallback path is untested
- Benchmark runner hangs for 30+ seconds on a single query

**Phase to address:** **P3 Criticals** (fix averaging, default off, fix fallback). **P7 Spark-v2** (Mini-4B migration must keep HyDE off by default even if faster). **P10 BEIR / P11 OCMemory-bench** (report HyDE-on vs HyDE-off separately). **P15 Paper** (honest HyDE-hurt result if found).

---

### Pitfall 6: MMR diversity destroys recall on high-relevance clusters (the 28% SciFact loss pattern)

**What goes wrong:**
MMR's formula is `MMR = (1−λ)·relevance − λ·max_similarity_to_selected`. At low λ (high diversity weight), if all relevant documents cluster in the same semantic region (common in scientific corpora, ArxivDL, SciFact, code repositories — all cases where near-duplicates or paraphrases are all relevant), the diversity penalty punishes the second, third, fourth correct answer because they're too similar to the first. The system then injects irrelevant "diverse" documents to fill the slots, and recall@10 craters. SciFact lost 28% on this project with aggressive diversity. The Qdrant/Elastic/OpenSearch docs recommend λ ≥ 0.7 — memory-spark's default was evidently too low.

**Why it happens on this project specifically:**
- Scientific and code corpora have near-duplicate relevant chunks by design.
- Agent memory has heavy semantic clustering (many memories about the same topic, all relevant).
- MMR on pooled results cross-fertilizes: if pool A has 5 near-duplicates all ranked 1-5, MMR drops 4 of them and fills with pool B's random #6, losing 4 correct chunks.
- The existing `adaptive MMR` feature (flagged in L8 as "undocumented") adjusts λ dynamically — but on what signal? If it's wrong, it makes the problem worse.

**How to avoid:**
1. **Benchmark MMR λ sweep on every dataset.** In **P10 BEIR**, run with λ ∈ {0.3, 0.5, 0.7, 0.9, 1.0 (MMR off)}. Report NDCG@10 + Recall@100 for each. Pick λ per dataset if necessary; do not assume one global value.
2. **Default λ = 0.7 or higher.** Industry consensus (Qdrant, Elastic). Start conservative.
3. **MMR OFF for high-R queries.** When `totalRelevant > 10` (NFCorpus-style), diversity hurts by design — the user wants all relevant results, not a diverse sample. Make MMR conditional on query type or turn it off for recall-optimized configs.
4. **Recall guard.** Assert: MMR stage may not remove any document that scores in top-K by cosine AND above threshold. Use the same position-preservation pattern as P1-B for reranker.
5. **Document the tradeoff.** MMR is for avoiding redundancy in presentation, not for retrieval quality. Explain this in docs.
6. **Re-verify after H1 fix.** Score clamping interacted with MMR too — once scores are clean, re-run the 28% SciFact loss measurement. It may already be partially recovered.

**Warning signs:**
- Recall@100 drops when MMR is enabled (it should at worst stay flat)
- NDCG@10 drops more than 5% between MMR-off and MMR-on on any dataset
- Golden dataset queries with 5+ relevant chunks have only 1-2 retrieved
- MMR default λ < 0.7
- "Adaptive MMR" exists but has no documented signal or test

**Phase to address:** **P3 Criticals** (verify post-H1 MMR behavior, fix default λ). **P10 BEIR** (λ sweep, report). **P15 Paper** (honest MMR failure mode in discussion).

---

### Pitfall 7: Cross-agent data leakage through unchecked storage-layer filters

**What goes wrong:**
The `ftsSearch()` WHERE filter was declared but never applied (`const q = ...; q.where(...)` without assignment). LanceDB's fluent API is immutable — `.where()` returns a new query object — so the filter silently did nothing. All agentId, pool, source, contentType filters were ignored on FTS path. The `school` agent could receive `meta` agent memories. This is a data-isolation bug, not a performance bug. The audit fixed it (C1), but the structural risk remains: any new storage method that uses the same fluent API pattern can silently regress.

**Why it happens on this project specifically:**
- LanceDB v0.25.3 (and earlier) has known silent-filter bugs: issue #3095 (prefilter arg silently inverted in hybrid search), issue #1656 (where-clause on FTS ignored when a scalar index exists), issue #470 (where without vector search doesn't work). The LanceDB fluent API has a track record of failing open.
- TypeScript's `const`-vs-`let` distinction doesn't catch this because `const q = q.where(...)` would fail a lint rule only if the caller remembers to use it.
- The project has no integration test that asserts agent isolation (a test that would have caught C1).
- **P5 Tool-inject** and **P6 Tool-surface** will add new storage queries. Each one is a fresh chance to reintroduce this bug.

**How to avoid:**
1. **Agent isolation integration test.** Seed the test DB with agent A (5 docs) and agent B (5 docs). Run every storage read path with agentId=A. Assert zero agent-B documents return. Run for: `ftsSearch`, `vectorSearch`, `hybridSearch`, `getByIds`, `readFile`, any new method added in P5/P6.
2. **ESLint `no-floating-promises` + `no-unused-expressions`** (audit already recommends this). A call to `.where()` whose return is discarded is either a floating promise or an unused expression; both lint rules catch it.
3. **LanceDB wrapper functions.** Wrap every LanceDB query-builder call in a helper: `function applyFilters(q, filters) { let result = q; for (const f of filters) result = result.where(f); return result; }`. No raw fluent-chain calls in storage code. Easier to audit.
4. **Pin LanceDB version.** Until issue #3095 and #1656 are resolved upstream, pin to a known-good version and track the upgrade explicitly.
5. **Defensive assertion in hot path.** After every storage read, sample-assert: `results.every(r => r.agentId === requestedAgentId || requestedAgentId === undefined)`. Log + drop violators. This won't fix the bug but it contains the blast radius.
6. **Audit every `.where(` call in `src/storage/`.** Not just the one the audit found. Right now.

**Warning signs:**
- Any `.where(...)` call whose return value is discarded
- Any storage method that takes `filters` but doesn't have a test asserting the filter is applied
- LanceDB version in package.json is behind current
- Tests that only check "a result came back" without checking "and it was the right agent's result"
- New storage paths added without agent-isolation tests

**Phase to address:** **P3 Criticals** (C1 already fixed, but add isolation test). **P4 High-sev** (ESLint rules). **P5 Tool-inject + P6 Tool-surface** (every new tool that reads from storage must include an isolation test in the same PR).

---

### Pitfall 8: Benchmark runner corrupts hybrid merge inputs (invalidating every "with hybrid" config)

**What goes wrong:**
Prior audit found the benchmark runner feeds corrupt inputs into hybrid merge. If the runner's data path differs from the production data path (different normalization, different dedup, different pool construction), the benchmark measures "memory-spark's hybrid merge on runner-specific inputs", not "memory-spark's production behavior." Every NDCG/MAP number becomes a measurement of the runner, not the system. All existing benchmark results published in the paper are suspect to this class of bug.

**Why it happens on this project specifically:**
- `scripts/run-beir-bench.ts` and `evaluation/benchmark-v2.ts` both exist and may diverge from the production `src/auto/recall.ts` orchestration.
- `evaluation/run.ts` and `evaluation/charts.ts` don't even compile (C5) — they import symbols that don't exist. This is "dead code" but means the benchmarker is code-split across files, making drift easy.
- The benchmark runner has its own mock of the capture / recall pipeline; any mismatch with production is a silent invalidator.

**How to avoid:**
1. **Benchmark runner uses the production code path.** No mocks, no parallel implementations. `run-beir-bench.ts` should call `MemorySparkManager.search()` directly, not re-implement the stages. Any deviation is a smell.
2. **Delete dead benchmark code.** `evaluation/run.ts` and `evaluation/charts.ts` are flagged as non-compiling (C5). Delete them in P3 Criticals.
3. **Input fixture dump + replay.** Have the benchmark runner dump, for 5 sampled queries per dataset, the exact scored lists entering every pipeline stage. Commit these fixtures. In future runs, compare against the fixture; fail if divergent unintentionally.
4. **Cross-check against reference implementation.** For BEIR, the pytrec_eval library is the canonical evaluator. Run the same qrels + results through pytrec_eval and assert memory-spark's internal metrics match. Any disagreement is a bug in metrics.ts (already found C4: MAP denominator).
5. **Benchmarks only publish from tagged commits.** Require the runner to refuse to run against a dirty working tree unless `FORCE=1`. All published numbers come from clean tagged commits.

**Warning signs:**
- Benchmark NDCG numbers move around when the runner is run twice on the same commit
- `run-beir-bench.ts` defines its own version of a function that exists in `src/`
- Metric values differ between `evaluation/metrics.ts` and `pytrec_eval` on the same input
- Dead code in `evaluation/` that doesn't compile
- No commit SHA stamped in results files

**Phase to address:** **P3 Criticals** (fix runner + delete dead code + MAP denominator C4). **P10 BEIR** (use pytrec_eval cross-check). **P11 OCMemory-bench** (same discipline).

---

### Pitfall 9: Documentation drift produces a public release whose docs are demonstrably wrong

**What goes wrong:**
Port numbers wrong in README + paper + CONFIGURATION.md (docs say 18081/18098, code uses 18091/18096). Version mismatch between package.json (0.4.0) and plugin manifest (0.1.0). Coverage badge claims 91% when real coverage is ~35%. CHANGELOG has PENDING markers on shipped features. Paper's infrastructure table has wrong ports. CONFIGURATION.md missing ALL Phase 8-12 features (rerankerGate, blendMode, rrfK, HyDE, queryExpansion, hierarchical chunking, adaptive MMR, embed queryInstruction). This is exactly the class of error a reviewer finds in the first 10 minutes of reading the paper and tweets about.

**Why it happens on this project specifically:**
- The project is brownfield. Code has drifted ahead of docs in some areas and behind plans in others. The existing audit catalogs 10+ instances.
- No CI job asserts doc-code consistency.
- The docs-site, paper, README, CONFIGURATION.md, and CHANGELOG all restate the same facts in different words, so any one of them can drift independently.

**How to avoid:**
1. **Single source of truth for port numbers, versions, config schema.** Generate README snippets, paper infrastructure table, CONFIGURATION.md from `src/config.ts` at build time. Use a `scripts/gen-docs.ts`. Commit the generated output; CI re-generates and diffs to catch drift.
2. **Version pin check.** CI job: assert `package.json.version === openclaw.plugin.json.version === src/index.ts VERSION constant`. One line test, catches M5 forever.
3. **Coverage badge auto-generated.** Run `vitest --coverage` in CI with `coverage.all: true`, extract the real number, update the README badge from CI. Never hand-edit.
4. **Port-number grep test.** A test that greps all `*.md` and `*.tex` files for known-wrong port numbers and fails. `grep -R "18081\|18098" docs/ README.md paper/` should return 0 matches.
5. **CHANGELOG linter.** CI fails if any unreleased heading has PENDING markers that map to shipped code (check by matching feature names against the commit log).
6. **Paper numbers come from result JSON, not LaTeX literals.** Use a Python/JS script to inject numbers into `paper/memory-spark.tex` at build time from `evaluation/results/`. No hand-typed numerics.
7. **"Drift audit" as a recurring phase.** **P13 Docs** is the scheduled rewrite. But the discipline has to persist; schedule a pre-release audit in **P16 Release**.

**Warning signs:**
- Any hand-edited numeric in README or paper
- Any doc that hasn't been touched since before the last feature shipped
- CHANGELOG "Unreleased" section is empty but commits have happened
- Badge images in README reference an external shields.io endpoint that isn't CI-driven
- Any grep for the old port number succeeds

**Phase to address:** **P13 Docs** (full rewrite from config.ts as source of truth). **P15 Paper** (numbers come from JSON). **P16 Release** (final drift audit + version pin check + port grep).

---

### Pitfall 10: Paper publishes claims without statistical significance, reviewers catch it, credibility dies

**What goes wrong:**
All comparisons in the current paper are raw point differences ("+0.94% NDCG"). No confidence intervals, no bootstrap, no p-values. With 300 queries (SciFact), a 0.005 NDCG delta is almost certainly within noise. Reviewers at any venue (TMLR, SIGIR, EMNLP, ACL findings, arXiv commentary) will call this out immediately. The audit already flagged this (paper issue #2) and recommends bootstrap CI or paired t-tests.

**Why it happens on this project specifically:**
- The existing paper reports 36-config ablation tables with raw deltas.
- No statistical testing infrastructure exists in the repo.
- The golden dataset is smaller than standard BEIR datasets (~500 QA pairs), which makes small differences even noisier.
- Single-dataset primary evaluation (SciFact only for main table) compounds the problem.

**How to avoid:**
1. **Bootstrap CI on every reported number.** Add `evaluation/bootstrap.ts`: resample queries with replacement 1000 times, compute NDCG@10 mean and 95% CI. Report as "0.7802 ± 0.0047 (95% CI)".
2. **Paired test between configs.** When comparing config A to config B, use the same query set, compute per-query NDCG, run paired t-test (or Wilcoxon signed-rank for non-parametric). Report p-value.
3. **Multiple-comparison correction.** 36 configs × ~3 metrics = 108 comparisons. Bonferroni or Benjamini-Hochberg correction required before claiming any "significant" result.
4. **Multi-dataset evaluation.** SciFact + NFCorpus + FiQA as separate tables. An effect that holds on one dataset is an anomaly; an effect that holds on three is a finding.
5. **Sensitivity analysis on hyperparameters.** τ_high, τ_low, RRF k, MMR λ. Audit flagged this (paper issue #10) — single-point hyperparameters without sensitivity is a reviewer flag.
6. **Don't claim what you can't measure.** If bootstrap CIs overlap, the comparison is "not statistically distinguishable." Say so. Do not claim improvement.
7. **Publish the bootstrap scripts.** Reviewers will want to reproduce. Ship `scripts/run-bootstrap.ts` and the random seed.

**Warning signs:**
- Any headline number in the paper without a ± uncertainty
- Comparisons between configs without p-values
- Claims that config A > config B when the CI overlap is >90%
- Single-dataset results used to make general claims
- No sensitivity analysis for any tuned hyperparameter

**Phase to address:** **P10 BEIR / P11 OCMemory-bench** (bootstrap infrastructure). **P15 Paper** (every number reported with CI + paired tests + correction; multi-dataset tables; sensitivity figures).

---

### Pitfall 11: Agent memory systems that auto-capture garbage and can't retrieve it later

**What goes wrong:**
Common failure modes in the agent-memory ecosystem (Mem0, Zep, Letta, LangMem, Supermemory) as of 2026:
- **Mem0**: indexing reliability issues at scale — memories not consistently added, context recall failures under load, limited ingestion pipeline beyond chat
- **Zep**: graph nodes and edges managed manually, slow iteration, bug surface
- **LangMem**: P95 search latency 59.82 seconds — unusable for real-time agents
- **Letta**: complex managed architecture, configuration time

The pattern: auto-capture is greedy. It captures garbage (short messages, acknowledgements, duplicate statements, transient thoughts). Query time comes around, the garbage drowns out the signal in retrieval. Users complain "memory doesn't work." The fix isn't better retrieval — it's better capture gating.

memory-spark already has classifiers (zero-shot + NER + quality + heuristic) but the audit found **zero test coverage on `capture.ts`** and classifiers untested (H4, H7). Garbage in, garbage out, and you can't tell which part is broken.

**Why it happens on this project specifically:**
- Auto-capture runs after every agent turn with no human review.
- Classifier pipeline has 4 stages (zero-shot, NER, quality, heuristic) and no integration test.
- Capture errors are silently swallowed (H3). If the classifier crashes, everything gets captured.
- The golden dataset (M17) has no capture-quality queries — nothing tests "this garbage should have been rejected."
- The `MAX_CAPTURES_PER_TURN` cap exists but isn't unit-tested.

**How to avoid:**
1. **Capture-quality golden dataset.** Add 20-30 queries of the form "did we incorrectly capture this?" Expected answer: "no chunk found" or "rejected by classifier." Run in **P9 Golden-set**.
2. **Classifier integration test.** `tests/capture.test.ts` with mocked embed + backend. Test: pure garbage → rejected. Duplicate of existing chunk → rejected. Valid new fact → accepted. Mixed-quality multi-message turn → only the valid parts accepted. Classifier crash → fallback to heuristic, NOT greedy accept.
3. **Capture ledger / observability.** Log every capture decision: `{timestamp, agentId, chunk, classifier_score, decision, reason}`. Persist to `capture-log.jsonl`. Review weekly to tune thresholds. Without observability, you cannot debug "memory doesn't work."
4. **Dedupe at capture, not at retrieval.** If a fact is already in the DB (Jaccard ≥ 0.85, or cosine ≥ 0.95), do not re-capture. The audit notes dedup runs at retrieval time — this is a symptom of under-dedup at capture.
5. **Stop silently swallowing errors.** H3: `catch { /* non-fatal */ }` → fix to log + metric. You can't tune what you can't see.
6. **Rate limiting: MAX_CAPTURES_PER_TURN.** Verify it's enforced. Add a test.
7. **Benchmark the capture pipeline end-to-end.** The OCMemory benchmark should test "inject these 100 messages through capture, then query — did the right things survive?" Not just "search over a pre-built index."

**Warning signs:**
- `capture.ts` test coverage is 0%
- Capture logs are silent on every turn
- Golden dataset has no "rejection" test cases
- User reports "I told it X, it can't find X" without reproduction steps (because there's no capture log)
- Same fact appears N times in the DB under different chunk_ids
- `MAX_CAPTURES_PER_TURN` exists but isn't in any test

**Phase to address:** **P3 Criticals** (fix silent error swallowing). **P4 High-sev** (write `tests/capture.test.ts`). **P9 Golden-set** (capture-quality queries). **P11 OCMemory-bench** (end-to-end capture-through-retrieval test).

---

### Pitfall 12: Website and docs ship claims that the repo cannot back up

**What goes wrong:**
The "sexy educational website" plan (P14) is the highest-marketing-risk artifact. It's the face of the project, it gets HN'd/lobste.rs'd, and it carries the biggest claims ("scientifically rigorous", "production-grade", "benchmarked"). If any claim is unsupported by the repo at that moment, the project loses credibility instantly.

The prior example on *this project*: README coverage badge claimed 91% when real was ~35%. That single badge is the kind of thing a skeptical reader checks first.

**Why it happens on this project specifically:**
- docs-site/ exists and will evolve. Each evolution is a chance for a new claim to drift from reality.
- The paper and website will share numbers. If the paper changes and the website doesn't (or vice versa), one is lying.
- Marketing copy is written to sound impressive; engineering reality is messier.

**How to avoid:**
1. **Every claim on the website has a source link** (to the repo, to a result file, to a commit SHA, to a specific test). No floating assertions.
2. **Benchmark numbers on the website come from the same JSON files the paper reads.** Build-time injection. Same `scripts/gen-docs.ts` as pitfall 9.
3. **Coverage badge, test count, benchmark numbers — all CI-generated.** No human edits.
4. **Claims audit before shipping.** Pre-release, walk through every claim on the website and tick: "is this backed by something in the repo I can link to?" If not, remove or soften.
5. **No marketing adjectives without receipts.** "Scientifically rigorous" requires a methodology page. "Production-grade" requires uptime metrics. "Fast" requires a latency chart. No hand-waving.
6. **Keep a CLAIMS.md** in the repo that lists every public claim and its source. Review in **P16 Release**.

**Warning signs:**
- Website has NDCG numbers that don't appear in any JSON file
- Latency numbers on website differ from paper
- Marketing adjectives without hyperlinks
- "X% faster than Y" claims without A/B methodology
- Feature listed on website as "available" when it's behind a config flag or unfinished

**Phase to address:** **P13 Docs** (single source of truth infrastructure). **P14 Website** (claims-audit discipline). **P15 Paper** (same numbers as website). **P16 Release** (CLAIMS.md review).

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Silent `catch { }` in capture / embed / rerank | Pipeline doesn't crash on transient failures | Can't debug "memory stopped working"; errors invisible; bugs fester for months (H3) | Never; always at minimum `console.warn` + metric |
| Clamping scores to [0,1] with `Math.min(1.0, x*w)` | Scores "always valid" | Ties collapse, gate misfires, hybrid merge breaks (H1, P1-A) | Only after normalizeScores(), and only for presentation |
| Hand-edited numbers in docs/paper/website | Quick updates without build infrastructure | Guaranteed drift (H8 ports, M8 charts, coverage badge) | Never in a paper heading for public release |
| Benchmark runner parallel to production code path | Decouples benchmark from refactors | Benchmarks measure a phantom system; all numbers invalidated (runner corruption bug) | Never; use the production path with fixtures |
| Hardcoded test fixtures (SCIFACT_RESULTS literal) | Benchmark independent of runtime | Guaranteed to drift from actual results (M8) | Only with a build-time assertion that literals match JSON |
| `return bool` instead of `expect(bool).toBe(true)` in tests | Compact test syntax | Tests always pass regardless of behavior (M6 — 20 tests) | Never; ESLint ban |
| `const q = q.where(...)` fluent API | Idiomatic JS | Silent filter discard; data leakage (C1) | Never; wrap all query-builder calls |
| Single-dataset paper primary eval | Fast to ship | Reviewers dismiss the paper instantly | Never for a release-grade paper |
| Hardcoded HyDE timeout | Simple config | Benchmark failure → production hang (P1-C root cause) | Only if <5s and with fallback path |
| Defaulting MMR on without λ tuned per dataset | "Diversity is good" | 28% recall loss on SciFact | Never; λ must be tuned; default to 0.7+ |
| Reranker on every query without gate | "More compute = better results" | 41s p50 latency, reranker shuffles noise (out-of-domain) | Never; gate always on, domain-validated |
| Capture without classifier tests | Fast iteration on classifier logic | Silent regression → garbage in DB → retrieval quality dies | Never; classifiers are the quality gate |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| LanceDB fluent API | `.where()` return value discarded → silent filter | Always `q = q.where(...)`; wrap in helper; add isolation integration test; pin version until upstream bugs resolved |
| LanceDB hybrid query + where | `prefilter` arg silently inverted in v0.25.3 (issue #3095) | Pin a known-good version; explicitly test prefilter/postfilter behavior on every upgrade |
| LanceDB FTS with scalar index | `.where()` ignored when field has a scalar index (issue #1656) | Test FTS + filter on every indexed field; don't assume filters compose |
| LanceDB Arrow types from `getByIds` | `as MemoryChunk[]` returns raw Arrow Vectors → NaN cosines (M1) | Convert with `.toArray()` / `toJsNumberArray` at boundary |
| Nemotron embed provider | Missing query instruction prefix on query path → asymmetric embedding → retrieval collapse | Assert query instruction is applied in a test; log first-char prefix on embed |
| Nemotron-Super HyDE LLM | 15s cold start → 100% timeout → hardcoded fallback corrupts benchmark inputs | Warm the model; short timeout (3-5s); explicit fallback that uses original query vector; test the fallback path |
| DGX Spark TCP to embed endpoint | Transient network failure at init → permanent brick (C2) | Clear `initPromise` on failure; retry on next call |
| LanceDB FTS tokenizer vs query tokenizer | Mismatched tokenization (e.g., stopwords, stemming) → BM25 scores noise | Test the same document via FTS and via ingest; assert both produce the same tokens |
| Reranker probe call | Probe response parsed differently than actual rerank call → mock-ordering-dependent tests (M20) | Named mock handlers, per-URL dispatch; no call-order mocks |
| `pytrec_eval` cross-check | Reimplementing NDCG/MAP and not validating against canonical implementation (C4) | Always cross-check at least one dataset's metrics against `pytrec_eval` |
| OpenClaw template expansion | `${SPARK_BEARER_TOKEN}` literal not expanded, used as API key (M12) | Guard: if token starts with `${`, warn + fallback to env; add test |
| NVIDIA local LLM for dataset generation | No cloud egress ≠ no PII leak; the LLM may paraphrase source PII into outputs | Scrub input + scrub output + canary test; never trust "it's local" |
| Chokidar file watcher | File events fire before writes complete → partial reads | Debounce + size-stable check before reading |
| EmbedCache under concurrent writers | Race on cache key → double-compute or stale read (M13 flags cache bypass logic untested) | Test concurrent hit/miss; add `tests/cached-provider.test.ts` |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Reranker on every query | p50 > 1s, NDCG gains marginal | Reranker gate + latency budget + domain validation | Any query; already broke at 41s p50 |
| HyDE with cold-start LLM | Agent turn p95 > 5s | HyDE default off; short timeout (3-5s); graceful fallback | Cold GPU / first query after idle |
| Sequential NER per chunk (M4) | Ingest pipeline O(n) when O(1) possible | `Promise.all(...)` | Any ingest of >50 chunks; blocks user-visible ingest |
| Unbounded capture per turn | DB growth unbounded; retrieval drowns in noise | `MAX_CAPTURES_PER_TURN` enforced + tested | After ~1000 turns at 10 captures/turn; 10k memories quickly |
| Multiquery + HyDE + reranker compound | Per-query latency = multiquery × HyDE × rerank → seconds | Enable only one at a time; benchmark each | Any "all-on" config |
| Unbounded pool search + dedup → reranker | Reranker input grows with DB → O(N) reranker calls | `MAX_RERANK_CANDIDATES` cap (40); cap tested | >10k memories in a single pool |
| LanceDB table without compaction | Query time degrades as fragmented | Periodic `table.optimize()` or compaction policy | After many upserts; ~50k docs |
| Parent-child expansion | Parent fetched per chunk → N+1 | Batch parent fetch or cache | Docs with many child chunks, e.g., long PDFs |
| PDF chunker without cap | 5085 chunks for cuda-programming-guide.pdf → Spark OOM + SSH death | PDF chunk cap (already on the Monday TODO) | Any large PDF |
| In-place chunk mutation on parent expansion (L1) | Dedup + parent expansion interact badly | Copy-before-mutate | When dedup runs after expansion |
| Vectored-only for long narrative queries | Recall low because semantic chunking doesn't align | Use hybrid + reranker (after latency is fixed) | Long-form queries |

---

## Security & Privacy Mistakes

(This is the highest-cost class on this project.)

| Mistake | Risk | Prevention |
|---------|------|------------|
| Committing real emails to the golden dataset | PII exposure, credibility loss, GDPR-like concerns, doxxing family | Input+output scrubber, canary test, pre-commit hook, gitleaks CI |
| Committing private subnet IPs | Network topology disclosure; attack surface mapping | IP regex in scrubber + pre-commit |
| Committing real person names | Doxxing; family members named | Internal block-list in scrubber (not reproduced in repo) |
| Committing cron schedules + agent internals | Reveals operational patterns; infra fingerprinting | Exclude `.claude/`, `.openclaw/`, `AGENTS.md` content from any corpus |
| Nemotron-generated QA pairs copying PII verbatim | Generative model smuggles source PII into "synthetic" output | Scrub output; canary PII in input with assertion it's absent from output |
| Hardcoded API key or bearer token | Credential theft | `${VAR}` template with resolve-time check (M12); gitleaks in CI |
| FTS WHERE filter silently discarded (C1) | Cross-agent data leakage (school gets meta's memories) | Agent isolation integration test; wrapper helper; ESLint rule |
| `escapeSql()` only escapes quotes (L3) | Injection via metadata filter field | Parameterized queries where LanceDB supports; otherwise strict allowlist of field values |
| No prompt injection tests for actual user content | LLM compromise via memory content | `tests/security.test.ts` for zero-width, homoglyph, `<<SYS>>` chained patterns (M18) |
| Capture path ingests anything (no classifier test — H4) | Garbage memories; potential ingest of adversarial text | Test classifier with adversarial inputs; reject low-quality; log decision |
| Public repo + planning docs + memory notes committed | `HEY-OPENCLAW-STOP-BEING-RETARDED.md` and memory/*.md may contain operational details or personal voice | Scrub planning docs; consider `.gitignore` `.planning/` subtree from public release branch |
| Benchmark results committed without reproducibility | Claims can't be verified; opens credibility attack | Commit SHA + seed in every result JSON; reproducibility script |

---

## "Looks Done But Isn't" Checklist

Items that appear complete in the codebase, docs, or plans but have hidden gaps. Each must be verified before cut.

- [ ] **Hybrid retrieval** — often missing: score normalization BEFORE fusion; test with different score distributions; verify gate behavior post-H1 fix
- [ ] **Reranker gate** — often missing: actual verification that the gate fires; skip-rate histogram; OCMemory-domain validation
- [ ] **Cross-agent isolation** — often missing: integration test that actually seeds two agents and asserts no bleed on every read path (not just the one the audit caught)
- [ ] **Golden dataset** — often missing: capture-quality queries, cross-pool boundary queries, HyDE-negative cases, temporal decay tests, multi-hop reasoning (M17)
- [ ] **Benchmark results** — often missing: commit SHA metadata; bootstrap CI; paired tests; multi-dataset; sensitivity analysis
- [ ] **Paper** — often missing: confidence intervals, multi-dataset primary results, honest limitations matching measured behavior
- [ ] **Research paper "Bug Archaeology" section** — framing: rename to "Failure Modes in Production RAG" for formal tone
- [ ] **Documentation** — often missing: port numbers correct, version numbers aligned, coverage badge reflects reality, CONFIGURATION.md current with all Phase 8-12 features
- [ ] **CHANGELOG** — often missing: v0.4.0 release entry, PENDING markers removed for shipped features (M9)
- [ ] **Classifier pipeline** — often missing: tests for quality/heuristic/zero-shot/NER; classifier crash fallback; observability of reject decisions
- [ ] **Capture pipeline** — often missing: test coverage (0% currently), MAX_CAPTURES enforcement test, dedup-at-capture, error logging
- [ ] **HyDE** — often missing: default-off, fallback on timeout, REPLACE-not-average vectors, HyDE-negative golden queries
- [ ] **MMR** — often missing: λ tuned per dataset, recall guard, disable for high-R queries, benchmark sweep
- [ ] **Reranker** — often missing: p95 latency budget, position preservation (P1-B GUARD-A/B/C), OCMemory-domain benchmark, candidate count tuning
- [ ] **Docker test environment** — often missing: config key names match actual schema (C3 — entire config was silently ignored); post-fix, re-verify with a schema-validation test
- [ ] **Circuit breaker in embed queue** — often missing: tests for all state transitions (CLOSED→OPEN→HALF_OPEN), retry exhaustion, concurrency (H5)
- [ ] **Dims-lock** — often missing: first-boot creation, mismatch detection, corrupt file handling (H6)
- [ ] **Plugin tool surface expansion** — often missing: for each new tool, an isolation test, an embed-failure test, a schema-validation test, a capture-path test
- [ ] **Website** — often missing: CLAIMS.md backing every public assertion; benchmark numbers from JSON not hand-typed; CI-generated badges

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Golden dataset PII leak (ALREADY OCCURRED) | HIGH | 1. `git filter-repo` to remove committed PII from history. 2. Force-push (risky — coordinate with collaborators, warn watchers). 3. Rotate any credentials exposed. 4. Assume emails and IPs are burned; update privacy stance accordingly. 5. Install scrubber + pre-commit hook to prevent recurrence. |
| Benchmark numbers invalidated by post-run bug fix | MEDIUM | Re-run benchmarks from the fixed commit; update all JSON result files; regenerate paper tables and website charts via `scripts/gen-docs.ts`; diff against old numbers and document the change in CHANGELOG or paper erratum |
| Reranker shipped as default, destroys OCMemory NDCG | MEDIUM | Flip default to `rerank.enabled = false`; retain as opt-in; document the finding in paper; re-benchmark |
| HyDE shipped enabled, agent turns hang 30s | LOW | Flip default off; ship hotfix; publish to plugin registry if applicable |
| Cross-agent data bleed in production (C1 recurrence) | HIGH | 1. Disable plugin gateway-wide. 2. Audit all recent recall logs to identify agents exposed to foreign memories. 3. Ship fix + isolation test. 4. Notify users. 5. Consider DB reindex if cross-contamination corrupted the data. |
| Score fusion bug inflates results | MEDIUM | Re-run benchmarks; paper/website erratum; updated numbers replace old ones; add score-invariant tests |
| MMR destroys recall on a new dataset | LOW | λ sweep; pick per-dataset value; add to golden-set regression |
| Nemotron generation produces PII | HIGH | Stop the generation; burn the output file (don't commit); run canary test on scrubber; fix scrubber; re-run |
| Documentation drift found at release time | LOW | `scripts/gen-docs.ts` regenerates all; verify port grep; verify coverage badge; ship |
| Paper reviewer finds a methodology error | MEDIUM | Publish a revised version (arXiv v2, TMLR revision); update repo; note in CHANGELOG as "paper-v2" |
| Website claim unbacked | LOW | Remove claim or add source link; do so before ship |
| Benchmark runner corruption caught post-paper | MEDIUM-HIGH | Re-run, re-chart, re-paper; prepare a "bug archaeology" section describing the failure; honesty becomes a feature |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase(s) | Verification |
|---------|--------------------|--------------|
| 1. Paper/benchmark invalidation by late bug fixes | P3 Criticals, P4 High-sev (gating); P10 BEIR, P11 OCMemory-bench, P15 Paper (consumers) | Commit SHA in every result JSON; CI asserts paper numbers match JSON; benchmarks only run post-P3 merge |
| 2. Golden dataset PII leak (load-bearing) | **P2 Privacy** (scrubber + canary + pre-commit); **P9 Golden-set** (scrubber in pipeline); **P16 Release** (final audit) | Seeded canary PII survives zero stages; gitleaks CI green; pre-commit hook blocks known-bad patterns; manual review of golden-ocmemory-scrubbed.json before commit |
| 3. Hybrid merge score distribution failures | P3 Criticals (fix known); P4 High-sev (score-invariant tests + branded Score01 type) | `tests/score-invariants.test.ts` passes; per-stage histograms in benchmark runner; no stage changes NDCG by <0.5% (dead stage check) |
| 4. Cross-encoder reranker destroys precision | P3 Criticals (latency + P1-B position preservation); P7 Spark-v2 (bge-reranker-v2-m3 re-validation); P11 OCMemory-bench (domain test); P15 Paper (honest report) | OCMemory NDCG delta reported; reranker p95 < 500ms; gate skip rate nonzero on easy queries |
| 5. HyDE hurts + burns latency | P3 Criticals (default off, fix averaging, fallback); P7 Spark-v2 (Mini-4B keeps default off); P10/P11 (benchmark both); P15 Paper (honest HyDE-hurts report) | Default `hyde.enabled: false`; HyDE-on vs HyDE-off rows in results; fallback path covered by test |
| 6. MMR diversity kills recall | P3 Criticals (default λ ≥ 0.7, verify post-H1); P10 BEIR (λ sweep); P15 Paper (findings) | λ sweep table in results; Recall@100 never decreases between MMR-off and MMR-on by >2% |
| 7. Cross-agent data leakage (C1 class) | P3 Criticals (C1 fix); P4 High-sev (ESLint + wrapper + isolation tests); P5 Tool-inject + P6 Tool-surface (per-tool isolation test) | `tests/isolation.test.ts` seeds 2 agents, runs every read path, asserts zero bleed; ESLint CI green |
| 8. Benchmark runner corrupts inputs | P3 Criticals (delete dead code C5, fix runner to use production path); P10 BEIR (pytrec_eval cross-check) | `pytrec_eval` vs `evaluation/metrics.ts` agreement on SciFact; dumped stage fixtures committed and replayable |
| 9. Documentation drift | **P13 Docs** (single source of truth via `scripts/gen-docs.ts`); P15 Paper (numbers from JSON); P16 Release (grep + version pin) | `grep "18081\|18098" docs/ paper/` returns 0; version pin assertion in CI; coverage badge auto-generated |
| 10. Paper without statistical significance | P10/P11 (bootstrap infra); P15 Paper (CI + p-values + multi-dataset + sensitivity) | Every paper number has ± CI; every comparison has paired test + p-value; multi-dataset tables |
| 11. Auto-capture garbage, retrieval drowns | P3 Criticals (fix silent errors H3); P4 High-sev (`tests/capture.test.ts`); P9 Golden-set (capture-quality queries); P11 OCMemory-bench (end-to-end capture test) | `src/auto/capture.ts` coverage > 70%; capture log `capture-log.jsonl` has rejection reasons; golden set has rejection test cases |
| 12. Website claims unbacked | P13 Docs (gen-docs infra); **P14 Website** (claims audit + CLAIMS.md); P16 Release (final walk) | CLAIMS.md exists and every public claim points to a repo artifact; no hand-edited numerics on website |

---

## Sources

### Project-internal (HIGH confidence — directly observed in this repo)
- `.planning/PROJECT.md` — milestone Active items, Context, Constraints, Key Decisions
- `docs/AUDIT-2026-04-02.md` — C1–C5, H1–H9, M1–M20, L1–L9 findings; paper rigor audit; PII leak enumeration; security audit
- `docs/PLAN-phase13.md` — P1-A through P3-D pipeline hardening items; severity corrections
- `docs/RESEARCH-SOTA-2026-VALIDATED.md` — SOTA findings (HyDE "not for us", reranker domain mismatch, RRF failure modes, Zep/Mem0 comparison, chunking impact)
- `memory/2026-03-29-benchmark-plan.md` — Two-DB benchmark strategy (`testDbBEIR` vs `testDbOCMemory`); "NEVER run BEIR queries against OCMemory" rule; Nemotron golden dataset plan

### External (MEDIUM confidence — 2025-2026 ecosystem research)
- [Best AI Agent Memory Frameworks 2026: Mem0, Zep, LangChain, Letta](https://atlan.com/know/best-ai-agent-memory-frameworks-2026/)
- [5 AI Agent Memory Systems Compared: Mem0, Zep, Letta, Supermemory (2026 Benchmark Data)](https://dev.to/varun_pratapbhardwaj_b13/5-ai-agent-memory-systems-compared-mem0-zep-letta-supermemory-superlocalmemory-2026-benchmark-59p3)
- [AI Agent Memory Systems in 2026: Mem0, Zep, Hindsight, Memvid — Compared](https://yogeshyadav.medium.com/ai-agent-memory-systems-in-2026-mem0-zep-hindsight-memvid-and-everything-in-between-compared-96e35b818da8)
- [Retrieval Is the Bottleneck: HyDE, Query Expansion, Multi-Query RAG (Jan 2026)](https://medium.com/@mudassar.hakim/retrieval-is-the-bottleneck-hyde-query-expansion-and-multi-query-rag-explained-for-production-c1842bed7f8a)
- [RAG Isn't Accuracy: 8 Confident Failure Modes (Mar 2026)](https://medium.com/@ThinkingLoop/rag-isnt-accuracy-8-confident-failure-modes-568cfe855694)
- [Weighted Reciprocal Rank Fusion in Elasticsearch](https://www.elastic.co/search-labs/blog/weighted-reciprocal-rank-fusion-rrf)
- [Reciprocal Rank Fusion — Azure AI Search](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)
- [BAAI/bge-reranker-v2-m3 — HuggingFace](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [BGE Reranker Cross-Encoder Reranking for RAG 2026](https://markaicode.com/bge-reranker-cross-encoder-reranking-rag/)
- [Maximum Marginal Relevance — Qdrant](https://qdrant.tech/blog/mmr-diversity-aware-reranking/)
- [Maximum Marginal Relevance — Elasticsearch Labs](https://www.elastic.co/search-labs/blog/maximum-marginal-relevance-diversify-results)
- [Enhancing RAG with MMR in Azure AI Search](https://farzzy.hashnode.dev/enhancing-rag-with-maximum-marginal-relevance-mmr-in-azure-ai-search)
- [LanceDB bug #3095 — prefilter silently inverted in hybrid search (March 2026)](https://github.com/lancedb/lancedb/issues/3095)
- [LanceDB bug #1656 — where clause ignored on FTS with scalar index](https://github.com/lancedb/lancedb/issues/1656)
- [LanceDB Metadata Filtering docs](https://docs.lancedb.com/search/filtering)
- [BEIR repository and evaluation harness](https://github.com/beir-cellar/beir)
- [BEIR benchmark — NeurIPS 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf)
- [ICML 2025: How Contaminated Is Your Benchmark?](https://icml.cc/virtual/2025/poster/43619)
- [Awesome Data Contamination — paper list](https://github.com/lyy1994/awesome-data-contamination)
- [Building a Golden Dataset for AI Evaluation](https://www.getmaxim.ai/articles/building-a-golden-dataset-for-ai-evaluation-a-step-by-step-guide/)
- [Synthetic Data for RAG Evaluation — Red Hat Developers (Feb 2026)](https://developers.redhat.com/articles/2026/02/23/synthetic-data-rag-evaluation-why-your-rag-system-needs-better-testing)
- [Fine-tuning LLMs on Sensitive Data: 19% PII Leakage](https://medium.com/secludy/fine-tuning-llm-on-sensitive-data-lead-to-19-pii-leakage-ee712d8e5821)
- [PII-Scope: Benchmark for Training Data PII Leakage (arXiv 2410.06704)](https://arxiv.org/html/2410.06704v1)
- [ProPILE: Probing Privacy Leakage in LLMs (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/420678bb4c8251ab30e765bc27c3b047-Paper-Conference.pdf)
- [Analyzing PII Leakage in LLMs (arXiv 2302.00539)](https://arxiv.org/abs/2302.00539)

---
*Pitfalls research for: scientifically-rigorous RAG memory plugin for LLM agents (memory-spark v1.0 milestone)*
*Researched: 2026-04-09*
