# Architecture Research — memory-spark v1.0

**Domain:** Production-grade agentic RAG memory plugin (brownfield TypeScript/Node, OpenClaw plugin SDK, LanceDB + GPU-backed Spark services)
**Researched:** 2026-04-09
**Confidence:** HIGH (primary source: existing code + `docs/ARCHITECTURE.md` + `docs/PLAN-spark-v2-architecture.md`)

---

## 1. Canonical Component Boundaries

memory-spark already has the right skeleton. The boundaries below are the ones implied by the existing `src/` tree and `src/manager.ts`, expressed as the canonical 2026 agentic-RAG layering. Future work should **consolidate inside these boundaries, not add new ones**.

```
┌──────────────────────────────────────────────────────────────────────┐
│                  OpenClaw Plugin Surface (index.ts)                   │
│  - Tool registrations (memory_search, memory_store, …14 total)        │
│  - Hook registrations (before_prompt_build, agent_end)                │
│  - Singleton state + init (getState)                                  │
├──────────────────────────────────────────────────────────────────────┤
│                  Plugin Core (src/manager.ts)                         │
│  - MemorySparkManager: search() + readFile() + status()               │
│  - Dependency container: backend, embed, reranker, queue              │
├──────────────────────────────────────────────────────────────────────┤
│   Recall path (src/auto/recall.ts)   │  Capture path (src/auto/       │
│                                       │  capture.ts, src/auto/        │
│                                       │  mistakes.ts)                 │
├──────────────────────────────────────┴───────────────────────────────┤
│                      Retrieval / Ingest Primitives                    │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────┐ │
│  │  query/   │ │   hyde/   │ │  rerank/  │ │ classify/ │ │ ingest/ │ │
│  │ expander  │ │ generator │ │ reranker  │ │ zero/ner/ │ │ pipeline│ │
│  └───────────┘ └───────────┘ └───────────┘ │ quality/  │ │ parsers │ │
│                                             │ heuristic │ │ watcher │ │
│                                             └───────────┘ │ sessions│ │
│                                                           │ workspace│ │
│                                                           └─────────┘ │
├──────────────────────────────────────────────────────────────────────┤
│                Embedding Subsystem (src/embed/)                       │
│  provider → cached-provider → queue (circuit breaker)                 │
│  dims-lock (invariant guard) │ cache (LRU) │ chunker                  │
├──────────────────────────────────────────────────────────────────────┤
│                Storage Subsystem (src/storage/)                       │
│  backend.ts (interface) ← lancedb.ts (impl)                           │
│  pool.ts (single source of truth for logical isolation)               │
├──────────────────────────────────────────────────────────────────────┤
│                Cross-cutting (src/security.ts, src/config.ts)         │
└──────────────────────────────────────────────────────────────────────┘
              │                           │                  │
              ▼                           ▼                  ▼
        LanceDB (local)       GPU Spark services       Filesystem
                              (HyDE LLM, embed,        (workspace +
                               rerank, VL, OCR)         sessions)
```

### Component Responsibilities (canonical boundaries)

| Component | Owns | Knows nothing about |
|-----------|------|---------------------|
| `index.ts` | OpenClaw plugin SDK contract, tool schemas, hook wiring | Retrieval math, LanceDB, Spark URLs |
| `src/manager.ts` | Plugin-facing search/readFile API, lifecycle, probes | Hook events, tool schemas |
| `src/auto/recall.ts` | Hook handler for `before_prompt_build`, full recall pipeline orchestration | Chunk table schema, LanceDB internals |
| `src/auto/capture.ts` | Hook handler for `agent_end`, classification routing, dedup | Chunk text, vector math |
| `src/ingest/pipeline.ts` | Extract → chunk → embed → store orchestration for files | Retrieval, hooks |
| `src/ingest/parsers.ts` | PDF/DOCX/Markdown/session extraction, OCR fallback chain | Pool routing |
| `src/ingest/watcher.ts` | Filesystem → ingest pipeline bridge | Classification |
| `src/embed/provider.ts` | Provider-agnostic embed contract (`spark`/`openai`/`gemini`/future `vl`) | Cache, queue |
| `src/embed/queue.ts` | Backpressure, retry/backoff, circuit breaker for embed calls | Cache policy |
| `src/embed/cached-provider.ts` | LRU cache layer over the queue for query embeddings only | Document embeddings |
| `src/embed/dims-lock.ts` | Invariant: dims of the store match the provider | Retrieval |
| `src/embed/chunker.ts` | Flat + hierarchical (parent-child) chunking | Embedding math |
| `src/classify/*` | Quality gate, zero-shot, NER, heuristic classifiers | Storage, retrieval |
| `src/hyde/generator.ts` | Hypothetical document generation via LLM | Retrieval pipeline orchestration |
| `src/query/expander.ts` | Multi-query reformulation | HyDE, classification |
| `src/rerank/reranker.ts` | Cross-encoder rerank w/ dynamic gate | Vector math, pools |
| `src/storage/backend.ts` | `StorageBackend` interface — the **narrow waist** of the project | Any specific DB |
| `src/storage/lancedb.ts` | LanceDB implementation, IVF_PQ + FTS index management | Classification, hooks |
| `src/storage/pool.ts` | **Single source of truth** for pool routing and isolation | Anything else |
| `src/security.ts` | Prompt-injection detection, XML-escaping for `<relevant-memories>` | Retrieval math |
| `src/config.ts` | `MemorySparkConfig` type, resolver, defaults | Everything else |

### The one boundary that is subtly wrong today

`src/manager.ts` `search()` reimplements a simplified copy of the recall pipeline (hybridMerge → sourceWeighting → temporalDecay → mmrRerank → rerank). That is a **duplicated orchestrator**. For v1.0 it should be refactored so that `recall.ts` exports a pure `runRecallPipeline(query, ctx, deps)` function that both the `before_prompt_build` hook and `manager.search()` call. This eliminates divergence, simplifies benchmarks (they can call the same entry point), and collapses the test surface. Put this on the roadmap in the "critical fixes" window, not as new features.

---

## 2. Data Flow — Capture Path vs Recall Path

These are the two load-bearing flows. Keep them strictly separate. Do not let a capture component import a recall component or vice-versa — they share only the `StorageBackend`, `EmbedProvider`, `config`, and `security` primitives.

### Capture Path (async, batched, fire-and-forget)

```
agent_end event  ──▶  createAutoCaptureHandler
    │                       │
    │                       ├─▶ extractCaptureMessages (message filter)
    │                       ├─▶ looksLikePromptInjection  (security gate 1)
    │                       ├─▶ looksLikeCaptureGarbage   (garbage gate)
    │                       ├─▶ scoreChunkQuality         (quality gate, ≥0.3)
    │                       ├─▶ classify (zero-shot → heuristic fallback)
    │                       ├─▶ resolvePool               (routing)
    │                       ├─▶ dedup query (vectorSearch, >0.92 = skip)
    │                       ├─▶ queue.embed               (circuit-breaker queue)
    │                       └─▶ backend.upsert(chunk)
    │
    └─ returns immediately (void) — never blocks the agent turn

Workspace file change (chokidar)  ──▶  ingest/watcher.ts
    └─▶ ingest/pipeline.ts (ingestFile)
            ├─▶ parsers.extractText / extractPdfBatched
            ├─▶ chunker.chunkDocument(Hierarchical)
            ├─▶ NER + qualityScore per chunk (batched)
            ├─▶ queue.embedBatch (backpressure at the queue)
            └─▶ backend.upsert (idempotent by id)
```

Characteristics:
- **Async/non-blocking.** Capture must never delay `agent_end` completion. All back-end work happens after the hook returns.
- **Batched embed.** Ingest calls `embedBatch` (not one-by-one); capture calls individual `embed` because a turn produces ≤3 chunks.
- **Backpressure in exactly one place:** `src/embed/queue.ts`. Every embed caller must go through the queue — direct `provider.embed()` calls are an anti-pattern (existing audit finding: "unguarded embed calls").
- **Idempotency:** ingest pipeline uses deterministic chunk IDs so re-ingest is an upsert, not duplication.

### Recall Path (synchronous, must return in ≤ agent tolerance)

```
before_prompt_build event
    │
    ▼
createAutoRecallHandler
    │
    ├─ 1. buildQuery(messages, queryMessageCount)
    ├─ 2. cleanQueryText            (strip LCM/metadata/recursive recall)
    ├─ 3. expandQuery               (multi-query, parallel)
    ├─ 4. HyDE generate             (LLM call, timeout-bounded, fallback on fail)
    ├─ 5. cachedEmbed.embedQuery    (LRU cache hit? return)
    │       └─▶ queue.embedQuery   (circuit breaker)
    │             └─▶ provider.embed("query" instruction prefix)
    │
    ├─ 6. Per-pool search (currently sequential per pool group — optimize):
    │       ├─▶ backend.vectorSearch(queryVec, {pools: agent_*})
    │       ├─▶ backend.ftsSearch(queryText, {pools: agent_*})
    │       └─▶ hybridMerge (RRF)
    │       … repeat for shared_* pools
    │
    ├─ 7. applySourceWeighting       (capture 1.5×, mistakes 1.6×, …)
    ├─ 8. applyTemporalDecay         (floor + (1-floor)·exp(-rate·age))
    ├─ 9. deduplicateSources         (Jaccard >0.85 within same source)
    ├─10. computeRerankerGate        (top-5 spread: skip if >0.08 or <0.02)
    ├─11. reranker.rerank            (cross-encoder, conditional)
    ├─12. RRF blend (vector ranks + rerank ranks)
    ├─13. mmrRerank (λ=0.9, cosine-on-vector w/ Jaccard fallback)
    ├─14. parent-child expansion     (child → parent text)
    ├─15. LCM dedup                  (skip chunks already in LCM summary)
    ├─16. looksLikePromptInjection   (security filter)
    ├─17. token-budget greedy fill   (≤ maxInjectionTokens)
    └─18. formatRecalledMemories     (<relevant-memories> XML)
          │
          ▼
    return { prependContext: xml }
```

Characteristics:
- **Synchronous path** — every ms is on the critical path. Budget is roughly 500-1500ms for the whole flow; HyDE alone must be <1s.
- **Degraded-mode contract:** if `queue.embedQuery` throws (Spark down), recall degrades to FTS-only.
- **No backpressure here** — recall is never queued. If embed queue is unhealthy, embed returns error immediately and we degrade.
- **Per-pool sequential search is a known inefficiency** — roadmap should include a "parallelize pool-group lookups" item.

---

## 3. Plugin Boundary — Isolating RAG Internals from OpenClaw

Treat the OpenClaw plugin SDK as a **thin adapter layer** over a RAG core that could, in principle, be extracted as its own npm package.

### Rules

1. **`index.ts` is the only file that imports from `openclaw/plugin-sdk`.** Nothing in `src/` may import plugin-sdk types.
2. **Hook handlers are factories, not functions.** `createAutoRecallHandler(deps)` returns the actual hook — explicit deps injection.
3. **Event types are the plugin boundary's only schema.** `BeforePromptBuildEvent` / `AgentEndEvent` / `HookContext` are declared locally, not imported from the SDK.
4. **The manager is the seam for `memory_*` tools.** All tool implementations in `index.ts` call `MemorySparkManager` methods.
5. **Config is a frozen snapshot.** `resolveConfig()` returns `MemorySparkConfig` passed to every factory. No lazy runtime reads.
6. **Singleton state is held only in `index.ts`.** `src/` modules are stateless by contract.

An ESLint `no-restricted-imports` rule scoped to `src/**` would enforce rule 1 and catch drift.

---

## 4. Test Harness Architecture for GPU-Dependent RAG

The existing Docker harness at `<external>/openclaw-plugin-test/` has the right shape but is currently broken by a config mismatch (C3).

### Three tiers of tests, three bring-up costs

| Tier | Runtime | External deps | When it runs |
|------|---------|---------------|--------------|
| **T1 Unit** (`tests/unit.ts`) | vitest, in-process, <5s total | None | Every commit, pre-push hook |
| **T2 Integration** (`tests/harness.ts`) | vitest + real LanceDB tmpdir + real Spark services | GPU Spark reachable | CI nightly + before benchmark runs |
| **T3 Benchmark + E2E** (`evaluation/`, docker compose) | Docker container boots full plugin inside OpenClaw | Full Spark stack + golden dataset + BEIR datasets | Manual + pre-release |

### T1: Unit tests must not touch the network

Fix the ~20 tests using `return bool` instead of `expect()` — pre-work for any benchmarking. A linter rule or test-env `global.fetch = () => { throw }` mock is the cheapest enforcement.

### T2: Integration tests as a fixture

- Ephemeral LanceDB dir per test file, torn down in `afterAll`
- Probe-then-skip pattern at `beforeAll`: ping each Spark port; skip suite with clear message if unreachable
- Test config profile that shrinks every knob
- Deterministic seeds where possible

### T3: Docker compose pattern

```
<external>/openclaw-plugin-test/
├── docker-compose.yml              # The plugin-under-test container
├── docker-compose.override.yml     # Optional local tweaks (gitignored)
├── Dockerfile                      # Plugin runtime (Node 22 + OpenClaw + plugin mount)
├── configs/
│   └── test-openclaw.json          # Writes Spark URLs via env interpolation
├── workspaces/
│   ├── test-agent/                 # Clean empty agent workspace
│   └── fixtures/                   # Golden dataset workspace (scrubbed, committed)
├── data/                           # Ephemeral LanceDB (gitignored)
└── scripts/
    ├── up.sh                       # Pre-flight: probe Spark → compose up
    ├── benchmark.sh                # docker compose run --rm → node evaluation/run-all.ts
    └── probe-spark.sh              # Independent health check of all ports
```

Key existing choices to keep: `network_mode: host`, plugin source mounted as volume, config file mounted read-only, data dir volume.

What's missing:
1. Service probe script (`scripts/probe-spark.sh`)
2. Separate compose profiles for BEIR vs OCMemory
3. Up-to-date bring-up doc listing every port, env var, required Spark model IDs

### SOTA service-soup pattern

- Treat every GPU service as an OpenAI-compatible HTTP endpoint
- Put a tiny "spark-health" aggregator in the config layer
- Version pin model IDs in config, not code; assert probed model ID matches
- Semantic versioning of the model pipeline — any dim/prompt/range change is a "v2" migration requiring dims-lock refusal to start

---

## 5. Multi-Corpus Data Layer — BEIR vs OCMemory Isolation

The most important architectural invariant for this milestone. The benchmark plan explicitly says mixing corpora invalidates results.

### The invariant

**Two physically separate LanceDB directories, selected via config at boot time, never co-resident.**

```
~/.openclaw/data/memory/               ← production / default
<external>/openclaw-plugin-test/data/
    ├── testDbBEIR/                    ← BEIR queries only
    └── testDbOCMemory/                ← golden dataset only
```

### Why not "one DB with a corpus pool"?

1. **IVF_PQ training is corpus-sensitive.** Optimal params for 5k BEIR chunks are wrong for 37k agent memory.
2. **FTS BM25 statistics are corpus-global.** Mixing BEIR and agent memory contaminates IDF vocabulary.
3. **Rerank score distributions differ per corpus.** Dynamic reranker gate was calibrated on one distribution.
4. **Structural guarantee is always better than runtime safety check.**

### The enforcement pattern

Add a `corpus-lock.json` file next to `dims-lock.json` in each LanceDB dir. On boot, if `cfg.corpus !== fileContents.corpus`, refuse to start with a fatal error. Defense-in-depth on top of directory separation, ~20 lines of code.

### Benchmark runner must enforce it

No "run both corpora" mode. BEIR or OCMemory, each with its own config, each writing to its own results dir:

```
evaluation/results/
├── beir/
│   ├── scifact-2026-04-09.json
│   ├── fiqa-2026-04-09.json
│   └── nfcorpus-2026-04-09.json
└── ocmemory/
    └── golden-2026-04-09.json
```

No unified "memory-spark overall score." This is the scientifically honest position.

---

## 6. Multi-Tenant / Multi-Corpus Isolation — Defense in Depth

Three layers, because agent data leaking between agents is a fatal class of bug.

### Layer 1 — Physical directory (corpus isolation)

Covered above. Different benchmark corpora = different `lancedbDir`.

### Layer 2 — Row-level pool + agent_id filter

Within a single `memory_chunks` table:

```sql
WHERE pool IN ('agent_memory','agent_tools','agent_mistakes')
  AND agent_id = :current_agent
UNION
WHERE pool IN ('shared_knowledge','shared_mistakes','shared_rules')
  -- no agent_id filter on shared pools
```

`src/storage/pool.ts` is the single source of truth for the pool enum. Every query site must use the `pools: string[]` option on `SearchOptions` — never a raw `where` string. **Missing `pool` must default to agent-only pools**. Lint-enforceable contract: `backend.vectorSearch`/`ftsSearch` calls must pass `pools`.

### Layer 3 — `user_id` for multi-user / gateway isolation

`MemoryChunk.user_id` already exists. Every search must carry current `user_id` from hook context into `SearchOptions.userId`.

### The privacy invariant (load-bearing)

From PROJECT.md: "No personal OpenClaw data may enter the repo, fixtures, benchmark corpora, or any public artifact — ever."

Architectural expression:

1. **Benchmark corpora live outside the repo** (verify `.gitignore`).
2. **LanceDB dirs live outside the repo**, under `<external>/openclaw-plugin-test/data/`.
3. **The scrubbing boundary is the golden dataset generator on DGX Spark**, not the plugin. Plugin never scrubs — it stores what it's given. Generator produces scrubbed output, committed scrubbed, only scrubbed touches benchmark.
4. **CI check** greps repo for known PII patterns (email domains, personal dirs under `~/`) and fails build if found.

---

## 7. Build Order — Dependency-Respecting Phase Sequence

Three hard rules shape the critical path:

- **You cannot benchmark what is broken.** Critical fixes must precede any benchmark run.
- **You cannot publish what leaks PII.** Privacy foundation must precede any public artifact generation.
- **You cannot validate SOTA claims without current sources.** SOTA validation pass must precede paper expansion and website copy.

### Recommended phase order

```
Phase 1: Read-only Recon               (blocks nothing, unblocks everything)
    │
Phase 2: Privacy Foundation            (must precede any data touch)
    │
Phase 3: Critical Bug Fixes            (blocks all benchmarking and logic upgrades)
    │   Includes: manager.search() refactor to share recall pipeline
    │
Phase 4: Test Harness Restoration      (depends on: Phase 3 fixes)
    │   Corpus isolation enforcement (corpus-lock.json) belongs here
    │
Phase 5: Logic Upgrades (Spark v2 Phase A)   (depends on: Phase 4 safety net)
    │   HyDE LLM swap, EasyOCR retirement, GLM-OCR port separation
    │   Parallelize per-pool search, tool-calling injection
    │
Phase 6: Benchmarks                    (depends on: Phases 3, 4, 5)
    │   Golden dataset gen → scrubbed commit; BEIR + OCMemory runs
    │
Phase 7: SOTA Validation               (can run in parallel with Phase 6)
    │   Verify every claim in existing research docs against 2026 literature
    │
Phase 8: Documentation Overhaul        (depends on: Phases 6, 7)
    │
Phase 9: Website                       (depends on: Phase 8 — docs must be stable)
    │
Phase 10: Paper Expansion              (depends on: Phases 6, 7, 8)
    │
Phase 11: v1.0 Release                 (depends on: all above)
```

### Deferred post-v1.0 (Spark v2 Phases B/C/D)

Out of scope for v1.0 per PROJECT.md. Don't gate v1.0 on VL embedder migration re-indexing or hardware partition.

### Critical dependency to not miss

**Phase 3 must include the `manager.search()` refactor** — otherwise Phase 6 benchmarks measure a fiction.

---

## Anti-Patterns Specific to This Project

1. **Duplicate pipeline orchestration** — `manager.search()` reimplements a simplified recall pipeline. Fix: extract `runRecallPipeline(query, ctx, deps)` from `recall.ts`; both callers invoke it.
2. **Unguarded embed calls** — direct `provider.embed()` instead of going through `EmbedQueue` bypasses circuit breaker. Audit every callsite.
3. **Searching without a pool filter** — reference pools leak into auto-recall results. Make `pools` required.
4. **Corpus mixing for "convenience"** — hard-enforce separate LanceDB dirs with `corpus-lock.json`.
5. **Module-level singletons inside `src/`** — breaks parallel tests, benchmark isolation, recovery paths. State lives in `index.ts` or factory closures.
6. **Plugin SDK types leaking into `src/`** — ESLint restricted-import rule to enforce.

---

## Integration Points — External GPU Services

| Service | v1 port | v2 port | Protocol | Integration pattern | Gotchas |
|---------|---------|---------|----------|---------------------|---------|
| HyDE LLM (Nemotron-Super-120B → Mini-4B / Qwen3-4B) | 18080 | 18080 | OpenAI /v1/chat/completions | Stateless HTTP, 5s timeout, fallback on failure | 120B cold-start blows budget; Phase 5 fixes |
| GLM-OCR (VL 0.9B) | 18080 (shared) | 18081 (dedicated) | OpenAI /v1/chat/completions (VL) | Called from parsers.ts PDF fallback | Shares VRAM with HyDE LLM in v1 |
| Embed (llama-embed-nemotron-8b → Qwen3-VL-Embedding-8B) | 18091 | 18091 | OpenAI /v1/embeddings | Via EmbedProvider → EmbedQueue → circuit breaker | Instruction prefix ("query"/"document") must match; dims-lock |
| Reranker (llama-nemotron-rerank-1b → bge-reranker-v2-m3 or Qwen3-VL-Reranker-2B) | 18096 | 18096 | Rerank HTTP API | Called only when dynamic gate passes (~22% of queries) | Score saturation at 1B scale |
| EasyOCR | 18097 | RETIRED | HTTP | PDF fallback in parsers.ts | Remove in Phase 5 |
| NER / zero-shot / summarizer | 18110–18113 | unchanged | HTTP | classify/*.ts | Can fail-open |

---

## Internal Boundaries (import contract)

| From → To | Allowed? | Notes |
|-----------|----------|-------|
| `index.ts` → `src/*` | Yes | Only file that may import `openclaw/plugin-sdk` |
| `src/auto/*` → `src/storage/*` | Yes | Through `StorageBackend` interface |
| `src/auto/*` → `src/embed/*` | Yes | Through `EmbedQueue` / `CachedEmbedProvider`, not `provider` |
| `src/auto/recall.ts` ↔ `src/auto/capture.ts` | **No** | Independent flows; share only primitives |
| `src/storage/*` → `src/embed/*` | **No** | Storage doesn't know about embedding providers |
| `src/storage/*` → `src/classify/*` | **No** | Storage is dumb persistence |
| `src/ingest/*` → `src/auto/*` | **No** | Ingest writes; recall reads. No direct path. |
| `src/*` → `openclaw/plugin-sdk` | **No** | Enforce via lint |
| `tests/*` → `src/*` and `index.ts` | Yes | Tests can reach anywhere |

---

## Roadmap Implications

1. **Phase 3 must include `manager.search()` pipeline refactor** — without this, Phase 6 benchmarks measure a fiction.
2. **Phase 4 must precede Phase 5** — changing logic without a test net is dangerous.
3. **Corpus isolation enforcement** belongs in Phase 4 (harness infrastructure), not Phase 6.
4. **Parallelize per-pool search** is a Phase 5 logic upgrade, not a Phase 3 critical fix.
5. **Privacy Phase 2 must add CI grep for PII patterns** — belt-and-suspenders.
6. **Retiring EasyOCR** is cheap — land in Phase 5, not deferred to Spark v2 Phase C.
7. **Plugin boundary lint rule** belongs in Phase 4 — prevents regression in Phase 5.

## Open Questions

- Exact VL embedding model selection (deferred to stack research — see STACK.md findings)
- Exact BGE reranker latency on hardware vs llama-nemotron-rerank — measurable in Phase 6
- LanceDB 0.27.1 FTS+WHERE fix coverage of all compound-query cases — validate during Phase 4
- Whether `memory_index_status` tool should aggregate Spark health into single endpoint — decide during Phase 5

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| Component boundaries | HIGH | Derived from existing code + docs/ARCHITECTURE.md |
| Capture/recall data flows | HIGH | Observed in `src/auto/recall.ts` and `src/auto/capture.ts` |
| Plugin boundary isolation pattern | HIGH | Already shape of `index.ts`; recommendation is "keep it" |
| Test harness pattern | MEDIUM-HIGH | Docker env exists; added probes + profiles are new |
| Multi-corpus isolation | HIGH | Explicitly decided in PROJECT.md; this doc codifies enforcement |
| Multi-agent isolation | HIGH | Pool + agent_id + user_id already in schema |
| Build order | HIGH | Concrete dependencies: can't benchmark broken, can't publish leaking, can't validate without sources |
| External service specifics (exact VL model, BGE latency) | LOW | Deferred to STACK.md + Phase 7 SOTA validation |

---

## Sources

- `docs/ARCHITECTURE.md` — authoritative current-state architecture (HIGH)
- `docs/PLAN-spark-v2-architecture.md` — service migration plan (HIGH)
- `.planning/PROJECT.md` — milestone scope, privacy constraint, corpus separation decision (HIGH)
- `src/manager.ts` — observed plugin core API (HIGH)
- `index.ts` — observed plugin boundary and init pattern (HIGH)
- `src/auto/recall.ts` — recall pipeline ordering and HyDE integration (HIGH)
- `src/auto/capture.ts` — capture gates (HIGH)
- `src/storage/pool.ts` — pool routing single source of truth (HIGH)
- `src/storage/backend.ts` — StorageBackend interface, user_id field (HIGH)
- `src/ingest/pipeline.ts` — ingest flow (HIGH)
- `<external>/openclaw-plugin-test/docker-compose.yml` — existing test harness shape (HIGH)
- `docs/PLAN-v040-release.md` — phase structure context (HIGH)
