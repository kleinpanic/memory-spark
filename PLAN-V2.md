# memory-spark Improvement Plan v2

> Written: 2026-03-26 14:00 EDT | Author: KleinClaw-Meta
> Status: **PROPOSAL — needs Klein approval**
> Repo: `~/codeWS/TypeScript/memory-spark/` (dev) → symlinked to `~/.openclaw/extensions/memory-spark/`
> Codebase: 25 source files, 6001 lines across `src/`, plus `index.ts` (entry), `evaluation/`, `tests/`

---

## System Understanding

### What memory-spark IS
An **OpenClaw plugin** (`kind: "memory"`) that occupies the `plugins.slots.memory` slot, replacing the built-in `memory-core`. It registers:

1. **7 tools** via `api.registerTool()` — `memory_search`, `memory_get`, `memory_store`, `memory_forget`, `memory_reference_search`, `memory_index_status`, `memory_forget_by_path`
2. **`before_prompt_build` hook** — auto-recall: injects `<relevant-memories>` XML into agent context before every turn (`src/auto/recall.ts`)
3. **`agent_end` hook** — auto-capture: extracts facts/preferences/decisions from conversation, stores them (`src/auto/capture.ts`)
4. **`after_compaction` hook** — re-indexes compacted session files
5. **Background service** (`memory-spark-watcher`) — chokidar file watcher + boot-pass indexer (`src/ingest/watcher.ts`)
6. **CLI** — `openclaw memory status|sync|migrate`
7. **`gateway_start` hook** — triggers migration on first boot

### Spark Services Used
| Service | Port | Module | Purpose |
|---------|------|--------|---------|
| Embed (Nemotron-Embed-8B) | 18091 | `src/embed/provider.ts` | 4096-dim embeddings via `/v1/embeddings` |
| Reranker (Nemotron-Rerank-1B) | 18096 | `src/rerank/reranker.ts` | Cross-encoder via `/v1/rerank` (Cohere-compatible) |
| Zero-shot (bart-large-mnli) | 18113 | `src/classify/zero-shot.ts` | Category classification via `/v1/classify` |
| NER | 18112 | `src/classify/ner.ts` | Entity extraction via `/v1/extract` |
| GLM-OCR (vLLM) | 18080 | `src/ingest/parsers.ts` | PDF/image OCR via vLLM chat completions |
| EasyOCR (legacy) | 18097 | `src/ingest/parsers.ts` | Fallback OCR |
| STT (Parakeet) | 18094 | `src/ingest/parsers.ts` | Audio transcription |

### Data Flow
```
File changes (chokidar) → watcher.ts → pipeline.ts
  → parsers.ts (extract text: md/txt/pdf/docx/audio)
  → chunker.ts (markdown-aware, 400 token max, 50 overlap)
  → quality.ts (noise filter: i18n, session dumps, casual chat, Discord metadata)
  → chunker.cleanChunkText() (strip metadata noise)
  → ner.ts (entity extraction via Spark NER)
  → embed/queue.ts → provider.ts (Spark /v1/embeddings → 4096-dim vector)
  → lancedb.ts upsert (mergeInsert by id, write-locked promise chain)

Agent turn → before_prompt_build hook → recall.ts
  → cleanQueryText() (strip Discord metadata, timestamps, <relevant-memories>)
  → embed query → vectorSearch + ftsSearch (parallel)
  → FTS filter (exclude sessions, apply minScore)
  → hybridMerge (preserve cosine scores + FTS rank boost)
  → applySourceWeighting (MISTAKES 1.6x, MEMORY.md 1.4x, captures 1.5x, sessions 0.5x, archives 0.4x)
  → applyTemporalDecay (0.8 + 0.2 * exp(-0.03 * ageDays), floor=0.8)
  → mmrRerank (Jaccard diversity, λ=0.7)
  → Spark cross-encoder reranker (top cfg.maxResults)
  → LCM recency suppression (>40% token overlap with recent messages → drop)
  → prompt injection filter
  → token budget enforcement (2000 max)
  → formatRecalledMemories() → XML injection into prependContext

Agent turn end → agent_end hook → capture.ts
  → extract user messages + assistant decision/fact patterns
  → prompt injection check
  → zero-shot classify (Spark) or heuristic fallback
  → dedup check (>0.92 cosine similarity = skip)
  → NER entity extraction
  → store with importance score
```

---

## Current Issues (Precise, Referenced)

### Issue 1: LanceDB Index Is Empty (CRITICAL)
- **State:** Table was wiped (old 2.2GB garbage deleted). Only 10 capture chunks remain.
- **Impact:** All tools return "no relevant memories found." Auto-recall injects nothing. Agents have zero memory.
- **Root cause:** I deleted `~/.openclaw/data/memory-spark/lancedb/memory_chunks.lance` to clear garbage, but never triggered the boot-pass reindex.
- **Fix:** Gateway restart → watcher boot-pass (`runBootPass()` in `src/ingest/watcher.ts:427`) auto-discovers all workspace files, compares mtime vs stored updated_at, indexes delta. Since the table is empty, everything is delta.

### Issue 2: `manager.search()` Uses Naive Merge (BUG)
- **File:** `src/manager.ts:64-123`
- **Problem:** The `memory_search` tool uses `manager.search()` which does a simple dedup-by-id + sort-by-score merge. It does NOT use `hybridMerge`, `applySourceWeighting`, `applyTemporalDecay`, or `mmrRerank`.
- **Contrast:** The auto-recall pipeline (`src/auto/recall.ts`) uses ALL of these. So agents using the `memory_search` tool get materially worse results than the auto-recall hook provides silently.
- **Fix:** Refactor `manager.search()` to use the same exported pipeline functions from `recall.ts`.

### Issue 3: Auto-Capture Barely Fires (83 captures vs 18K chunks)
- **File:** `src/auto/capture.ts:44` — `minMessageLength` defaults to 30 in config but the code at line 56 uses `cfg.minMessageLength ?? 80` as fallback.
- **File:** `src/auto/capture.ts:78` — `effectiveMinConfidence` is 0.6 for heuristic results but `cfg.minConfidence` default is 0.75 in config.ts.
- **Problem:** The zero-shot classifier was unhealthy (container healthcheck failing, now fixed). Heuristic fallback caps at 0.70 score. With default `minConfidence: 0.75`, heuristic captures NEVER pass. The `effectiveMinConfidence` adjustment to 0.6 only kicks in when `result.score <= 0.7`, which helps, but the pipeline is fragile.
- **Also:** `extractCaptureMessages()` only captures user messages + assistant messages matching specific regex patterns (`containsDecisionPattern`, `containsFactPattern`). Many valuable assistant outputs don't match these narrow regexes.
- **Fix:** Lower `minConfidence` default to 0.6. Broaden `containsFactPattern` regex. Monitor capture rate after fix.

### Issue 4: Eval Suite Still Has Mock Mode Code (MISLEADING)
- **File:** `evaluation/run.ts:160-240` — `mockQueryResult()`, `mockConfigStrength()`, `scoreCandidate()` etc.
- **Problem:** The README quotes mock-mode numbers (NDCG@10: 0.889). Klein was right to call this out. Mock mode uses hardcoded "strength" multipliers per config flag, not real retrieval.
- **Current state:** `runLiveEvaluation()` now correctly uses production pipeline functions (`hybridMerge`, `applySourceWeighting`, `applyTemporalDecay`, `mmrRerank`, real Spark reranker). This is fixed but not committed.
- **Fix:** Remove mock mode entirely (or clearly label it as "synthetic baseline only"). Default to live eval.

### Issue 5: Ground Truth Dataset Is Shallow
- **File:** `evaluation/ground-truth.json` — 60 queries, relevance defined by `path_contains` + `snippet_contains`
- **Problem:** Relevance matching is done by checking if retrieved doc paths/text contain the expected strings. This doesn't validate whether the retrieved context actually helps the agent answer correctly.
- **Fix:** Phase 4 — golden answers + weak model test.

### Issue 6: No Agent-Level Validation
- **Problem:** All benchmarks measure IR metrics (NDCG, MRR, Recall). None measure "does the agent actually perform better with memory-spark enabled vs disabled?" Which is the whole point.
- **Fix:** Phase 4 — E2E agent benchmark using Nemotron-Super or Codex.

---

## Execution Plan

### Phase 0: Clean Slate (No Code Changes)
**Actions:**
1. Commit current valid changes (recall.ts, eval/run.ts, tests, scripts/index-audit.ts, research docs)
2. Request gateway restart via `oc-restart` — triggers boot-pass reindex on empty LanceDB
3. Monitor: `openclaw memory status` shows chunk count growing
4. Once boot-pass completes: run `npx tsx evaluation/run.ts` (live mode) for baseline
5. Record baseline metrics

**Validation:** `memory_index_status` tool shows >1000 chunks across multiple agents

### Phase 1: Fix the Three Bugs
**1a. Manager search parity** (`src/manager.ts`)
- Import and use `hybridMerge`, `applySourceWeighting`, `applyTemporalDecay`, `mmrRerank` from `recall.ts`
- Test: `memory_search("How do I restart the gateway?")` returns AGENTS.md oc-restart section

**1b. Auto-capture fix** (`src/auto/capture.ts` + `src/config.ts`)
- Change `minMessageLength` fallback from 80 to 30 (match config default)
- Change `minConfidence` default from 0.75 to 0.60
- Broaden `containsFactPattern`: add patterns like `\b(the issue|root cause|fixed by|the problem|resolved|the answer|the solution)\b`
- Test: Run a session with facts → verify capture count increases

**1c. Remove mock eval mode** (`evaluation/run.ts`)
- Delete `mockQueryResult`, `mockConfigStrength`, `scoreCandidate`, `buildDistractors`, `mulberry32`, `hashString`, `similarityScore`, `inferQuality`, `inferUpdatedAt`, `mergeCandidates` functions
- Remove `--mock` flag from `parseArgs`
- Default to live evaluation
- Test: `npx tsx evaluation/run.ts` runs live by default

**Validation:** Run tests (`npx tsx tests/unit.ts`), run live eval, compare to Phase 0 baseline

### Phase 2: Quality Improvements
**2a. Embedding cache** — new `src/embed/cache.ts`
- In-memory LRU cache keyed by hash(text), TTL 1 hour
- Wrap `EmbedQueue.embedQuery()` to check cache first
- Reduces Spark load for repeated queries

**2b. Query cleaning improvements** (`src/auto/recall.ts:cleanQueryText`)
- Add: strip `[System: ...]` prefixes, `HEARTBEAT_OK`, `NO_REPLY` patterns
- Add: strip `## Current Task Queue` blocks (injected by oc-tasks)
- Test: Feed a real agent message through cleanQueryText, verify noise stripped

**2c. `memory_health` tool** — new tool registration in `index.ts`
- Returns: embed queue stats (`queue.stats`), Spark endpoint connectivity (embed.probe + reranker.probe), index stats (backend.status + backend.getStats), staleness report
- Useful for immune agent health checks

### Phase 3: Bootstrap Bloat Reduction
- Audit each agent's AGENTS.md token count
- Move static facts (host names, IP addresses, port numbers, service inventory) into reference library files under `~/Documents/OpenClaw/ReferenceLibrary/`
- Tag them with `content_type: reference` so `memory_reference_search` retrieves them
- Test: Agent asks about Spark endpoints → `memory_reference_search` returns correct info → AGENTS.md doesn't need to contain it

### Phase 4: E2E Agent Benchmark (THE REAL TEST)

This is what actually matters. IR metrics are necessary but not sufficient.

**4a. Expand golden dataset** to 100+ queries with **gold answers** (the correct response text).

**4b. Weak model test using Nemotron-Super:**
```
For each query in golden dataset:
  1. Run query through memory_search → get retrieved context
  2. Build prompt: "Given this context: {retrieved}. Answer: {query}"
  3. Send to Nemotron-Super (spark-vllm, port 18096)
  4. Compare answer to gold answer using LLM-as-judge (same model)
  5. Score: correct/partial/wrong
```
This tests whether retrieved context ACTUALLY helps the model answer correctly.

**4c. A/B agent comparison:**
- Spawn a benchmark agent session with memory-spark enabled → run 20 representative tasks
- Spawn same agent with `plugins.slots.memory = "none"` → run same 20 tasks
- Compare: factual accuracy, repeated mistakes, task completion quality
- Use **Codex** (`openai-codex/gpt-5.3-codex`) or **Nemotron-Super** (`nemotron-super`) as the benchmark agent model
- This is the definitive test: does memory-spark make agents measurably smarter?

**4d. MISTAKES.md recall test:**
- Create a deliberate mistake scenario (e.g., "agent tries to edit openclaw.json directly during heartbeat")
- Verify MISTAKES.md entry is recalled with 1.6x boost
- Verify agent avoids repeating the mistake

**Implementation:** Write `evaluation/e2e-benchmark.ts` that:
1. Takes a model param (codex or nemotron-super)
2. Runs golden queries through the full tool chain
3. Scores answers with LLM-as-judge
4. Produces a report: `evaluation/results/e2e-{model}-{timestamp}.json`

---

## What to Skip (and Why)

| Item | Reason |
|------|--------|
| HyDE | Adds 200-500ms per query (extra LLM call). Our latency budget is <100ms for the embed step. Research confirms: agent memories are already answer-shaped, HyDE helps most with ambiguous web queries. |
| Parent-Child Chunking | Our content is short distilled facts (~50-200 tokens). Chunker already does 400-token markdown-aware splits. Parent-child matters for long documents, not memory files. |
| Dimension Reduction | 4096-dim Nemotron-Embed-8B is local (zero API cost), #1 on MTEB for these dims. Reducing dims saves disk but costs retrieval quality. |
| Memory Linking (graph) | Requires new table + schema + UI. Nice-to-have but complex. Zep uses Neo4j for this — we don't have a graph DB. |
| Tool Suggestions | Speculative. Agents already have TOOLS.md. No evidence this would improve behavior. |

---

## Testing Strategy

### Unit Tests (`tests/unit.ts`)
- Currently 128 tests (+ 7 new for hybridMerge/sourceWeighting/temporalDecay)
- Run: `npx tsx tests/unit.ts`
- Covers: security, chunking, quality scoring, heuristic classification, config resolution
- **Add:** Tests for manager search parity, capture threshold changes

### Live Evaluation (`evaluation/run.ts`)
- 60 queries across 8 categories
- Full ablation suite: toggles rerank/decay/fts/quality/context/mistakes
- Measures: NDCG@10, MRR, Recall@5, p95 latency
- Run: `npx tsx evaluation/run.ts`

### E2E Agent Benchmark (`evaluation/e2e-benchmark.ts` — NEW)
- Model: Nemotron-Super (local, free) or Codex (API, highest quality)
- 100+ queries with gold answers
- LLM-as-judge scoring
- A/B: memory ON vs OFF
- Run: `npx tsx evaluation/e2e-benchmark.ts --model nemotron-super`

### Build Verification
- `npm run build` → clean TypeScript compilation
- Symlink verification: `~/.openclaw/extensions/memory-spark → ~/codeWS/TypeScript/memory-spark`

---

## Immediate Next Steps (For Klein Approval)

1. **Commit current changes** (Phase 0)
2. **Gateway restart** to trigger boot-pass reindex (Phase 0)
3. **Fix 3 bugs** in order: manager search, capture thresholds, remove mock eval (Phase 1)
4. **Re-run live eval** on clean index (Phase 1 validation)
5. **Build E2E benchmark** with Nemotron-Super (Phase 4)
6. **Run A/B test** — memory ON vs OFF with real agent tasks (Phase 4)
