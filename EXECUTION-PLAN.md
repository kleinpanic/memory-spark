# memory-spark Comprehensive Execution Plan

**Date:** 2026-03-26
**Compiled from:** RAG-AUDIT.md, PHASE2-ROADMAP.md, OVERHAUL-PLAN.md, Klein's directives
**Goal:** Execute all improvements tonight. Parallel where safe, sequential where dependent.

---

## Current State

- **Index:** 37,159 chunks (purged from 63K)
- **Tests:** 82/82 passing
- **Commit:** `c559d2f` on main
- **Tools exposed:** 4 (`memory_search`, `memory_get`, `memory_store`, `memory_forget`)
- **Key gaps:** No vector indexes, no temporal decay, no chunk overlap, no eval metrics, no reference library tools, no MISTAKES.md enforcement, stale reference data

---

## Phase A: Vector Indexes + Temporal Decay (FOUNDATION)
**Dependency:** None — can start immediately
**Parallelizable:** Yes (indexes + decay are independent)

### A1. Create Vector + Scalar Indexes
**File:** `src/storage/lancedb.ts`
**What:**
- After table open, check `listIndices()` — if no vector index exists, create one
- IVF_PQ: `table.createIndex({ type: "ivf_pq", column: "vector", metric: "cosine", numPartitions: 10, numSubVectors: 512 })`
- Scalar: Bitmap on `source`, `agent_id`; BTree on `updated_at`
- FTS: Ensure FTS index on `text` column exists
- Add `refine_factor: 20` to vector search queries
- Call `table.optimize()` periodically (on boot + every 6 hours via interval)

**Validation:**
- `table.listIndices()` returns expected indexes
- Vector search returns results in <50ms (was 100ms+)
- Search results quality unchanged (spot check 5 queries)

### A2. Temporal Decay Scoring
**File:** `src/auto/recall.ts`
**What:**
- Add temporal decay to the scoring pipeline, AFTER reranking but BEFORE final selection
- Formula: `decayed_score = score * (0.8 + 0.2 * Math.exp(-0.03 * age_days))`
  - 0 days old: score × 1.0
  - 7 days old: score × 0.96
  - 30 days old: score × 0.89
  - 90 days old: score × 0.81
  - 365 days old: score × 0.80 (floor)
- The 0.8 floor prevents old-but-gold facts (MEMORY.md) from being completely suppressed
- Exception: `source === "capture"` gets no decay (captures are already timestamped knowledge)

**Validation:**
- Unit test: verify decay formula at 0, 7, 30, 90, 365 days
- Unit test: captures are not decayed
- Manual test: query about "Spark node IP" should prefer recent MEMORY.md over ancient daily note

### A3. Chunk Overlap
**File:** `src/embed/chunker.ts`
**What:**
- Add overlap parameter to chunker config (default: 50 tokens)
- When splitting markdown sections that exceed maxTokens, include last N tokens of previous chunk at start of next
- Only apply overlap to within-section splits, not between-section boundaries
- Update `start_line` calculation to account for overlap

**Validation:**
- Unit test: overlap tokens present in consecutive chunks
- Unit test: total text coverage preserved (no content lost)
- Unit test: section boundaries don't get overlap

---

## Phase B: Reference Library + New Tools (FEATURE)
**Dependency:** A1 (indexes should exist for reference search)
**Parallelizable with:** Phase C

### B1. Schema Evolution — Add `content_type` Field
**File:** `src/storage/lancedb.ts`
**What:**
- Use `table.addColumns()` to add:
  - `content_type: string` (values: "knowledge", "capture", "reference", "daily-note", "mistakes")
  - `quality_score: float` (0.0-1.0, from quality scorer)
  - `token_count: number` (for budgeting)
- Set defaults via SQL expressions for existing rows:
  - `"'knowledge'"` for content_type
  - `"0.5"` for quality_score
  - `"0"` for token_count
- Create Bitmap index on `content_type`

**Validation:**
- `table.schema()` shows new columns
- Existing chunks have correct defaults
- New chunks get proper values set during ingest

### B2. Reference Library Ingest Script
**File:** `scripts/ingest-reference.ts`
**What:**
- CLI: `npx tsx scripts/ingest-reference.ts --path <file-or-dir> --agent <agentId> --tag <tag> [--chunk-size 800]`
- Accepts: PDF, MD, TXT, DOCX
- Chunks with larger size (800 tokens default vs 400 for notes) — reference material benefits from more context per chunk
- Sets `content_type: "reference"`, `source: "ingest"`
- Stores tag metadata for filtering (e.g., "textbook:calculus", "docs:openclaw", "docs:vllm")
- Version-stamps with ISO date so stale references can be identified

**Validation:**
- Ingest the Anthropic prompting guide (already in knowledge-base)
- Verify chunks have correct content_type and tags
- Search for content from the ingested doc

### B3. New Plugin Tools
**File:** `index.ts`
**What — add 3 new tools:**

**`memory_reference_search`** — Search ONLY the reference library (content_type: "reference")
- Params: `{ query: string, tag?: string, maxResults?: number }`
- Description: "Search reference documentation (textbooks, manuals, API docs). Use instead of web search when you know the answer is in indexed reference materials."
- Filters: `content_type = 'reference'` + optional tag filter
- Returns: formatted results with source path + page/section info

**`memory_index_status`** — Show index health/stats
- Params: `{ agentId?: string }`
- Description: "Show memory index statistics: chunk counts by type, freshness distribution, index health."
- Returns: total chunks, by content_type, by source, avg age, index status, staleness warnings

**`memory_forget_by_path`** — Delete all chunks for a specific file path
- Params: `{ path: string, agentId?: string }`
- Description: "Remove all indexed chunks from a specific file. Use when a reference doc is outdated."
- Uses existing `deleteByPath()` backend method

**Validation:**
- All 3 tools register and appear in tool list
- `memory_reference_search` returns only reference content
- `memory_index_status` returns accurate stats
- `memory_forget_by_path` removes chunks and they don't appear in subsequent searches

---

## Phase C: MISTAKES.md Enforcement (FEATURE)
**Dependency:** B1 (needs content_type field)
**Parallelizable with:** Phase B2/B3

### C1. MISTAKES.md Watcher + Enforcer
**File:** `src/ingest/watcher.ts` + new `src/auto/mistakes.ts`
**What:**
- On boot pass, for each discovered agent workspace:
  - Check if `mistakes.md` or `MISTAKES.md` exists
  - If not, create a template:
    ```markdown
    # Mistakes Log
    
    Track recurring errors and lessons learned to avoid repeating them.
    
    ## Format
    - **Date:** When the mistake was identified
    - **What happened:** Brief description
    - **Root cause:** Why it happened
    - **Fix:** How to avoid it next time
    ```
  - Index with `content_type: "mistakes"`, high source weight (1.6x in recall)
- Watcher: watch `mistakes.md` / `MISTAKES.md` in each workspace with high priority re-indexing

### C2. Source Weight for Mistakes
**File:** `src/auto/recall.ts` (in `applySourceWeighting`)
**What:**
- Add: `if (chunkPath.toLowerCase().includes('mistakes')) weight *= 1.6;`
- This ensures mistakes are always surfaced highly when relevant
- Combined with the quality scorer already boosting captures, this gives mistakes proper weight

**Validation:**
- Unit test: MISTAKES.md chunks get 1.6x weight
- Test: create a mistakes.md with "Never use config.patch for agents.list array mutations", query for "agents.list config", verify it surfaces

---

## Phase D: Contextual Embeddings (SECRET WEAPON)
**Dependency:** A1 (indexes), B1 (content_type)
**Parallelizable:** No — this touches the core ingest pipeline

### D1. Template-Based Contextual Prepend
**File:** `src/embed/chunker.ts`
**What:**
- Before embedding, prepend a context template to each chunk:
  ```
  [Source: {source} | File: {path} | Agent: {agentId} | Section: {parent_heading}]
  ```
- Example: `[Source: memory | File: MEMORY.md | Agent: meta | Section: System Facts] The Spark node runs at 10.99.1.1...`
- This is the "cheap" version of Anthropic's contextual retrieval — no LLM needed
- The embedded vector now captures WHERE the information lives, not just WHAT it says
- Only prepend for embedding, not for stored text (don't pollute the display text)

**Implementation detail:**
- In `pipeline.ts`, before calling `embed.embedBatch()`, create a parallel array of `contextualizedTexts` with the prefix prepended
- Embed the contextualized text, but store the original text in the chunk
- This means the vector represents `[context] + text` but the retrieved text is clean

**Validation:**
- Unit test: contextualized text is longer than original
- Unit test: stored chunk text does NOT contain the prefix
- Manual test: search for "meta agent system facts" should rank MEMORY.md higher than before

### D2. Parent Heading Extraction
**File:** `src/embed/chunker.ts`
**What:**
- When splitting markdown by sections, track the last `## Heading` seen
- Pass it through as `parent_heading` in chunk metadata
- Use in contextual prefix (D1) and in displayed results

**Validation:**
- Unit test: chunks under "## System Facts" have parent_heading "System Facts"
- Unit test: top-level chunks have empty/root parent_heading

---

## Phase E: Evaluation & Validation Suite (MEASUREMENT)
**Dependency:** A1-A3 (need working pipeline to measure)
**Parallelizable with:** D (can build eval while D is being implemented)

### E1. Benchmark Dataset
**File:** `scripts/benchmark.ts` (or expand existing `benchmark.ts`)
**What:**
- 50 test queries with known relevant documents/chunks
- Categories: system facts, decisions, preferences, code patterns, infrastructure, mistakes
- Example queries:
  - "What IP does the Spark node run on?" → expects MEMORY.md chunk about 10.99.1.1
  - "What model should dev use for complex coding?" → expects MEMORY.md chunk about opus
  - "How do you restart the gateway?" → expects MEMORY.md or AGENTS.md chunk about oc-restart
  - "What caused the model alias incident?" → expects memory/2026-03-02-model-alias.md
- Store as JSON: `{ query, expectedPaths: string[], expectedSnippets: string[] }`

### E2. Metrics Implementation
**File:** `scripts/benchmark.ts`
**What — compute these metrics:**
- **Recall@5:** Of the 5 results returned, how many expected documents were found?
- **Precision@5:** Of the 5 results returned, how many were actually relevant?
- **MRR (Mean Reciprocal Rank):** Average of 1/rank of first relevant result
- **Freshness score:** Average age of recalled chunks (lower = better)
- **Noise rate:** Percentage of recalled chunks that fail quality gate

**Output:** JSON + human-readable summary

### E3. Regression Test Integration
**File:** `test-unit.ts` additions + `package.json` scripts
**What:**
- Add `npm run benchmark` script
- Integrate into the test suite as a separate stage
- Set minimum thresholds: Recall@5 ≥ 0.6, MRR ≥ 0.5, Noise rate ≤ 5%
- Future: run before/after any pipeline change to measure impact

**Validation:**
- Benchmark runs end-to-end
- Metrics are computed and printed
- Regressions fail CI

---

## Phase F: Verify RRF + Cleanup (POLISH)
**Dependency:** All above phases complete

### F1. Verify RRF Merge Logic
**File:** `src/auto/recall.ts` or `src/storage/lancedb.ts` (wherever merge happens)
**What:**
- Audit the vector + FTS merge logic
- If it's dedup-only: implement proper RRF: `score = Σ 1/(k + rank_i)` where k=60
- If RRF exists: verify k parameter and fusion weights

### F2. Stale Reference Cleanup
**What:**
- Run `memory_forget_by_path` for known stale content:
  - The-Linux-Command-Line.pdf (2.1MB of irrelevant content)
  - Old OpenClaw docs if they're indexed from early versions
  - vllm-engine-args.md and vllm-quickstart.md (14 bytes each — empty stubs)
- Re-ingest current OpenClaw docs as reference material with version tag

### F3. LCM 0.5.2 Update
**What:**
- `cd ~/.openclaw/extensions/lossless-claw && npm install @martian-engineering/lossless-claw@0.5.2`
- Test: start gateway, verify LCM loads without errors
- Separate from memory-spark work

### F4. Table Optimize
**What:**
- Stop the gateway watcher temporarily (or find a quiet window)
- Run `table.optimize()` to compact fragments after all deletes
- Re-index FTS after optimization

---

## Execution Order (Tonight)

```
PARALLEL TRACK 1 (Core Pipeline)     PARALLEL TRACK 2 (Features)
─────────────────────────────        ────────────────────────────
A1. Vector indexes (30min)           B1. Schema evolution (20min)
A2. Temporal decay (30min)           C1. MISTAKES.md enforcer (30min)
A3. Chunk overlap (30min)            C2. Source weight (10min)
    ↓                                    ↓
D1. Contextual embeddings (1hr)      B2. Reference ingest script (1hr)
D2. Parent heading (30min)           B3. New plugin tools (1hr)
    ↓                                    ↓
    └──────── MERGE ────────────────────┘
                    ↓
            E1. Benchmark dataset (1hr)
            E2. Metrics implementation (1hr)
            E3. Regression integration (30min)
                    ↓
            F1. Verify RRF (30min)
            F2. Stale cleanup (20min)
            F3. LCM update (20min)
            F4. Table optimize (10min)
                    ↓
              FINAL: Build + Test + Push
```

**Estimated total:** 8-10 hours of work, ~5-6 hours wall clock with parallelism.
**Target tests:** 82 → ~130+ (new tests for each phase)
**Target tools:** 4 → 7

---

## Success Criteria

| Metric | Before | Target |
|--------|--------|--------|
| Index chunks | 37,159 | ~37K (same data, better indexed) |
| Vector search speed | ~100ms (brute) | <10ms (IVF_PQ) |
| Plugin tools | 4 | 7 |
| Unit tests | 82 | 130+ |
| Recall@5 (benchmark) | unknown | ≥ 0.6 |
| Noise rate | ~1/500 (0.2%) | <1% |
| Stale data in top-5 | frequent | rare (decay + cleanup) |
| MISTAKES.md coverage | 1/8 agents | 8/8 agents |
| Reference library | 0 docs | ≥ 3 (OpenClaw, prompting guide, LanceDB) |

---

## Files to Touch

| File | Phases | Changes |
|------|--------|---------|
| `src/storage/lancedb.ts` | A1, B1 | Indexes, schema evolution, optimize |
| `src/auto/recall.ts` | A2, C2, F1 | Temporal decay, mistakes weight, RRF verify |
| `src/embed/chunker.ts` | A3, D1, D2 | Overlap, contextual prefix, parent heading |
| `src/ingest/pipeline.ts` | D1 | Contextualized embedding (embed prefix, store clean) |
| `src/ingest/watcher.ts` | C1 | MISTAKES.md discovery + creation |
| `src/auto/mistakes.ts` | C1 | NEW: MISTAKES.md template + enforcement |
| `index.ts` | B3 | 3 new tools: reference_search, index_status, forget_by_path |
| `scripts/ingest-reference.ts` | B2 | NEW: Reference material CLI |
| `scripts/benchmark.ts` | E1, E2 | NEW or expand: Benchmark + metrics |
| `test-unit.ts` | ALL | New tests per phase |
| `src/config.ts` | A3 | Overlap config param |
