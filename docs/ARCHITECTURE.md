# memory-spark v0.4.0 Architecture

**Status:** Production  
**Authors:** KleinClaw-Meta (Opus 4.6) + Dev (Codex 5.4)  
**Last Updated:** 2026-04-04

## Design Principles

1. **No workarounds.** Every implementation must be the proper solution, not a temporary hack.
2. **Scientifically reproducible.** All benchmarks, evaluations, and experiments must produce identical results given identical inputs.
3. **Configurable over hardcoded.** Expose tunable parameters via config, not buried constants.
4. **Data isolation with shared access.** Each agent owns its data. Cross-agent access is explicit and auditable.
5. **LLM-assisted classification.** Use our serving infrastructure (Nemotron, zero-shot classifiers) for content routing, not regex heuristics alone.
6. **Reference ≠ Auto-inject.** Reference documents (PDFs, docs) are retrieved on-demand via tool calls, never silently injected into context.
7. **Monorepo.** Code, evaluations, docs, scripts — all in one repo with clear structure.

---

## 1. Single-Table + Pool Column Architecture

### Design Decision

LanceDB [officially recommends](https://lancedb.com/lp/vector-db-guide/) single-table with metadata columns over multiple tables. Benefits: zero-copy schema evolution, better partition distribution, less infrastructure overhead. Our dataset (~37k chunks) is well below the 10M+ threshold where multi-table IVF_PQ would help.

The FTS+WHERE bug that originally motivated multi-table was **fixed in LanceDB 0.27.1** (upgraded from 0.14.1). Validated with a 5-test suite confirming all compound queries work.

### Single Table: `memory_chunks`

```
Columns:
  id:             VARCHAR (UUID, 16-char)
  text:           VARCHAR (chunk text, max ~2000 tokens)
  vector:         VECTOR(4096)  -- Nvidia Llama-Embed-Nemotron-8B via Spark
  path:           VARCHAR (relative to agent workspace or capture path)
  source:         VARCHAR ("capture" | "workspace" | "session" | "memory")
  agent_id:       VARCHAR (owning agent or "shared")
  pool:           VARCHAR (logical section — see Pool Types below)
  content_type:   VARCHAR ("knowledge" | "decision" | "preference" | "fact" | "code" | "tool" | "mistake" | "rule" | "reference" | "reference_code")
  category:       VARCHAR (classifier label: fact, preference, decision, code-snippet, etc.)
  confidence:     FLOAT (classification confidence, 0.0–1.0)
  entities:       VARCHAR (JSON array of extracted entities)
  parent_heading: VARCHAR (markdown heading context above this chunk)
  quality_score:  FLOAT (chunk quality at ingest time, 0.0–1.0)
  token_count:    INT (estimated token count)
  start_line:     INT
  end_line:       INT
  updated_at:     TIMESTAMP (file mtime for workspace, capture time for captures)

Indexes:
  - IVF_PQ vector index (numPartitions=10, numSubVectors=64, cosine)
  - FTS index on `text` column (BM25)
```

### Pool Types

The `pool` column provides logical data isolation within the single table. All filtering uses native LanceDB WHERE clauses (no workarounds).

| Pool | Purpose | Scope | Auto-Inject? | Access |
|------|---------|-------|-------------|--------|
| `agent_memory` | Workspace files, captures, daily notes | Per-agent (`agent_id` filter) | ✅ Relevance-gated | Auto-recall + `memory_search` |
| `agent_tools` | TOOLS.md, tool schemas | Per-agent (`agent_id` filter) | ✅ Relevance-gated | Auto-recall |
| `agent_mistakes` | Per-agent mistake entries | Per-agent (`agent_id` filter) | ✅ Relevance-gated (1.6x boost) | Auto-recall + `memory_mistakes_search` |
| `shared_knowledge` | Cross-agent facts, infra docs | All agents | ✅ Relevance-gated | Auto-recall + `memory_search` |
| `shared_mistakes` | Cross-agent shared mistakes | All agents | ✅ Relevance-gated (1.6x boost) | Auto-recall + `memory_mistakes_search` |
| `shared_rules` | Global rules & preferences | All agents | ✅ Relevance-gated | Auto-recall + `memory_rules_search` |
| `reference_library` | PDFs, documentation | All agents | ❌ **Never** | `memory_reference_search` only |
| `reference_code` | Code examples, patterns | All agents | ❌ **Never** | `memory_reference_search` only |

### Pool Routing

`src/storage/pool.ts` — single source of truth. `resolvePool()` determines pool from `content_type` + `path`:

1. Explicit `pool` on chunk → use it
2. `content_type === "tool"` or `TOOLS.md` path → `agent_tools`
3. `MISTAKES.md` path or `content_type === "mistake"` → `agent_mistakes`
4. `content_type === "rule"` or `"preference"` → `shared_rules`
5. `content_type === "reference"` → `reference_library`
6. `content_type === "reference_code"` → `reference_code`
7. Default → `agent_memory`

### Why This Architecture

1. **Single table follows LanceDB best practices.** Zero-copy schema evolution, no per-table index overhead, simpler operations.
2. **Pool column enables efficient WHERE filtering.** `WHERE pool = 'shared_rules' AND agent_id = 'meta'` — native, no workaround.
3. **FTS+WHERE works in LanceDB 0.27.1.** The Arrow panic that forced the 3x overfetch workaround is fixed. Validated with compound query tests.
4. **Reference separation prevents context pollution.** `reference_library` and `reference_code` are NEVER auto-injected. Tool-call only.
5. **Per-agent mistakes + shared mistakes.** Agents see their own mistakes first, shared mistakes second. Promotion is explicit via `shared: true` param.
6. **Single backend simplicity.** `LanceDBBackend` is the sole backend — no multi-table indirection, no routing mismatch risk.

---

## 2. Retrieval Pipeline (v0.4.0)

```
Query → Clean → HyDE (optional) → Multi-Query Expand → Embed
  → Per-Pool Search (Vector + FTS) → RRF Hybrid Merge
  → Source Weighting → Temporal Decay → Source Dedup
  → Dynamic Reranker Gate → Cross-Encoder Rerank → RRF Blend
  → MMR Diversity → Parent-Child Expansion → LCM Dedup
  → Security Filter → Token Budget → <relevant-memories> XML
```

### Stage Details

1. **Query Cleaning** (`cleanQueryText`)
   - Strip Discord metadata, oc-tasks blocks, system prefixes, media paths
   - Remove LCM summary blocks and `<relevant-memories>` (prevent recursive recall)
   - Normalize whitespace, collapse excessive newlines

2. **HyDE — Hypothetical Document Embeddings** (optional)
   - Generate a hypothetical answer document via LLM
   - Embed as a DOCUMENT (no instruction prefix) → lands in document subspace
   - Bridges the question↔answer semantic gap
   - Quality gate: reject refusals, too-short responses
   - Fallback: raw query embedding on failure/timeout

3. **Multi-Query Expansion** (Phase 11B)
   - Generate 3 LLM reformulations of the query
   - Embed each reformulation in parallel
   - All query vectors used in subsequent vector search (union by chunk ID)

4. **Per-Pool Search** (5 pool groups searched sequentially in the current implementation)
   - agent_memory + agent_tools (filtered by agent_id)
   - agent_mistakes (filtered, 1.6× boost)
   - shared_mistakes (cross-agent, 1.6× boost)
   - shared_knowledge (cross-agent, 0.8× weight)
   - shared_rules (cross-agent)
   - Each pool group: Vector search + FTS search → RRF merge
   - Note: query reformulation embedding can run in parallel, but the top-level pool-group lookups in `src/auto/recall.ts` are currently awaited one after another

5. **RRF Hybrid Merge** (`hybridMerge`)
   - Reciprocal Rank Fusion: `score(d) = Σ weight / (k + rank)`
   - Scale-invariant — no normalization of BM25 or cosine scores
   - Configurable k (default 60), vector/FTS weights
   - Dual-source documents get natural rank-sum boost

6. **Source Weighting** (`applySourceWeighting`)
   - Captures: 1.5×, Mistakes: 1.6×, Sessions: 0.5×, Archive: 0.4×
   - Path-specific weights (MEMORY.md 1.4×, TOOLS.md 1.3×, etc.)
   - Configurable via `autoRecall.weights`

7. **Temporal Decay** (`applyTemporalDecay`)
   - Formula: `floor + (1 - floor) × exp(-rate × ageDays)`
   - Default: floor=0.8, rate=0.03 → gentle decay, 80% floor
   - NaN-safe: invalid timestamps skip decay

8. **Source Dedup** (`deduplicateSources`)
   - Within same source: Jaccard token overlap > 0.85 = deduplicate
   - Cross-source chunks never deduped

9. **Dynamic Reranker Gate** (Phase 12, `computeRerankerGate`)
   - Compute top-5 vector score spread (max − min)
   - **Hard gate** (default): skip reranker if spread > 0.08 (confident) or < 0.02 (tied)
   - **Soft gate**: dynamically scale vector weight from spread
   - Benchmark: 78% queries skip reranking with no NDCG loss

10. **Cross-Encoder Rerank** (`nvidia/llama-nemotron-rerank-1b-v2`)
    - Fires only when gate passes (spread 0.02–0.08)
    - Rescores all candidates against original query text
    - Automatic query normalization (declarative → interrogative)

11. **RRF Blend** (Phase 12)
    - Fuses original vector ranks with reranker ranks via RRF
    - Configurable vector/reranker weights and k
    - Scale-invariant — avoids the min-max normalization problems of score blending

12. **MMR Diversity** (`mmrRerank`, λ=0.9)
    - Cosine similarity on embedding vectors (semantic diversity)
    - Fallback: Jaccard on token sets when vectors unavailable
    - Defense-in-depth: validates vector type before cosine path

13. **Parent-Child Expansion**
    - Child chunks retrieved precisely → replaced with parent text for context
    - Dedup: multiple children from same parent → keep highest-scoring

14. **LCM Dedup**
    - Skip chunks that overlap >40% with recent conversation messages
    - Skip chunks that overlap with LCM summary content already in context

15. **Security Filter** + **Token Budget**
    - Prompt injection detection (regex patterns)
    - Greedy fill: highest-scored chunks first until `maxInjectionTokens` budget exhausted

---

## 3. Content Classification Pipeline

Use our LLM serving infrastructure for proper classification, not just regex.

### Ingest-Time Classification

```
Raw Text → Language Detection → Quality Scoring → LLM Classification → 
  → Pool Routing → Chunking → Embedding → Storage
```

1. **Language Detection** (heuristic + confidence threshold)
2. **Quality Scoring** (`scoreChunkQuality`)
   - Noise patterns: session headers, Discord metadata, media paths
   - Information density: keyword diversity, sentence structure
   - Minimum threshold: 0.3 (configurable)
3. **LLM Classification** (zero-shot via Nemotron)
   - Categories: fact, preference, decision, code-snippet, tool-definition, mistake, infrastructure
   - Route to appropriate pool based on classification
4. **Pool Routing Logic**:
   - Classification and path rules ultimately resolve into a logical `pool` inside the single `memory_chunks` table
   - `tool-definition` / `TOOLS.md`-style content → `agent_tools`
   - `mistake` / `MISTAKES.md`-style content → `agent_mistakes` unless explicitly promoted to shared mistakes
   - shared infrastructure / cross-agent knowledge → `shared_knowledge`
   - rules and preferences → `shared_rules`
   - default agent-scoped recall content → `agent_memory`

### Capture-Time Classification

Same pipeline but triggered by `agent_end` hook for conversation captures.
Additional gate: garbage detection (30+ regex patterns) before LLM classification.

---

## 4. Plugin Tools

| Tool | Description | Table(s) |
|------|-------------|----------|
| `memory_search` | Search agent's own memory + shared knowledge | agent memory + shared |
| `memory_reference_search` | Search reference library (docs, PDFs, code) | reference_library, reference_code |
| `memory_store` | Store a new memory (agent-scoped) | agent memory |
| `memory_forget` | Delete a specific memory by ID | agent memory |
| `memory_forget_by_path` | Delete all chunks from a file path | agent memory |
| `memory_get` | Get a specific memory by ID | any table |
| `memory_index_status` | Health check: embed queue, table stats, Spark connectivity | all tables |
| `memory_mistakes` | Search/add to mistakes log | shared_mistakes |

---

## 5. Monorepo Structure

```
memory-spark/
├── src/                    # Core plugin source
│   ├── auto/               # Auto-recall and auto-capture
│   ├── classify/           # Quality scoring, LLM classification
│   ├── embed/              # Embedding provider, queue, cache, chunker
│   ├── ingest/             # File ingestion pipeline, watcher, workspace scanner
│   ├── rerank/             # Cross-encoder reranking
│   ├── security/           # Prompt injection detection, memory escaping
│   ├── storage/            # LanceDB backend, pool routing
│   └── config.ts           # Centralized configuration
├── evaluation/             # Evaluation framework
│   ├── metrics.ts          # BEIR metrics (NDCG, MRR, MAP, Recall, Precision)
│   ├── benchmark.ts        # BEIR benchmark runner
│   ├── pipeline-eval.ts    # E2E pipeline tests
│   ├── golden-dataset.json # 107-query evaluation corpus
│   └── run-all.ts          # Unified evaluation entry point
├── tests/                  # Test suites
│   ├── unit.ts             # Unit tests (no external deps)
│   ├── harness.ts          # Integration tests (requires Spark)
│   └── integration.ts      # Full E2E tests
├── tools/                  # Operational scripts
│   ├── indexer.ts          # Standalone re-indexing tool
│   ├── purge.ts            # Garbage purge tool
│   ├── audit.ts            # Data quality audit
│   └── migrate.ts          # Table migration tools
├── docs/                   # User-facing documentation
│   ├── README.md           # Plugin overview and setup
│   ├── ARCHITECTURE.md     # This file
│   ├── CONFIGURATION.md    # All config options documented
│   ├── EVALUATION.md       # How to run and interpret benchmarks
│   └── CHANGELOG.md        # Version history
├── index.ts                # OpenClaw plugin entry point
├── package.json
├── tsconfig.json
└── .eslintrc.json
```

---

## 6. Configuration Schema

The authoritative config type is `MemorySparkConfig` in `src/config.ts`. See `docs/CONFIGURATION.md` for the full reference with all defaults. Key top-level sections:

| Section | Purpose |
|---------|---------|
| `backend` / `lancedbDir` | Storage backend (LanceDB only) |
| `embed` | Embedding provider, model, dimensions, instruction prefix |
| `rerank` | Cross-encoder reranking, RRF blend, dynamic gate |
| `fts` | BM25 full-text search tuning |
| `autoRecall` | Memory injection: agents, weights, MMR, HyDE, temporal decay, query expansion |
| `autoCapture` | Fact extraction: categories, classifier, confidence |
| `watch` | Filesystem watcher for auto-indexing |
| `ingest` | Language filtering, quality thresholds |
| `reference` | Reference library paths and chunking |
| `hyde` | Hypothetical Document Embeddings config |
| `chunk` | Chunking sizes (flat + hierarchical parent-child) |
| `embedCache` | Query embedding LRU cache |
| `search` | ANN tuning (refineFactor, IVF_PQ params) |
| `spark` | Individual Spark service endpoint URLs |

Pool routing is NOT configurable — it's determined by `src/storage/pool.ts` based on content type and path.

---

## 7. Implementation Phases

### Historical note on phases
The phase list below is retained as historical context, but the authoritative current-state operational handoff is `docs/STATE.md`.

### Phase 0: Bug Fixes ✅ DONE
- Critical early retrieval and evaluation bugs were fixed during the v0.4.0 cycle
- See `docs/CHANGELOG.md`, `docs/AUDIT-2026-04-02.md`, and `docs/STATE.md` for current validated status

### Phase 1: Pool Architecture ✅ COMPLETE
- [x] Pool column + `resolvePool()` routing
- [x] LanceDB 0.27.1 upgrade (FTS+WHERE fixed natively)
- [x] Pool filtering in vector search + FTS (native WHERE)
- [x] Pool propagation through recall, capture, ingest, manager
- [x] Backend consolidation: `LanceDBBackend` sole backend (MultiTableBackend deleted)
- [x] Rules/mistakes tools with pool-based isolation
- [x] Covered by the current test suite (see `docs/STATE.md` for latest validated counts)

### Phase 2: Reference Library
- [ ] Implement reference ingestion (PDF, markdown, HTML, code)
- [ ] `memory_reference_search` tool (not auto-injected)
- [ ] PDF text extraction pipeline
- [ ] Version tracking for reference documents
- [ ] FTS coverage for reference pool content

### Phase 3: Classification Pipeline
- [ ] Integrate zero-shot classifier (Nemotron) for content routing
- [ ] Replace heuristic-only classification with LLM+heuristic hybrid
- [ ] Implement pool routing based on classification output
- [ ] Quality scoring refinement with LLM feedback

### Phase 4: Evaluation Overhaul
- [ ] Fix all metric bugs (Codex findings)
- [ ] Calibrate BM25 sigmoid from corpus score distribution
- [ ] Build golden dataset per table (not just global)
- [ ] Docker-based reproducible benchmarking
- [ ] A/B framework for pipeline ablation studies

### Phase 5: Production Deployment
- [ ] Full reindex with pool assignment on existing chunks
- [ ] Config migration tool for existing installations
- [ ] Performance profiling and optimization
- [ ] Documentation complete
- [ ] v1.0 release

---

## 8. Testing Strategy

Every component must have:
1. **Unit tests** — pure logic, no external deps, fast
2. **Integration tests** — requires Spark, tests real embeddings
3. **E2E pipeline tests** — tests the full ingest→search→inject lifecycle
4. **Benchmark tests** — BEIR metrics with golden dataset

Test coverage targets:
- Storage layer: pool-based CRUD, search, filtering
- Retrieval pipeline: each stage in isolation + full pipeline
- Classification: routing accuracy for each content type
- Security: prompt injection detection, memory escaping
- Configuration: all defaults, overrides, edge cases
- Error handling: Spark down, invalid data, concurrent access
