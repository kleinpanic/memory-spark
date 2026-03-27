# memory-spark v1.0 Architecture

**Status:** Planning → Implementation  
**Authors:** KleinClaw-Meta (Opus 4.6) + Dev (Codex 5.4)  
**Last Updated:** 2026-03-27

## Design Principles

1. **No workarounds.** Every implementation must be the proper solution, not a temporary hack.
2. **Scientifically reproducible.** All benchmarks, evaluations, and experiments must produce identical results given identical inputs.
3. **Configurable over hardcoded.** Expose tunable parameters via config, not buried constants.
4. **Data isolation with shared access.** Each agent owns its data. Cross-agent access is explicit and auditable.
5. **LLM-assisted classification.** Use our serving infrastructure (Nemotron, zero-shot classifiers) for content routing, not regex heuristics alone.
6. **Reference ≠ Auto-inject.** Reference documents (PDFs, docs) are retrieved on-demand via tool calls, never silently injected into context.
7. **Monorepo.** Code, evaluations, docs, scripts — all in one repo with clear structure.

---

## 1. LanceDB Multi-Table Schema

### Tables

| Table | Purpose | Scope | Auto-Inject? | Access Method |
|-------|---------|-------|-------------|--------------|
| `agent_{id}_memory` | Agent-specific captures, workspace files, daily notes | Per-agent isolated | ✅ Yes (before_prompt_build) | Auto-recall + `memory_search` tool |
| `agent_{id}_tools` | TOOLS.md, tool schemas, tool policies | Per-agent isolated | ✅ Yes (tool-context injection) | Auto-inject relevant tools per query |
| `shared_knowledge` | Cross-agent facts, infrastructure docs, system config | All agents (read), meta (write) | ✅ Yes (limited budget) | Auto-recall with lower priority |
| `shared_mistakes` | MISTAKES.md entries from all agents | All agents (read/write) | ✅ Yes (1.6x boost, always in context) | Auto-inject top-N mistakes |
| `reference_library` | PDFs, documentation, SDK docs, OpenClaw docs | All agents (read) | ❌ No | `memory_reference_search` tool only |
| `reference_code` | Code snippets, examples, patterns | All agents (read) | ❌ No | `memory_reference_search` tool only |

### Per-Agent Table: `agent_{id}_memory`

```
Columns:
  id:            VARCHAR (UUID)
  text:          VARCHAR (chunk text)
  vector:        VECTOR(4096)  -- Nvidia Llama-Embed-Nemotron-8B
  path:          VARCHAR (relative to agent workspace)
  source:        VARCHAR ("capture" | "workspace" | "session")
  content_type:  VARCHAR ("knowledge" | "decision" | "preference" | "fact" | "code")
  category:      VARCHAR (LLM-classified: fact, preference, decision, code-snippet, etc.)
  confidence:    FLOAT (classification confidence)
  entities:      VARCHAR (JSON array of extracted entities)
  parent_heading: VARCHAR (markdown heading context)
  start_line:    INT
  end_line:      INT
  updated_at:    TIMESTAMP
  created_at:    TIMESTAMP
  quality_score: FLOAT (chunk quality at ingest time)
  
Indexes:
  - IVF_PQ vector index (nprobes=20, npartitions=auto)
  - FTS index on `text` column (BM25, English stemming)
```

### Shared Knowledge Table: `shared_knowledge`

```
Columns:
  (same as agent memory, plus:)
  contributed_by: VARCHAR (agent_id that created it)
  visibility:     VARCHAR ("all" | "agents:dev,meta" | custom ACL)
  
Access Pattern:
  - Auto-recalled with 0.8x weight relative to agent's own memory
  - Useful for: infrastructure facts, system config, shared decisions
```

### Shared Mistakes Table: `shared_mistakes`

```
Columns:
  id:            VARCHAR (UUID)
  text:          VARCHAR (mistake description + root cause + fix)
  vector:        VECTOR(4096)
  agent_id:      VARCHAR (who made the mistake)
  severity:      VARCHAR ("critical" | "high" | "medium" | "low")
  date:          TIMESTAMP
  category:      VARCHAR ("config" | "deployment" | "data" | "logic" | "security")
  lessons:       VARCHAR (key takeaways)
  
Indexes:
  - IVF_PQ vector index
  - FTS index on `text`
  
Recall:
  - Always 1.6x weight boost
  - Top-3 most relevant mistakes injected per turn
  - Pinned mistakes always present regardless of query relevance
```

### Reference Library Table: `reference_library`

```
Columns:
  id:            VARCHAR (UUID)
  text:          VARCHAR (chunk text)
  vector:        VECTOR(4096)
  path:          VARCHAR (original file path)
  source:        VARCHAR ("pdf" | "markdown" | "html" | "code")
  title:         VARCHAR (document title)
  section:       VARCHAR (section/chapter heading)
  page_number:   INT (for PDFs)
  doc_version:   VARCHAR (version string if applicable)
  content_type:  VARCHAR ("documentation" | "api-reference" | "tutorial" | "spec")
  updated_at:    TIMESTAMP
  
Indexes:
  - IVF_PQ vector index
  - FTS index on `text`

Access:
  - NEVER auto-injected
  - Retrieved ONLY via `memory_reference_search` tool call
  - Agent must explicitly decide to search references
```

### Why This Schema

1. **Per-agent isolation eliminates the FTS WHERE workaround.** Currently, FTS + WHERE panics in LanceDB. With per-agent tables, the WHERE clause for `agent_id` filtering is unnecessary — you're already querying the right table.

2. **Reference separation prevents context pollution.** PDFs and docs are large. Auto-injecting them wastes token budget on potentially irrelevant content. Tool-call-only access means agents retrieve exactly what they need.

3. **Mistakes get priority treatment.** A dedicated table means mistakes can have their own recall logic (always boost, pin critical ones) without competing with general knowledge.

4. **Shared knowledge enables cross-agent learning** without mixing everyone's workspace files together.

---

## 2. Retrieval Pipeline (Revised)

```
Query → Clean → Embed → [Per-Agent Search] → [Shared Search] → Merge → 
  → Source Weight → Temporal Decay → MMR Diversity → Cross-Encoder Rerank → Budget Trim → Inject
```

### Stage Details

1. **Query Cleaning** (`cleanQueryText`)
   - Strip oc-tasks blocks, system prefixes, media paths
   - Normalize whitespace, remove markdown formatting artifacts
   - Extract intent keywords for FTS boost

2. **Parallel Table Search** (per-agent + shared)
   - Agent memory: Vector + FTS on `agent_{id}_memory`
   - Agent tools: Vector on `agent_{id}_tools` (for tool-context queries)
   - Shared knowledge: Vector + FTS on `shared_knowledge`
   - Shared mistakes: Vector + FTS on `shared_mistakes`
   - NO reference tables searched (unless tool call)

3. **Hybrid Merge** (`hybridMerge`)
   - Preserve raw cosine similarity scores from vector search
   - Normalize FTS BM25 scores via calibrated sigmoid
   - Dual-source bonus: +0.15 for results appearing in both Vector and FTS
   - Cross-table dedup by content hash

4. **Source Weighting**
   - Agent own memory: 1.0x (baseline)
   - Shared knowledge: 0.8x
   - Shared mistakes: 1.6x
   - Agent tools: 1.3x
   - Configurable per-agent via `autoRecall.weights`

5. **Temporal Decay**
   - Formula: `0.8 + 0.2 * exp(-0.03 * ageDays)`
   - 80% floor: old knowledge is never fully suppressed
   - NaN-safe: invalid timestamps skip decay (score preserved)

6. **MMR Diversity** (λ=0.7)
   - Maximal Marginal Relevance to avoid redundant chunks
   - Ensures diverse coverage across topics

7. **Cross-Encoder Rerank** (Nemotron Rerank 1B v2)
   - Top-20 candidates → reranked by relevance to original query
   - Runs on Spark GPU when available, graceful degradation to skip

8. **Token Budget Trim**
   - `maxInjectionTokens` (default: 2000) limits total injection
   - Greedy fill: highest-scored chunks first until budget exhausted

---

## 3. Content Classification Pipeline

Use our LLM serving infrastructure for proper classification, not just regex.

### Ingest-Time Classification

```
Raw Text → Language Detection → Quality Scoring → LLM Classification → 
  → Table Routing → Chunking → Embedding → Storage
```

1. **Language Detection** (heuristic + confidence threshold)
2. **Quality Scoring** (`scoreChunkQuality`)
   - Noise patterns: session headers, Discord metadata, media paths
   - Information density: keyword diversity, sentence structure
   - Minimum threshold: 0.3 (configurable)
3. **LLM Classification** (zero-shot via Nemotron)
   - Categories: fact, preference, decision, code-snippet, tool-definition, mistake, infrastructure
   - Route to appropriate table based on classification
4. **Table Routing Logic**:
   - `tool-definition` → `agent_{id}_tools`
   - `mistake` → `shared_mistakes`
   - `infrastructure` + shared flag → `shared_knowledge`
   - Everything else → `agent_{id}_memory`

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
│   ├── storage/            # LanceDB backend, multi-table manager
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

## 6. Configuration Schema (Target)

```typescript
interface MemorySparkConfig {
  // Storage
  backend: "lancedb";
  lancedbDir: string;
  
  // Tables
  tables: {
    agentPrefix: string;        // Default: "agent_"
    sharedKnowledge: string;    // Default: "shared_knowledge"
    sharedMistakes: string;     // Default: "shared_mistakes"
    referenceLibrary: string;   // Default: "reference_library"
    referenceCode: string;      // Default: "reference_code"
  };
  
  // Embedding
  embed: {
    provider: "spark" | "openai" | "local";
    model: string;
    dimensions: number;
    batchSize: number;
    cache: { enabled: boolean; maxSize: number; };
  };
  
  // Reranking
  rerank: {
    enabled: boolean;
    provider: "spark" | "none";
    model: string;
    topN: number;
    timeoutMs: number;
  };
  
  // FTS
  fts: {
    enabled: boolean;
    sigmoidMidpoint: number;     // BM25 normalization center (default: 3.0)
    language: string;            // Stemming language
    maxTokenLength: number;
  };
  
  // Auto-Recall
  autoRecall: {
    enabled: boolean;
    agents: string[];
    ignoreAgents: string[];
    maxResults: number;
    minScore: number;
    queryMessageCount: number;
    maxInjectionTokens: number;
    weights: {
      sources: Record<string, number>;
      paths: Record<string, number>;
      pathPatterns: Record<string, number>;
    };
    temporalDecay: {
      enabled: boolean;
      floor: number;             // Default: 0.8
      rate: number;              // Default: 0.03
    };
    mmr: {
      enabled: boolean;
      lambda: number;            // Default: 0.7
    };
  };
  
  // Auto-Capture
  autoCapture: {
    enabled: boolean;
    agents: string[];
    ignoreAgents: string[];
    categories: string[];
    minConfidence: number;
    minMessageLength: number;
    useClassifier: boolean;      // Use LLM zero-shot vs heuristic
    maxCapturesPerTurn: number;
    deduplicationThreshold: number; // Cosine similarity for skip (default: 0.92)
  };
  
  // Ingest
  ingest: {
    language: string;
    languageThreshold: number;
    minQuality: number;
    watch: {
      enabled: boolean;
      fileTypes: string[];
      indexOnBoot: boolean;
      debounceMs: number;
    };
  };
  
  // Reference Library
  reference: {
    enabled: boolean;
    paths: string[];
    chunkSize: number;
    tags: Record<string, string[]>;
    autoIndex: boolean;          // Reindex on boot?
  };
  
  // Spark endpoints
  spark: {
    embed: string;
    rerank: string;
    ocr: string;
    ner: string;
    zeroShot: string;
  };
  
  // Security
  security: {
    promptInjectionDetection: boolean;
    escapeMemoryText: boolean;
    maxChunkSizeBytes: number;
  };
}
```

---

## 7. Implementation Phases

### Phase 0: Bug Fixes ✅ DONE (commit 8c46531, 9770281)
- 14 critical bugs fixed
- Codex audit passed
- 159/159 unit tests passing

### Phase 1: Multi-Table Schema (Current)
- [ ] Refactor LanceDB backend for multi-table support
- [ ] Implement table routing logic in ingest pipeline
- [ ] Migrate `rowToSearchResult` to handle per-table queries
- [ ] Remove FTS WHERE workaround (per-table eliminates need)
- [ ] Add table management tools (create, migrate, stats)
- [ ] Update config schema with table configuration
- [ ] Tests for multi-table search, routing, isolation

### Phase 2: Reference Library
- [ ] Implement reference ingestion (PDF, markdown, HTML, code)
- [ ] `memory_reference_search` tool (not auto-injected)
- [ ] PDF text extraction pipeline
- [ ] Version tracking for reference documents
- [ ] Separate FTS index per reference table

### Phase 3: Classification Pipeline
- [ ] Integrate zero-shot classifier (Nemotron) for content routing
- [ ] Replace heuristic-only classification with LLM+heuristic hybrid
- [ ] Implement table routing based on classification output
- [ ] Quality scoring refinement with LLM feedback

### Phase 4: Evaluation Overhaul
- [ ] Fix all metric bugs (Codex findings)
- [ ] Calibrate BM25 sigmoid from corpus score distribution
- [ ] Build golden dataset per table (not just global)
- [ ] Docker-based reproducible benchmarking
- [ ] A/B framework for pipeline ablation studies

### Phase 5: Production Deployment
- [ ] Full reindex with new multi-table schema
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
- Storage layer: multi-table CRUD, search, filtering
- Retrieval pipeline: each stage in isolation + full pipeline
- Classification: routing accuracy for each content type
- Security: prompt injection detection, memory escaping
- Configuration: all defaults, overrides, edge cases
- Error handling: Spark down, invalid data, concurrent access
