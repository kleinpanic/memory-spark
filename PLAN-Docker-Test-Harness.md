# PLAN: Docker Test Harness for Memory-Spark

**Created:** 2026-03-29 20:35 EDT
**Updated:** 2026-03-29 20:50 EDT
**Goal:** Move all testing/benchmarking to Docker, isolated from production OpenClaw

---

## End Goal: Professional README

Reference: https://github.com/kleinpanic/memory-spark (old version, but README style is the target)

**Target README features:**
- Centered header with badges (NDCG@10, MRR, Recall, Coverage)
- Architecture flowchart (mermaid)
- SVG charts (ablation, recall curve, temporal decay, latency)
- Key Results table (Full Pipeline vs Vanilla)
- vs. BEIR SOTA comparison
- Ablation study table
- **NEW:** testDbOC results showing OpenClaw-specific gains
- **NEW:** testDbRef results (if established benchmarks exist)
- Citation (bibtex)

**Two scientific claims to prove:**
1. **Performs well in standard RAG benchmarks** → testDbBEIR
2. **Performs well for OpenClaw specifically** → testDbOC

---

## Clarifications (from Klein)

| Point | Detail |
|-------|--------|
| Spark auth | API bearer token required — include in config |
| testDbBEIR | ALL BEIR datasets (SciFact, NFCorpus, FiQA) — scientifically rigorous |
| testDbOC | Prove OpenClaw agents perform better with memory-spark |
| testDbRef | Check for established PDF Q&A benchmarks; if none, plan custom |
| Docker | Overhaul first, before any benchmark work |
| Scripts | TypeScript, NOT bash-hell |
| Execution order | BEIR first (may take long time) → OC → Ref |
| Test agents | 4 agents (main, meta, dev, school/cortex), not 9 |
| Production | All 12 agents will be used |

---

## Current State (Problems)

| Issue | Current | Problem |
|-------|---------|---------|
| Agent model | `anthropic/claude-sonnet-4-6` | Not local, not Nemotron |
| Test data | Production `~/.openclaw/data/` | Contaminates real data |
| BEIR index | Mixed with workspace data | Config bug + no isolation |
| Fixtures | Empty `workspaces/fixtures/` | No reference library tests |
| Benchmarks | Run on host | Uses production instance |

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Container                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  OpenClaw Gateway (port 18899 → 18789)                      ││
│  │  └─ Agent: "local" (Nemotron on Spark)                     ││
│  │     └─ Plugin: memory-spark                                ││
│  │        ├─ lancedbDir: /home/node/.openclaw/data/testDbBEIR ││
│  │        └─ lancedbDir: /home/node/.openclaw/data/testDbOC   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  /home/node/.openclaw/data/                                      │
│  ├── testDbBEIR/lancedb/     ← BEIR corpus (SciFact, etc.)      │
│  ├── testDbOC/lancedb/       ← Agent workspace chunks           │
│  └── testDbRef/lancedb/      ← Reference library fixtures       │
│                                                                  │
│  /home/node/fixtures/                                            │
│  ├── docs/                   ← PDF/DOCX for reference tests     │
│  ├── messages/               ← JSON messages for classifier     │
│  └── entities/               ← Text with known entities (NER)   │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
   Spark Services (10.99.1.1) — ALL require Bearer Auth
   ├── 18091  Embeddings (Llama-Embed-Nemotron-8B)
   ├── 18096  Reranker (Llama-Nemotron-Rerank-1B-v2)
   ├── 18112  NER (BART-large-NER)
   ├── 18113  Zero-shot (BART-large-MNLI)
   └── 18081  LLM (Nemotron-120B)
   
   Auth: X-SPARK-BEARER-TOKEN header (from ~/.openclaw/.env or env var)
```

---

## Phase 1: Docker Config Overhaul

### 1.1 Update test-openclaw.json

**Changes:**
```json
{
  "agents": {
    "list": [
      {
        "id": "local",
        "name": "Local Test Agent",
        "model": "spark-vllm/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        "workspace": "/home/node/.openclaw/workspace"
      }
    ]
  },
  "plugins": {
    "entries": {
      "memory-spark": {
        "config": {
          "sparkHost": "10.99.1.1",
          "lancedbDir": "/home/node/.openclaw/data/testDbOC/lancedb",
          // ... rest of config
        }
      }
    }
  }
}
```

### 1.2 Create Multi-Database Configs

**New configs:**
- `configs/test-beir.json` — Points to `testDbBEIR`
- `configs/test-agent.json` — Points to `testDbOC`
- `configs/test-reference.json` — Points to `testDbRef`

### 1.3 Docker Compose Updates

**New volume mounts:**
```yaml
volumes:
  # Test databases (pre-populated)
  - ./data/testDbBEIR:/home/node/.openclaw/data/testDbBEIR:ro
  - ./data/testDbOC:/home/node/.openclaw/data/testDbOC:ro
  - ./data/testDbRef:/home/node/.openclaw/data/testDbRef:ro
  
  # Fixtures
  - ./workspaces/fixtures:/home/node/fixtures:ro
```

---

## Phase 2: Test Databases

### 2.1 testDbBEIR (BEIR Corpus) — DO FIRST

**Purpose:** Validate retrieval pipeline against academic benchmarks — prove standard RAG quality

**Contents (ALL BEIR datasets):**
- SciFact: 5,183 docs, 300 queries
- NFCorpus: 3,633 docs, 323 queries
- FiQA: 5,716 docs, 648 queries
- **Total:** ~14,500 docs, ~1,300 queries

**Why ALL datasets:**
- Single dataset = not statistically meaningful
- Multiple datasets = scientifically rigorous, comparable to papers
- BEIR 2.0 uses 18 datasets — we start with 3, expand later

**Build script (TypeScript):**
```typescript
// scripts/build-beir-testdb.ts
// 1. Load ALL BEIR corpora from evaluation/beir-datasets/
// 2. Index into ./data/testDbBEIR/lancedb/
// 3. Include qrels for evaluation
// 4. NO env vars — direct path in code
```

**Config override:**
```typescript
const cfg = resolveConfig({ 
  lancedbDir: "/home/node/.openclaw/data/testDbBEIR/lancedb",
  sparkBearerToken: process.env.SPARK_BEARER_TOKEN
});
```

**Expected results:**
- NDCG@10 >= 0.65 (vector-only)
- NDCG@10 >= 0.70 (full pipeline with rerank)
- Comparable to BEIR SOTA baselines

### 2.2 testDbOC (Agent Memory) — DO SECOND

**Purpose:** Prove OpenClaw agents perform better with memory-spark

**Contents (4 core agents):**
| Agent | Why included | Est. files |
|-------|--------------|------------|
| meta | Config, system knowledge | ~200 |
| main | General knowledge, user profile | ~150 |
| dev | Code patterns, project context | ~100 |
| school/cortex | Academic, coursework | ~80 |
| **Total** | | ~530 files |

**Why 4 agents, not 9:**
- Other 5 agents barely used — not enough data
- 4 core = representative, statistically meaningful
- Production uses all 12, but test needs density

**Build script (TypeScript):**
```typescript
// scripts/build-oc-testdb.ts
// 1. Copy 4 agent workspaces to ./workspaces/fixtures/agents/
// 2. Index into ./data/testDbOC/lancedb/
// 3. Generate LLM benchmark with Nemotron (queries from chunks)
```

**Benchmark generation (LLM-based):**
```typescript
// scripts/gen-oc-golden.ts
// For each chunk:
//   1. Call Nemotron: "Write 3 questions this text answers"
//   2. Record: { query, relevant_chunk_ids, agent }
// Output: evaluation/golden-ocmemory-llm.json
```

**Scientific claim to prove:**
"OpenClaw agents with memory-spark retrieve relevant context X% better than vanilla RAG"

### 2.3 testDbRef (Reference Library) — DO THIRD

**Purpose:** Test reference pool retrieval — prove document understanding

**Research needed:**
Are there established PDF Q&A benchmarks? Let me check...

**Options:**
| Option | Description | Est. effort |
|--------|-------------|-------------|
| **Established benchmark** | Use existing PDF Q&A dataset (e.g., QASPER, Natural Questions) | Low if exists |
| **Custom benchmark** | Build from fixtures + LLM-generated questions | High |

**Build script (TypeScript):**
```typescript
// scripts/build-ref-testdb.ts
// 1. Index ./workspaces/fixtures/docs/ into ./data/testDbRef/lancedb/
// 2. All chunks go to pool: "reference_library"
// 3. Include page numbers, section headers in metadata
```

---

## Phase 3: Test Suites

### 3.1 BEIR Benchmark Test

**File:** `evaluation/tests/beir.test.ts`

```typescript
describe("BEIR Retrieval Quality", () => {
  beforeAll(async () => {
    // Verify testDbBEIR is populated
    // Connect to Docker container's testDbBEIR
  });

  it("should achieve NDCG@10 >= 0.65 on SciFact (vector-only)", async () => {
    // Run vector-only retrieval
    // Compare against BEIR qrels
  });

  it("should achieve NDCG@10 >= 0.70 on SciFact (full pipeline)", async () => {
    // Run full pipeline (vector + FTS + rerank)
  });
});
```

**Run:** `docker exec oc-plugin-test npx tsx evaluation/tests/beir.test.ts`

### 3.2 Agent Memory Benchmark Test

**File:** `evaluation/tests/agent-memory.test.ts`

```typescript
describe("Agent Memory Retrieval", () => {
  it("should find correct agent for factual queries", async () => {
    // Query: "What timezone is Klein in?"
    // Expect: meta/USER.md in top 3
  });

  it("should respect pool isolation", async () => {
    // Query with pool: "agent_memory"
    // Expect: no reference_library results
  });
});
```

### 3.3 Reference Library Test

**File:** `evaluation/tests/reference.test.ts`

```typescript
describe("Reference Library Retrieval", () => {
  beforeAll(async () => {
    // Verify testDbRef has fixtures
  });

  it("should retrieve from reference pool only", async () => {
    // Query about fixture document content
    // Expect: all results from pool: "reference_library"
  });

  it("should handle PDF extraction correctly", async () => {
    // Query about PDF content
    // Verify chunk metadata has correct page numbers
  });
});
```

### 3.4 Zero-Shot Classifier Test

**File:** `evaluation/tests/classifier.test.ts`

```typescript
describe("Zero-Shot Classification", () => {
  const fixtures = loadFixtures("./workspaces/fixtures/messages/");

  it("should classify fact correctly", async () => {
    const msg = "Klein lives in Blacksburg, VA.";
    const result = await classifyForCapture(msg, cfg);
    expect(result.category).toBe("fact");
    expect(result.confidence).toBeGreaterThan(0.75);
  });

  it("should classify preference correctly", async () => {
    const msg = "I prefer using TypeScript over JavaScript.";
    const result = await classifyForCapture(msg, cfg);
    expect(result.category).toBe("preference");
  });

  it("should fall back to heuristic when confidence low", async () => {
    const msg = "Hello world"; // Too short
    const result = await classifyForCapture(msg, cfg);
    // Should use heuristic fallback
    expect(result.category).toBeDefined();
  });
});
```

### 3.5 NER Test

**File:** `evaluation/tests/ner.test.ts`

```typescript
describe("Named Entity Recognition", () => {
  it("should extract person names", async () => {
    const text = "Klein and Brodie discussed the plan.";
    const entities = await extractEntities(text, cfg);
    expect(entities).toContainEqual(
      expect.objectContaining({ entity_group: "PER", word: expect.stringMatching(/Klein|Brodie/) })
    );
  });

  it("should extract locations", async () => {
    const text = "The server is in Blacksburg, VA.";
    const entities = await extractEntities(text, cfg);
    expect(entities).toContainEqual(
      expect.objectContaining({ entity_group: "LOC" })
    );
  });
});
```

---

## Phase 4: Fixture Data

### 4.1 Reference Documents

**Location:** `workspaces/fixtures/docs/`

```
fixtures/docs/
├── pdf/
│   ├── sample-paper.pdf          # Academic paper for retrieval test
│   └── technical-manual.pdf      # Multi-section manual
├── docx/
│   └── meeting-notes.docx        # Meeting transcript
└── txt/
    └── knowledge-base.txt        # Plain text reference
```

### 4.2 Classifier Messages

**Location:** `workspaces/fixtures/messages/`

```json
[
  {"text": "Klein lives in Blacksburg, VA.", "expected": "fact"},
  {"text": "I prefer dark mode over light mode.", "expected": "preference"},
  {"text": "We decided to use PostgreSQL for the database.", "expected": "decision"},
  {"text": "function add(a, b) { return a + b; }", "expected": "code-snippet"},
  {"text": "Hello, how are you?", "expected": "none"}
]
```

### 4.3 NER Fixtures

**Location:** `workspaces/fixtures/entities/`

```json
[
  {
    "text": "Klein panic lives in Blacksburg, VA and works at Virginia Tech.",
    "expected": {
      "PER": ["Klein panic"],
      "LOC": ["Blacksburg, VA", "Virginia Tech"]
    }
  }
]
```

---

## Phase 5: Scripts (TypeScript — NO BASH-HELL)

### 5.1 Build Test Databases

```typescript
// scripts/build-beir-testdb.ts
// Single TypeScript script to build testDbBEIR
// Usage: npx tsx scripts/build-beir-testdb.ts

// scripts/build-oc-testdb.ts
// Build testDbOC from 4 agent workspaces
// Usage: npx tsx scripts/build-oc-testdb.ts

// scripts/build-ref-testdb.ts
// Build testDbRef from fixtures
// Usage: npx tsx scripts/build-ref-testdb.ts
```

### 5.2 Run Tests

```typescript
// scripts/run-beir-test.ts
// Run BEIR benchmark against testDbBEIR
// Output: evaluation/results/beir-YYYYMMDD.json

// scripts/run-oc-test.ts
// Run agent memory benchmark against testDbOC
// Output: evaluation/results/oc-YYYYMMDD.json

// scripts/run-ref-test.ts
// Run reference library benchmark against testDbRef
// Output: evaluation/results/ref-YYYYMMDD.json
```

### 5.3 Generate Charts

```typescript
// evaluation/charts.ts (EXISTS — update for new results)
// Generate SVG charts from results
// Output: docs/figures/*.svg
```

---

## Implementation Order

**Phase 0: Docker Overhaul (MUST BE FIRST)**
| Step | Task | Est. Time |
|------|------|-----------|
| 0.1 | Update test-openclaw.json for Nemotron + Bearer token | 15 min |
| 0.2 | Create multi-database configs (BEIR, OC, Ref) | 30 min |
| 0.3 | Update docker-compose.yml with volume mounts | 30 min |
| 0.4 | Verify Spark connectivity from container | 15 min |

**Phase 1: testDbBEIR (DO FIRST — May Take Long Time)**
| Step | Task | Est. Time |
|------|------|-----------|
| 1.1 | Create `scripts/build-beir-testdb.ts` | 1 hr |
| 1.2 | Build testDbBEIR (all 3 datasets) | 2-4 hr (indexing) |
| 1.3 | Create `scripts/run-beir-test.ts` | 1 hr |
| 1.4 | Run BEIR benchmark | 30 min |
| 1.5 | Verify NDCG@10 >= 0.65 | — |

**Phase 2: testDbOC (DO SECOND)**
| Step | Task | Est. Time |
|------|------|-----------|
| 2.1 | Create `scripts/build-oc-testdb.ts` | 1 hr |
| 2.2 | Copy 4 agent workspaces to fixtures | 30 min |
| 2.3 | Build testDbOC | 1 hr |
| 2.4 | Create `scripts/gen-oc-golden.ts` (LLM benchmark) | 2 hr |
| 2.5 | Create `scripts/run-oc-test.ts` | 1 hr |
| 2.6 | Run OC benchmark | 30 min |

**Phase 3: testDbRef (DO THIRD — Research First)**
| Step | Task | Est. Time |
|------|------|-----------|
| 3.1 | Research established PDF Q&A benchmarks | 1 hr |
| 3.2 | Decide: established vs custom | — |
| 3.3 | Create build/run scripts | 1-3 hr |
| 3.4 | Run reference benchmark | 30 min |

**Phase 4: README & Documentation**
| Step | Task | Est. Time |
|------|------|-----------|
| 4.1 | Update charts.ts for new results | 1 hr |
| 4.2 | Generate SVG charts | 30 min |
| 4.3 | Write professional README | 2 hr |
| 4.4 | Add badges, flowchart, tables | 1 hr |

---

## Research: PDF Q&A Benchmarks for testDbRef

**Question:** Are there established frameworks for PDF retrieval evaluation?

**Candidates to investigate:**

| Benchmark | Description | Suitability |
|-----------|-------------|-------------|
| **QASPER** | Questions from NLP papers | ✅ Academic papers |
| **Natural Questions** | Google search queries | ⚠️ Web, not PDF-specific |
| **MS MARCO** | Bing search queries | ⚠️ Web, not PDF-specific |
| **SQuAD** | Reading comprehension | ⚠️ Short passages, not retrieval |
| **HotpotQA** | Multi-hop reasoning | ⚠️ Wikipedia, not PDF |
| **FinanceBench** | Financial document Q&A | ✅ Document-focused |
| **LegalBench** | Legal document reasoning | ✅ Document-focused |

**If none suitable:**
Build custom benchmark:
1. Collect 20-50 PDF documents (papers, manuals, reports)
2. Use Nemotron to generate questions from each
3. Create evaluation set with ground truth answers

---

## Files to Remove

| File | Reason |
|------|--------|
| ~~GAPS.md~~ | ✅ Already deleted |

(Other markdowns kept per Klein's request)

---

## Success Criteria

### Docker
- [ ] Container runs with Nemotron agent
- [ ] Spark bearer auth configured
- [ ] Test databases mounted correctly
- [ ] No contamination of production data

### testDbBEIR
- [ ] All 3 BEIR datasets indexed
- [ ] NDCG@10 >= 0.65 (vector-only)
- [ ] NDCG@10 >= 0.70 (full pipeline)
- [ ] Results comparable to BEIR SOTA baselines

### testDbOC
- [ ] 4 agent workspaces indexed
- [ ] LLM-generated benchmark created
- [ ] Proves OpenClaw gains over vanilla RAG

### testDbRef
- [ ] Research completed on established benchmarks
- [ ] Decision made: established vs custom
- [ ] Benchmark implemented

### README
- [ ] Professional header with badges
- [ ] Architecture flowchart (mermaid)
- [ ] SVG charts generated
- [ ] Key results tables
- [ ] Ablation study
- [ ] vs. BEIR SOTA comparison
- [ ] Citation (bibtex)
