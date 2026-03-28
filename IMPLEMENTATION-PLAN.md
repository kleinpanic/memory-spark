# memory-spark Implementation Plan v2
## 2026-03-28

---

## Core Agent List (Benchmark Scope)

Only these agents get indexed in the dev benchmark instance and tested:

| Agent | Role | Chunks (prod) | Reason |
|-------|------|---------------|--------|
| **meta** | Config architect | 15,674 | Primary maintenance agent |
| **main** | General assistant | 1,948 | Most user-facing |
| **dev** | Developer | 477 | Core coding agent |
| **school** | Academic | 405 | Active loop agent |
| **immune** | Security/health | 67 | Assignment agent |
| **recovery** | Recovery ops | 39 | Assignment agent |
| **ghost** | Messaging | 102 | Assignment agent |
| **research** | Research | 48 | Assignment agent |
| **taskmaster** | Task orchestration | 131 | Assignment agent |
| **shared** | Cross-agent knowledge | 6,370 | Reference library |

**Excluded from benchmarks:** bench, claude, codex, coach, finance, legal, lens, local, mika, node, pantry

The golden dataset and dev index will ONLY contain content from these 10 agents.

---

## Phase 1: Evaluation Overhaul

### 1A. Curated Dev Index

**What:** Build a separate reindex script that only ingests the 10 core agents.

**File:** `scripts/reindex-benchmark.ts`

**Logic:**
```
1. Read workspace dirs for ONLY the 10 core agents
2. Also include shared/ reference library
3. Index into test-data/ (separate from production)
4. Expected size: ~25k chunks (meta + shared dominate)
```

### 1B. LLM-as-Judge Evaluator

**What:** Use Nemotron-Super-120B to grade retrieval quality.

**File:** `evaluation/llm-judge.ts`

**API:** Direct HTTP to `http://10.99.1.1:18080/v1/chat/completions`
- Model: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
- Thinking: enabled with `budget_tokens: 100` (keeps it fast, ~4.5s/call)
- Batch: 5 query-passage pairs per call (saves overhead)
- Parallelism: 3 concurrent HTTP calls (Spark can handle it)
- Output: JSON `{"score": N}` where N=0-3

**Evaluation flow:**
```
For each query in golden dataset:
  1. Run full retrieval pipeline → top 10 results
  2. For each result, call Nemotron-Super judge:
     "Query: {query}\nPassage: {chunk_text}\nScore 0-3"
  3. Parse scores, compute graded NDCG, MRR, MAP
  4. Also compute: answer quality (does top-1 actually answer the question?)
```

**Speed estimate:** 200 queries × 10 results = 2000 judgments
- Batched (5/call) = 400 calls × 4.5s = 30 min serial
- With 3x parallelism = ~10 min per full eval run

### 1C. Golden Dataset v3

**What:** Generate 200+ queries covering the 10 core agents properly.

**File:** `evaluation/golden-dataset-v3.json`

**Categories:**
- 60 factual queries (who/what/where about Klein, agents, infra)
- 40 procedural queries (how to restart, config workflow, task lifecycle)
- 30 technical queries (ports, IPs, models, services)
- 20 cross-agent queries (shared knowledge, common patterns)
- 20 mistakes/learning queries (past incidents, safety rules)
- 15 security/negative queries (prompt injection, out-of-scope)
- 15 edge cases (short queries, typos, ambiguous)

**Generation method:**
1. Sample 500 chunks from the 10 core agents
2. For each chunk, use Nemotron-Super to generate 1-2 natural queries
3. Store: query, source_chunk_ids, graded_relevance, gold_answer text
4. Human review pass (Klein spot-checks a sample)

### 1D. Multiple BEIR Datasets

**File:** `evaluation/beir-adapter.ts` (already exists, extend)

**Add datasets:**
- SciFact (already done) — scientific claim verification
- NQ (Natural Questions) — Wikipedia-based QA
- FiQA — financial opinion QA (tests domain transfer)
- TREC-COVID — pandemic info retrieval

Each dataset downloaded once, cached in `evaluation/datasets/`.

### 1E. Pipeline Telemetry

**What:** Log every stage's contribution per query.

**File:** `evaluation/telemetry.ts`

**Output per query:**
```json
{
  "query": "What timezone is Klein in?",
  "stages": {
    "vector": {"count": 40, "top1_score": 0.406, "top1_path": "finance:USER.md"},
    "fts": {"count": 40, "top1_score": 1.0, "top1_path": "ghost:USER.md"},
    "hybrid": {"count": 40, "top1_score": 0.534, "promoted": 3, "demoted": 2},
    "source_weight": {"max_boost": 1.6, "max_penalty": 0.5},
    "temporal_decay": {"avg_decay": 0.87},
    "mmr": {"removed_duplicates": 5},
    "reranker": {"top1_score": 0.971, "top1_path": "finance:USER.md", "reorder_count": 7}
  },
  "judge_score": 3,
  "latency_ms": 1250
}
```

---

## Phase 2: HyDE (Hypothetical Document Embeddings)

### What
Before embedding a query, ask Nemotron-Super to generate a hypothetical passage that would answer the question. Embed THAT instead of the raw query. This bridges the vocabulary gap between short questions and detailed documents.

### Architecture

**File:** `src/hyde/generator.ts`

```
User query: "What timezone is Klein in?"
    ↓
HyDE prompt → Nemotron-Super: "Write a short factual passage answering: What timezone is Klein in?"
    ↓
Hypothetical doc: "Klein's timezone is America/New_York (Eastern Time). He lives in Blacksburg, Virginia."
    ↓
Embed hypothetical doc → 4096-dim vector
    ↓
Vector search with HyDE embedding (+ original query embedding for FTS)
```

### Config
```json
{
  "hyde": {
    "enabled": true,
    "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "endpoint": "http://10.99.1.1:18080/v1/chat/completions",
    "maxTokens": 150,
    "temperature": 0.3,
    "cacheEnabled": true,
    "cacheTTLMs": 300000
  }
}
```

### Integration Points
1. `src/auto/recall.ts` → before `backend.vectorSearch()`, generate HyDE embedding
2. Use HyDE vector for vector search, original query text for FTS
3. Cache HyDE results (same query within 5 min = same hypothetical doc)
4. Fallback: if HyDE generation fails, use original query embedding

### Expected Impact
Published HyDE results show +15-25% NDCG improvement for domain-specific retrieval. Our gap between query language ("What timezone?") and document language ("America/New_York") is exactly the pattern HyDE addresses.

### Latency Budget
- HyDE generation: ~4s (Nemotron-Super with thinking)
- This adds to the recall pipeline latency
- Mitigation: cache aggressively, pre-warm common patterns
- Total pipeline: vector(1s) + HyDE(4s) + rerank(2s) = ~7s worst case
- With cache: vector(1s) + HyDE(0ms) + rerank(2s) = ~3s

---

## Phase 3: Parent-Child Chunking

### What
Replace flat 400-token chunks with a two-level hierarchy:
- **Children** (200 tokens): Small, precise chunks for vector search
- **Parents** (2000 tokens): Large context windows returned to the agent

Search matches children, but the agent sees the parent context.

### Schema Changes

**LanceDB schema additions:**
```
parent_id: string | null    — child→parent reference
chunk_type: "parent" | "child" | "standalone"
```

### Chunking Pipeline

**File:** `src/embed/chunker-v2.ts`

```
Document (e.g., MEMORY.md, 5000 tokens)
    ↓
Split into PARENTS (2000 tokens, markdown-section-aware)
    ↓
Each parent splits into CHILDREN (200 tokens, with 30 token overlap)
    ↓
Store both in LanceDB:
  - Parent: full text, chunk_type="parent", no vector (saves embedding cost)
  - Children: small text, chunk_type="child", parent_id=parent.id, HAS vector
```

### Retrieval Changes

**File:** `src/auto/recall.ts` modifications

```
1. Vector/FTS search → matches CHILDREN only (WHERE chunk_type='child')
2. For each matched child → look up parent_id
3. Deduplicate parents (multiple children may match same parent)
4. Return PARENT text to the agent (not child text)
5. Reranker scores parent text (what the agent actually sees)
```

### Benefits
- **Better precision:** Small 200-token children match queries more precisely
- **Better context:** 2000-token parents give the agent enough surrounding info
- **Less redundancy:** Multiple children from same section → one parent

### Migration
- Full re-index required (changes chunk structure)
- Parents are ~5x larger but don't need vectors → net storage is similar
- Expected chunk count: ~130k children + ~26k parents = ~156k rows total
- Embedding cost: same (only children get embedded)

---

## Phase 4: Integration Testing

After all three features are built:

1. Re-index dev instance with parent-child chunks (10 core agents only)
2. Run golden dataset v3 with LLM-as-judge
3. A/B test configurations:
   - Baseline: current pipeline (no HyDE, flat chunks)
   - +HyDE only
   - +Parent-child only
   - +HyDE +Parent-child (full)
4. Run BEIR datasets for standardized comparison
5. Generate ablation matrix

### Success Criteria
- Custom domain NDCG@10 ≥ 0.65 (currently ~0.45 with broken eval)
- LLM-as-judge answer quality ≥ 80% (top-1 correctly answers query)
- Reranker shows positive lift (currently negative due to eval bug)
- BEIR SciFact stays above 0.75

---

## Phase 5: Production Deploy

1. Copy verified config to `~/.openclaw/extensions/memory-spark/`
2. Rebuild production index with parent-child chunks
3. Enable in openclaw.json with the Docker dev gateway initially
4. Monitor auto-recall latency and quality for 24h
5. Full production enable

---

## Execution Order

```
Phase 1A: Curated reindex script          → 1h
Phase 1B: LLM-as-judge evaluator          → 2h  
Phase 1C: Golden dataset v3 generation     → 2h (mostly LLM time)
Phase 1D: BEIR datasets                    → 1h
Phase 1E: Pipeline telemetry               → 1h
Phase 2:  HyDE                             → 2h
Phase 3:  Parent-child chunking            → 4h
Phase 4:  Integration testing              → 3h
Phase 5:  Production deploy                → 1h
                                     Total: ~17h
```

Start with Phase 1 (eval fixes) — everything else depends on having reliable metrics.
