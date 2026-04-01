# PLAN: Phase 11 — Post-10B Improvements

**Date:** 2026-04-01
**Author:** KleinClaw-Meta
**Task:** `00889d7b` — Deep recon: BEIR benchmark underperformance
**Baseline:** Config Q (Vector → Logit Blend α=0.5) = **0.7863 NDCG@10** (+2.0% over vector-only)

---

## Phase 10B Results (Current State)

| Config | Strategy | NDCG@10 | vs A | Recall@10 | p95 Latency |
|--------|----------|---------|------|-----------|-------------|
| **Q** | Logit Blend α=0.5 | **0.7863** | +2.0% 🏆 | 0.9143 | 1563ms |
| **S** | Conditional + Blend α=0.3 | 0.7792 | +1.1% | 0.9099 | 1670ms |
| **T** | Logit Blend α=0.3 | 0.7756 | +0.6% | 0.9032 | 1956ms |
| **A** | Vector-Only (old baseline) | 0.7709 | — | 0.9037 | 1055ms |
| **R** | Pure Reranker α=0 | 0.7395 | -4.1% | 0.8924 | 1738ms |

**Key insight:** The reranker adds real signal when blended with vector scores (α=0.5 optimal so far). Pure reranker remains net-negative.

---

## Phase 11A: Alpha Sweep (Quick Win)

**Goal:** Find the optimal blending alpha between vector and reranker scores.
**Effort:** ~1 hour (code + benchmark)
**Expected impact:** +0.5-1.0% NDCG@10

### Rationale
We only tested α ∈ {0.0, 0.3, 0.5}. The optimal alpha could be anywhere in [0.4, 0.7]. This is the cheapest experiment — no new logic, just new configs.

### New Configs

| Config | α | Label |
|--------|---|-------|
| U | 0.4 | Logit Blend α=0.4 |
| V | 0.6 | Logit Blend α=0.6 |
| W | 0.7 | Logit Blend α=0.7 |
| X | 0.8 | Logit Blend α=0.8 (sanity — should converge to vector) |

### Implementation
1. Add 4 config entries to `scripts/run-beir-bench.ts` (copy Q, change alpha + id)
2. Run all 4 on SciFact
3. Plot the alpha curve: α vs NDCG@10
4. If peak is between tested values (e.g., 0.55), run one more interpolation

### Success Criteria
- Find the alpha that maximizes NDCG@10
- Confirm the curve is smooth (no weird discontinuities = healthy blending math)

---

## Phase 11B: Multi-Query Expansion (Biggest Potential Gain)

**Goal:** Break the 11% retrieval ceiling by generating query reformulations.
**Effort:** ~3-4 hours (new module + benchmark configs)
**Expected impact:** +2-5% NDCG@10 (addresses hard recall ceiling)

### The Problem
11% of relevant documents don't appear in the top-40 vector results. No amount of reranking/blending can recover these — they're invisible to the pipeline. Different phrasings of the same query can surface completely different documents.

### Architecture

```
User Query
    │
    ├──────────────────┐
    │                  │
    ▼                  ▼
  LLM: Generate     Original Query
  2-3 reformulations    │
    │                  │
    ▼                  ▼
  Embed each        Embed original
    │                  │
    ▼                  ▼
  Vector search     Vector search
  (top-40 each)     (top-40)
    │                  │
    └──────┬───────────┘
           │
           ▼
     Union + Dedupe
     (by doc ID, keep best score)
           │
           ▼
     Reranker (logit blend)
           │
           ▼
     Top-10 results
```

### Implementation

1. **New file: `src/query/expander.ts`**
   ```typescript
   export interface QueryExpansionConfig {
     enabled: boolean;
     llmUrl: string;
     model: string;
     numReformulations: number; // default: 3
     maxTokens: number;        // default: 100
     temperature: number;      // default: 0.7 (diverse)
     timeoutMs: number;        // default: 10000
     apiKey?: string;
   }

   export async function expandQuery(
     query: string,
     config: QueryExpansionConfig
   ): Promise<string[]>;  // returns [original, ...reformulations]
   ```

2. **System prompt for reformulation:**
   ```
   Generate {n} alternative phrasings of the following search query.
   Each rephrasing should use different vocabulary while preserving the
   original meaning. Return one rephrasing per line, no numbering.
   Focus on: synonyms, passive/active voice, and different abstraction levels.
   ```

3. **Modify `recall.ts`:** After query expansion, run vector search for each reformulation, union results by doc ID (keep highest score), then proceed to reranker.

4. **New benchmark configs:**
   | Config | Strategy |
   |--------|----------|
   | MQ-A | Multi-Query (3) → Vector → Top-10 |
   | MQ-B | Multi-Query (3) → Vector → Logit Blend α=optimal |
   | MQ-C | Multi-Query (3) → Vector → Logit Blend + Conditional |

5. **Model choice:** Use Nemotron-Super (only model on vLLM). Query expansion prompts are simple — should be fast at 100 max tokens.

### Risks
- **Latency:** 3 extra embed calls + 3 extra vector searches. Mitigation: run in parallel (`Promise.all`).
- **Noise:** Bad reformulations could add irrelevant candidates. Mitigation: reranker filters these out (that's its job now that it works).
- **LLM timeout:** Same issue HyDE had. Mitigation: configurable timeout + graceful fallback to original query only.

### Success Criteria
- Recall@10 improves from 0.9143 → 0.94+
- NDCG@10 improves over best alpha from Phase 11A
- Latency stays under 3s p95

---

## Phase 11C: HyDE with Fast Model (Medium Gain)

**Goal:** Activate HyDE as a retrieval enhancement, not a replacement.
**Effort:** ~2 hours (config + testing)
**Expected impact:** +1-3% NDCG@10

### The Problem
HyDE timed out in all previous benchmarks because Nemotron-Super takes ~10.5s per generation. The concept is sound — embedding a hypothetical answer bridges the question↔document vocabulary gap — but the latency makes it impractical.

### Strategy

**Option A: Use Nemotron-Super with aggressive settings**
- Drop `maxTokens` from 150 → 50 (2-3 sentences is enough)
- Set `temperature: 0.1` (fast, deterministic)
- Increase `timeoutMs` from 10s → 20s
- Pros: No new infra. Cons: Still slow.

**Option B: Spin up a smaller model on Spark Ollama**
- `qwen3-4b` or `nemotron-nano-9b` on Ollama
- Much faster generation (< 1s)
- Pros: Production-viable latency. Cons: Needs model pull + config.

**Option C: Use HyDE only as a recall expander (not a replacement)**
- Current implementation replaces the query vector with HyDE vector
- Instead: treat HyDE vector as an additional query (like multi-query expansion)
- Run both original + HyDE vectors, union results
- This is safer — HyDE can only add candidates, never remove them

### Recommended: Option C on top of Option A/B
- HyDE as an additional retrieval path, not a replacement
- If Ollama has a fast model → use it. If not → aggressive Nemotron-Super settings.

### New benchmark configs:
| Config | Strategy |
|--------|----------|
| HY-A | Vector + HyDE (additive) → Top-10 |
| HY-B | Vector + HyDE (additive) → Logit Blend |
| HY-C | Multi-Query + HyDE → Logit Blend (the full kitchen sink) |

### Success Criteria
- HyDE generation latency < 3s p95
- HyDE provides positive NDCG@10 lift over non-HyDE equivalents
- No regressions on queries where vector-only was already optimal

---

## Execution Order

```
Phase 11A: Alpha Sweep          ← Do first (30 min, no code changes, biggest confidence)
    │
    ▼
Phase 11B: Multi-Query Expansion ← Do second (3-4 hrs, new module, biggest ceiling-breaker)
    │
    ▼
Phase 11C: HyDE Fast Model      ← Do third (2 hrs, builds on 11B infrastructure)
    │
    ▼
Phase 12: Cross-Dataset Validation
    └── Run winning configs on FiQA + NFCorpus to confirm generalization
```

### Timeline (estimated)
| Phase | Effort | Cumulative |
|-------|--------|------------|
| 11A | 30 min | 30 min |
| 11B | 3-4 hrs | 4-4.5 hrs |
| 11C | 2 hrs | 6-6.5 hrs |
| 12 | 1 hr (benchmark only) | 7-7.5 hrs |

---

## Git Strategy
- Branch: `feat/phase11-improvements` (off `fix/phase10b-unified-reranker`)
- One commit per sub-phase
- Squash merge to main when Phase 12 validates

---

## Target
**Current best:** 0.7863 NDCG@10 (Config Q, α=0.5)
**Target after Phase 11:** 0.82+ NDCG@10 with Recall@10 > 0.95
