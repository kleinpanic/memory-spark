# PLAN Addendum: Reorientation & Revised Strategy

> Written: 2026-03-26 13:30 EDT | Author: KleinClaw-Meta
> Context: Session regression audit. Original PLAN.md is still valid but needs reframing.
> Status: **PROPOSAL — needs Klein approval before any execution**

---

## What Went Wrong This Session

### Root Problems
1. **Lost the frame.** Treated memory-spark as a standalone RAG system instead of what it is: an **OpenClaw plugin** that uses Spark services to improve agent memory. Every change must be evaluated through "does this make agents smarter?"
2. **Didn't read the source.** Made assumptions about LanceDB, the plugin loader, and the watcher instead of reading OpenClaw's plugin SDK docs and memory-spark's actual code first.
3. **Premature optimization.** Jumped to fixing the RRF merge algorithm and rewriting the eval harness before having a clean index to test against. Can't benchmark a broken index.
4. **Flailed on infra.** Wasted time debugging tsx top-level-await, stale LanceDB reads, and reindex scripts that don't persist, instead of using the plugin's own boot-pass indexer (the watcher service).

### What Was Actually Accomplished (Salvageable)
- **recall.ts fixes:** `hybridMerge` (preserves cosine scores), early source weighting, FTS session filtering — these are sound and backed by validated research. 128/128 tests pass.
- **eval/run.ts rewrite:** Now uses real production pipeline functions + actual Spark reranker instead of fake `scoreCandidate()` simulation. This is correct.
- **Research reports:** SOTA landscape (Mem0, Zep, Letta), HyDE verdict (don't use), cross-encoder failure modes, tool injection patterns. Saved in `docs/`.
- **Live eval baseline:** First real numbers. Full pipeline NDCG@10=0.293, vanilla=0.559, no-FTS=0.615. These are honest measurements.
- **Spark zero-shot healthcheck:** Fixed (5s→15s timeout). Container healthy.
- **LanceDB wipe:** Old 2.2GB of garbage deleted. Clean slate for proper reindex.

### What Regressed
- Index is now nearly empty (10 chunks from captures only). The watcher boot-pass needs to run a full reindex on next gateway restart.
- No gateway restart was done (correctly — meta shouldn't restart during autonomous sessions), so the boot-pass hasn't fired.

---

## Reorientation: memory-spark Is an OpenClaw Plugin

### What It DOES (OpenClaw Plugin Contract)
1. **Occupies the `memory` slot** (`plugins.slots.memory = "memory-spark"`) — replaces memory-core
2. **Registers 7 tools:** `memory_search`, `memory_get`, `memory_store`, `memory_forget`, `memory_reference_search`, `memory_index_status`, `memory_forget_by_path`
3. **`before_prompt_build` hook:** Auto-recall — injects `<relevant-memories>` XML into agent context before every turn
4. **`agent_end` hook:** Auto-capture — extracts facts/preferences/decisions from agent turns, stores them
5. **`after_compaction` hook:** Re-indexes compacted session files
6. **Background service:** File watcher + boot-pass indexer
7. **CLI:** `openclaw memory status|sync|migrate`

### What It Interfaces With (Spark Services)
| Service | Port | Purpose |
|---------|------|---------|
| Embed (Nemotron-Embed-8B) | 18095 | 4096-dim embeddings for vector search |
| Reranker (Nemotron-Rerank-1B) | 18098 | Cross-encoder reranking of candidates |
| Zero-shot classifier | 8013 (internal) | Auto-capture category classification |
| NER | internal | Entity extraction for stored memories |
| OCR (GLM-OCR) | 18096 (vLLM) | PDF/image text extraction for ingest |
| vLLM (Nemotron-Super-120B) | 18096 | NOT directly used by memory-spark (agents use this separately) |

### The Goal (Why This Plugin Exists)
**Make agents smarter by giving them the right context at the right time.**

- Agent asks about Spark config → recall injects the relevant MEMORY.md section about Spark endpoints
- Agent makes a mistake → MISTAKES.md entries are boosted 1.6x so the error is recalled next time
- Agent discovers a fact → auto-capture stores it so it persists across sessions
- Agent needs reference docs → `memory_reference_search` retrieves from indexed textbooks/API docs
- Prompt injection in recalled text → security filter blocks it before injection

---

## Revised Phase Plan

### Phase 0: Foundation (MUST DO FIRST — No Code Changes)
**Goal:** Get a clean, properly indexed LanceDB table and honest baseline metrics.

1. **Trigger the boot-pass reindex.** NOT by writing a custom script. The watcher service already does this perfectly — it discovers all agent workspaces, compares mtime vs updated_at, and indexes delta. The table was wiped, so everything is delta. Just restart the gateway (via Klein approval + oc-restart).

2. **After reindex completes, run live eval.** Use `npx tsx evaluation/run.ts` (which now calls the real production pipeline). This gives honest baseline numbers on clean data.

3. **Commit the current recall.ts + eval/run.ts + tests changes.** These are validated and correct. Don't mix them with future work.

### Phase 1: Fix What's Actually Broken (From Eval Data)
**Goal:** Address root causes identified by live eval.

The live eval showed:
- No-FTS (vector-only + reranker + decay + quality + context + mistakes) = **0.615 NDCG@10** — BEST
- Full pipeline = 0.293 — FTS is dragging it down

**But wait** — the eval ran on a near-empty index (10 chunks). These numbers are meaningless. We need Phase 0 first.

After Phase 0 reindex, the eval will tell us which pipeline stage actually hurts. Then we fix that stage. Don't pre-optimize.

**Specific things to investigate after reindex:**
- Is FTS still poisonous with clean data? (No more session garbage → FTS might actually help)
- Does the reranker help or hurt with clean data? (Research says cross-encoder can hurt on domain-specific content)
- Is the quality gate too aggressive? (Some memory files got all chunks filtered)
- What's the recall on MISTAKES.md content? (1.6x boost — is it enough?)

### Phase 2: Improve What Agents Actually Need (From PLAN.md Phases 1-2)
**Goal:** The features that directly improve agent behavior.

From the original PLAN.md, reordered by **impact on agent performance:**

**2a. Query Cleaning Improvements** (recall.ts)
- The `cleanQueryText` already strips Discord metadata, but agents send messy prompts
- Add: strip `[System: ...]` prefixes, `NO_REPLY` patterns, repeated whitespace
- Low effort, direct recall quality improvement

**2b. Manager Search Parity** (manager.ts)  
- `manager.search()` uses a naive merge (dedup by id, sort by score) — it should use `hybridMerge` + `applySourceWeighting` + `applyTemporalDecay` like the auto-recall pipeline
- Currently tools (`memory_search`) get worse results than auto-recall (`before_prompt_build`)
- This is a bug, not a feature

**2c. Auto-Capture Improvement** (capture.ts)
- Only 83 captures vs 18K memory chunks — barely firing
- The zero-shot classifier on Spark is working (healthy now), but the trigger threshold might be too high
- Review `minConfidence` and `minMessageLength` settings
- Test: manually trigger captures, verify they appear in recall

**2d. Embedding Cache** (PLAN.md Phase 2.3)
- Same query from same agent within 1 hour → skip embed call
- Simple LRU cache, reduces Spark load, <1 hour of work

### Phase 3: New Capabilities (From PLAN.md Phases 4-5)
**Goal:** Expand what memory-spark can do for agents.

**3a. `memory_health` tool** — gives agents visibility into the memory system
- Embed queue depth, Spark connectivity, staleness report
- Immune agent can use this in health checks

**3b. `memory_watch` tool** — hot-add directories at runtime
- Agent discovers a new project dir → `memory_watch(path)` → indexed automatically
- Persists across restarts via `watched-paths.json`

**3c. Bootstrap Bloat Reduction** (PLAN.md Phase 5)
- Move static facts from AGENTS.md (20KB+) into reference library
- Agents retrieve via `memory_reference_search` instead of burning context tokens
- This is where memory-spark pays for itself: smaller system prompts + better recall

### Phase 4: Evaluation & Hardening (PLAN.md Phase 6)
**Goal:** Credible, reproducible benchmarks.

**4a. Golden Dataset Expansion**
- 100+ queries with gold answers and source paths
- Query types: factoid, procedural, troubleshooting, cross-agent
- Adversarial: paraphrases, negation, entity swaps

**4b. Weak Model Testing**
- Feed recalled context to Nemotron-Super (the running model on Spark)
- If the model answers correctly WITH context but NOT without → retrieval is adding value
- This is the ultimate test of whether memory-spark is useful

**4c. A/B Agent Testing**
- Run same agent session with memory-spark ON vs OFF
- Measure: task completion rate, factual accuracy, repeated mistakes
- This measures real-world impact, not just IR metrics

### Phase 5: Skip / Deprioritize
These items from the original PLAN.md are either premature or wrong for our use case:

- **HyDE** (Phase 2.2) — Skip. Research confirmed: latency budget doesn't allow it, agent memories are already answer-shaped.
- **Parent-Child Chunking** (Phase 3) — Deprioritize. Research confirmed: agent memories are short distilled facts (~50-200 tokens). Chunking matters for long documents, not memory stores. Our 500-token chunks are close to the validated 512-token default.
- **Dimension Reduction** (Phase 7.1) — Deprioritize. 4096-dim is local (no API cost), Nemotron-Embed-8B is #1 on MTEB at these dims. No reason to reduce.
- **Memory Linking** (Phase 4.3) — Deprioritize. Nice-to-have but complex. Zep does this with a graph DB, we don't have one.
- **Context-Aware Tool Suggestions** (Phase 4.4) — Deprioritize. Interesting but speculative. Agents already have tools listed in TOOLS.md.

---

## Execution Order (Revised)

```
Phase 0: Gateway restart → boot-pass reindex → live eval baseline
    ↓
Phase 1: Fix what eval data tells us (not pre-optimizing)
    ↓
Phase 2a-2d: Query cleaning, manager parity, capture, embed cache
    ↓
Phase 3a-3c: New tools, bootstrap bloat reduction
    ↓
Phase 4: Golden dataset, weak model testing, A/B
```

**Key principles:**
1. **Measure before changing.** Get clean data + honest baseline before any optimization.
2. **Test through agents.** The metric is "do agents behave better?" not NDCG.
3. **Use the plugin system.** Boot-pass indexer, `before_prompt_build`, `agent_end` — these are the interfaces. Don't build standalone scripts that bypass them.
4. **Don't touch `~/.openclaw/` directly.** Dev copy is `~/codeWS/TypeScript/memory-spark/`. Changes propagate via symlink.

---

## Immediate Next Steps (For Klein Approval)

1. **Commit current changes** to the dev repo (recall.ts, eval/run.ts, tests, research docs)
2. **Request gateway restart** (via oc-restart) to trigger boot-pass reindex on clean LanceDB
3. **Wait for reindex to complete** (monitor via `openclaw memory status`)
4. **Run live eval** on clean data
5. **Share results** and decide Phase 1 priorities based on what the data says
