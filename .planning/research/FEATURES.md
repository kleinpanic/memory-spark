# Feature Research — memory-spark v1.0

**Domain:** LLM agent memory plugin (RAG + long-term memory)
**Researched:** 2026-04-09
**Confidence:** HIGH for ecosystem survey, MEDIUM for specific Mem0/Zep internal tool counts (web-search sourced, not Context7-verified)
**Research mode:** Ecosystem + comparison
**Reference systems:** Mem0 (OSS + cloud), Zep/Graphiti (cloud + CE), Letta/MemGPT (framework), LangMem (LangGraph), langgraph-bigtool, Supermemory, Cognee, Hindsight

---

## Executive Opinion

**What memory-spark already does better than SOTA peers:**
1. **Scientific rigor / reproducibility** — BEIR harness + planned golden dataset. Nobody in the Mem0/Zep/Letta cluster publishes comparable in-repo benchmarks; most SOTA claims are marketing PDFs.
2. **Local-first, zero-egress** — Spark-backed Nemotron stack. Mem0 cloud / Zep cloud / Supermemory are all API-based. Only Letta and the LangMem local path are comparable. This is a **genuine differentiator** for the paper and website.
3. **13-stage recall pipeline introspection** — the proposed `memory_recall_debug` is rarer than the research papers suggest. LangSmith and Langfuse trace *LLM calls*, not *retrieval internals*. This is differentiator territory.
4. **Reranker gate telemetry** — completely novel in the OSS agent-memory space. Nobody else exposes gate decisions as a first-class concept. This IS your paper's contribution.

**What memory-spark is missing that peers have:**
1. **LLM-driven extraction UPDATE/DELETE pipeline** (Mem0's signature) — you capture, you don't consolidate. See "gap" section below.
2. **Bitemporal modeling** — Zep's invalidation-of-fact pattern. Worth addressing in paper as "out of scope, future work" rather than building.
3. **Reflections / consolidation** (Generative Agents, LangMem procedural) — importance-threshold-triggered summarization. Real gap but L-effort.
4. **Memory types taxonomy** — LangMem's semantic/episodic/procedural distinction. You have `contentType` (knowledge/decision/preference/mistake) which is actually *better* aligned to coding agents, but you don't brand it.

---

## Ground Truth: Current memory-spark Tool Surface

**IMPORTANT for downstream consumers:** `docs/PLAN-v040-release.md` Phase C is **stale** — it claims tool expansion from 9 → 14 as TODO, but `src/index.ts` already registers **18 tools** including all 5 "Phase C" tools. Verified by grep against `src/index.ts` on 2026-04-09:

```
memory_search, memory_get, memory_store, memory_forget, memory_reference_search,
memory_index_status, memory_forget_by_path, memory_inspect, memory_reindex,
memory_mistakes_search, memory_mistakes_store, memory_rules_store, memory_rules_search,
memory_recall_debug, memory_bulk_ingest, memory_temporal, memory_related, memory_gate_status
```

The v1.0 release milestone work is therefore **not "implement the 5 Phase C tools"** — it's **"audit, test, document, benchmark, and justify the 18 tools already shipped"**. This reframes the research question for the downstream roadmap.

`docs/PLUGIN-API.md` already claims 18 tools; `docs/TOOLS.md` also claims 18 (both documents mutually consistent). PLAN-v040-release.md is the drifted document.

---

## Feature Landscape

### Table Stakes (Every SOTA System Has These)

Missing any of these = product feels broken to users coming from Mem0/Zep/Letta.

| Feature | Why Expected | memory-spark Status | Complexity | Notes |
|---------|--------------|---------------------|------------|-------|
| **Semantic search over memories** | Core primitive; all 6 SOTA systems expose this | SHIPPED (`memory_search`) | — | Hybrid vector+FTS exceeds Mem0/Zep (vector-only + keyword) |
| **Explicit store** (agent-commanded) | Letta `archival_memory_insert`, Mem0 `add`, LangMem `create_manage_memory_tool` | SHIPPED (`memory_store`) | — | Matches pattern |
| **Explicit delete/forget** | Mem0 `delete`, Zep `delete_edge`, LangMem `manage_memory(action="delete")` | SHIPPED (`memory_forget`, `memory_forget_by_path`) | — | Two forget paths (query + path) is slightly richer than peers |
| **Auto-capture after agent turn** | Mem0 extraction phase, Zep episode ingest, LangMem background manager | SHIPPED (`src/auto/capture.ts`) | — | Matches pattern |
| **Auto-recall before agent turn** | Context engineering is THE 2026 trend (Zep v3 rebrand, Mem0 docs) | SHIPPED (`src/auto/recall.ts`) | — | Matches pattern |
| **Stats / health endpoint** | All systems expose some form (Mem0 `list_entities`, Zep `get_graph_info`) | SHIPPED (`memory_index_status`, `memory_gate_status`) | — | Gate status is richer than peers |
| **Content typing / categorization** | Letta memory blocks, LangMem semantic/episodic/procedural, Mem0 `categories` | SHIPPED (`contentType`: knowledge/decision/preference/mistake) | — | Domain-specific typing is a quiet advantage |
| **Per-user/agent isolation** | Mem0 `user_id`/`agent_id`, Zep `session_id`, Letta agents | SHIPPED (`security.ts` cross-agent filters) | — | Matches pattern |
| **Pipeline introspection** (simulate recall) | Emerging in 2026; Langfuse/LangSmith trace it externally | SHIPPED (`memory_inspect`, `memory_recall_debug`) | — | Two-level introspection is ahead of peers |
| **Reindex / invalidation** | Mem0 `update`, Zep graph rebuild | SHIPPED (`memory_reindex`) | — | Matches pattern |
| **Temporal filtering** (after/before) | Table stakes in 2026 per LongMemEval emphasis | SHIPPED (`memory_temporal`) | — | Matches pattern; NOT bitemporal (valid-time vs transaction-time) — that's Zep-only |

**Verdict:** memory-spark meets or exceeds table-stakes bar on every dimension. No gaps here.

---

### Differentiators (Competitive Advantage for Paper/Website)

These are where you compete. Align to Core Value: "scientifically-rigorous, measurable, 2026-current."

| Feature | Value Proposition | memory-spark Status | Complexity | Notes |
|---------|-------------------|---------------------|------------|-------|
| **Reproducible BEIR harness in-repo** | Nobody in Mem0/Zep/Letta cluster ships this. Their papers cite LoCoMo/LongMemEval but you can't reproduce without their servers. | SHIPPED (bugs present) | L to fix | **Top paper contribution.** Fix the runner bugs before v1.0. |
| **Reranker gate telemetry** (`memory_gate_status`) | First-class exposure of reranker skip decisions. No peer exposes this. | SHIPPED | — | Paper-worthy. Rename to `memory_pipeline_gates` if future gates added. |
| **13-stage recall trace** (`memory_recall_debug`) | LangSmith/Langfuse trace LLM calls; this traces *retrieval internals* (vector scores → hybrid merge → gate → rerank → MMR → token budget). | SHIPPED | — | **Biggest website demo asset.** Build an animated/interactive version for docs-site. |
| **Local-first with Spark hardware** | Nemotron embed + rerank + HyDE on owned GPU. Mem0/Zep require cloud API; Letta requires their runtime. | SHIPPED | — | Privacy story for paper. Pitch as "memory with zero data egress." |
| **HyDE query expansion** (configurable) | Most agent memory systems skip HyDE (latency cost). You ship it as opt-in. | SHIPPED | — | Be HONEST in paper: you previously concluded "NOT for us" in RESEARCH-SOTA-2026-VALIDATED.md (latency). Either ship it measured-and-defended or document the deprecation. DO NOT ship both positions. |
| **MMR diversity stage** | Most systems stop at rerank. Diversity matters for agents (no duplicate memories). | SHIPPED | — | Include in pipeline diagram. |
| **Quality-gated classification** (zero-shot + NER + heuristic) | Mem0 uses monolithic LLM extractor. You use a multi-stage classifier with gates. | SHIPPED | — | Pitch as "Mem0 extraction without the extraction-time LLM cost." |
| **content_type="tool" semantic tool retrieval** | langgraph-bigtool is Python-only and LangGraph-locked. You can be the TS/OpenClaw equivalent. | IN SCOPE (active) | M | Covered in RESEARCH-TOOLS-INJECTION-2026.md. Big paper contribution if measured. |
| **Mistake-oriented memory type** (`memory_mistakes_*`) | None of Mem0/Zep/Letta have "mistakes" as a first-class category. Coding agent native. | SHIPPED | — | Differentiator for coding-agent positioning. |
| **Rule-oriented memory type** (`memory_rules_*`) | Analogue of LangMem "procedural memory" but as a tool, not prompt-rewriting. | SHIPPED | — | Validate via test that rules actually get retrieved on relevant turns. |

**Verdict:** 10 genuine differentiators. Frame these as the paper's "contributions" section.

---

### Anti-Features (DO NOT Build — Seductive but Problematic)

Features that look like 2026 SOTA but would be wrong for memory-spark specifically.

| Anti-Feature | Why Requested | Why Problematic for memory-spark | What to Do Instead |
|--------------|---------------|----------------------------------|--------------------|
| **Full temporal knowledge graph (Graphiti-style)** | Zep's bitemporal model is the state of the art for LongMemEval temporal reasoning. Tempting to match. | H-complexity rewrite. Invalidates your LanceDB-centric architecture. Zep has a 3-person team dedicated to Graphiti; you are one person. | Ship flat temporal filter (`memory_temporal`, already done). Document "bitemporal invalidation: future work" in paper limitations. |
| **Cloud-hosted managed service** | Mem0/Zep/Supermemory all do this. Monetization temptation. | Kills your privacy/local-first differentiator. Out of v1.0 scope anyway. | Stay local-first. The differentiator is the product. |
| **Memory compression / "memvid" video encoding** | Memvid (2026) compresses memory into video frames — buzzy. | Unproven, single implementation, 60%+ performance loss per independent tests. | Stay with LanceDB + reranker. |
| **Multi-language embeddings** | Mem0 and Zep support multilingual via multilingual E5. | Nemotron is English-optimized. Switching embedder voids your BEIR numbers. | Explicitly out-of-scope in PROJECT.md; keep it that way. |
| **Fine-tuning the embedder** | Best-in-class systems (OpenAI, Anthropic memory) fine-tune. | Out-of-scope per PROJECT.md. Also, fine-tuning kills reproducibility. | Stay with off-the-shelf Nemotron. |
| **Federated memory across agents** | Hot topic in multi-agent systems papers. | memory-spark is single-agent with cross-agent READ; federation adds auth/sync complexity. | Stay with per-agent isolation + cross-agent read flag. |
| **LLM-driven UPDATE/DELETE (Mem0 signature)** on every capture | Mem0's 4-operation UPDATE phase is a differentiator FOR THEM. | Costs an LLM call per capture. You're latency-sensitive. Also: your classifier-based dedup is cheaper. | **BUT**: consider low-frequency offline consolidation (see gaps below). |
| **Real-time streaming ingestion** | Some 2026 systems (Supermemory) pitch this. | Batching is more efficient for embed queue. Your existing watcher + queue pattern is correct. | Keep file watcher + bulk ingest. Don't chase "real-time." |
| **"Active learning" from user feedback** | Listed in PROJECT.md as deferred. Several SOTA systems pitch this. | Requires feedback infrastructure that doesn't exist yet. Deferred correctly. | Stay deferred. |
| **Complete RAGAS integration in v1.0** | RAGAS is the 2026 RAG eval standard. | Per PROJECT.md key decision: stretch goal only. Golden dataset + BEIR is sufficient for v1.0. | Stay deferred. Mention in paper as future work. |

---

## Gaps memory-spark Should Consider (Features Peers Have, You Don't)

Filed honestly, with opinionated recommendations.

### GAP-1: LLM-Driven Memory Consolidation (Mem0 UPDATE/DELETE Phase)

**What Mem0 does:** After extraction, each new fact is compared against top-K similar existing memories. LLM decides: `ADD / UPDATE / DELETE / NOOP`.

**What memory-spark does:** Captures → chunks → embeds → stores. Deduplication is cosine-similarity-based (planned in Phase D.4), not LLM-arbitrated.

**Recommendation:** **Defer to v1.1.** Rationale:
- Would cost an LLM call per capture (latency tax you've rejected for HyDE).
- Offline background consolidation (run nightly, not per-capture) is a better fit. File as v1.1.
- For v1.0: document honestly in paper: "memory-spark dedups via cosine similarity threshold; LLM-arbitrated UPDATE is deferred."

**Complexity if built:** M. Requires capture queue gate + offline consolidation worker + UPDATE_edge in LanceDB.

### GAP-2: Reflections / Hierarchical Consolidation (Generative Agents pattern)

**What Stanford Generative Agents do:** When accumulated importance score exceeds threshold, trigger LLM reflection. Reflections ARE memories, stored back in same DB. Prevents unbounded growth via hierarchical abstraction.

**What LangMem does:** Background "memory manager" abstracts clusters of episodic memories into semantic summaries.

**What memory-spark does:** Nothing. Every chunk stays atomic.

**Recommendation:** **Defer to v1.1**, but add it to the paper's "future work" section with specific Stanford + LangMem citations.

**Complexity if built:** L (background worker + importance threshold + LLM call to summarize + re-embed summary as new chunk with `contentType="reflection"`).

### GAP-3: Bitemporal Modeling (Zep signature)

**What Zep does:** Every fact has `valid_time` (when fact became true in the world) and `transaction_time` (when system learned it). Invalidation of old facts without deletion.

**What memory-spark does:** Flat `indexed_at` timestamp. No separation of valid vs transaction time.

**Recommendation:** **Do NOT build.** Zep's 18-month head start on Graphiti is uncatchable for a solo developer. Instead: in paper, call this out as the principled alternative, cite Zep's 63.8% vs Mem0's 49.0% on LongMemEval, and frame memory-spark as taking a different path (local-first + scientific rigor over temporal reasoning sophistication).

### GAP-4: LangMem-Style Memory Type Taxonomy (Branding Gap)

**What LangMem does:** Semantic / Episodic / Procedural as first-class types with different retrieval behaviors.

**What memory-spark does:** Has `contentType` of knowledge/decision/preference/mistake/rule which is **functionally equivalent but better aligned to coding agents**. You just don't brand it that way.

**Recommendation:** **Document only.** Add a section to the paper mapping memory-spark content types to cognitive memory taxonomy:
- `knowledge` ≈ semantic memory
- `decision` ≈ episodic memory (specific past events)
- `preference` ≈ procedural memory (how-to/behavioral)
- `mistake` ≈ episodic memory (what went wrong)
- `rule` ≈ procedural memory (internalized policy)

Zero code change. Pure framing win for the paper.

### GAP-5: Task-ID Based Progress Tracking for Batch Operations

**What Zep does:** `thread.add_messages_batch` returns a `task_id`, agent polls `get_ingestion_status(task_id)`.

**What memory-spark does:** `memory_bulk_ingest` (per PLAN-v040) is synchronous.

**Recommendation:** **Nice-to-have for v1.0, not blocker.** If bulk ingest time exceeds 30s for typical payloads, add async path with task_id. Otherwise defer.

---

## Validation of the 5 "Phase C" Tools (from PLAN-v040-release.md)

Downstream consumer asked: confirm need / reject with 2026 evidence for each. **All 5 are already implemented** in `src/index.ts` — the question is whether they're *justified* for the paper and docs.

### 1. `memory_recall_debug` — VALIDATED ✓ (DIFFERENTIATOR)

**2026 evidence:** Langfuse and LangSmith are the dominant agent observability tools in 2026. Both trace LLM calls and tool invocations. Neither traces retrieval pipeline internals (vector scores → RRF merge → reranker gate → MMR). The Mindra observability article calls out: "Most agent failures aren't model failures — they're memory failures" — exactly this gap.

**Keep.** This is arguably memory-spark's strongest differentiator. Build an animated version for the docs-site. Make sure it surfaces: query embedding similarity to top-K, hybrid merge intermediate scores, reranker gate decision + spread, reranker deltas per chunk, MMR diversity penalty, token budget cut point.

**Complexity:** Already L-effort done. Polish for docs-site visualization.

### 2. `memory_bulk_ingest` — VALIDATED ✓ (TABLE STAKES)

**2026 evidence:** Zep explicitly recommends batch ingestion for backfills/document collections (`thread.add_messages_batch` up to 30 msgs, `graph.add_episodes_batch` up to 20 episodes). Mem0 REST API supports bulk add. This is table stakes.

**Keep.** Document the batch size limits. Consider adding optional `taskId` async mode if synchronous latency becomes a pain point (defer to v1.1 unless measured pain).

**Complexity:** Already done.

### 3. `memory_temporal` — VALIDATED ✓ (TABLE STAKES)

**2026 evidence:** Temporal reasoning is THE 2026 battleground per LongMemEval emphasis. "Who owned the budget before Q3?" is the canonical example. Zep wins here via Graphiti; Mem0g competes via graph-augmented extraction.

**Keep, but be honest about scope.** Your `memory_temporal` is flat time-window filtering, not bitemporal reasoning. In docs: "Supports `after`/`before` ISO date filters against ingestion time. Does not model fact validity intervals (bitemporal) — see Zep for that capability." Don't oversell.

**Complexity:** Already done.

### 4. `memory_related` — VALIDATED ✓ (TABLE STAKES)

**2026 evidence:** Neighborhood search ("given this memory, find similar ones") is a ubiquitous pattern across Mem0, Zep, LangMem, Cognee, AgentCore. The Mem0 vs agentmemory comparison shows memories linked with relationships (supersedes, extends, related) traversable up to N hops. Your vector-neighborhood version is the simplest instance of this.

**Keep.** Consider as v1.1 enhancement: add relationship types (not just vector similarity). For v1.0, vector neighborhood is fine and honest.

**Complexity:** Already done.

### 5. `memory_gate_status` — VALIDATED ✓ (DIFFERENTIATOR)

**2026 evidence:** Nobody else ships this. Langfuse traces the agent loop, Mindra traces tool calls, LangSmith traces LLM. None trace retrieval gate decisions as a first-class concept.

**Keep.** This is your unique signal. Make sure the tool returns: current gate mode (hard/soft/off), thresholds (gateThreshold/gateLowThreshold), skip rate over last N queries, average spread distribution histogram, vector-weight-multiplier histogram.

**Complexity:** Already done.

**Overall verdict on Phase C:** All 5 tools are justified by 2026 SOTA evidence. 3 of 5 are table stakes; 2 of 5 (`memory_recall_debug`, `memory_gate_status`) are genuine differentiators worth highlighting in paper and website.

---

## Convergent Tool Patterns Across SOTA Systems

What every major 2026 agent-memory system exposes (cross-system convergence):

| Canonical Operation | Mem0 | Zep | Letta | LangMem | memory-spark |
|---------------------|------|-----|-------|---------|--------------|
| Add / Store | `add()` | `graph.add_episode()` | `archival_memory_insert` | `create_manage_memory_tool(action=create)` | `memory_store` ✓ |
| Search (semantic) | `search()` | `graph.search()` | `archival_memory_search` | `create_search_memory_tool` | `memory_search` ✓ |
| Get by ID | `get()` | `get_edge()` | — | via store | `memory_get` (path-based) ✓ |
| Update | `update()` | implicit (graph update) | `core_memory_replace` | `manage(action=update)` | via reindex + `memory_forget_by_path` ~ |
| Delete | `delete()` | `delete_edge()` | — | `manage(action=delete)` | `memory_forget` ✓ |
| List / Enumerate | `get_all()` | `list_users()` | `get_memory()` | — | via index_status ~ |
| Stats / Health | — | `get_graph_info()` | — | — | `memory_index_status`, `memory_gate_status` ✓ |
| Bulk ingest | REST API | `add_episodes_batch` | — | — | `memory_bulk_ingest` ✓ |
| Temporal filter | via metadata | bitemporal native | — | — | `memory_temporal` ✓ |
| Related memories | via search+filter | graph traversal | — | — | `memory_related` ✓ |

**Finding:** memory-spark's 18-tool surface is **broader** than any individual competitor. Mem0 exposes 7 core operations; Zep exposes ~10; Letta exposes ~6 memory tools; LangMem exposes 2 core tools + manager primitives.

**Caveat:** broader is not automatically better. Each tool adds to the tool-injection token budget. Be ready to defend tool count in the paper (citing langgraph-bigtool's semantic tool retrieval as the answer to "when do you have too many tools?").

---

## Auto-Capture Behaviors (SOTA Comparison)

| Behavior | Mem0 | Zep | Letta | memory-spark |
|----------|------|-----|-------|--------------|
| Trigger | After conversation | After message | Agent-driven (tool call) | After agent turn (auto) |
| Extraction method | LLM extractor | Entity+relation LLM extractor | Agent decides | Classifier (zero-shot + NER + heuristic + quality gate) |
| Deduplication | LLM UPDATE decision | Entity resolution | Agent-managed | cosine threshold (planned D.4) |
| Quality gating | LLM judgment | entity confidence | LLM judgment | classifier quality score (existing) |
| Failure mode | LLM hallucinates fact | entity resolution errors | agent forgets to capture | classifier false negatives |

**memory-spark's advantage:** Quality gating via classifier is **cheaper** than LLM extraction and **more auditable**. Pitch this in the paper as "classifier-gated capture" vs "LLM-extracted capture."

**memory-spark's weakness:** No UPDATE logic. Every capture is ADD. Over time, this creates redundancy. Address via offline consolidation in v1.1 (GAP-1).

---

## Auto-Recall Behaviors (SOTA Comparison)

| Behavior | Mem0 | Zep | Letta | memory-spark |
|----------|------|-----|-------|--------------|
| Trigger | User query | User query | User query + agent decision | Before agent turn (auto) |
| Context injection | Top-K memories in prompt | Context block w/ facts+entities | Core memory + archival search | `<relevant-memories>` XML |
| Token budget | Configurable per call | Rebranded as "context engineering" | Agent loops until done | `maxInjectionTokens` (default 2000) |
| Hybrid retrieval | vector (+ graph in Mem0g) | graph + semantic + bm25 + temporal | vector | vector + FTS + HyDE + rerank + MMR |
| Latency p95 | 1.44s (claimed), 7-8s (reported by Vectorize) | ~4s (reported) | agent-dependent | <100ms target (pre-gate); gate-skip path faster |

**memory-spark's advantage:** Lowest latency target in the comparison. Your 13-stage pipeline with gate-skip path is engineered for sub-100ms recall. Make sure v1.0 benchmarks actually demonstrate this.

**memory-spark's risk:** Latency claims must be measurable. If the docs say "<100ms" and the benchmark shows 500ms, you've undone the core value. Phase A of PLAN-v040 must fix this.

---

## Temporal Handling: The Honest Picture

| System | Temporal Model | Verdict |
|--------|---------------|---------|
| Zep/Graphiti | **Bitemporal** (valid_time + transaction_time), LLM-arbitrated fact invalidation | SOTA. 15-point LongMemEval lead over Mem0. |
| Mem0 / Mem0g | Metadata timestamps + LLM UPDATE | Decent. Handles most cases. |
| Letta | Conversation search is time-ordered | Weak. Relies on agent calling `conversation_search`. |
| LangMem | Store-level timestamps only | Weak. |
| memory-spark (current) | `indexed_at` flat filter via `memory_temporal` | Weak (flat filter), BUT honest. |

**Recommendation:** Ship `memory_temporal` as a flat filter. Do NOT attempt bitemporal. In the paper, spend one paragraph explaining the tradeoff (local-first + reproducible vs bitemporal sophistication) and cite Zep as the alternative.

---

## Tool Retrieval / Semantic Tool Discovery (Emerging Pattern)

Per `docs/RESEARCH-TOOLS-INJECTION-2026.md` and my 2026 web research:

**What's proven (HIGH confidence):**
- langgraph-bigtool ships this pattern in production (March 2025+)
- Retrieval Models Aren't Tool-Savvy benchmark (arXiv 2503.01763)
- 99.6% token reduction claim from arxiv 2603.20313 (cited in your existing research doc; verify before paper)
- "Lost in the Middle" effect is documented and real

**What's speculation (LOW confidence):**
- Specific hit-rate numbers for tool retrieval generalize across models/domains
- 97.1% hit@K=3 on 121 tools is promising but single-study
- Cross-domain generalization (is this as good for coding tools as it is for ERP tools?)

**Recommendation for memory-spark v1.0:**
1. **Ship `content_type="tool"` chunks** (in scope per PROJECT.md Active). Measure on your golden dataset — don't claim the arxiv 2603 numbers until you replicate them.
2. **Consider a separate `memory_tools` tool** (future) that specifically retrieves tool docs for an agent query. langgraph-bigtool calls this `retrieve_tools`. This is the two-phase pattern.
3. **For v1.0:** boost tool chunks in recall ranking. Do NOT implement two-phase tool retrieval yet — measure first.

---

## Debugging / Observability (The memory-spark Opportunity)

**2026 landscape:**
- LangSmith / Langfuse / Arize / AgentOps dominate agent-loop tracing.
- None of them trace **retrieval pipeline internals**.
- The Mindra observability post explicitly calls this out: "Most agent failures aren't model failures — they're memory failures."

**memory-spark's unique angle:**
- `memory_recall_debug` traces the 13-stage pipeline for a specific query.
- `memory_gate_status` exposes aggregate gate decisions.
- `memory_inspect` simulates what would be recalled for a query without calling the LLM.

**Together, these three tools constitute a retrieval-layer observability suite that no peer has.** This is the strongest paper+website angle. Build a visual pipeline trace for docs-site showing exactly what happens to "what is the reranker gate threshold?" as it flows through the 13 stages.

**Recommended additions (LOW cost, HIGH differentiator value):**
1. **Per-stage latency histograms** in `memory_gate_status` → add stage names (embed, vector-search, fts-search, hybrid-merge, gate-check, rerank, mmr, budget-cut). Show p50/p95/p99 per stage over rolling window.
2. **Query traces DB** — optional debug mode that stores the last N recall traces to a side table for post-hoc analysis. This is the data backing `memory_recall_debug`.
3. **Gate skip-rate chart** in the docs-site (live updating from API or static from benchmark). "We skip the reranker 78% of the time because the vector distribution is confident."

---

## Benchmarking Datasets (Beyond BEIR)

The downstream question: what does the field actually use for *agent memory* (not vanilla RAG)?

| Dataset | What it tests | 2026 status | Recommended for memory-spark? |
|---------|---------------|-------------|-------------------------------|
| **BEIR** (SciFact, FiQA, NFCorpus, etc.) | Retrieval quality on heterogeneous corpora | Industry standard for retrieval | YES — already in scope, fix the runner |
| **LongMemEval** (arxiv 2410.10813) | 5 long-term memory abilities: info extraction, multi-session, temporal, updates, abstention | **Primary agent-memory benchmark in 2026.** 500 questions, up to 1.5M token conversations. Used by Zep (63.8%), Mem0 (49.0%), Supermemory (85.4%), OMEGA (95.4%), TiMem (76.88%), EverMemOS (83.0%). | **STRONGLY YES.** Replaces ambiguous "OCMemory" dataset in your plan. This is THE dataset your paper needs. |
| **LoCoMo** | 10 extended conversations, ~600 dialogues each, ~26k tokens | Used by Mem0. Smaller than LongMemEval, well-controlled. | MAYBE — easier entry point than LongMemEval |
| **DMR (Deep Memory Retrieval)** | MemGPT benchmark | Used by Zep (94.8%). Older, simpler. | Optional |
| **LOCOMO** | Long-term conversation memory | Cited in Mem0 paper as 26% uplift benchmark | Optional |
| **MTEB English v2** | Embedding quality | 72.31 for Nemotron-8B (your embedder) | Cite in paper, don't re-run |
| **Your golden dataset** | OpenClaw-specific QA pairs | In scope per PROJECT.md | YES — unique contribution |

**Top recommendation for the paper:** Add **LongMemEval** to the benchmark suite if possible. It is the 2026 benchmark your paper MUST cite. Running it would put you on the same scoreboard as Zep, Mem0, Supermemory, OMEGA, TiMem, EverMemOS, and that is exactly the positioning you need for scientific credibility.

**Scope warning:** LongMemEval takes significant compute (115k-1.5M token conversations x 500 questions). If your DGX Spark can handle it, this is the single highest-value benchmark addition. If not, running LoCoMo (smaller) is a reasonable second-best.

---

## Feature Dependencies

```
memory_search ──requires──> embed pipeline ──requires──> Nemotron-8B (port 18091)
                      └──requires──> LanceDB vector + FTS ──requires──> dims-lock

memory_store ──requires──> classify pipeline ──requires──> zero-shot + NER + quality gate
                      └──requires──> embed pipeline
                      └──requires──> LanceDB write path

auto-capture ──requires──> classify + memory_store + quality threshold
auto-recall  ──requires──> memory_search + token budget + injection XML

memory_recall_debug ──requires──> recall trace instrumentation ──requires──> per-stage latency tracking
memory_gate_status  ──requires──> recall trace instrumentation + rolling window stats

memory_bulk_ingest  ──requires──> embed queue + circuit breaker + (optional) task_id tracker
memory_temporal     ──requires──> indexed_at metadata + LanceDB WHERE clause
memory_related      ──requires──> memory_get (to resolve ID) + vector neighborhood search

memory_mistakes_*   ──enhances──> memory_search (via boosted ranking for coding-agent queries)
memory_rules_*      ──enhances──> memory_search (via boosted ranking for policy queries)

content_type="tool" ──enhances──> memory_search (boost tool chunks for tool-related queries)
                    ──conflicts──> none

LongMemEval benchmark ──requires──> golden dataset pipeline + eval harness
                       └──requires──> Nemotron-Super-3-122B on DGX Spark (per PROJECT.md)

Bitemporal modeling  ──conflicts──> LanceDB-centric architecture (would need rewrite)
LLM-driven UPDATE    ──conflicts──> <100ms recall budget (tax too high for per-capture)
RAGAS integration    ──conflicts──> v1.0 timeline (explicit stretch goal per PROJECT.md)
```

---

## MVP Definition for v1.0

### Launch With (v1.0) — Already Shipped, Needs Validation/Docs

All 18 tools already in `src/index.ts`. Scope for v1.0 is **validation + documentation + benchmarks**, not new features.

- [x] **18-tool plugin surface** — shipped; needs test coverage + PLUGIN-API docs sync
- [x] **Hybrid vector + FTS + RRF + gate + rerank + MMR pipeline** — shipped; needs critical bug fixes (11 items from ISSUES.md)
- [x] **Auto-capture + classify + quality gate** — shipped; needs test coverage per AUDIT-2026-04-02
- [x] **Auto-recall + token budget + XML injection** — shipped; needs benchmark validation
- [x] **content_type="tool" semantic tool retrieval** — partial; boost logic needs wiring
- [ ] **Fix BEIR runner bugs** — critical for paper credibility (P1)
- [ ] **Generate + benchmark golden dataset** — in PROJECT.md Active (P1)
- [ ] **Add LongMemEval benchmark** — NEW recommendation; positions you on SOTA scoreboard (P1)
- [ ] **Audit + rewrite PLAN-v040-release.md** — it's stale re: tool count (P2)
- [ ] **Fix TOOLS.md vs PLUGIN-API.md consistency** — both claim 18 tools; verify examples match code (P2)
- [ ] **Reranker gate telemetry dashboard** in docs-site (P2)
- [ ] **13-stage pipeline visualization** in docs-site (P1 — top differentiator)

### Add After v1.0 Validation (v1.1)

- [ ] **Offline consolidation worker** (GAP-1) — LLM-arbitrated UPDATE/DELETE once per day
- [ ] **Reflections** (GAP-2) — importance-threshold-triggered summarization
- [ ] **Per-stage latency histograms** in `memory_gate_status`
- [ ] **Query trace side-table** for post-hoc debugging
- [ ] **Async bulk_ingest with task_id** if measured latency pain
- [ ] **Relationship types in `memory_related`** (supersedes, extends, contradicts)
- [ ] **Two-phase tool retrieval** (`memory_tools` — langgraph-bigtool TS port)
- [ ] **RAGAS integration** (deferred stretch goal from v1.0)

### Future Consideration (v2+)

- [ ] **Bitemporal modeling** — only if you're convinced it's worth a rewrite (probably not)
- [ ] **Multi-language embeddings** — requires embedder swap
- [ ] **Active learning from user feedback** — requires feedback infrastructure
- [ ] **Federated cross-agent memory** — requires auth/sync layer

---

## Feature Prioritization Matrix for v1.0

| Item | User Value | Impl Cost | Priority | Rationale |
|------|------------|-----------|----------|-----------|
| Fix 11 critical bugs from ISSUES.md | HIGH | MEDIUM | **P0** | Every benchmark claim rides on these |
| Fix BEIR runner | HIGH | LOW | **P0** | Primary paper validation |
| Privacy audit + .gitignore scrub | HIGH | LOW | **P0** | Project is public; failure mode is catastrophic |
| LongMemEval benchmark integration | HIGH | MEDIUM | **P1** | Positions you on 2026 SOTA scoreboard |
| Golden dataset generation | HIGH | MEDIUM | **P1** | In PROJECT.md, unique contribution |
| 13-stage pipeline visualization | HIGH | MEDIUM | **P1** | Biggest website differentiator |
| PLAN-v040 rewrite (drift fix) | MEDIUM | LOW | **P1** | Drift hurts trust; quick win |
| TOOLS.md / PLUGIN-API.md audit | MEDIUM | LOW | **P1** | Quick win |
| Reranker gate telemetry dashboard | HIGH | MEDIUM | **P2** | Paper contribution |
| Documentation overhaul (README, ARCH, etc.) | MEDIUM | HIGH | **P1** | v1.0 gate per PROJECT.md |
| Spark v2 migration (Nemotron-Mini-4B HyDE) | MEDIUM | HIGH | **P2** | Defer if timeline slips |
| Expanded paper (golden + LongMemEval results) | HIGH | MEDIUM | **P1** | v1.0 gate |
| Offline consolidation (GAP-1) | MEDIUM | MEDIUM | **P3** | v1.1 |
| Reflections (GAP-2) | LOW | LOW | **P3** | v1.1 |
| RAGAS integration | LOW | HIGH | **P3** | v1.1+ |

---

## Competitor Feature Analysis

| Feature | Mem0 | Zep | Letta | LangMem | langgraph-bigtool | memory-spark Approach |
|---------|------|-----|-------|---------|-------------------|-----------------------|
| Storage model | Vector DB + (optional) graph | Temporal knowledge graph | Tiered (core/recall/archival) | LangGraph store | LangGraph store | **LanceDB vector + FTS hybrid** |
| Extraction | LLM extractor | LLM entity+relation | Agent-driven | Background manager | N/A | **Classifier gated (cheaper, auditable)** |
| Dedup/Update | LLM UPDATE decision | Entity resolution | Agent-managed | manage() action | N/A | **cosine threshold (v1.0) + LLM consolidation (v1.1)** |
| Temporal | Metadata timestamps | **Bitemporal (SOTA)** | Conversation search | Timestamps | N/A | Flat filter (honest) |
| Auto-capture | Yes | Yes | No (tool-driven) | Yes (background) | N/A | **Yes** |
| Auto-recall | Yes | Yes | Partial | Yes | N/A | **Yes** |
| Hybrid retrieval | Vector only (Mem0), graph (Mem0g) | graph + semantic + BM25 + temporal | Vector | Vector | Vector | **Vector + FTS + HyDE + rerank + MMR (13 stages)** |
| Local/self-hosted | OSS + cloud | CE + cloud | OSS | OSS | OSS | **OSS only, local-first** |
| Pipeline debug tool | No | No | No | No | No | **`memory_recall_debug` ✓ differentiator** |
| Gate telemetry | No | No | No | No | No | **`memory_gate_status` ✓ differentiator** |
| Benchmark in-repo | No | No | No | No | No | **BEIR harness ✓ differentiator (needs fixes)** |
| LongMemEval score | 49.0% (GPT-4o) | 63.8% (GPT-4o) | — | — | — | **TBD — running is key paper contribution** |
| p95 latency | 1.44s claimed / 7-8s reported | ~4s reported | agent-loop dependent | unknown | <100ms (tool lookup) | **<100ms target (gate-skip path)** |

**Key observation:** memory-spark is the only system in this cluster that combines: local-first + hybrid retrieval + in-repo benchmarks + retrieval-layer introspection. Each of those individually has peers, but the combination is unique. That IS the paper's positioning.

---

## Sources

### Primary (HIGH confidence — official docs / arxiv)
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory (arxiv 2501.13956)](https://arxiv.org/abs/2501.13956)
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory (arxiv 2504.19413)](https://arxiv.org/html/2504.19413v1)
- [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory (arxiv 2410.10813)](https://arxiv.org/abs/2410.10813)
- [LongMemEval GitHub (xiaowu0162)](https://github.com/xiaowu0162/LongMemEval)
- [LangMem Documentation](https://langchain-ai.github.io/langmem/)
- [LangMem SDK launch post](https://blog.langchain.com/langmem-sdk-launch/)
- [langgraph-bigtool GitHub](https://github.com/langchain-ai/langgraph-bigtool)
- [Letta / MemGPT Documentation — Research background](https://docs.letta.com/concepts/letta/)
- [Letta legacy MemGPT tools](https://docs.letta.com/guides/legacy/memgpt_agents_legacy)
- [Mem0 REST API Server docs](https://docs.mem0.ai/open-source/features/rest-api)
- [Mem0 GitHub (mem0ai/mem0)](https://github.com/mem0ai/mem0)
- [Mem0 MCP Server](https://github.com/mem0ai/mem0-mcp)
- [Zep batch add data documentation](https://help.getzep.com/adding-batch-data)
- [Zep quick start](https://help.getzep.com/quick-start-guide)

### Secondary (MEDIUM confidence — analysis posts, verified against primary)
- [5 AI Agent Memory Systems Compared 2026 (DEV)](https://dev.to/varun_pratapbhardwaj_b13/5-ai-agent-memory-systems-compared-mem0-zep-letta-supermemory-superlocalmemory-2026-benchmark-59p3)
- [Mem0 vs Letta (MemGPT) Compared 2026 (Vectorize)](https://vectorize.io/articles/mem0-vs-letta)
- [Best AI Agent Memory Frameworks 2026 (Atlan)](https://atlan.com/know/best-ai-agent-memory-frameworks-2026/)
- [AI Agent Memory Systems in 2026 (Dev Genius)](https://blog.devgenius.io/ai-agent-memory-systems-in-2026-mem0-zep-hindsight-memvid-and-everything-in-between-compared-96e35b818da8)
- [Best AI Agent Memory Systems 2026 (Vectorize)](https://vectorize.io/articles/best-ai-agent-memory-systems)
- [State of AI Agent Memory 2026 (Mem0 blog)](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- [AI Memory Research 26% Accuracy Boost (Mem0)](https://mem0.ai/research)
- [Generative Agents: Interactive Simulacra of Human Behavior (ACM)](https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763)
- [Mem0 vs Zep vs LangMem vs MemoClaw 2026 Comparison](https://dev.to/anajuliabit/mem0-vs-zep-vs-langmem-vs-memoclaw-ai-agent-memory-comparison-2026-1l1k)
- [AI Agent Observability Debugging (Mindra)](https://mindra.co/blog/ai-agent-observability-tracing-and-debugging-in-production)
- [Top 6 AI Agent Memory Frameworks (DEV)](https://dev.to/nebulagg/top-6-ai-agent-memory-frameworks-for-devs-2026-1fef)
- [Survey of AI Agent Memory Frameworks (Graphlit)](https://www.graphlit.com/blog/survey-of-ai-agent-memory-frameworks)

### Internal references
- `.planning/PROJECT.md`
- `docs/PLAN-v040-release.md` (stale re: Phase C)
- `docs/RESEARCH-TOOLS-INJECTION-2026.md`
- `docs/RESEARCH-SOTA-2026-VALIDATED.md`
- `docs/TOOLS.md`
- `docs/PLUGIN-API.md`
- `src/index.ts` (verified 18 registered `memory_*` tools)

---

## Confidence Notes

**HIGH confidence claims:**
- 2026 SOTA cluster is Mem0, Zep, Letta, LangMem, Supermemory, Cognee (convergent across ~15 sources)
- LongMemEval is the 2026 agent-memory benchmark (arxiv + HuggingFace + multiple system papers)
- Zep's bitemporal model leads on LongMemEval (arxiv 2501.13956)
- Mem0's extraction-then-update pipeline is the reference pattern for capture/consolidation (arxiv 2504.19413)
- langgraph-bigtool is the production reference for semantic tool retrieval (GitHub, PyPI, LangChain blog)
- Letta's archival/recall/core tiering is the MemGPT pattern (Letta docs)

**MEDIUM confidence claims:**
- Specific latency numbers ("Mem0 1.44s vs 17.12s", "Zep ~4s", "Mem0 7-8s") — reported by third-party reviews; vendors dispute. Safe to cite "1-2 order of magnitude improvements claimed" without specific numbers.
- "121 MCP tools / 99.6% reduction / 97.1% hit@K=3" from arxiv 2603.20313 — cited in your existing research doc; I did not re-verify the arxiv ID. Recommend verifying before paper publication.
- Specific LongMemEval scores for Supermemory (85.4%), OMEGA (95.4%), TiMem (76.88%), EverMemOS (83.0%) — web-sourced, vendor-reported; treat with skepticism until verified against official leaderboard.

**LOW confidence claims:**
- Exact tool counts for Mem0/Zep internal APIs (tool surfaces change between releases)
- "Mem0 has no consolidation tool" — may have landed post my web search results

**What might I have missed:**
- OpenAI/Anthropic memory APIs (cloud provider first-party memory) — explicitly not in your comparator cluster but relevant for "is memory a framework or a capability?" framing
- Cognee and Graphlit as graph-based alternatives — mentioned in peripheral sources but not deeply researched
- AWS Bedrock AgentCore long-term memory — mentioned once in results, not investigated
- Microsoft AutoGen memory plugins
- NVIDIA NeMo Retriever as a RAG-specific alternative (would intersect your Nemotron stack)

---

*Feature research for: LLM agent memory plugin (RAG + long-term memory)*
*Researched: 2026-04-09*
*Next consumer: roadmap + requirements definition for memory-spark v1.0 release milestone*
