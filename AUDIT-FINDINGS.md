# memory-spark Comprehensive Audit — Klein's Questions

## 1. Current Tools (7 exposed)

| Tool | What it does | Status |
|------|-------------|--------|
| `memory_search` | Vector + FTS search | ✅ Working |
| `memory_get` | Read file by path + line range | ✅ Working |
| `memory_store` | Manually store a fact/preference | ✅ Working |
| `memory_forget` | Remove memories matching query | ✅ Working |
| `memory_reference_search` | Search reference docs (tagged) | ✅ Working |
| `memory_index_status` | Health dashboard + stats | ✅ Enhanced (cache, probes) |
| `memory_forget_by_path` | Remove all chunks for a file | ✅ Working |

### Tools that SHOULD exist but DON'T:
- `memory_inspect` — show what auto-recall would inject for a given query (debug/visibility)
- `memory_reindex` — force re-index a specific path or all paths
- `memory_sync` — sync reference docs (OpenClaw docs, git-latest) — THIS WAS sync-rag.ts

## 2. What's Configurable vs Hardcoded

### ✅ Configurable (in plugin config):
- autoRecall: enabled, agents, ignoreAgents, maxResults, minScore, queryMessageCount, maxInjectionTokens
- autoCapture: enabled, agents, ignoreAgents, minConfidence, categories, minMessageLength, useClassifier
- embed: spark/openai/gemini provider settings
- rerank: enabled, provider settings
- reference: enabled, paths, chunkSize, tags
- watch: enabled, paths
- ingest: chunkSize, chunkOverlap, sessionIndexing

### ❌ HARDCODED (should be configurable):
- Source weights: capture=1.5x, sessions=0.5x
- Path weights: MEMORY.md=1.4x, MISTAKES=1.6x, TOOLS.md=1.3x, AGENTS.md=1.2x, SOUL.md=1.2x, USER.md=1.3x, archive=0.4x, learnings=0.1x
- Temporal decay formula: `0.8 + 0.2 * exp(-0.03 * ageDays)` (floor=0.8)
- Dedup threshold: 0.92
- MMR lambda: 0.7
- LCM overlap threshold: 40%
- Max captures per turn: 3
- Embed cache size: 256, TTL: 30min

## 3. MISTAKES.md Handling — INCOMPLETE

### What works:
- ✅ `enforceMistakesFiles()` creates MISTAKES.md + mistakes/ dir in every workspace
- ✅ Indexer picks up MISTAKES.md and mistakes/*.md files
- ✅ Source weighting gives 1.6x boost during recall

### What DOESN'T work:
- ❌ No "always inject" for mistakes — they only show up IF the query is relevant
- ❌ No session-start forced injection of recent mistakes
- ❌ Agent has to be "smart enough" to get a query that matches mistake content
- ❌ No mistake aggregation (if 5 agents have different mistakes, no cross-agent view)

## 4. Auto-injection vs Agentic Tool Use

### How it currently works:
1. **Auto-recall** (before_prompt_build): ALWAYS runs. Embeds recent messages, searches LanceDB, injects top results as `<relevant-memories>` XML. Agent sees them but doesn't choose them.
2. **Tool-based** (memory_search etc.): Agent must DECIDE to call the tool. Relies on the agent being smart enough to know when to search.

### Problems with this:
- Auto-injection is the primary mechanism. Tool use is secondary.
- If auto-injection quality is bad, agents get bad context whether they want it or not.
- No way for agents to opt out of bad recalls mid-session.
- Tool descriptions could be clearer about when to use them.

## 5. LCM Overlap / Conflict

### Current coordination:
- ✅ Session JSONL indexing is OFF by default (`sessionIndexing: false`) — LCM owns sessions
- ✅ LCM recency suppression: if a recalled chunk has >40% token overlap with recent messages, it's filtered out
- ✅ Quality gate filters LCM summary blocks from capture
- ✅ Query cleaning strips LCM summaries before embedding

### What's MISSING:
- ❌ No awareness of what LCM already has in context — memory-spark might inject something LCM already provided
- ❌ No token budget coordination — both LCM and memory-spark compete for context window
- ❌ No dedup between LCM expanded content and memory-spark recalled content
- ❌ If LCM compacts a conversation, memory-spark might recall the pre-compaction version

## 6. sync-rag.ts — NEEDS RESTORATION

Deleted in lint cleanup. This was the doc sync functionality that:
- Copies OpenClaw installed docs to knowledge-base
- Syncs git-latest docs
- Cleans up old versions (keeps newest 2)
- Creates VERSIONS.md index

This should be restored as a script AND as a tool (`memory_sync`).

## 7. Benchmark / Validation Gaps

### What we CAN test:
- ✅ Vector search quality (quick-eval-v2: 90%)
- ✅ Hybrid search quality (80%)
- ✅ Unit tests (141/141)
- ✅ Garbage capture detection

### What we CAN'T test yet:
- ❌ Full injection visibility (what does the agent actually see?)
- ❌ A/B: agent performance WITH vs WITHOUT memory-spark
- ❌ MISTAKES.md retrieval rate (do agents actually get mistakes when relevant?)
- ❌ Cross-agent recall quality (does meta see school's memories when needed?)
- ❌ Production latency under load
- ❌ Token budget impact (how much context does injection consume?)
