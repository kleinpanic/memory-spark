# Tool Knowledge & Context Injection Research — March 2026
> Source: Subagent research (nick-sonnet). Key claims validated against cited papers.

## Key Findings for memory-spark

### 1. Semantic Tool Retrieval Is Proven (arxiv 2603.20313)
- 121 MCP tools indexed with dense embeddings, retrieving top-3 per query
- **99.6% token reduction**, **97.1% hit rate at K=3**, MRR=0.91
- Sub-100ms retrieval latency
- Each tool schema: 200-800 tokens. 100 tools = 20K-80K tokens overhead

### 2. LangChain langgraph-bigtool Pattern
- Two-phase approach: agent starts with `retrieve_tools` as only tool
- Calls `retrieve_tools(query)` → gets top-K tool names → those tools become available
- Supports hundreds/thousands of tools
- This is the most mature production pattern for large toolsets

### 3. OpenClaw-Specific Implications
- Our TOOLS.md approach (inject all tools) is the naive baseline — works until it doesn't
- Prompt caching mitigates cost for STABLE tool lists, but not accuracy/noise
- "Lost in the Middle" effect: 5 relevant tools near user message > 70 tools mid-context
- Switching to retrieval improves accuracy AND frees context space

### 4. learnings.md / Reflections Pattern
- Stanford "Generative Agents": composite score (recency × importance × relevance)
- Reflections generated when accumulated importance exceeds threshold
- Key: reflections ARE retrievable memories — stored back in same DB
- Prevents unbounded growth via hierarchical abstraction

## Actionable for memory-spark
1. Add `content_type="tool"` for tool documentation chunks
2. At recall time, detect tool-related queries and boost tool chunks
3. Future: expose a `memory_tools` tool that agents can call to get relevant tool docs
4. For learnings: implement periodic reflection/distillation on high-importance captures
