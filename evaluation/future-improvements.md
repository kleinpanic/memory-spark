# Future Improvements — memory-spark Retrieval Pipeline

**Created:** 2026-03-30  
**Source:** Phase 4 cross-reference research against NVIDIA RAG Blueprint and modern papers  

---

## 1. Query Decomposition

**What:** Break complex multi-aspect queries into sub-queries, run retrieval for each, merge results via RRF.

**Why:** A query like "What monitoring does Klein have set up, and what's the infrastructure it runs on?" actually asks two distinct things. A single embedding averages both intents, retrieving chunks that partially match both but fully match neither. Decomposing into `["What monitoring is configured?", "What infrastructure does Klein use?"]` retrieves precise chunks for each.

**How:**
- Use the HyDE LLM endpoint (already available) to decompose queries into 2-4 sub-queries
- Run `recall()` for each sub-query in parallel
- Merge results via RRF across all sub-query result sets
- Deduplicate before reranking

**Complexity:** Medium. The LLM call adds ~1-2s latency. Can be gated behind a complexity heuristic (only decompose queries > 10 words or containing conjunctions).

**References:**
- NVIDIA RAG Blueprint includes optional query decomposition
- "Query2doc" (Wang et al., 2023): https://arxiv.org/abs/2303.07678
- LlamaIndex SubQuestionQueryEngine: https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/

---

## 2. Query Rewriting for Multi-Turn Conversations

**What:** Rewrite follow-up queries to be standalone before embedding. "What about the other one?" → "What is the configuration for the second Spark node?"

**Why:** Agents operate in conversational contexts. When a query references prior conversation ("that thing", "the other one", "same as before"), the raw query embedding lands nowhere useful in vector space. Rewriting it into a standalone question dramatically improves retrieval.

**How:**
- Feed the last 2-3 conversation turns + current query to the HyDE LLM
- System prompt: "Rewrite this follow-up question as a standalone question that contains all necessary context."
- Use the rewritten query for embedding, keep original for display
- Gate behind a detection heuristic: only rewrite if query contains pronouns, demonstratives, or is < 5 words

**Complexity:** Low-medium. Similar infrastructure to HyDE (same LLM endpoint). The hard part is getting the conversation context into the rewriter without bloating the call.

**References:**
- NVIDIA RAG Blueprint: query rewriting module
- "Rewrite-Retrieve-Read" (Ma et al., 2023): https://arxiv.org/abs/2305.14283

---

## 3. Self-Reflection / Answer Validation

**What:** After retrieval, have an LLM verify that the retrieved chunks actually answer or inform the query before injecting them into the context window.

**Why:** Embedding similarity is imperfect. High cosine similarity doesn't guarantee relevance — a chunk about "agent configuration" might match "how do I configure agents?" but contain only boilerplate setup instructions, not the specific config Klein is asking about. A quick LLM pass can catch these false positives.

**How:**
- After reranking, pass each top-k chunk + query to a cheap LLM (Gemini Flash or Nemotron)
- Prompt: "Does this passage directly answer or inform the question? Reply YES or NO."
- Drop chunks that get NO. If all chunks are dropped, fall back to the original set (avoid returning nothing).
- Budget: ~100 tokens per validation call × k chunks = ~500-1000 tokens total

**Complexity:** Medium. Adds latency (parallel LLM calls help). Cost is low with a cheap model. Risk of over-filtering if the LLM is too strict.

**When to implement:** After the core pipeline is optimized (Phases 1-4). This is a precision polish step, not a fundamental architecture fix.

**References:**
- NVIDIA RAG Blueprint: self-reflection module
- "Self-RAG" (Asai et al., 2023): https://arxiv.org/abs/2310.11511
- Anthropic RAG cookbook: context relevance filtering

---

## 4. GPU-Accelerated ANN Search (cuVS)

**What:** Replace CPU-based IVF_PQ vector search with GPU-accelerated approximate nearest neighbor search using NVIDIA cuVS.

**Why:** LanceDB currently runs CPU-based IVF_PQ indexing. On Spark DGX with available GPU resources, cuVS can provide 10-100x speedup for vector search, especially as the corpus grows beyond 100k chunks.

**How:**
- Option A: Use Milvus as the vector backend (native cuVS support)
- Option B: Use FAISS with GPU support (drop-in replacement for the ANN step)
- Option C: Wait for LanceDB GPU support (on their roadmap)
- All options require significant backend refactoring

**Complexity:** High. Requires either swapping the vector DB backend or adding a parallel GPU-accelerated index alongside LanceDB.

**When to implement:** When corpus size makes CPU search a bottleneck (>500k chunks) or when search latency becomes the pipeline bottleneck. Currently not the limiting factor — reranker latency (41s) is 100x worse than search latency.

**References:**
- NVIDIA cuVS: https://github.com/rapidsai/cuvs
- NVIDIA RAG Blueprint uses Milvus with cuVS
- LanceDB GPU roadmap: https://github.com/lancedb/lancedb/issues

---

## 5. Dynamic MMR Lambda (Deferred from Phase 4)

**What:** Adjust MMR lambda per-query based on query characteristics rather than using a fixed value.

**Why:** A specific factual query ("what port does the gateway run on?") benefits from near-zero diversity penalty (lambda → 1.0). A broad exploratory query ("tell me about the agent system") benefits from more diversity (lambda → 0.8). A fixed lambda is a compromise.

**Proposed heuristic:**
- Baseline: config value (0.9)
- Specific queries (short, contains identifiers/paths/numbers): +0.05 (max 0.95)
- Broad queries (long, interrogative patterns, abstract terms): -0.05 (min 0.85)
- Effective range: 0.85 – 0.95

**Why deferred:** No production RAG systems implement dynamic lambda. The research subagent found zero examples in NVIDIA, LangChain, LlamaIndex, or academic papers. The heuristics are plausible but unvalidated. Need empirical evidence first:
1. Run BEIR ablations at lambda = 0.85, 0.9, 0.95, 1.0
2. If optimal lambda varies significantly by query type within a dataset → implement dynamic
3. If a single fixed value works well across all datasets → skip dynamic, keep it simple

**Complexity:** Low to implement, but validation is the bottleneck.

**Prerequisite:** Complete Phase 4A-4E first (fix the fundamental MMR bugs). Dynamic lambda on a broken MMR is meaningless.
