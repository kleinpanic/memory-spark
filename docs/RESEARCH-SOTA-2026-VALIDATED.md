# SOTA RAG Research Report — March 2026
> Source: Subagent research (nick-sonnet). Key claims need validation against cited sources.
> Generated: 2026-03-26

## Key Takeaways for memory-spark

### 1. HyDE — NOT for us
- Extra LLM inference call blows our <100ms recall budget
- Agent memories are already "answer-shaped" — query-document gap is small
- Consider HyPE (index-time hypothetical embeddings) as alternative if needed later

### 2. Cross-Encoder Reranking — Known Failure Modes (matches our regression)
- Domain mismatch: Nemotron-Rerank-1B trained on MS MARCO/BEIR, not agent memory
- Small candidate pool: with K=5-10, reranker shuffles noise instead of adding signal
- TREC TOT 2025 (arXiv:2601.15518): reranking improved bulk recall but DESTROYED top-1 precision (RR fell from 0.2838 to 0.0601) — matches our NDCG collapse
- LanceDB blog (April 2025): FTS results saw "mixed improvements, with some models showing up to 20.65% gains and others suffering performance degradation"

### 3. RRF Failure Modes — Well Documented
- Score compression destroys quality signal from embeddings
- When one retriever returns garbage, RRF gives it equal weight
- Known to hurt when retrievers have very different quality distributions

### 4. Chunking — Largely Irrelevant for Agent Memory
- Vectara/NAACL 2025: chunking config had MORE influence than embedding model choice
- BUT: agent memories are already short distilled facts (~50-200 tokens)
- For long doc ingestion: 512 tokens with 25% overlap is the validated default
- Parent-child useful for multi-topic docs, not for memory stores

### 5. Agent Memory Architecture Landscape (2026)
| System | Pattern | Temporal | Dedup |
|--------|---------|----------|-------|
| Mem0 | Extract → Vector DB | LLM UPDATE/DELETE | LLM-based |
| Zep | Temporal Knowledge Graph | Bitemporal edges | Entity resolution |
| Letta/MemGPT | LLM-managed memory tools | Agent-driven | Agent-driven |

- Zep: 94.8% on DMR benchmark, +18.5% accuracy on LongMemEval
- Mem0: 67.13% LLM-as-Judge, 91% token reduction, 200ms p95
- Mem0's extract-then-store is most applicable to our architecture

### 6. What Actually Matters for RAG Quality (ranked by impact)
1. **Chunking strategy** — surprisingly high impact (Vectara finding)
2. **Embedding model quality** — but diminishing returns past good-enough
3. **Query cleaning / preprocessing** — garbage in = garbage out
4. **Reranking** — helps ONLY when well-calibrated for domain
5. **Hybrid search** — helps ONLY when FTS quality is controlled
6. **Metadata filtering** — cheap and effective when available

### 7. Embedding Model Comparison
- nvidia/llama-embed-nemotron-8b: 72.31 on MTEB English (v2), strong
- Voyage-3: 67.83 MTEB
- OpenAI text-embedding-3-large: 64.59 MTEB
- 4096 dims is fine — not overkill for a local model with no API cost

## Action Items Derived
1. ✅ Fix hybridMerge to preserve vector scores (already implementing)
2. ✅ Move source weighting before reranker (already implementing)  
3. ✅ Filter sessions from FTS (already implementing)
4. Consider making reranker optional / test with it OFF
5. Don't implement HyDE — latency budget doesn't allow it
6. Keep our 500-token chunks for workspace docs (close to validated 512 default)
7. Consider Mem0-style LLM extraction for auto-capture improvement
