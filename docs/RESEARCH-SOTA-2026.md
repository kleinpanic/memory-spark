# Advanced RAG SOTA Research Notes (2026-03-26)

## Sources
1. Microsoft Learn: "Build Advanced RAG Systems" (2026-01-30)
2. PremAI: "Building Production RAG" (2026-03 guide)
3. BEIR 2.0 Leaderboard (2025)
4. arxiv: "Comprehensive Survey of RAG Architectures" (2506.00054)
5. arxiv: "Engineering the RAG Stack" (2601.05264)
6. RAGFlow: "Rise and Evolution of RAG in 2024"

## Key SOTA Techniques We're Missing

### 1. HyDE (Hypothetical Document Embeddings)
Have LLM generate a hypothetical answer, embed THAT, search with it.
Bridges query-document semantic gap. Especially effective for short queries.
**Impact:** +5-15% recall on ambiguous queries.

### 2. Query Rewriting / Expansion
Rewrite user queries before retrieval: expand acronyms, add synonyms, step-back prompting.
**We have:** Nothing. Raw agent prompts go straight to embedding.
**Impact:** +3-8% NDCG on production queries.

### 3. Parent-Child (Hierarchical) Chunking
Store small child chunks (200 tokens) + large parent chunks (2000 tokens).
Retrieve by child similarity, return parent for context.
**We have:** Flat 500-char chunks with no hierarchy.
**Impact:** Best of both worlds — precision retrieval + contextual generation.

### 4. Proposition Chunking
Break content into atomic factual statements. Each chunk = one fact.
Highest retrieval precision for factoid queries.
**We have:** Line-based chunking from markdown files.
**Impact:** +20-30% precision on factoid queries (but expensive to generate).

### 5. Chunk Overlap
10-20% overlap between adjacent chunks prevents information loss at boundaries.
**We have:** Zero overlap.
**Recent finding:** 2026 study found overlap provides NO benefit with SPLADE retrieval.
                     Only matters for dense-only retrieval on long-context queries.
**Impact:** Minimal for our hybrid setup. Test before implementing.

### 6. Reranking Pool Size
SOTA reranks top-100 candidates, not top-10.
Cross-encoders (MiniLM) or late-interaction (ColBERT) for scoring.
**We have:** Rerank top-10 only.
**Impact:** +3-5% NDCG from broader candidate pool.

### 7. Embedding Model Choice
SOTA: Voyage-3-large (1024 dims, 32K context, $0.06/M tokens)
**We have:** Nemotron-Embed-8B (4096 dims, self-hosted)
**Tradeoff:** Our model is free but 4x the dimensions = 4x storage.
             Voyage leads MTEB by 9.74% over OpenAI.
**Impact:** Consider dimension reduction or PQ compression.

### 8. RAGAS Evaluation Framework
Standard RAG eval with 4 core metrics:
- Context Precision: Are retrieved chunks relevant?
- Context Recall: Were all relevant chunks retrieved?
- Faithfulness: Is the answer grounded in context?
- Answer Relevance: Does the answer address the query?
Uses LLM-as-judge for automated scoring.
**We have:** Mock IR metrics only. No LLM-as-judge.
**Impact:** Critical for credible evaluation.

### 9. ARES Framework
Like RAGAS but uses:
- Synthetic data generation for training
- Trained judge models (not just prompting)
- Confidence intervals on metrics
- PPI (Prediction-Powered Inference) for statistical rigor
**Impact:** More rigorous than RAGAS, suitable for academic claims.

### 10. Golden Dataset
50-100 representative query-answer pairs with:
- Approved answers
- Source document links
- Multiple phrasings per query
- Topic and query type metadata
**We have:** ground-truth.json with 60 queries but no gold answers.
**Impact:** Foundation for all evaluation.

## Chunking Decision Framework (from PremAI)
| Document Type | Query Pattern | Strategy |
|---|---|---|
| Short FAQ/tickets | Factoid | No chunking (embed whole) |
| News/blogs | General | Fixed-size 512 |
| Technical docs/wikis | Mixed | Recursive or hierarchical |
| Legal/compliance | Clause-specific | Semantic or proposition |
| Research papers | Specific facts | Proposition chunking |
| Multi-topic long docs | Mixed | Hierarchical parent-child |

## Storage Efficiency
- Voyage: 1024 dims → 4 KB per vector
- Us: 4096 dims → 16 KB per vector (4x overhead)
- PQ compression: 16x reduction typical
- FP16 vectors: 2x reduction, negligible quality loss

## Production RAG Checklist (from research)
- [x] Hybrid search (dense + sparse)
- [x] Reranking
- [x] Quality filtering at ingest
- [x] Metadata per chunk (source, type, timestamp)
- [ ] Query rewriting / HyDE
- [ ] Parent-child hierarchical chunks
- [ ] Chunk overlap (test first)
- [ ] Rerank pool expansion (10→50+)
- [ ] RAGAS/ARES evaluation
- [ ] Golden dataset with gold answers
- [ ] Adversarial query set
- [ ] Embedding cache
- [ ] Change detection (hash-based)
- [ ] Incremental re-indexing
- [ ] Observability / monitoring dashboard
