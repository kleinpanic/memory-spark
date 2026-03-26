# Feature Gap Analysis — Promised vs Delivered

*Last updated: 2026-03-26 (v0.2.1)*

## Delivered ✅ (24 features)

| Feature | Source | Status |
|---------|--------|--------|
| Hybrid search (vector + FTS) | PLAN.md | ✅ Working |
| Cross-encoder reranking | PLAN.md | ✅ Nemotron-Rerank-1B |
| Temporal decay | PLAN.md | ✅ 0.8 floor, exp decay |
| Source weighting | PLAN.md | ✅ Fully configurable |
| MISTAKES.md boost | PLAN.md | ✅ 1.6x (configurable) |
| Dynamic mistakes injection | Session fix | ✅ Separate filtered search |
| Auto-recall (before_prompt_build) | PLAN.md | ✅ 13-stage pipeline |
| Auto-capture (agent_end) | PLAN.md | ✅ Quality gated |
| Garbage capture defense | Session fix | ✅ 30+ patterns |
| LCM dedup | Session fix | ✅ Summary extraction |
| Prompt injection detection | PLAN.md | ✅ Security filter |
| Token budget enforcement | PLAN.md | ✅ Default 2000 tokens |
| Embedding cache | PLAN.md 2.3 | ✅ LRU, 256 entries |
| EmbedQueue | PLAN-V2 | ✅ Retry, backoff, health |
| 9 plugin tools | PLAN.md Phase 4 | ✅ All working |
| FTS Arrow bug fix | Session fix | ✅ Post-filter workaround |
| mtime preservation | Session fix | ✅ Temporal decay works |
| 144 unit tests | PLAN-V3 | ✅ All passing |
| E2E benchmark | PLAN-V3 4b | ✅ 7/7 dev, 6/7 prod |
| Documentation | Session work | ✅ 6 docs + README |
| Configurable weights | Session fix | ✅ Sources, paths, patterns |
| Provider attribution | Session fix | ✅ memory-spark: prefix |
| Bootstrap file weighting | Session fix | ✅ TOOLS/USER/AGENTS/SOUL |
| Query cleaning | Session fix | ✅ Strips noise before embed |

## Partially Delivered ⚠️ (4 features)

| Feature | Source | Status | What's Missing |
|---------|--------|--------|----------------|
| Reference library | PLAN.md Phase B | ⚠️ | Infrastructure exists, no paths configured |
| NER | PLAN.md | ⚠️ | Used in capture, not in recall/search enrichment |
| OCR (GLM/EasyOCR) | PLAN.md | ⚠️ | Parsers exist, not tested in E2E |
| sync-rag | Original feature | ⚠️ | Restored from git, not wired as tool/cron |

## Not Delivered ❌ (17 features)

### High Priority (should do next)

| Feature | Source | Impact | Effort |
|---------|--------|--------|--------|
| content_type="tool" for TOOLS.md | Research doc | High — agents can't search tool docs | Low |
| memory_health dedicated tool | PLAN.md 4.2 | Medium — merged into index_status | Low |
| Golden dataset with gold answers | PLAN-V3 4a | High — can't measure real quality | Medium |
| A/B testing (with/without) | PLAN-V3 4c | High — don't know if memory helps | Medium |
| Reference library activation | PLAN.md Phase B | High — OpenClaw docs not searchable | Low |

### Medium Priority (quality improvements)

| Feature | Source | Impact | Effort |
|---------|--------|--------|--------|
| Query rewriting | PLAN.md 2.1 | Medium — improves recall for vague queries | Medium |
| Parent-child chunks | PLAN.md 3.1 | Medium — better context for matches | High |
| Contextual prefix | PLAN.md 3.3 | Medium — preserves doc structure | Medium |
| Adversarial queries | PLAN.md 6.4 | Medium — security hardening | Medium |
| Bootstrap bloat reduction | PLAN-V2 Phase 3 | Medium — saves context tokens | Medium |

### Low Priority (optimization/speculative)

| Feature | Source | Impact | Effort |
|---------|--------|--------|--------|
| HyDE | PLAN.md 2.2 | Low — research says skip | High |
| memory_watch (hot-add dirs) | PLAN.md 4.1 | Low — watcher handles this | Medium |
| memory_link (relationship graph) | PLAN.md 4.3 | Low — speculative value | High |
| Context-aware tool suggestions | PLAN.md 4.4 | Low — agents have TOOLS.md | High |
| RAGAS integration | PLAN.md 6.2 | Low — custom eval works | Medium |
| Dimension reduction | PLAN.md 7.1 | Low — storage not a bottleneck | Medium |
| FP16 vectors | PLAN.md 7.3 | Low — 4096d storage is fine | Low |
| LanceDB compaction | PLAN.md 7.2 | Low — not hitting limits | Low |
| Periodic reflection/distillation | Research | Medium — prevents unbounded growth | High |

## SOTA Comparison (from RAG-AUDIT.md)

| SOTA Feature | Our Status |
|-------------|------------|
| Dense embeddings (#1 MMTEB model) | ✅ llama-embed-nemotron-8b |
| Cross-encoder reranking | ✅ Nemotron-Rerank-1B |
| Hybrid search (dense + sparse) | ✅ Vector + FTS |
| Contextual retrieval (Anthropic) | ⚠️ Partial (source weighting, not chunk-level context) |
| HyDE / query rewriting | ❌ Not implemented |
| Parent-child retrieval | ❌ Flat chunks only |
| Semantic tool retrieval | ❌ No tool indexing |
| Multi-hop / chain-of-retrieval | ❌ Single-hop only |
| Reflections / learnings | ❌ No periodic distillation |
| Self-hosted (no cloud) | ✅ Fully local on DGX Spark |
