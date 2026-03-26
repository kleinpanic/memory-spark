# Changelog

All notable changes to memory-spark are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- `mistakes/` directory structure enforcement alongside `MISTAKES.md` index
- Workspace scanner discovers and indexes `mistakes/` subdirectory
- File watcher monitors `mistakes/` for live changes
- ESLint, Prettier, and knip tooling
- Test fixtures: `tests/fixtures/test-config.ts`, sample docs, ground truth
- `docs/TECHNICAL-REPORT.md` — academic-style architecture documentation
- `memory_reference_search` tool for filtered documentation queries
- `memory_index_status` tool for aggregate chunk statistics
- `memory_forget_by_path` tool for targeted path deletion
- IVF_PQ vector indexing (64 sub-vectors for 4096-dim embeddings)
- Full-text search (FTS) indexing in LanceDB
- Contextual retrieval: source/path/heading prepended before embedding
- Schema evolution: `content_type`, `quality_score`, `token_count`, `parent_heading`
- Non-linear temporal decay (`0.8 + 0.2 * exp(-0.03 * ageDays)`, floor 0.8)
- RRF (Reciprocal Rank Fusion) for hybrid vector + FTS results
- MMR (Maximal Marginal Relevance) diversity filtering
- `pending-embed.jsonl` queue for retry during Spark downtime
- Practical evaluation suite (16 scenarios, 94% pass rate)
- A/B evaluation suite (12 scenarios, 100% lift over baseline)

### Changed
- `MISTAKES.md` template now creates lean index + `mistakes/` directory (was flat file)
- Test files moved from root to `tests/` directory
- Hardcoded IPs replaced with localhost/TEST-NET-1 in source and tests
- Brute-force kNN replaced by indexed ANN search
- Search minimum score lowered from 0.65 to 0.2 (configurable)

### Removed
- `OVERHAUL-PLAN.md`, `EXECUTION-PLAN.md`, `PHASE2-ROADMAP.md`, `RAG-AUDIT.md`
- `ISSUES.txt`, `CLAUDE.md`
- Root-level `benchmark.ts` (duplicate of `scripts/benchmark.ts`)
- Stale result JSON files (`ab-eval-results.json`, `benchmark-results.json`)

## [0.1.0] — 2026-03-22

### Added
- Initial release: LanceDB-backed vector memory for OpenClaw agents
- Spark-hosted embedding (nvidia/llama-embed-nemotron-8b)
- Spark-hosted reranking (nvidia/llama-nemotron-rerank-1b-v2)
- Chokidar-based file watching for workspace directories
- Auto-recall and auto-capture with per-agent filtering
- Prompt injection detection and sanitization
- Quality scoring for ingest filtering
- `memory_search`, `memory_get`, `memory_store`, `memory_forget` tools
