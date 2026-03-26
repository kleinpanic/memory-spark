# memory-spark Phase 2 Roadmap — Klein's Vision

**Date:** 2026-03-25
**Status:** PLANNING — Needs recon + feasibility assessment

---

## Klein's Requests (organized from raw input)

### 1. ✅ Execute Purge (IN PROGRESS)
- Backup → delete 26K noise chunks → optimize
- **Status:** Running now

### 2. Validation & Test Suite
- **Goal:** Ongoing behavioral tests that catch regressions over time
- Not just unit tests — validation that recall quality stays high
- Integration tests with real queries against live DB
- Possibly a CI-style "recall quality benchmark" script

### 3. Vector Index Creation (no more brute-force)
- **Current state:** 63K × 4096-dim vectors, NO index — every search is brute-force kNN
- **Need:** IVF_PQ vector index for cosine distance search
- **Also need:** Scalar indexes (Bitmap on source/agent_id, BTree on updated_at)
- LanceDB supports all of these — confirmed in Phase 0 validation
- After purge (~37K rows), create indexes on the clean data

### 4. MISTAKES.md — Agent Learning System
- **Goal:** Plugin enforces creation + ingestion of MISTAKES.md per agent workspace
- Every agent gets a `MISTAKES.md` in their workspace (like AGENTS.md, SOUL.md)
- Plugin watches and indexes it with high priority (source weight boost)
- Plugin should enforce creation on first boot if missing
- This is a CROSS-AGENT feature — not just meta, ALL agents benefit
- Mistakes should be high-value memories that persist and get recalled

### 5. LCM Update to 0.5.2
- **Current version:** Need to check
- Update `@martian-engineering/lossless-claw` to 0.5.2
- Test for breaking changes
- Separate from memory-spark work but related

### 6. Clean Embedded Data Validation
- **Goal:** Guarantee no Discord metadata, formatting noise, or raw JSON gets stored
- The `cleanChunkText()` function exists now — need to validate it works on real data
- Post-purge: sample chunks and verify they're clean
- Add a validation script: scan index, flag any remaining noise patterns

### 7. Temporal Staleness Management
- **Problem:** Old data gets recalled as if current. A fact from February gets injected alongside a fact from yesterday.
- **Need:** Beyond temporal decay (which we have), add:
  - "Last verified" timestamps
  - Ability to mark chunks as superseded/stale
  - Version-aware recall: prefer latest chunk for same topic
  - Possibly: periodic re-validation sweep that checks if indexed content still matches source files

### 8. Stale Reference Material Cleanup
- **Problem:** Early OpenClaw docs PDF embedded into memories from an ancient release
- **Need:** Identify and purge outdated reference materials
- Keep reference materials versioned — when we re-index, mark old versions as stale

### 9. Reference Library — Specialized Knowledge Sections
- **Goal:** Dedicated section of the vector DB for large reference documents
- **Use cases:**
  - **School agent:** Textbook analysis, course materials
  - **Dev agent:** Tool documentation (vLLM, NVIDIA, framework docs)
  - **Meta agent:** OpenClaw documentation, Spark docs, LanceDB docs
- **Design considerations:**
  - Separate `content_type: "reference"` vs `"knowledge"` vs `"capture"`
  - Reference materials should be version-tagged (doc version, date indexed)
  - Higher chunk sizes for reference material (800-1000 tokens vs 400 for notes)
  - Reference recall should be query-aware: only inject when the query relates to the reference domain
  - Ingest CLI: `npx tsx scripts/ingest-reference.ts --path ~/docs/book.pdf --agent school --tag "textbook:calculus"`

### 10. Meaningful Memory Scaling
- **Klein's core concern:** "Not everything needs to be a memory"
- Memory should IMPROVE agents over time, not just grow the DB
- Quality > quantity — the purge proves this
- Capture pipeline improvements (Phase 4) already help: heuristic fallback, assistant capture
- Next step: periodic memory consolidation — merge similar/overlapping chunks into distilled summaries

---

## Priority Ordering (my recommendation)

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| 1 | ✅ Purge execution | Done | Immediate — 41% noise removed |
| 2 | Vector + scalar indexes | 1hr | Massive — search speed + accuracy |
| 3 | Stale reference cleanup | 30min | Medium — remove ancient docs |
| 4 | Reference library system | 2hr | High — enables textbook/doc ingestion |
| 5 | MISTAKES.md enforcement | 1hr | High — agent learning over time |
| 6 | Validation suite | 2hr | High — prevents regressions |
| 7 | Temporal staleness | 2hr | Medium — prevents stale injection |
| 8 | LCM 0.5.2 update | 30min | Medium — separate concern |
| 9 | Memory consolidation | 4hr | Future — depends on LLM latency |

---

## Recon Needed

- [ ] Check current LCM version
- [ ] Identify the stale OpenClaw docs PDF in the index
- [ ] Check what reference materials currently exist in the index
- [ ] Research LanceDB schema evolution for `content_type` field
- [ ] Check if MISTAKES.md exists in any agent workspace currently
- [ ] Verify the purge results are clean
