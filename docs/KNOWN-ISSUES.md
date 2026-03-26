# Known Issues & Workarounds

## Active Issues

### 1. LanceDB FTS + WHERE Clause Arrow Panic
**Status:** Workaround applied (v0.2.1)
**Severity:** High (was causing complete recall failure)
**Root cause:** LanceDB native binary panics in `arrow-array cast.rs:758` when combining FTS search with SQL `.where()` clauses. The `ExecNode(Take)` operation fails on the Arrow cast.
**Workaround:** FTS queries now skip `.where()` entirely. We overfetch 3x and post-filter in JavaScript. Vector search `.where()` is NOT affected and still uses SQL.
**Impact:** FTS queries are slightly slower (fetch 3x, filter in JS) but functionally correct.
**Upstream:** This appears to be a LanceDB issue with tantivy FTS + Arrow integration.

### 2. Embedding Queue Concurrency with LanceDB
**Status:** Fixed (v0.2.1)
**Root cause:** Running FTS and vector search concurrently via `Promise.all` allowed the FTS Arrow panic to corrupt the shared LanceDB native connection, causing vector search to also fail.
**Fix:** Search is now sequential (vector first, then FTS). This adds ~50ms latency but ensures vector results are always safe.

### 3. Production Index Has Garbage Data
**Status:** Pending reindex
**Impact:** The production LanceDB index (22K+ chunks) still contains garbage from before the quality gate improvements. ~3000 session dumps and ~200 zh-CN docs need purging.
**Fix:** Run a full reindex after deploying the new code. The quality gate now blocks these automatically.

### 4. MISTAKES.md Not Force-Injected
**Status:** Partially addressed (dynamic injection added)
**Detail:** Mistakes get a dedicated vector search with lower threshold (0.7x normal), but they still need to be semantically relevant to the query. A truly irrelevant mistake won't be injected even with the boost.
**Future:** Consider a "pinned chunks" feature that always injects the top N mistakes regardless of query relevance.

## Resolved Issues

### LanceDB Schema Mismatch on Append (v0.2.0)
Fixed by creating a seed record with ALL fields on table creation, including new columns (content_type, quality_score, token_count, parent_heading). Arrow nullability is strict — all fields must be present from table creation.

### Auto-Capture Storing Garbage (v0.2.0)
Fixed with 30+ garbage detection patterns, quality gate integration, and LCM summary block filtering. 91% of captures were noise (42/46 audited).

### Temporal Decay Not Working (v0.2.0)
Fixed by using file `mtime` for `updated_at` instead of indexing time. Every reindex was resetting all timestamps to NOW, making decay useless.
