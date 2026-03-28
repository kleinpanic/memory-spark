# GAPS.md — memory-spark Pre-Integration Gap Analysis

**Written:** 2026-03-27 (post overnight benchmarks, full codebase audit)  
**Context:** Codebase is feature-complete for Phase 0/1. All prior benchmarks ran on the HOST
against a no-pool-column index, not inside Docker. Docker is the target environment for all
future testing before production integration.

---

## 🚨 BLOCKER: All Benchmarks Ran on Host, Not Docker

Every result in `evaluation/results/` was produced by running `scripts/run-overnight.sh` on the
host against `test-data/lancedb` (created 2026-03-26 14:40, **before** the pool column commit at
23:06). The Docker results directory (`~/codeWS/Docker/openclaw-plugin-test/results/`) is
**completely empty**.

This means:
- **No valid Docker benchmark exists yet** — the isolated test environment has never run end-to-end
- Tier 1 numbers are from a degraded index (pool filtering silently returns empty → every recall
  query returns 0 results → metrics reflect random-ish vector similarity, not the real pipeline)
- Tier 2 passed 13/14 but the 1 failure ("Short meaningful → has results") is the pool bug
  surfacing: `WireGuard IP?` returns 0 because all pool-filtered recall is silently empty
- Tier 3 hard-crashed: `LanceError(Schema): Schema error: No field named pool`

**The one clean result** is BEIR SciFact (evaluation/beir-datasets/scifact-index, created
2026-03-27 02:46, **after** pool commit) — that index has the pool column and those numbers
(vector-only NDCG@10 0.768) are valid.

---

## GAP 1 — Docker Workspace Discovery Broken (BLOCKER)

**File:** `tools/indexer.ts`, `scripts/run-benchmark.sh`, `docker-compose.yml`

**Problem:**  
`discoverAllAgents()` in `src/ingest/workspace.ts` scans `~/.openclaw/` for directories
named `workspace-<agentId>`. In Docker, the test workspace is mounted as:

```yaml
- ./workspaces/test-agent:/home/node/.openclaw/workspace:ro
```

There is no `workspace-test` or `workspace-bench` directory — just `workspace`. So
`discoverAllAgents()` finds **0 agents** and indexes **0 files**. The Docker LanceDB volume
has only a `dims-lock.json` and an empty lancedb directory.

`run-benchmark.sh` passes `--workspace /home/node/.openclaw/workspace` to the indexer, but
`tools/indexer.ts` has **no `--workspace` CLI argument** — the flag is silently ignored.

**Fix options (pick one):**
1. Update `docker-compose.yml`: rename mount to `workspace-test` so discovery finds it
2. Add `--workspace <path> --agent-id <id>` CLI args to `tools/indexer.ts`
3. Add `MEMORY_SPARK_WORKSPACE_OVERRIDE` env var to `resolveConfig()` for Docker-specific paths

**Recommended:** Option 1 — rename the mount. It's one line and matches the expected convention:
```yaml
# Change:
- ./workspaces/test-agent:/home/node/.openclaw/workspace:ro
# To:
- ./workspaces/test-agent:/home/node/.openclaw/workspace-test:ro
```
Then update `configs/test-openclaw.json` agent workspace to `/home/node/.openclaw/workspace-test`.

---

## GAP 2 — No Pool Column in Custom Corpus / test-data Index (BLOCKER)

**File:** `test-data/lancedb/`, production `~/.openclaw/data/memory-spark/lancedb/`

**Problem:**  
The `test-data/` index was last written 2026-03-26 14:40, before the pool column was added
(commit `85e173a`, 23:06). The schema has: `id, path, source, agent_id, ..., parent_heading`
— no `pool` field.

`ensureSchema()` in `lancedb.ts` detects the missing column and sets `schemaHasNewColumns = false`.
New upserts silently skip pool assignment. Reads that filter by `pool` via WHERE clauses throw
a hard LanceDB `Schema error: No field named pool`. The production recall pipeline wraps all
pool-filtered searches in `.catch(() => [])`, so errors are silently swallowed and every recall
returns 0 memories.

**Impact:**
- Tier 3 benchmark crashes completely
- All Tier 2 tests that depend on recalled content receive empty context
- The "Short meaningful → has results" edge case fails
- Real-world recall in the Docker test agent would return nothing

**Fix:** Full wipe + reindex. `ensureSchema()` deliberately does NOT auto-migrate (see the
comment in lancedb.ts — Arrow nullability mismatch causes mergeInsert to fail). The clean
path is `tools/reindex.ts` or `tools/purge.ts` followed by a fresh index pass.

```bash
# After fixing GAP 1 (workspace discovery):
docker exec oc-plugin-test bash -c "
  cd /home/node/.openclaw/extensions/memory-spark
  MEMORY_SPARK_DATA_DIR=/home/node/.openclaw/data/memory-spark \
  npx tsx tools/purge.ts --confirm
  MEMORY_SPARK_DATA_DIR=/home/node/.openclaw/data/memory-spark \
  SPARK_HOST=10.99.1.1 npx tsx tools/indexer.ts
"
```

---

## GAP 3 — run-benchmark.sh Doesn't Use Docker for Benchmarks

**File:** `scripts/run-benchmark.sh`

**Problem:**  
`run-benchmark.sh` runs `docker exec` for indexing, but the actual benchmark is also run
inside the container — however it only copies the golden dataset in, not the BEIR dataset.
The BEIR SciFact data lives at `evaluation/beir-datasets/` on the host and is never copied
in. So `beir-benchmark.ts` would fail inside Docker with a missing dataset path.

Additionally, `run-benchmark.sh` only copies `golden-dataset.json` — but `benchmark.ts` also
needs the index to be populated (GAP 1+2 above).

**Fix:**
```bash
# Add to run-benchmark.sh before benchmark call:
docker cp "$(pwd)/../../../TypeScript/memory-spark/evaluation/beir-datasets/" \
  "${CONTAINER}:${PLUGIN_DIR}/evaluation/beir-datasets/"
```
Or add the beir-datasets dir as a read-only bind mount in `docker-compose.yml`:
```yaml
- ../../TypeScript/memory-spark/evaluation/beir-datasets:/home/node/.openclaw/extensions/memory-spark/evaluation/beir-datasets:ro
```

---

## GAP 4 — Reference Library: Architecture Exists, Ingestion Doesn't

**Files:** `src/ingest/parsers.ts`, `src/config.ts`, `tools/indexer.ts`

**What exists:**
- `reference_library` and `reference_code` pool values with correct routing
- `memory_reference_search` tool — correctly queries only these pools
- `parsers.ts` supports: `md`, `txt`, `rst`, `pdf` (pdftotext + pdf-parse + GLM-OCR fallback),
  `docx`, `mp3/wav/m4a/ogg/flac/opus` (audio via Spark STT)
- `reference.paths` config with auto-discovery of OpenClaw docs at install time
- Indexer handles reference paths when `cfg.reference.paths` is non-empty

**What's missing:**
- In Docker, `~/.local/share/npm/lib/node_modules/openclaw/docs` doesn't exist → `reference.paths = []`
- No test fixtures in `workspaces/fixtures/` — the fixtures dir is empty
- `docker-compose.yml` mounts fixtures at `/home/node/fixtures` but the indexer doesn't
  know to pick them up (it only uses `cfg.reference.paths`)
- No `--ref-path` CLI arg on indexer to inject paths at runtime
- Phase 2 subtasks (`76bfb098` PDF pipeline, `23082484` reference_search wiring) are
  marked incomplete in CHECKLIST.md, but based on code review, the infrastructure IS mostly
  there — it just needs fixtures and a Docker config that points at them

**Fix:** Add reference paths to Docker test config and populate `workspaces/fixtures/`:
```json
// In configs/test-openclaw.json, under plugins.entries.memory-spark.config:
"reference": {
  "enabled": true,
  "paths": ["/home/node/fixtures/docs"],
  "chunkSize": 800
}
```
Then add a few test `.md` and `.pdf` files to `workspaces/fixtures/docs/`.

---

## GAP 5 — Zero-Shot Classifier: Wired but Unconfigured in Docker

**File:** `src/classify/zero-shot.ts`, `src/auto/capture.ts`, `src/config.ts`

**What exists:**
- Full implementation: `classifyForCapture()` POSTs to Spark's BART-large-MNLI endpoint
  at `${cfg.spark.zeroShot}/v1/classify` (default: `http://<sparkHost>:18113/v1/classify`)
- Service confirmed UP (port 18113 is reachable from Docker container)
- Heuristic fallback if zero-shot returns "none"

**What's missing:**
- `test-openclaw.json` doesn't set `spark.zeroShot` override — defaults to `http://10.99.1.1:18113`
  which IS the right address, so this should work. But it's **not been validated** — no test
  exercises the classifier path end-to-end inside Docker
- `useClassifier` defaults to `true` in config but no capture test in Tier 2 exercises it
- Phase 3A/3B (LLM classification gates) are listed as incomplete in CHECKLIST.md, but
  based on code, the classifier itself IS complete — what's incomplete are the
  **quality gate thresholds** that block low-confidence captures

**Fix:** Add a Tier 2 test case that sends a message containing a clear fact and verifies
it gets captured with a non-"none" category. Low effort.

---

## GAP 6 — A/B/C/D Experiment Harness Is a Stub

**File:** `evaluation/abcd-benchmark.ts`

The overnight runner created this file but it's a 3-line stub that prints a warning and exits.
The formal A/B/C/D statistical comparison (Phase 8B) is completely unimplemented.

Without this, the benchmarks show ablation numbers but no statistical significance — any
paper or presentation claim would be unsupported.

---

## GAP 7 — "Short meaningful → has results" Edge Case Failure

**File:** `evaluation/benchmark.ts` Tier 2 edge_cases

The test sends `"WireGuard IP?"` and expects `memoryCount > 0`. This fails because:
1. The test-data index has no pool column → all pool-filtered recall returns empty
2. Even after pool reindex, this test requires that WireGuard IP content actually be in the
   index. The test-agent workspace (`workspaces/test-agent/MEMORY.md`) needs to contain
   a WireGuard IP or network config entry for this to pass

**Fix:** After pool reindex (GAP 2), verify MEMORY.md in the test-agent workspace contains
relevant WireGuard content. If not, add it or change the test query to match actual content.

---

## GAP 8 — No End-to-End Auto-Recall/Capture Test via OpenClaw API

**Files:** `evaluation/pipeline-eval.ts`, `evaluation/run-all.ts`

`pipeline-eval.ts` is a comprehensive test suite (9 categories, ~40 test cases) but it
exercises the recall pipeline by calling `createAutoRecallHandler` directly — not through
the actual OpenClaw plugin `before_prompt_build` / `agent_end` hooks.

This means the Plugin SDK integration (Phase 6) has **never been validated** end-to-end.
The hooks are wired in `index.ts` and appear correct (verified against OpenClaw source), but
there's no test that actually:
1. Sends a message to the Docker OpenClaw gateway
2. Verifies that memories were injected into the system prompt
3. Sends a follow-up turn and verifies the capture hook fired

This is the gap between unit-testing the pipeline and integration-testing the plugin.

**Fix:** Add a small HTTP integration test in `tests/integration/plugin-e2e.ts` that:
1. POSTs to `http://localhost:18899` (Docker gateway)  
2. Checks gateway logs or a `/debug/memory-spark/last-recall` diagnostic endpoint

---

## GAP 9 — Tier 2 Tests Don't Verify Recall Content Quality

**File:** `evaluation/benchmark.ts`, `evaluation/pipeline-eval.ts`

Tier 2 "pipeline integration" tests verify:
- Garbage is not in results ✅
- Token budget is respected ✅
- Prompt injection is blocked ✅
- Edge cases handled ✅

But none verify that **relevant content IS recalled** — only that bad content is NOT recalled.
Tier 1 covers retrieval metrics, but there's no test that says "when I ask about X, I should
get memory Y back." `pipeline-eval.ts` has those tests but it's not wired into the Docker
benchmark runner.

---

## GAP 10 — BM25 Sigmoid Calibration Is Hardcoded (Not Validated)

**File:** `src/storage/lancedb.ts`, `src/config.ts`

`BM25_SIGMOID_MIDPOINT = 3.0` is the default. The comment in code says to calibrate from
corpus score distribution. It hasn't been done. On BEIR SciFact, vector-only (0.768) already
beats hybrid (0.758) — strongly suggesting FTS/BM25 is adding noise rather than signal on
that corpus with the current sigmoid midpoint.

A wrong midpoint converts BM25 scores to either near-0 or near-1, destroying the signal.
Without calibration, every corpus is effectively using an uncalibrated FTS component.

**Fix:** Run `scripts/test-spark.ts` or a small one-off to collect 100 BM25 scores from the
test corpus, find the median, set that as the sigmoid midpoint in Docker config.

---

## Summary Table

| # | Gap | Severity | Fix Effort | Blocks |
|---|-----|----------|------------|--------|
| 1 | Docker workspace discovery broken | 🔴 BLOCKER | 5 min | Everything |
| 2 | No pool column in any benchmark index | 🔴 BLOCKER | 30 min (reindex) | Tiers 1+2+3 |
| 3 | BEIR dataset not in Docker | 🟠 High | 5 min | BEIR benchmarks |
| 4 | Reference library needs fixtures + Docker config | 🟠 High | 1-2h | Phase 2 validation |
| 5 | Zero-shot classifier uncalibrated/untested | 🟡 Medium | 1h | Phase 3 validation |
| 6 | A/B/C/D harness is a stub | 🟡 Medium | 4-8h | Phase 8B |
| 7 | "WireGuard IP?" edge case | 🟡 Medium | 15 min | Tier 2 clean pass |
| 8 | No E2E plugin hook integration test | 🟡 Medium | 2-3h | Phase 6 validation |
| 9 | Tier 2 only tests rejection, not recall quality | 🟡 Medium | 2h | Full pipeline confidence |
| 10 | BM25 sigmoid uncalibrated | 🟡 Medium | 1h | FTS contribution accuracy |

---

## Recommended Execution Order

1. **Fix GAP 1** (docker-compose mount rename + indexer arg) — 5 min
2. **Fix GAP 2** (purge + reindex inside Docker) — depends on embed speed, ~30 min
3. **Fix GAP 7** (check MEMORY.md content) — 5 min
4. **Run `./scripts/run-benchmark.sh`** — first clean Docker benchmark
5. **Fix GAP 3** (BEIR dataset in Docker) + run BEIR — 15 min
6. **Fix GAP 4** (reference fixtures) + validate `memory_reference_search` — 1-2h
7. **Fix GAP 10** (BM25 calibration) + re-run benchmarks — 1h
8. **Fix GAP 8** (E2E plugin test) — 2-3h
9. **Implement GAP 6** (A/B/C/D harness) — Phase 8B
10. **Production integration** — only after all above pass

---

## What's Actually Done and Solid

- ✅ Phase 0: All 14 bugs fixed, Codex-audited, regression tests in place
- ✅ Phase 1: Pool architecture, single-table LanceDB 0.27.1, pool routing, pool propagation
- ✅ Backend consolidation: MultiTableBackend deleted, LanceDB is sole backend
- ✅ 13 tools registered and wired (search, get, store, forget ×2, reference_search,
  index_status, inspect, reindex, mistakes ×2, rules ×2)
- ✅ 221/221 unit tests passing, 0 type errors, 0 lint errors
- ✅ BEIR SciFact baseline: vector-only 0.768 NDCG@10 (on a clean pool-aware index)
- ✅ Docker harness running and healthy (10h uptime, plugin mounted, all Spark services up)
- ✅ Zero-shot classifier implemented and Spark endpoint is reachable
- ✅ Reference library parser supports PDF/DOCX/audio (pdftotext + GLM-OCR fallback)
- ✅ Security layer: prompt injection detection + escape before context injection
- ✅ Pool isolation design correct — reference pools never auto-injected
