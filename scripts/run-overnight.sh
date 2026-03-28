#!/usr/bin/env bash
# Overnight benchmark runner — chains all suites in order.
# Waits for any already-running benchmark.ts to finish first.
#
# Suites:
#   1. benchmark.ts        — custom corpus: full ablation study (Tiers 1/2/3)
#   2. beir-benchmark.ts   — BEIR SciFact with reranker ON
#   3. abcd-benchmark.ts   — A/B/C/D experiment matrix (Phase 8B)
#
# Results: evaluation/results/
# Log: /tmp/overnight-benchmark-full.log

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$REPO/evaluation/results"
LOG="/tmp/overnight-benchmark-full.log"
ENV_PREFIX="MEMORY_SPARK_DATA_DIR=./test-data SPARK_HOST=10.99.1.1"

cd "$REPO"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

mkdir -p "$RESULTS"
echo "=========================================" >> "$LOG"
echo "  memory-spark Overnight Run" >> "$LOG"
echo "  Started: $(date)" >> "$LOG"
echo "=========================================" >> "$LOG"

# ── 1. Custom corpus benchmark (Tiers 1+2+3) ─────────────────────────────────
# Skip if already running or if a fresh result exists from this session
BENCH_RESULT="$RESULTS/benchmark-$(date '+%Y-%m-%d')*.json"
if ls $BENCH_RESULT 1>/dev/null 2>&1; then
  log "✅ benchmark.ts result already exists for today — skipping (delete to re-run)"
else
  log "▶ Suite 1/3: Custom corpus benchmark (all tiers, reranker ON)..."
  if env MEMORY_SPARK_DATA_DIR=./test-data SPARK_HOST=10.99.1.1 npx tsx evaluation/benchmark.ts >> "$LOG" 2>&1; then
    log "✅ Suite 1 complete"
  else
    log "❌ Suite 1 failed (see log for details) — continuing"
  fi
fi

# ── 2. BEIR SciFact with reranker ON ─────────────────────────────────────────
log "▶ Suite 2/3: BEIR SciFact (reranker ON, full pipeline)..."
if env MEMORY_SPARK_DATA_DIR=./test-data SPARK_HOST=10.99.1.1 \
    npx tsx evaluation/beir-benchmark.ts \
    --dataset scifact \
    --reranker \
    --output "$RESULTS/beir-scifact-reranked-$(date '+%Y-%m-%dT%H-%M-%S').json" \
    >> "$LOG" 2>&1; then
  log "✅ Suite 2 complete"
else
  log "❌ Suite 2 failed — continuing"
fi

# ── 3. A/B/C/D experiment matrix ─────────────────────────────────────────────
log "▶ Suite 3/3: A/B/C/D experiment matrix..."
if [ -f "evaluation/abcd-benchmark.ts" ]; then
  if env MEMORY_SPARK_DATA_DIR=./test-data SPARK_HOST=10.99.1.1 \
      npx tsx evaluation/abcd-benchmark.ts >> "$LOG" 2>&1; then
    log "✅ Suite 3 complete"
  else
    log "❌ Suite 3 failed — see log"
  fi
else
  log "⚠️  evaluation/abcd-benchmark.ts not found — writing stub and skipping"
  cat > evaluation/abcd-benchmark.ts << 'ABCD_STUB'
#!/usr/bin/env npx tsx
/**
 * A/B/C/D Experiment Matrix — Phase 8B
 *
 * Formal comparison between pipeline configurations with paired t-test
 * statistical significance testing.
 *
 * Groups:
 *   A — Vanilla: vector-only, no reranker, no pools, no decay
 *   B — Hybrid:  vector + FTS + reranker, no pools, no decay
 *   C — Full:    pools + hybrid + reranker + temporal decay + MMR
 *   D — SOTA+:   C + late interaction (HyDE disabled — too slow for agent memory)
 *
 * TODO: implement after overnight run confirms current baselines.
 * Blocked on: pool-column reindex of custom corpus.
 */

console.log("⚠️  A/B/C/D harness not yet implemented (Phase 8B).");
console.log("   Run after: pool-column reindex of custom corpus.");
console.log("   See CHECKLIST.md Phase 8B for design spec.");
process.exit(0);
ABCD_STUB
  log "  Stub created at evaluation/abcd-benchmark.ts — implement with Opus tomorrow"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
log ""
log "========================================="
log "  Overnight run complete: $(date)"
log "  Results in: $RESULTS/"
log "  Full log: $LOG"
log "========================================="
ls -lh "$RESULTS"/*.json 2>/dev/null | tail -10 | tee -a "$LOG"
