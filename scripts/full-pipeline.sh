#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  memory-spark Full Pipeline — Reindex + Benchmark + Validate
# ═══════════════════════════════════════════════════════════════════════
#
#  Run in tmux:
#    tmux new -s bench
#    cd ~/codeWS/TypeScript/memory-spark
#    ./scripts/full-pipeline.sh 2>&1 | tee /tmp/memory-spark-pipeline.log
#
#  Stages:
#    1. Pre-flight checks (Spark connectivity, build, tests)
#    2. Purge old index (no pool column)
#    3. Full reindex with pool column
#    4. Verify pool column exists
#    5. Run benchmark suite (all tiers)
#    6. Run BEIR SciFact benchmark
#    7. Print summary
#
#  Env:
#    MEMORY_SPARK_DATA_DIR=./test-data  (isolated from production)
#    SPARK_HOST=10.99.1.1               (DGX Spark for embeddings)
#
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

export MEMORY_SPARK_DATA_DIR=./test-data
export SPARK_HOST=10.99.1.1

LOG="/tmp/memory-spark-pipeline.log"
RESULTS="$REPO/evaluation/results"
mkdir -p "$RESULTS"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

banner() { echo -e "\n${CYAN}═══════════════════════════════════════════${NC}"; echo -e "  ${BOLD}$1${NC}"; echo -e "${CYAN}═══════════════════════════════════════════${NC}\n"; }
ok()     { echo -e "  ${GREEN}✅ $1${NC}"; }
fail()   { echo -e "  ${RED}❌ $1${NC}"; }
warn()   { echo -e "  ${YELLOW}⚠️  $1${NC}"; }
step()   { echo -e "  ${BOLD}▶ $1${NC}"; }

PIPELINE_START=$(date +%s)
STAGE_START=0
stage_time() {
  local now
  now=$(date +%s)
  if [ "$STAGE_START" -gt 0 ]; then
    echo -e "  ⏱  Stage: $(( now - STAGE_START ))s | Total: $(( now - PIPELINE_START ))s"
  fi
  STAGE_START=$now
}

# ── Stage 1: Pre-flight ─────────────────────────────────────────────

banner "Stage 1: Pre-flight Checks"
stage_time

step "Checking Spark embed endpoint..."
if curl -sf --connect-timeout 5 http://10.99.1.1:18091/health > /dev/null 2>&1; then
  ok "Spark Embed (18091) — UP"
else
  fail "Spark Embed (18091) — DOWN. Cannot continue."
  exit 1
fi

step "Checking Spark rerank endpoint..."
if curl -sf --connect-timeout 5 http://10.99.1.1:18096/health > /dev/null 2>&1; then
  ok "Spark Rerank (18096) — UP"
else
  warn "Spark Rerank (18096) — DOWN. Benchmarks will skip reranker tests."
fi

step "Checking Spark zero-shot endpoint..."
if curl -sf --connect-timeout 5 http://10.99.1.1:18113/health > /dev/null 2>&1; then
  ok "Spark Zero-Shot (18113) — UP"
else
  warn "Spark Zero-Shot (18113) — DOWN. Classification tests will use heuristics."
fi

step "Running unit tests (unit only — integration tests run with benchmark)..."
TEST_OUT=$(npx vitest run tests/unit.test.ts --reporter=dot 2>&1) || true
TEST_RESULT=$(echo "$TEST_OUT" | grep -oP '\d+ passed' | head -1)
if echo "$TEST_OUT" | grep -q "Tests.*passed" && ! echo "$TEST_OUT" | grep -q "failed"; then
  ok "Unit tests: $TEST_RESULT"
else
  echo "$TEST_OUT" | tail -15
  fail "Unit tests failed — fix before benchmarking"
  exit 1
fi

step "Type check..."
TSC_OUT=$(npx tsc --noEmit 2>&1) || true
if [ -z "$TSC_OUT" ]; then
  ok "TypeScript — no errors"
else
  echo "$TSC_OUT" | tail -10
  fail "TypeScript errors found"
  exit 1
fi

stage_time

# ── Stage 2: Purge old index ────────────────────────────────────────

banner "Stage 2: Purge Old Index"
stage_time

if [ -d "test-data/lancedb/memory_chunks.lance" ]; then
  step "Backing up old index..."
  mv test-data/lancedb/memory_chunks.lance "test-data/lancedb/memory_chunks.lance.bak.$(date +%s)"
  ok "Old index backed up"
else
  ok "No existing index to purge"
fi

# Also remove dims-lock if it exists (will be recreated)
rm -f test-data/lancedb/dims-lock.json 2>/dev/null || true

stage_time

# ── Stage 3: Full Reindex ───────────────────────────────────────────

banner "Stage 3: Full Reindex (with pool column)"
stage_time

step "Starting indexer... (this takes a while — ~5 files/sec with Spark embeddings)"
echo ""

if npx tsx tools/indexer.ts 2>&1; then
  ok "Reindex complete"
else
  fail "Reindex failed"
  exit 1
fi

stage_time

# ── Stage 4: Verify pool column ────────────────────────────────────

banner "Stage 4: Verify Pool Column"
stage_time

cat > /tmp/ms-verify-pool.ts << 'VERIFY_EOF'
import lancedb from "@lancedb/lancedb";
async function main() {
  const db = await lancedb.connect("test-data/lancedb");
  const t = await db.openTable("memory_chunks");
  const schema = await t.schema();
  const fields = schema.fields.map((f: any) => f.name);
  const hasPool = fields.includes("pool");
  console.log("Schema fields: " + fields.join(", "));
  console.log("HAS_POOL: " + hasPool);
  if (!hasPool) {
    console.error("FATAL: pool column missing after reindex!");
    process.exit(1);
  }
  // Sample a few rows to check pool values
  const rows = await t.query().limit(10).toArray();
  const pools = [...new Set(rows.map((r: any) => r.pool).filter(Boolean))];
  console.log("Sample pools: " + (pools.length > 0 ? pools.join(", ") : "(all null — pool routing may not be working)"));
  const count = await t.countRows();
  console.log("Total chunks: " + count);
}
main().catch(e => { console.error(e); process.exit(1); });
VERIFY_EOF

cp /tmp/ms-verify-pool.ts verify-pool.ts
if npx tsx verify-pool.ts 2>&1; then
  ok "Pool column verified"
else
  fail "Pool column verification failed"
  exit 1
fi
rm -f verify-pool.ts

stage_time

# ── Stage 5: Benchmark Suite ───────────────────────────────────────

banner "Stage 5: Benchmark Suite (Tiers 1+2+3)"
stage_time

step "Running full benchmark..."
echo ""

if npx tsx evaluation/benchmark.ts 2>&1; then
  ok "Benchmark complete"
else
  warn "Benchmark had errors (check results)"
fi

stage_time

# ── Stage 6: BEIR SciFact ──────────────────────────────────────────

banner "Stage 6: BEIR SciFact Benchmark"
stage_time

# Check if BEIR index exists
if [ -d "evaluation/beir-datasets/scifact-index" ]; then
  step "Running BEIR SciFact (reusing existing index)..."
  if npx tsx evaluation/beir-benchmark.ts --dataset scifact --skip-index 2>&1; then
    ok "BEIR SciFact complete"
  else
    warn "BEIR SciFact had errors"
  fi
else
  warn "BEIR SciFact index not found — skipping (run beir-benchmark.ts first to build index)"
fi

stage_time

# ── Stage 7: Summary ──────────────────────────────────────────────

banner "Stage 7: Summary"

echo ""
echo -e "${BOLD}Results:${NC}"
ls -lh "$RESULTS"/*.json 2>/dev/null | tail -10
echo ""

# Parse latest benchmark result
LATEST=$(ls -t "$RESULTS"/benchmark-*.json 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo -e "${BOLD}Latest Benchmark:${NC} $LATEST"
  python3 << PYEOF
import json
with open("$LATEST") as f:
    d = json.load(f)
print(f"  Chunks: {d.get('indexChunks', '?')}")
print(f"  Reranker: {d.get('rerankerEnabled', '?')}")
t1 = d.get('tier1', {})
for cfg_name in ['vector_only', 'fts_only', 'hybrid_no_reranker', 'full_pipeline']:
    r = t1.get(cfg_name, {})
    ndcg = r.get('ndcg', {}).get('@10', '?')
    mrr = r.get('mrr', {}).get('@10', '?')
    if ndcg != '?':
        print(f"  {cfg_name}: NDCG@10={ndcg:.4f}  MRR@10={mrr:.4f}")
err = d.get('error')
if err:
    print(f"  ⚠️ Error: {str(err)[:120]}")
PYEOF
fi

echo ""
TOTAL_ELAPSED=$(( $(date +%s) - PIPELINE_START ))
echo -e "${GREEN}${BOLD}Pipeline complete in ${TOTAL_ELAPSED}s!${NC}"
echo -e "Full log: $LOG"
echo ""
echo "Next steps:"
echo "  1. Review results in evaluation/results/"
echo "  2. If NDCG improved: proceed to production integration"
echo "  3. If reranker still regresses: investigate candidate window size"
