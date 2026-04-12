#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# memory-spark eval harness entrypoint
# Usage: docker run ... [scifact|ncf|all|core|benchmark|shell]
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

BEIR_DATA_DIR="${BEIR_DATA_DIR:-/data/beir-datasets}"
EVAL_DB_DIR="${EVAL_DB_DIR:-/data/eval-lancedb}"
RESULTS_DIR="${RESULTS_DIR:-/data/results}"
CONFIGS="${BEIR_CONFIGS:-all}"
DISABLE_OCR="${DISABLE_OCR:-true}"

# Always create fresh eval DB dir (never touch production)
export MEMORY_SPARK_DB_PATH="$EVAL_DB_DIR/eval.db"
mkdir -p "$EVAL_DB_DIR" "$RESULTS_DIR"

echo "═══════════════════════════════════════════════════"
echo "memory-spark eval harness"
echo "  BEIR data:  $BEIR_DATA_DIR"
echo "  Eval DB:    $EVAL_DB_DIR"
echo "  Results:    $RESULTS_DIR"
echo "  Configs:    $CONFIGS"
echo "  OCR:        ${DISABLE_OCR}"
echo "═══════════════════════════════════════════════════"

# ── Validate BEIR data exists ─────────────────────────────────────────────────
check_datasets() {
  local datasets="${1:-scifact}"
  local missing=0
  for ds in $(echo "$datasets" | tr ',' ' '); do
    if [ ! -f "$BEIR_DATA_DIR/$ds/corpus.jsonl" ]; then
      echo "❌ Missing dataset: $ds (expected at $BEIR_DATA_DIR/$ds/corpus.jsonl)"
      ((missing++))
    fi
  done
  if [ $missing -gt 0 ]; then
    echo "❌ $missing dataset(s) missing. Run: docker run -v /path/to/beir-data:$BEIR_DATA_DIR ..."
    echo "   Available datasets:"
    ls "$BEIR_DATA_DIR" 2>/dev/null || echo "   (directory empty or not mounted)"
    exit 1
  fi
  echo "✅ All datasets validated"
}

# ── Check vLLM connectivity ───────────────────────────────────────────────────
check_vllm() {
  local url="$1"
  local name="$2"
  if curl -sf --max-time 3 "$url/models" > /dev/null 2>&1; then
    echo "✅ $name: reachable"
    return 0
  else
    echo "⚠️  $name: unreachable at $url (will use CPU fallback if available)"
    return 1
  fi
}

# ── Commands ───────────────────────────────────────────────────────────────────
cmd="${1:-benchmark}"

case "$cmd" in
  shell|bash)
    echo "Dropping into shell..."
    exec bash
    ;;

  benchmark)
    check_datasets "${BEIR_DATASETS}"
    echo ""
    check_vllm "${SPARK_EMBED_URL}" "Embedder (${SPARK_EMBED_URL})" || true
    check_vllm "${SPARK_RERANK_URL}" "Reranker (${SPARK_RERANK_URL})" || true
    check_vllm "${SPARK_HYDE_URL}"   "HyDE LLM (${SPARK_HYDE_URL})"   || true
    echo ""
    echo "🚀 Running benchmark..."
    exec npx tsx evaluation/harness/run-benchmark.ts \
      --datasets "${BEIR_DATASETS}" \
      --configs "${CONFIGS}" \
      --output "$RESULTS_DIR" \
      --db-dir "$EVAL_DB_DIR"
    ;;

  scifact)
    export BEIR_DATASETS=scifact
    check_datasets scifact
    echo ""
    echo "🚀 Running SciFact benchmark..."
    exec npx tsx evaluation/harness/run-benchmark.ts \
      --datasets scifact \
      --configs all \
      --output "$RESULTS_DIR" \
      --db-dir "$EVAL_DB_DIR"
    ;;

  core)
    export BEIR_DATASETS=scifact,nfcorpus,fiqa
    check_datasets "$BEIR_DATASETS"
    echo ""
    echo "🚀 Running CORE (3-dataset) benchmark..."
    exec npx tsx evaluation/harness/run-benchmark.ts \
      --datasets scifact,nfcorpus,fiqa \
      --configs all \
      --output "$RESULTS_DIR" \
      --db-dir "$EVAL_DB_DIR"
    ;;

  ncf)
    # NFCorpus only — for smoke testing
    export BEIR_DATASETS=nfcorpus
    check_datasets nfcorpus
    echo ""
    echo "🚀 Running NFCorpus smoke test..."
    exec npx tsx evaluation/harness/run-benchmark.ts \
      --datasets nfcorpus \
      --configs A,B,C \
      --output "$RESULTS_DIR" \
      --db-dir "$EVAL_DB_DIR"
    ;;

  index)
    # Pre-index datasets into LanceDB for faster subsequent runs
    shift
    datasets="${*:-$BEIR_DATASETS}"
    check_datasets "$datasets"
    echo ""
    echo "📚 Pre-indexing datasets..."
    exec npx tsx evaluation/harness/index-datasets.ts \
      --datasets "$datasets" \
      --db-dir "$EVAL_DB_DIR"
    ;;

  *)
    echo "Usage: $0 [benchmark|scifact|core|ncf|index|shell]"
    echo ""
    echo "Commands:"
    echo "  benchmark  — Run full benchmark on BEIR_DATASETS (default: scifact,nfcorpus,fiqa)"
    echo "  scifact    — Run only SciFact (fast smoke test)"
    echo "  core       — Run the 3 paper datasets (SciFact + NFCorpus + FiQA)"
    echo "  ncf        — NFCorpus smoke test with configs A,B,C only"
    echo "  index      — Pre-index datasets into LanceDB (faster subsequent runs)"
    echo "  shell      — Drop into bash shell"
    exit 1
    ;;
esac
