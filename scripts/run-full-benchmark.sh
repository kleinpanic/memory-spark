#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/.openclaw/.env
export SPARK_HOST SPARK_BEARER_TOKEN

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOGDIR="evaluation/results"

echo "═══════════════════════════════════════════════════════"
echo "  memory-spark v0.4.0 — Full BEIR Benchmark Suite"
echo "  Started: $(date)"
echo "  Configs: ALL (36)"
echo "  Datasets: SciFact, FiQA, NFCorpus"
echo "═══════════════════════════════════════════════════════"

for dataset in scifact fiqa nfcorpus; do
  echo ""
  echo "▶▶▶ Dataset: $dataset — $(date)"
  echo ""
  npx tsx scripts/run-beir-bench.ts --dataset "$dataset" 2>&1 | tee "$LOGDIR/full-run-${dataset}-${TIMESTAMP}.log"
  echo ""
  echo "✅ $dataset complete — $(date)"
  echo ""
done

echo "═══════════════════════════════════════════════════════"
echo "  Full benchmark complete: $(date)"
echo "═══════════════════════════════════════════════════════"
