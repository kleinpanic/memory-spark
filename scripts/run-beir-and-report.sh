#!/usr/bin/env bash
# BEIR datasets + Report — chain after main benchmark finishes
set -uo pipefail
cd "$HOME/codeWS/TypeScript/memory-spark"
export MEMORY_SPARK_DATA_DIR=./test-data
SMS="$HOME/.openclaw/hooks/sms-alert.sh"
TIMESTAMP=$(date +%Y%m%d-%H%M)
RESULTS_DIR="evaluation/results/run-${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/beir.log"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }
sms_send() { rm -f /tmp/.openclaw-sms-cooldown; bash "$SMS" "$1" || true; }

cp evaluation/results/quick-*.log evaluation/results/full-*.log evaluation/results/pipeline-*.log evaluation/results/benchmark-v2-*.json "$RESULTS_DIR/" 2>/dev/null || true

sms_send "Main benchmarks done. Running 3 BEIR datasets now..."

for DS in scifact nfcorpus fiqa; do
  log "BEIR — $DS"
  npx tsx evaluation/beir-benchmark.ts --dataset "$DS" --batch-size 32 2>&1 | tee "$RESULTS_DIR/beir-${DS}.log"
  NDCG=$(grep -oP 'NDCG@10[:\s]+\K[0-9.]+' "$RESULTS_DIR/beir-${DS}.log" | tail -1 || echo "?")
  log "$DS NDCG@10: $NDCG"
  eval "${DS^^}_NDCG=$NDCG"
done

# Extract main bench numbers
F=$(ls -t "$RESULTS_DIR"/full-*.log 2>/dev/null | head -1)
ext() { grep -A15 "$1" "$F" | grep 'ndcg' | head -1 | awk '{print $NF}' 2>/dev/null || echo "?"; }
VEC=$(ext 'Vector-Only'); FTS=$(ext 'FTS-Only'); HYB=$(ext 'Hybrid (No Reranker)')
FULL=$(ext 'Full Pipeline'); HYDE=$(ext 'HyDE'); PROD=$(ext 'Production Mirror')
JUDGE=$(grep -oP 'Judge.*?:\s*\K[0-9.]+' "$F" 2>/dev/null | tail -1 || echo "?")

log ""; log "ALL DONE"
log "Custom: Vec=$VEC FTS=$FTS Hyb=$HYB Full=$FULL HyDE=$HYDE Prod=$PROD Judge=$JUDGE/5"
log "BEIR: SciFact=${SCIFACT_NDCG} NFCorpus=${NFCORPUS_NDCG} FiQA=${FIQA_NDCG}"

sms_send "ALL DONE! Custom: Vec=$VEC Full=$FULL HyDE=$HYDE Prod=$PROD Judge=$JUDGE/5. BEIR: Sci=${SCIFACT_NDCG} NF=${NFCORPUS_NDCG} FiQA=${FIQA_NDCG}. Results: $RESULTS_DIR/"
