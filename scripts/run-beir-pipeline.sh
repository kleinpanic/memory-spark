#!/usr/bin/env bash
#
# BEIR Benchmark Pipeline — Index + Benchmark A-G
# ================================================
# Full BEIR evaluation: 66,454 docs across SciFact, NFCorpus, FiQA
# Configs A-G: Vector, FTS, Hybrid, Reranker, MMR, HyDE, Full
#
# Usage:
#   tmux new-session -d -s BEIR-pipeline "bash scripts/run-beir-pipeline.sh --all"
#   tmux attach -t BEIR-pipeline
#
# Flags:
#   --all           # scifact + nfcorpus + fiqa (66,454 docs)
#   --without-fiqa  # scifact + nfcorpus (8,816 docs)
#   --fiqa-only     # fiqa only (57,638 docs)
#

set -uo pipefail

PROJECT="$HOME/codeWS/TypeScript/memory-spark"
cd "$PROJECT"

# Source env vars (SPARK_HOST, SPARK_BEARER_TOKEN)
[[ -f "$HOME/.openclaw/.env" ]] && set -a && source "$HOME/.openclaw/.env" && set +a
[[ -f .env ]] && set -a && source .env && set +a

export BEIR_LANCEDB_DIR="$HOME/.openclaw/data/testDbBEIR/lancedb"

SMS="$HOME/.openclaw/hooks/sms-alert.sh"
LOGDIR="evaluation/results"
LOGFILE="$LOGDIR/BEIR-pipeline-$(date +%Y%m%d-%H%M%S).log"
STATEFILE="/tmp/BEIR-pipeline.state"
mkdir -p "$LOGDIR"

# ── Parse flags ───────────────────────────────────────────────────────────

MODE="all"
case "${1:-}" in
    --without-fiqa) MODE="without-fiqa" ;;
    --fiqa-only)    MODE="fiqa-only" ;;
esac

case "$MODE" in
    all)           DATASETS="scifact nfcorpus fiqa";    TOTAL_DOCS=66454 ;;
    without-fiqa)  DATASETS="scifact nfcorpus";        TOTAL_DOCS=8816 ;;
    fiqa-only)     DATASETS="fiqa";                    TOTAL_DOCS=57638 ;;
esac

# ── Helpers ───────────────────────────────────────────────────────────────

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOGFILE"; }

sms() {
  local msg="$1"
  rm -f /tmp/.openclaw-sms-cooldown
  log "SMS: $msg"
  [[ -x "$SMS" ]] && bash "$SMS" "$msg" >> "$LOGFILE" 2>&1 || true
}

die() {
  log "FATAL: $1"
  echo "FAILED: $1" > "$STATEFILE"
  sms "(OpenClaw) BEIR Pipeline FAILED: $1"
  exit 1
}

# ── Signal Traps for BEIR Pipeline ────────────────────────────────────────

_CURRENT_PHASE="startup"
_LAST_FILE=""

update_phase() { 
  _CURRENT_PHASE="$1"
  echo "PHASE: $1" > "$STATEFILE"
  log "→ Phase: $1"
}

update_file() { 
  _LAST_FILE="$1"
  echo "FILE: $1" >> "$STATEFILE"
}

on_exit() {
  local EC=$?
  local SIGNAL_NAME=""
  
  case $EC in
    130) SIGNAL_NAME=" (SIGINT/Ctrl-C)" ;;
    137) SIGNAL_NAME=" (SIGKILL — OOM or killed)" ;;
    143) SIGNAL_NAME=" (SIGTERM)" ;;
  esac

  if [[ $EC -ne 0 ]]; then
    log ""
    log "╔══════════════════════════════════════════════╗"
    log "║  BEIR PIPELINE CRASHED — exit ${EC}${SIGNAL_NAME}"
    log "║  Phase:     ${_CURRENT_PHASE}"
    log "║  Last file: ${_LAST_FILE:-N/A}"
    log "║  Time:      $(ts)"
    log "╚══════════════════════════════════════════════╝"
    sync
    sms "(OpenClaw) BEIR Pipeline CRASHED (exit ${EC}${SIGNAL_NAME}). Phase: ${_CURRENT_PHASE}. Check: tmux attach -t BEIR-pipeline"
  fi
}

trap on_exit EXIT
trap 'log "Caught SIGTERM — shutting down"; exit 143' TERM
trap 'log "Caught SIGINT — shutting down"; exit 130' INT
trap 'log "Caught SIGHUP — continuing (detached)";' HUP

# ── Preflight ─────────────────────────────────────────────────────────────

log "═══════════════════════════════════════════════════"
log "  BEIR Benchmark Pipeline"
log "═══════════════════════════════════════════════════"
log "Mode:       $MODE"
log "Datasets:   $DATASETS"
log "Total docs: $TOTAL_DOCS"
log "Configs:    A/B/C/D/E/F/G"
log "State:      $STATEFILE"
log "Log:        $LOGFILE"
log "PID:        $$"
log ""

command -v npx &>/dev/null || die "npx not found"
command -v jq &>/dev/null || die "jq not found"
[[ -f evaluation/index-beir.ts ]] || die "evaluation/index-beir.ts missing"
[[ -f evaluation/run-beir-bench.ts ]] || die "evaluation/run-beir-bench.ts missing"

echo "RUNNING: preflight OK" > "$STATEFILE"

# ── Phase 1: Build ───────────────────────────────────────────────────────

update_phase "build"
log ""
log "═══════════════════════════════════════════════════"
log "  Phase 1: Build"
log "═══════════════════════════════════════════════════"

npm run build 2>&1 | tee -a "$LOGFILE" || die "Build failed"
log "✅ Build complete"

# ── Phase 2: Index BEIR Datasets ────────────────────────────────────────

update_phase "index"
log ""
log "═══════════════════════════════════════════════════"
log "  Phase 2: Index BEIR Datasets"
log "═══════════════════════════════════════════════════"
log "Target: $BEIR_LANCEDB_DIR"
log "Resume: enabled (checkpoint.json per dataset)"

INDEX_START=$(date +%s)

for ds in $DATASETS; do
  update_phase "index-$ds"
  log ""
  log "Indexing $ds..."
  
  NODE_OPTIONS="--max-old-space-size=8192" \
    npx tsx evaluation/index-beir.ts --dataset "$ds" --resume 2>&1 \
    | while IFS= read -r line; do
        echo "$line"
        echo "$line" >> "$LOGFILE"
        # Track progress for crash diagnosis
        if [[ "$line" =~ \[([a-z]+)\]\ ([0-9]+)/([0-9]+) ]]; then
          echo "PROGRESS: ${BASH_REMATCH[1]} ${BASH_REMATCH[2]}/${BASH_REMATCH[3]}" > "$STATEFILE"
        fi
      done
  
  local EC=${PIPESTATUS[0]}
  if [[ "$EC" -ne 0 ]]; then
    log "WARN: $ds indexing exited $EC (continuing...)"
  fi
done

INDEX_END=$(date +%s)
INDEX_DURATION=$(( (INDEX_END - INDEX_START) / 60 ))

log ""
log "✅ Indexing complete (${INDEX_DURATION}m)"
echo "DONE: indexing ${INDEX_DURATION}m" > "$STATEFILE"

sms "(OpenClaw) BEIR indexing done! ${TOTAL_DOCS} docs in ${INDEX_DURATION}m. Running A-G benchmarks now..."

# ── Phase 3: Benchmark A-G ───────────────────────────────────────────────

declare -A NDCG_RESULTS
declare -A MRR_RESULTS

for ds in $DATASETS; do
  update_phase "benchmark-$ds"
  log ""
  log "═══════════════════════════════════════════════════"
  log "  Benchmark: $ds (Configs A-G)"
  log "═══════════════════════════════════════════════════"
  
  NODE_OPTIONS="--max-old-space-size=8192" \
    npx tsx evaluation/run-beir-bench.ts --dataset "$ds" 2>&1 | tee -a "$LOGFILE"
  
  # Extract metrics from summary
  SUMMARY=$(ls -t evaluation/results/beir-${ds}-summary-*.json 2>/dev/null | head -1)
  if [[ -n "$SUMMARY" ]]; then
    # Get Config G (Full Pipeline) metrics
    NDCG=$(jq -r '.results[] | select(.config=="G") | .ndcg' "$SUMMARY" 2>/dev/null || echo "?")
    MRR=$(jq -r '.results[] | select(.config=="G") | .mrr' "$SUMMARY" 2>/dev/null || echo "?")
    NDCG_RESULTS[$ds]=$NDCG
    MRR_RESULTS[$ds]=$MRR
    
    log ""
    log "$ds Results (Config G - Full Pipeline):"
    log "  NDCG@10: $NDCG"
    log "  MRR@10:  $MRR"
    
    sms "(OpenClaw) BEIR $ds done. NDCG=$NDCG MRR=$MRR"
  fi
done

# ── Phase 4: Final Report ────────────────────────────────────────────────

update_phase "report"
log ""
log "═══════════════════════════════════════════════════"
log "  Phase 4: Final Report"
log "═══════════════════════════════════════════════════"

SCIFACT_NDCG="${NDCG_RESULTS[scifact]:-?}"
SCIFACT_MRR="${MRR_RESULTS[scifact]:-?}"
NFCORPUS_NDCG="${NDCG_RESULTS[nfcorpus]:-?}"
NFCORPUS_MRR="${MRR_RESULTS[nfcorpus]:-?}"
FIQA_NDCG="${NDCG_RESULTS[fiqa]:-?}"
FIQA_MRR="${MRR_RESULTS[fiqa]:-?}"

REPORT="$LOGDIR/BEIR-report-$(date +%Y%m%d-%H%M%S).md"
cat > "$REPORT" << REPORT_EOF
# BEIR Benchmark Report

**Date:** $(date '+%Y-%m-%d %H:%M %Z')
**Mode:** $MODE
**Datasets:** $DATASETS
**Total Docs:** $TOTAL_DOCS
**Index Duration:** ${INDEX_DURATION}m
**Configs:** A/B/C/D/E/F/G (Vector → Full Pipeline)

## Results (Config G - Full Pipeline)

| Dataset | NDCG@10 | MRR@10 |
|---------|---------|--------|
| SciFact | ${SCIFACT_NDCG} | ${SCIFACT_MRR} |
| NFCorpus | ${NFCORPUS_NDCG} | ${NFCORPUS_MRR} |
| FiQA | ${FIQA_NDCG} | ${FIQA_MRR} |

## Files

- Log: \`$(basename "$LOGFILE")\`
- Summaries: \`evaluation/results/beir-*-summary-*.json\`

REPORT_EOF

log ""
log "╔══════════════════════════════════════════════╗"
log "║  BEIR PIPELINE COMPLETE"
log "║  $(ts)"
log "║"
log "║  SciFact:   NDCG=${SCIFACT_NDCG}  MRR=${SCIFACT_MRR}"
log "║  NFCorpus:  NDCG=${NFCORPUS_NDCG}  MRR=${NFCORPUS_MRR}"
log "║  FiQA:      NDCG=${FIQA_NDCG}  MRR=${FIQA_MRR}"
log "║"
log "║  Report: $REPORT"
log "╚══════════════════════════════════════════════╝"

echo "DONE: all phases complete" > "$STATEFILE"
sync

sms "(OpenClaw) BEIR Pipeline COMPLETE! SciFact=${SCIFACT_NDCG} NFCorpus=${NFCORPUS_NDCG} FiQA=${FIQA_NDCG}. Report: $REPORT"

log ""
log "Press Ctrl+C to exit, or detach with Ctrl+B D"
exec bash
