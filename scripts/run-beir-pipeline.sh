#!/usr/bin/env bash
#
# BEIR Benchmark Pipeline — Index + Benchmark A-G
# ================================================
# Robust version: tmux-visible, state tracking, signal traps, SMS alerts.
#
# Via tmux (preferred):
#   tmux new-session -d -s beir-pipeline "bash scripts/run-beir-pipeline.sh --all"
#   tmux attach -t beir-pipeline
#
# Flags:
#   --all           # scifact + nfcorpus + fiqa (66,454 docs)
#   --without-fiqa  # scifact + nfcorpus (8,816 docs)
#   --fiqa-only     # fiqa only (57,638 docs)
#

set -uo pipefail

PROJECT="$HOME/codeWS/TypeScript/memory-spark"
cd "$PROJECT"

# Source env vars
[[ -f "$HOME/.openclaw/.env" ]] && set -a && source "$HOME/.openclaw/.env" && set +a
[[ -f .env ]] && set -a && source .env && set +a

export BEIR_LANCEDB_DIR="$HOME/.openclaw/data/testDbBEIR/lancedb"

SMS="$HOME/.openclaw/hooks/sms-alert.sh"
LOGDIR="evaluation/results"
LOGFILE="$LOGDIR/pipeline-$(date +%Y%m%d-%H%M%S).log"
STATEFILE="/tmp/beir-pipeline.state"
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
  sms "(OpenClaw) BEIR FAILED: $1"
  exit 1
}

# ── Signal Traps ──────────────────────────────────────────────────────────

_CURRENT_PHASE="startup"

update_phase() { _CURRENT_PHASE="$1"; echo "PHASE: $1" > "$STATEFILE"; log "→ $1"; }

on_exit() {
  local EC=$?
  if [[ $EC -ne 0 ]]; then
    log "CRASHED (exit $EC) in phase: $_CURRENT_PHASE"
    sms "(OpenClaw) BEIR CRASHED (exit $EC). Phase: $_CURRENT_PHASE"
  fi
}
trap on_exit EXIT

# ── Preflight ─────────────────────────────────────────────────────────────

log "═══════════════════════════════════════════════════"
log "  BEIR Benchmark Pipeline"
log "═══════════════════════════════════════════════════"
log "Mode:     $MODE"
log "Datasets: $DATASETS"
log "Docs:     $TOTAL_DOCS"
log "State:    $STATEFILE"
log ""

command -v npx &>/dev/null || die "npx not found"
[[ -f evaluation/index-beir.ts ]] || die "index-beir.ts missing"
[[ -f evaluation/run-beir-bench.ts ]] || die "run-beir-bench.ts missing"

# ── Phase 1: Build ───────────────────────────────────────────────────────

update_phase "build"
log "▶ Build"
npm run build 2>&1 | tee -a "$LOGFILE" || die "Build failed"
log "✅ Build done"

# ── Phase 2: Index ───────────────────────────────────────────────────────

update_phase "index"
log "▶ Index BEIR datasets"
log "  Target: $BEIR_LANCEDB_DIR"

for ds in $DATASETS; do
  update_phase "index-$ds"
  log "  Indexing $ds..."
  npx tsx evaluation/index-beir.ts --dataset "$ds" --resume 2>&1 | tee -a "$LOGFILE" || log "WARN: $ds indexing had issues"
done

log "✅ Indexing done"
sms "(OpenClaw) BEIR indexing done ($TOTAL_DOCS docs). Benchmarking A-G starting..."

# ── Phase 3: Benchmark ───────────────────────────────────────────────────

declare -A NDCG_RESULTS

for ds in $DATASETS; do
  update_phase "bench-$ds"
  log "▶ Benchmark $ds"
  
  npx tsx evaluation/run-beir-bench.ts --dataset "$ds" 2>&1 | tee -a "$LOGFILE"
  
  # Get latest summary
  SUMMARY=$(ls -t evaluation/results/beir-${ds}-summary-*.json 2>/dev/null | head -1)
  if [[ -n "$SUMMARY" ]]; then
    NDCG=$(jq -r '.results[] | select(.config=="G") | .ndcg' "$SUMMARY" 2>/dev/null || echo "?")
    NDCG_RESULTS[$ds]=$NDCG
    log "$ds NDCG@10 (Config G): $NDCG"
  fi
done

# ── Done ─────────────────────────────────────────────────────────────────

update_phase "done"

SCIFACT_NDCG="${NDCG_RESULTS[scifact]:-?}"
NFCORPUS_NDCG="${NDCG_RESULTS[nfcorpus]:-?}"
FIQA_NDCG="${NDCG_RESULTS[fiqa]:-?}"

log ""
log "╔══════════════════════════════════════════════╗"
log "║  ALL DONE — $(ts)"
log "║  Sci=$SCIFACT_NDCG  NF=$NFCORPUS_NDCG  FiQA=$FIQA_NDCG"
log "╚══════════════════════════════════════════════╝"

sms "(OpenClaw) BEIR COMPLETE! Sci=$SCIFACT_NDCG NF=$NFCORPUS_NDCG FiQA=$FIQA_NDCG"

echo "DONE: all phases" > "$STATEFILE"
sync

log ""
log "Press Ctrl+C to exit"
exec bash
