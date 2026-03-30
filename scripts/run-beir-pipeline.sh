#!/bin/bash
# BEIR Benchmark Pipeline — Index + Benchmark with text notification
#
# Usage:
#   ./scripts/run-beir-pipeline.sh [dataset]
#
# Datasets: scifact (default), nfcorpus, fiqa, all
#
# Runs in tmux session 'beir-pipeline' with full logging.

set -euo pipefail

DATASET="${1:-scifact}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/evaluation/results"
LOG_FILE="$LOG_DIR/pipeline-$(date +%Y%m%d-%H%M%S).log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Check if already running in tmux
if [ -n "${TMUX:-}" ]; then
    echo "[INFO] Already in tmux session, proceeding..."
else
    # Start or attach to tmux session
    if tmux has-session -t beir-pipeline 2>/dev/null; then
        echo "[INFO] Attaching to existing beir-pipeline session"
        tmux attach -t beir-pipeline
        exit 0
    else
        echo "[INFO] Starting new tmux session 'beir-pipeline'"
        tmux new-session -d -s beir-pipeline -x 200 -y 50 \
            "cd '$PROJECT_ROOT' && bash '$SCRIPT_DIR/run-beir-pipeline.sh' '$DATASET'"
        echo "[INFO] Session started. Attach with: tmux attach -t beir-pipeline"
        echo "[INFO] Logs: $LOG_FILE"
        exit 0
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Functions
# ─────────────────────────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

send_notification() {
    local message="$1"
    log "Sending notification: $message"
    
    # Try BlueBubbles via OpenClaw message tool
    if command -v openclaw &>/dev/null; then
        openclaw message send --to klein --message "$message" 2>/dev/null || true
    fi
    
    # Fallback: write to notification file for pickup
    echo "$message" > "$PROJECT_ROOT/evaluation/results/.notification-pending"
}

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

log "═══════════════════════════════════════════════════════════════"
log "  BEIR Benchmark Pipeline"
log "  Dataset: $DATASET"
log "═══════════════════════════════════════════════════════════════"

cd "$PROJECT_ROOT"

# Phase 1: Build
log ""
log "▶ Phase 1: Build"
npm run build 2>&1 | tee -a "$LOG_FILE"

# Phase 2: Index BEIR corpus
log ""
log "▶ Phase 2: Index BEIR corpus ($DATASET)"

if [ "$DATASET" = "all" ]; then
    DATASETS="scifact nfcorpus fiqa"
else
    DATASETS="$DATASET"
fi

for ds in $DATASETS; do
    log "  Indexing $ds..."
    BEIR_LANCEDB_DIR=/home/node/.openclaw/data/testDbBEIR/lancedb \
        npx tsx evaluation/index-beir.ts --dataset "$ds" --resume 2>&1 | tee -a "$LOG_FILE"
done

# Phase 3: Run benchmarks
log ""
log "▶ Phase 3: Run A/B/C/D/E/F/G benchmarks"

for ds in $DATASETS; do
    log "  Benchmarking $ds..."
    BEIR_LANCEDB_DIR=/home/node/.openclaw/data/testDbBEIR/lancedb \
        npx tsx evaluation/run-beir-bench.ts --dataset "$ds" 2>&1 | tee -a "$LOG_FILE"
done

# Phase 4: Summary
log ""
log "═══════════════════════════════════════════════════════════════"
log "  Pipeline Complete"
log "═══════════════════════════════════════════════════════════════"

# Find latest summary files
log ""
log "Results summary:"
for ds in $DATASETS; do
    latest_summary=$(ls -t evaluation/results/beir-${ds}-summary-*.json 2>/dev/null | head -1)
    if [ -n "$latest_summary" ]; then
        log "  $ds: $latest_summary"
        cat "$latest_summary" | tee -a "$LOG_FILE"
    fi
done

# Send notification
send_notification "✅ BEIR Benchmark Complete

Dataset: $DATASET
Configs tested: A/B/C/D/E/F/G

Results saved to: evaluation/results/
Log file: $LOG_FILE

Attach to review: tmux attach -t beir-pipeline"

log ""
log "✅ Pipeline complete. Logs: $LOG_FILE"
log "Press Ctrl+C to exit, or 'tmux detach' to leave running."

# Keep session alive for review
exec bash
