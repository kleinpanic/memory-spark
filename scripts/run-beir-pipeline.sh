#!/bin/bash
# BEIR Benchmark Pipeline — Index ALL BEIR + Benchmark A-G + Notifications
#
# Datasets: scifact (5183), nfcorpus (3633), fiqa (57638) = 66,454 docs total
# Configs: A/B/C/D/E/F/G with NDCG@10, MRR, Recall@10, MAP@10, p95 latency
#
# Usage:
#   ./scripts/run-beir-pipeline.sh
#
# Notifications:
#   1. Indexing complete / Benchmarking started
#   2. Each dataset benchmark complete with results
#   3. Fully done with summary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/evaluation/results"
LOG_FILE="$LOG_DIR/pipeline-$(date +%Y%m%d-%H%M%S).log"
BEIR_LANCEDB_DIR="${BEIR_LANCEDB_DIR:-/home/node/.openclaw/data/testDbBEIR/lancedb}"

# All BEIR datasets
DATASETS="scifact nfcorpus fiqa"

mkdir -p "$LOG_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Functions
# ─────────────────────────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

notify() {
    local message="$1"
    log "NOTIFICATION: $message"
    
    # Write to notification file for OpenClaw to pick up
    local notify_file="$LOG_DIR/.notification"
    echo "$(date -Iseconds)|$message" > "$notify_file"
    
    # Also try to send via message tool if available
    if command -v openclaw &>/dev/null; then
        openclaw message send --to klein --message "$message" 2>/dev/null || true
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Start in tmux if not already
# ─────────────────────────────────────────────────────────────────────────────

if [ -z "${TMUX:-}" ]; then
    if tmux has-session -t beir-pipeline 2>/dev/null; then
        echo "[INFO] Attaching to existing beir-pipeline session"
        exec tmux attach -t beir-pipeline
    else
        echo "[INFO] Starting new tmux session 'beir-pipeline'"
        tmux new-session -d -s beir-pipeline -x 200 -y 50 \
            "cd '$PROJECT_ROOT' && bash '$SCRIPT_DIR/run-beir-pipeline.sh'"
        echo "[INFO] Session started. Attach with: tmux attach -t beir-pipeline"
        echo "[INFO] Logs: $LOG_FILE"
        exit 0
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

cd "$PROJECT_ROOT"

log "═══════════════════════════════════════════════════════════════"
log "  BEIR Benchmark Pipeline"
log "  Datasets: $DATASETS"
log "  Total docs: 66,454"
log "  Configs: A/B/C/D/E/F/G"
log "═══════════════════════════════════════════════════════════════"

# Phase 1: Build
log ""
log "▶ Phase 1: Build"
npm run build 2>&1 | tee -a "$LOG_FILE" || {
    log "❌ Build failed!"
    notify "❌ BEIR Pipeline FAILED at build stage"
    exit 1
}
log "✅ Build complete"

# Phase 2: Index ALL BEIR datasets
log ""
log "▶ Phase 2: Index ALL BEIR datasets to testDbBEIR"
log "  Target: $BEIR_LANCEDB_DIR"

TOTAL_DOCS=0
for ds in $DATASETS; do
    log ""
    log "  Indexing $ds..."
    BEIR_LANCEDB_DIR="$BEIR_LANCEDB_DIR" \
        npx tsx evaluation/index-beir.ts --dataset "$ds" --resume 2>&1 | tee -a "$LOG_FILE"
    # Count docs
    docs=$(wc -l < "evaluation/beir-datasets/$ds/corpus.jsonl")
    TOTAL_DOCS=$((TOTAL_DOCS + docs))
done

log ""
log "✅ Indexing complete: $TOTAL_DOCS docs indexed"
notify "✅ BEIR Indexing Complete

Indexed: $TOTAL_DOCS documents across 3 datasets
- scifact: 5,183 docs
- nfcorpus: 3,633 docs
- fiqa: 57,638 docs

Database: testDbBEIR
Benchmarking A/B/C/D/E/F/G starting now..."

# Phase 3: Run benchmarks for each dataset
log ""
log "▶ Phase 3: Run A/B/C/D/E/F/G benchmarks"

for ds in $DATASETS; do
    log ""
    log "─────────────────────────────────────────────────────────────"
    log "  Benchmarking: $ds"
    log "─────────────────────────────────────────────────────────────"
    
    BEIR_LANCEDB_DIR="$BEIR_LANCEDB_DIR" \
        npx tsx evaluation/run-beir-bench.ts --dataset "$ds" 2>&1 | tee -a "$LOG_FILE"
    
    # Find latest summary
    latest_summary=$(ls -t evaluation/results/beir-${ds}-summary-*.json 2>/dev/null | head -1)
    if [ -n "$latest_summary" ]; then
        log "  Results: $latest_summary"
        
        # Extract key metrics for notification
        notify "📊 BEIR $ds Complete

$(cat "$latest_summary" | jq -r '.results[] | "Config \(.config): NDCG=\(.ndcg | tostring | .[0:6]) MRR=\(.mrr | tostring | .[0:6]) Recall=\(.recall | tostring | .[0:6])"' 2>/dev/null || cat "$latest_summary")"
    fi
done

# Phase 4: Final summary
log ""
log "═══════════════════════════════════════════════════════════════"
log "  Pipeline Complete"
log "═══════════════════════════════════════════════════════════════"

# Aggregate results
log ""
log "Final Results Summary:"
for ds in $DATASETS; do
    latest=$(ls -t evaluation/results/beir-${ds}-summary-*.json 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        log ""
        log "=== $ds ==="
        cat "$latest" | tee -a "$LOG_FILE"
    fi
done

# Find telemetry files
log ""
log "Telemetry files:"
ls -la evaluation/results/*telemetry*.json 2>/dev/null | tee -a "$LOG_FILE" || true

notify "✅ BEIR Benchmark Pipeline COMPLETE

All 3 datasets benchmarked with A-G configs:
- scifact (1,109 queries)
- nfcorpus (3,237 queries)
- fiqa (6,648 queries)

Results saved to: evaluation/results/
Log file: $LOG_FILE

Attach to review: tmux attach -t beir-pipeline"

log ""
log "✅ Pipeline complete."
log "   Logs: $LOG_FILE"
log "   Results: evaluation/results/"
log ""
log "Press Ctrl+C to exit, or 'tmux detach' to leave running."

# Keep session alive
exec bash
