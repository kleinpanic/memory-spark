#!/usr/bin/env bash
#
# Memory-Spark: Full Reindex + Benchmark Pipeline
# ================================================
# Robust version: systemd-managed, verbose crash logging, signal traps.
#
# Via systemd (preferred):
#   systemctl --user daemon-reload
#   systemctl --user start ms-reindex
#   journalctl --user -u ms-reindex -f
#
# Manual:
#   setsid nohup bash scripts/run-after-reindex.sh > /tmp/ms-pipeline-manual.log 2>&1 &
#

set -uo pipefail

PROJECT="$HOME/codeWS/TypeScript/memory-spark"
cd "$PROJECT"

export MEMORY_SPARK_DATA_DIR=./test-data

# Source env vars (SPARK_HOST, SPARK_BEARER_TOKEN) — needed when running via systemd
[[ -f "$HOME/.openclaw/.env" ]] && set -a && source "$HOME/.openclaw/.env" && set +a
[[ -f .env ]] && set -a && source .env && set +a

SMS="$HOME/.openclaw/hooks/sms-alert.sh"
LOGFILE="evaluation/results/pipeline-$(date +%Y%m%d-%H%M).log"
STATEFILE="/tmp/ms-pipeline.state"
mkdir -p evaluation/results

# ── Helpers ───────────────────────────────────────────────────────────────

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOGFILE"; }

sms() {
  local msg="$1"
  # Clear cooldown so crash/completion SMSes always go through
  rm -f /tmp/.openclaw-sms-cooldown
  log "SMS: $msg"
  if bash "$SMS" "$msg" >> "$LOGFILE" 2>&1; then
    log "SMS sent OK"
  else
    log "WARN: sms-alert.sh returned non-zero (email/himalaya issue?)"
    # Try a second time after a moment
    sleep 5 && bash "$SMS" "$msg" >> "$LOGFILE" 2>&1 || true
  fi
}

die() {
  local msg="$1"
  log "FATAL: $msg"
  echo "FAILED: $msg" > "$STATEFILE"
  sms "(OpenClaw Alert) memory-spark FAILED: $msg. Check: journalctl --user -u ms-reindex"
  exit 1
}

# ── Signal Traps ──────────────────────────────────────────────────────────
# Runs on ANY exit. Captures exit code and last signal for diagnosis.

_CURRENT_PHASE="startup"
_LAST_FILE=""

update_phase() { _CURRENT_PHASE="$1"; echo "PHASE: $1" > "$STATEFILE"; log "→ Phase: $1"; }
update_file()  { _LAST_FILE="$1";  echo "FILE: $1" >> "$STATEFILE"; }

on_exit() {
  local EC=$?
  local SIGNAL_NAME=""
  case $EC in
    130) SIGNAL_NAME=" (SIGINT/Ctrl-C)" ;;
    137) SIGNAL_NAME=" (SIGKILL — killed by OS/cgroup/OOM)" ;;
    143) SIGNAL_NAME=" (SIGTERM)" ;;
    *) ;;
  esac

  if [[ $EC -ne 0 ]]; then
    log ""
    log "╔══════════════════════════════════════════════╗"
    log "║  PIPELINE CRASHED — exit code ${EC}${SIGNAL_NAME}"
    log "║  Phase:     ${_CURRENT_PHASE}"
    log "║  Last file: ${_LAST_FILE}"
    log "║  Time:      $(ts)"
    log "║  State:     $(cat "$STATEFILE" 2>/dev/null | tr '\n' '|')"
    log "╚══════════════════════════════════════════════╝"
    # Flush all buffers
    sync
    sms "(OpenClaw Alert) memory-spark CRASHED (exit ${EC}${SIGNAL_NAME}). Phase: ${_CURRENT_PHASE}. Last file: ${_LAST_FILE}. Check: journalctl --user -u ms-reindex"
  fi
}

trap on_exit EXIT
trap 'log "Caught SIGTERM — shutting down"; exit 143' TERM
trap 'log "Caught SIGINT — shutting down"; exit 130' INT
trap 'log "Caught SIGHUP — continuing (detached run)"; ' HUP

# ── Preflight ─────────────────────────────────────────────────────────────

log "═══════════════════════════════════════════════════"
log "  Memory-Spark Full Pipeline"
log "═══════════════════════════════════════════════════"
log "PID:      $$"
log "PPID:     $PPID"
log "User:     $(whoami)"
log "Project:  $PROJECT"
log "Data dir: $MEMORY_SPARK_DATA_DIR"
log "Log:      $LOGFILE"
log "State:    $STATEFILE"
log "Cgroup:   $(cat /proc/$$/cgroup 2>/dev/null | head -3 | tr '\n' ' ')"
log ""

command -v npx      &>/dev/null || die "npx not found"
command -v himalaya &>/dev/null || die "himalaya not found"
[[ -x "$SMS" ]]                 || die "sms-alert.sh missing or not executable"
[[ -f scripts/reindex-benchmark.ts ]] || die "reindex script missing"
[[ -f evaluation/benchmark-v2.ts ]]   || die "benchmark script missing"

# Check Spark embed service is alive
log "Checking Spark embed service..."
if ! curl -sf --max-time 10 "http://$(grep -oP 'SPARK_HOST=\K\S+' .env 2>/dev/null || echo 'localhost'):8080/health" &>/dev/null; then
  # Non-fatal — just warn. The indexer handles retries internally.
  log "WARN: Spark embed service health check failed or unreachable — proceeding anyway"
else
  log "Spark embed service: OK"
fi

log "Preflight OK"
echo "RUNNING: preflight OK" > "$STATEFILE"

# ── Phase 1: Wipe & Reindex ──────────────────────────────────────────────

update_phase "reindex"

log ""
log "═══════════════════════════════════════════════════"
log "  Phase 1: Clean Reindex"
log "═══════════════════════════════════════════════════"

if [[ -d "$MEMORY_SPARK_DATA_DIR/lancedb" ]]; then
  STALE_SIZE=$(du -sh "$MEMORY_SPARK_DATA_DIR/lancedb" | cut -f1)
  log "Wiping stale index ($STALE_SIZE)..."
  rm -rf "$MEMORY_SPARK_DATA_DIR/lancedb"
fi
mkdir -p "$MEMORY_SPARK_DATA_DIR/lancedb"

log "Starting fresh reindex... (logging to both stdout and $LOGFILE)"
REINDEX_START=$(date +%s)

# Run reindex; each file processed updates state for crash diagnosis
NODE_OPTIONS="--max-old-space-size=4096" npx tsx scripts/reindex-benchmark.ts 2>&1 \
  | while IFS= read -r line; do
      echo "$line"
      echo "$line" >> "$LOGFILE"
      # Track the last file being processed for crash diagnosis
      if [[ "$line" =~ memory-spark:\ ([^[:space:]]+)\ →\ ([0-9]+)\ chunks ]]; then
        echo "FILE: ${BASH_REMATCH[1]}" > "$STATEFILE"
      fi
    done
REINDEX_EXIT=${PIPESTATUS[0]}

REINDEX_END=$(date +%s)
REINDEX_DURATION=$(( (REINDEX_END - REINDEX_START) / 60 ))

if [[ "$REINDEX_EXIT" -ne 0 ]]; then
  die "Reindex failed (exit $REINDEX_EXIT) after ${REINDEX_DURATION}m. Last file: ${_LAST_FILE}"
fi

CHUNK_COUNT=$(grep -oP 'Total chunks: \K\d+' "$LOGFILE" | tail -1 || echo "?")
log ""
log "✅ Reindex complete: ${CHUNK_COUNT} chunks in ${REINDEX_DURATION}m"
echo "DONE: reindex ${CHUNK_COUNT} chunks ${REINDEX_DURATION}m" > "$STATEFILE"

sms "(OpenClaw Alert) memory-spark reindex done (${CHUNK_COUNT} chunks, ${REINDEX_DURATION}m). Running FULL benchmark suite now — all tiers, all ablations, all BEIR datasets. ETA 1-2h."

TIMESTAMP=$(date +%Y%m%d-%H%M)
RESULTS_DIR="evaluation/results/run-${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# ── Phase 2A: Full Benchmark v2 ──────────────────────────────────────────

update_phase "benchmark-v2"

log ""
log "═══════════════════════════════════════════════════"
log "  Phase 2A: Benchmark v2 — All Tiers + HyDE + Judge"
log "═══════════════════════════════════════════════════"

FULL_LOG="$RESULTS_DIR/benchmark-v2-full.log"

NODE_OPTIONS="--max-old-space-size=4096" npx tsx evaluation/benchmark-v2.ts \
  --skip-preflight --hyde --judge 2>&1 | tee "$FULL_LOG" | tee -a "$LOGFILE"
FULL_EXIT=${PIPESTATUS[0]}

[[ "$FULL_EXIT" -ne 0 ]] && log "⚠️  Benchmark v2 exited $FULL_EXIT (continuing to BEIR)"

cp evaluation/results/benchmark-v2-*.json "$RESULTS_DIR/" 2>/dev/null || true

VECTOR_NDCG=$(grep -A12 'Vector-Only'           "$FULL_LOG" | grep 'ndcg' | head -1 | awk '{print $NF}' || echo "?")
FTS_NDCG=$(   grep -A12 'FTS-Only'              "$FULL_LOG" | grep 'ndcg' | head -1 | awk '{print $NF}' || echo "?")
HYBRID_NDCG=$(grep -A12 'Hybrid (No Reranker)'  "$FULL_LOG" | grep 'ndcg' | head -1 | awk '{print $NF}' || echo "?")
FULL_NDCG=$(  grep -A12 'Full Pipeline'          "$FULL_LOG" | grep 'ndcg' | head -1 | awk '{print $NF}' || echo "?")
HYDE_NDCG=$(  grep -A12 'HyDE'                  "$FULL_LOG" | grep 'ndcg' | head -1 | awk '{print $NF}' || echo "?")
JUDGE_AVG=$(  grep -oP 'Judge.*?:\s*\K[0-9.]+' "$FULL_LOG" | tail -1 || echo "?")
COMPOSITE=$(  grep -oP 'Composite.*?:\s*\K[0-9.]+' "$FULL_LOG" | tail -1 || echo "?")

log "Custom corpus: Vec=${VECTOR_NDCG} FTS=${FTS_NDCG} Hyb=${HYBRID_NDCG} Full=${FULL_NDCG} HyDE=${HYDE_NDCG} Judge=${JUDGE_AVG}/5"

sms "(OpenClaw Alert) Phase 2A done. Custom NDCG@10: Vec=${VECTOR_NDCG} Full=${FULL_NDCG} HyDE=${HYDE_NDCG} Judge=${JUDGE_AVG}/5. Starting BEIR..."

# ── Phase 2B-D: BEIR ─────────────────────────────────────────────────────

run_beir() {
  local DATASET="$1"
  update_phase "beir-${DATASET}"
  log ""
  log "═══════════════════════════════════════════════════"
  log "  BEIR — ${DATASET}"
  log "═══════════════════════════════════════════════════"
  local BDIR="$RESULTS_DIR/beir-${DATASET}.log"
  NODE_OPTIONS="--max-old-space-size=4096" npx tsx evaluation/beir-benchmark.ts \
    --dataset "$DATASET" --batch-size 32 2>&1 | tee "$BDIR" | tee -a "$LOGFILE"
  local EC=${PIPESTATUS[0]}
  local NDCG
  NDCG=$(grep -oP 'NDCG@10[:\s]+\K[0-9.]+' "$BDIR" | tail -1 || echo "?")
  log "${DATASET} NDCG@10: ${NDCG} (exit ${EC})"
  echo "$NDCG"
}

SCIFACT_NDCG=$(run_beir scifact)
NFCORPUS_NDCG=$(run_beir nfcorpus)
FIQA_NDCG=$(run_beir fiqa)

# ── Phase 3: Report ──────────────────────────────────────────────────────

update_phase "report"

REPORT="$RESULTS_DIR/REPORT.md"
cat > "$REPORT" <<REPORT_EOF
# memory-spark Benchmark Report
**Date:** $(date '+%Y-%m-%d %H:%M %Z')
**Index:** ${CHUNK_COUNT} chunks | Reindex: ${REINDEX_DURATION}m
**Architecture:** Hierarchical Parent-Child, Hybrid Search, HyDE, Cross-Encoder Rerank, LLM-as-Judge

## Custom Corpus (139 queries)

| Config | NDCG@10 |
|--------|---------|
| Vector-Only | ${VECTOR_NDCG} |
| FTS-Only | ${FTS_NDCG} |
| Hybrid (No Reranker) | ${HYBRID_NDCG} |
| Full Pipeline | ${FULL_NDCG} |
| HyDE | ${HYDE_NDCG} |
| LLM-Judge avg | ${JUDGE_AVG}/5 |
| Composite (70/30) | ${COMPOSITE} |

## BEIR Benchmarks

| Dataset | NDCG@10 |
|---------|---------|
| SciFact | ${SCIFACT_NDCG} |
| NFCorpus | ${NFCORPUS_NDCG} |
| FiQA | ${FIQA_NDCG} |

## Files
- Full log: \`$(basename "$LOGFILE")\`
- Results: \`$(basename "$RESULTS_DIR")/\`
REPORT_EOF

echo "DONE: all phases complete" > "$STATEFILE"

log ""
log "╔══════════════════════════════════════════════╗"
log "║  ALL DONE — $(ts)"
log "║  Vec=${VECTOR_NDCG}  Full=${FULL_NDCG}  HyDE=${HYDE_NDCG}"
log "║  Sci=${SCIFACT_NDCG}  NF=${NFCORPUS_NDCG}  FiQA=${FIQA_NDCG}"
log "║  Report: ${REPORT}"
log "╚══════════════════════════════════════════════╝"

cp "$LOGFILE" "$RESULTS_DIR/" 2>/dev/null || true

sms "(OpenClaw Alert) memory-spark COMPLETE! Custom: Vec=${VECTOR_NDCG} Full=${FULL_NDCG} HyDE=${HYDE_NDCG} Judge=${JUDGE_AVG}/5. BEIR: Sci=${SCIFACT_NDCG} NF=${NFCORPUS_NDCG} FiQA=${FIQA_NDCG}. Report: evaluation/results/run-${TIMESTAMP}/REPORT.md"
