#!/usr/bin/env bash
# run-full-benchmark.sh — Full BEIR benchmark orchestration
# Designed for long-running execution as a detached background process.
#
# Usage:
#   ./run-full-benchmark.sh              # Full 15-dataset benchmark
#   ./run-full-benchmark.sh --paper      # Paper datasets only (SciFact, NFCorpus, FiQA, Arguana, scidocs)
#   ./run-full-benchmark.sh --index-only # Index datasets, don't run benchmarks
#   ./run-full-benchmark.sh --resume     # Resume from last checkpoint
#
# Output:
#   evaluation/results/full-benchmark.json  — combined results
#   evaluation/results/<dataset>/            — per-dataset JSON results
#
# Check progress:
#   cat evaluation/results/status.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/evaluation/results"
INDEX_DIR="${BEIR_LANCEDB_DIR:-$HOME/.openclaw/data/testDbBEIR/lancedb}"
BEIR_DATA_DIR="${BEIR_DATA_DIR:-$SCRIPT_DIR/evaluation/beir-datasets}"
LOG_FILE="$RESULTS_DIR/benchmark.log"

# Paper datasets (5 — for paper revision)
PAPER_DATASETS="scifact,nfcorpus,fiqa,arguana,scidocs"

# Extra datasets for comprehensive evaluation (10 more)
EXTRA_DATASETS="nq,hotpotqa,trec-covid-beir,dbpedia-entity,fever,climate-fever,cqadupstack,quora,msmarco,webis-touche2020"

# All 15 indexable datasets
ALL_DATASETS="$PAPER_DATASETS,$EXTRA_DATASETS"

mkdir -p "$RESULTS_DIR"

# ── Logging ────────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M')] $*" | tee -a "$LOG_FILE"; }
log_section() { log "══════════════════════════════════════════════════"; log "$*"; log "══════════════════════════════════════════════════"; }

# ── Status tracking ───────────────────────────────────────────────────────────

STATUS_FILE="$RESULTS_DIR/status.json"

update_status() {
    local phase="$1"
    local dataset="${2:-}"
    local progress="${3:-}"
    local elapsed="${4:-}"
    echo "{\"phase\":\"$phase\",\"dataset\":\"$dataset\",\"progress\":\"$progress\",\"elapsed\":\"$elapsed\",\"updated\":\"$(date -Iseconds)\"}" > "$STATUS_FILE"
}

# ── Indexing ───────────────────────────────────────────────────────────────────

run_indexing() {
    local datasets="${1:-ALL}"
    log_section "INDEXING: $datasets"
    
    # Set env for spark
    export SPARK_HOST="${SPARK_HOST:-10.99.1.1}"
    export SPARK_BEARER_TOKEN="${SPARK_BEARER_TOKEN:-}"
    export BEIR_DATA_DIR
    export BEIR_LANCEDB_DIR="$INDEX_DIR"
    
    if [ "$datasets" = "ALL" ]; then
        log "Indexing all 15 datasets..."
        npx tsx "$SCRIPT_DIR/scripts/index-beir.ts" 2>&1 | tee -a "$LOG_FILE"
    elif [ "$datasets" = "PAPER" ]; then
        log "Indexing paper datasets: $PAPER_DATASETS"
        for ds in $(echo $PAPER_DATASETS | tr ',' ' '); do
            log "Indexing $ds..."
            update_status "indexing" "$ds" ""
            npx tsx "$SCRIPT_DIR/scripts/index-beir.ts" --dataset "$ds" 2>&1 | tee -a "$LOG_FILE"
        done
    else
        log "Indexing specified: $datasets"
        IFS=',' read -ra DS <<< "$datasets"
        for ds in "${DS[@]}"; do
            log "Indexing $ds..."
            update_status "indexing" "$ds" ""
            npx tsx "$SCRIPT_DIR/scripts/index-beir.ts" --dataset "$ds" 2>&1 | tee -a "$LOG_FILE"
        done
    fi
    log "✅ Indexing complete"
}

# ── Benchmarking ──────────────────────────────────────────────────────────────

run_benchmark() {
    local datasets="${1:-ALL}"
    log_section "BENCHMARKING: $datasets"
    
    export SPARK_HOST="${SPARK_HOST:-10.99.1.1}"
    export SPARK_BEARER_TOKEN="${SPARK_BEARER_TOKEN:-}"
    export BEIR_DATA_DIR
    export BEIR_LANCEDB_DIR="$INDEX_DIR"
    export RESULTS_DIR
    
    if [ "$datasets" = "ALL" ]; then
        datasets="$ALL_DATASETS"
    elif [ "$datasets" = "PAPER" ]; then
        datasets="$PAPER_DATASETS"
    fi
    
    IFS=',' read -ra DS <<< "$datasets"
    for ds in "${DS[@]}"; do
        local out="$RESULTS_DIR/${ds}.json"
        if [ -f "$out" ]; then
            log "⏭️  Skipping $ds (results exist)"
            continue
        fi
        log "Benchmarking $ds..."
        update_status "benchmarking" "$ds" ""
        npx tsx "$SCRIPT_DIR/evaluation/harness/run-benchmark.ts" \
            --datasets "$ds" \
            --configs all \
            --output "$RESULTS_DIR" \
            --db-dir "$INDEX_DIR" \
            2>&1 | tee -a "$LOG_FILE"
        log "✅ $ds complete → $out"
    done
}

# ── Combine results ─────────────────────────────────────────────────────────────

combine_results() {
    log_section "COMBINING RESULTS"
    python3 -c "
import json, os, glob
from datetime import datetime

results = {'timestamp': datetime.now().isoformat(), 'datasets': {}}
for f in sorted(glob.glob('$RESULTS_DIR/*.json')):
    name = os.path.basename(f).replace('.json','')
    if name in ('status', 'full-benchmark'): continue
    with open(f) as fh:
        d = json.load(fh)
        results['datasets'][name] = d

with open('$RESULTS_DIR/full-benchmark.json', 'w') as out:
    json.dump(results, out, indent=2)

# Summary table
print()
print('RESULTS SUMMARY')
print('=' * 70)
print(f'{'Dataset':<25} {'NDCG@10':>10} {'MAP@10':>10} {'Recall@10':>10} {'Latency':>10}')
print('-' * 70)
for name, d in sorted(results['datasets'].items()):
    configs = d.get('configs', [])
    best = max(configs, key=lambda c: c.get('ndcg10', 0)) if configs else {}
    print(f'{name:<25} {best.get(\"ndcg10\", 0):>10.4f} {best.get(\"map10\", 0):>10.4f} {best.get(\"recall10\", 0):>10.4f} {best.get(\"latencyMs\", 0):>10.0f}ms')
"
}

# ── Main ───────────────────────────────────────────────────────────────────────

MODE="${1:-PAPER}"

case "$MODE" in
    --paper)
        log_section "PAPER BENCHMARK (5 datasets)"
        run_indexing PAPER
        run_benchmark PAPER
        combine_results
        ;;
    --index-only)
        log_section "INDEXING ONLY (all 15 datasets)"
        run_indexing ALL
        ;;
    --all)
        log_section "FULL BENCHMARK (15 datasets —预计 3-7 days)"
        run_indexing ALL
        run_benchmark ALL
        combine_results
        ;;
    --resume)
        log_section "RESUMING from checkpoint"
        run_benchmark ALL
        combine_results
        ;;
    *)
        echo "Usage: $0 [--paper|--index-only|--all|--resume]"
        ;;
esac

log "🎉 Done!"
