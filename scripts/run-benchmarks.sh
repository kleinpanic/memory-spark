#!/usr/bin/env bash
#
# Memory-Spark: Full Benchmark Suite
# ===================================
# Runs ALL benchmarks: v2 (all tiers, all ablations, HyDE, Judge) + 3 BEIR datasets.
# Writes everything to evaluation/results/run-<timestamp>/.
# Texts Klein via sms-alert.sh at each phase boundary.
#
# Usage:  bash scripts/run-benchmarks.sh [chunk_count] [reindex_duration_min]
#

set -uo pipefail

PROJECT="$HOME/codeWS/TypeScript/memory-spark"
cd "$PROJECT"

export MEMORY_SPARK_DATA_DIR=./test-data
SMS="$HOME/.openclaw/hooks/sms-alert.sh"
CHUNK_COUNT="${1:-?}"
REINDEX_DURATION="${2:-?}"

TIMESTAMP=$(date +%Y%m%d-%H%M)
RESULTS_DIR="evaluation/results/run-${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
LOGFILE="$RESULTS_DIR/pipeline.log"

# ── Helpers ───────────────────────────────────────────────────────────────

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOGFILE"; }

sms() {
  rm -f /tmp/.openclaw-sms-cooldown
  bash "$SMS" "$1" 2>&1 | tee -a "$LOGFILE" || log "WARN: sms failed"
}

die() {
  log "FATAL: $1"
  sms "memory-spark benchmarks FAILED: $1"
  exit 1
}

trap 'EC=$?; if [ $EC -ne 0 ]; then sms "memory-spark benchmarks crashed (exit $EC). tmux attach -t bench"; fi' EXIT

log "═══════════════════════════════════════════════════"
log "  Memory-Spark Full Benchmark Suite"
log "═══════════════════════════════════════════════════"
log "Index: ${CHUNK_COUNT} chunks (reindexed in ${REINDEX_DURATION}m)"
log "Results: $RESULTS_DIR"
log ""

sms "memory-spark reindex done (${CHUNK_COUNT} chunks, ${REINDEX_DURATION}m). Running FULL benchmark suite — all tiers, all ablations, 3 BEIR datasets. ETA ~1-2h."

# ── Phase A: Full Benchmark v2 — All Tiers, All Ablations, HyDE, Judge ──

log "═══════════════════════════════════════════════════"
log "  Phase A: Benchmark v2 — All Tiers + HyDE + Judge"
log "  Tier 1: 10+ Retrieval Ablations (NDCG/MRR/MAP)"
log "  Tier 2: Pipeline Integration (garbage, security)"
log "  Tier 3: Pool Isolation (agent scoping)"
log "  Tier 4: Parent-Child Expansion"
log "═══════════════════════════════════════════════════"

V2_LOG="$RESULTS_DIR/benchmark-v2-full.log"

npx tsx evaluation/benchmark-v2.ts \
  --skip-preflight --hyde --judge 2>&1 | tee "$V2_LOG"
V2_EXIT=${PIPESTATUS[0]}

# Grab any JSON artifacts
cp evaluation/results/benchmark-v2-*.json "$RESULTS_DIR/" 2>/dev/null || true

if [[ "$V2_EXIT" -ne 0 ]]; then
  log "⚠️  Benchmark v2 exited with $V2_EXIT (continuing to BEIR)"
fi

# Extract headline numbers
extract() { grep -A15 "$1" "$V2_LOG" | grep "$2" | head -1 | awk '{print $NF}'; }

VECTOR_NDCG=$(extract 'Vector-Only' 'ndcg' || echo "?")
FTS_NDCG=$(extract 'FTS-Only' 'ndcg' || echo "?")
HYBRID_NDCG=$(extract 'Hybrid (No Reranker)' 'ndcg' || echo "?")
HYBRID_NO_TD=$(extract 'Temporal Decay' 'ndcg' || echo "?")
HYBRID_NO_SW=$(extract 'Source Weighting' 'ndcg' || echo "?")
HYBRID_NO_MMR=$(extract 'MMR Diversity' 'ndcg' || echo "?")
FULL_NDCG=$(extract 'Full Pipeline' 'ndcg' || echo "?")
HYDE_NDCG=$(extract 'HyDE' 'ndcg' || echo "?")
MMR_05=$(extract 'λ=0.5' 'ndcg' || echo "?")
MMR_09=$(extract 'λ=0.9' 'ndcg' || echo "?")
PROD_NDCG=$(extract 'Production Mirror' 'ndcg' || echo "?")
JUDGE_AVG=$(grep -oP 'Judge.*?:\s*\K[0-9.]+' "$V2_LOG" | tail -1 || echo "?")
COMPOSITE=$(grep -oP 'Composite.*?:\s*\K[0-9.]+' "$V2_LOG" | tail -1 || echo "?")

log ""
log "Custom Corpus NDCG@10 Ablation Matrix:"
log "  A. Vector-Only:          ${VECTOR_NDCG}"
log "  B. FTS-Only:             ${FTS_NDCG}"
log "  C. Hybrid (no reranker): ${HYBRID_NDCG}"
log "  D. Hybrid − Temp Decay:  ${HYBRID_NO_TD}"
log "  E. Hybrid − Src Weight:  ${HYBRID_NO_SW}"
log "  F. Hybrid − MMR:         ${HYBRID_NO_MMR}"
log "  G. Full Pipeline:        ${FULL_NDCG}"
log "  H. Full + HyDE:          ${HYDE_NDCG}"
log "  I. MMR λ=0.5:            ${MMR_05}"
log "  J. MMR λ=0.9:            ${MMR_09}"
log "  K. Production Mirror:    ${PROD_NDCG}"
log "  Judge: ${JUDGE_AVG}/5 | Composite: ${COMPOSITE}"

sms "Phase A done! Custom NDCG@10: Vec=${VECTOR_NDCG} Hyb=${HYBRID_NDCG} Full=${FULL_NDCG} HyDE=${HYDE_NDCG} Prod=${PROD_NDCG} Judge=${JUDGE_AVG}/5. Running 3 BEIR datasets next..."

# ── Phase B: BEIR SciFact ─────────────────────────────────────────────────

log ""
log "═══════════════════════════════════════════════════"
log "  Phase B: BEIR — SciFact (5,183 docs, scientific)"
log "═══════════════════════════════════════════════════"

SCIFACT_LOG="$RESULTS_DIR/beir-scifact.log"
npx tsx evaluation/beir-benchmark.ts \
  --dataset scifact --batch-size 32 2>&1 | tee "$SCIFACT_LOG"
SCIFACT_EXIT=${PIPESTATUS[0]}
SCIFACT_NDCG=$(grep -oP 'NDCG@10[:\s]+\K[0-9.]+' "$SCIFACT_LOG" | tail -1 || echo "?")
log "SciFact NDCG@10: ${SCIFACT_NDCG} (exit: $SCIFACT_EXIT)"

# ── Phase C: BEIR NFCorpus ────────────────────────────────────────────────

log ""
log "═══════════════════════════════════════════════════"
log "  Phase C: BEIR — NFCorpus (3,633 docs, medical)"
log "═══════════════════════════════════════════════════"

NFCORPUS_LOG="$RESULTS_DIR/beir-nfcorpus.log"
npx tsx evaluation/beir-benchmark.ts \
  --dataset nfcorpus --batch-size 32 2>&1 | tee "$NFCORPUS_LOG"
NFCORPUS_EXIT=${PIPESTATUS[0]}
NFCORPUS_NDCG=$(grep -oP 'NDCG@10[:\s]+\K[0-9.]+' "$NFCORPUS_LOG" | tail -1 || echo "?")
log "NFCorpus NDCG@10: ${NFCORPUS_NDCG} (exit: $NFCORPUS_EXIT)"

# ── Phase D: BEIR FiQA ───────────────────────────────────────────────────

log ""
log "═══════════════════════════════════════════════════"
log "  Phase D: BEIR — FiQA (57,638 docs, financial)"
log "═══════════════════════════════════════════════════"

FIQA_LOG="$RESULTS_DIR/beir-fiqa.log"
npx tsx evaluation/beir-benchmark.ts \
  --dataset fiqa --batch-size 32 2>&1 | tee "$FIQA_LOG"
FIQA_EXIT=${PIPESTATUS[0]}
FIQA_NDCG=$(grep -oP 'NDCG@10[:\s]+\K[0-9.]+' "$FIQA_LOG" | tail -1 || echo "?")
log "FiQA NDCG@10: ${FIQA_NDCG} (exit: $FIQA_EXIT)"

# ── Phase E: Generate Report ─────────────────────────────────────────────

log ""
REPORT="$RESULTS_DIR/REPORT.md"

cat > "$REPORT" <<REPORT_EOF
# memory-spark Benchmark Report
**Date:** $(date '+%Y-%m-%d %H:%M %Z')
**Index:** ${CHUNK_COUNT} chunks | Reindex time: ${REINDEX_DURATION}m
**Architecture:** Hierarchical Parent-Child, Hybrid Search, HyDE, Cross-Encoder Rerank
**Embedding:** nvidia/llama-embed-nemotron-8b (4096d) on Spark (127.0.0.1:18091)
**Reranker:** nvidia/llama-nemotron-rerank-1b-v2 on Spark (127.0.0.1:18096)
**Judge:** Nemotron-Super-120B on Spark (127.0.0.1:18080)

---

## 1. Custom Corpus — Retrieval Ablation Matrix (NDCG@10)

139 queries, 116 corpus docs, 9 agents.

| # | Configuration | NDCG@10 | Notes |
|---|--------------|---------|-------|
| A | Vector-Only | ${VECTOR_NDCG} | Pure cosine similarity, no pipeline |
| B | FTS-Only | ${FTS_NDCG} | Pure BM25, no pipeline |
| C | Hybrid (No Reranker) | ${HYBRID_NDCG} | Vector+FTS merged, no cross-encoder |
| D | Hybrid − Temporal Decay | ${HYBRID_NO_TD} | Ablation: temporal decay disabled |
| E | Hybrid − Source Weighting | ${HYBRID_NO_SW} | Ablation: source weights disabled |
| F | Hybrid − MMR Diversity | ${HYBRID_NO_MMR} | Ablation: MMR disabled |
| G | Full Pipeline | ${FULL_NDCG} | Hybrid+TD+SW+MMR+Rerank |
| H | Full + HyDE | ${HYDE_NDCG} | Hypothetical Document Embeddings |
| I | MMR λ=0.5 (diverse) | ${MMR_05} | More diversity, less relevance |
| J | MMR λ=0.9 (relevant) | ${MMR_09} | Less diversity, more relevance |
| K | Production Mirror | ${PROD_NDCG} | Multi-pool, exact prod code path |

### LLM-as-Judge (Nemotron-Super-120B)
- **Mean Judge Score:** ${JUDGE_AVG}/5
- **Composite (0.7·BEIR + 0.3·Judge):** ${COMPOSITE}

---

## 2. BEIR Standard Benchmarks (NDCG@10)

Cross-domain retrieval quality on public IR datasets.

| Dataset | Domain | Docs | NDCG@10 | Exit |
|---------|--------|------|---------|------|
| SciFact | Scientific claims | 5,183 | ${SCIFACT_NDCG} | ${SCIFACT_EXIT} |
| NFCorpus | Medical/nutrition | 3,633 | ${NFCORPUS_NDCG} | ${NFCORPUS_EXIT} |
| FiQA | Financial QA | 57,638 | ${FIQA_NDCG} | ${FIQA_EXIT} |

---

## 3. Tier 2-4 Results
See \`benchmark-v2-full.log\` for detailed output:
- **Tier 2:** Pipeline Integration (garbage rejection, security filters, token budgets)
- **Tier 3:** Pool Isolation (agent-scoped retrieval, shared vs private pools)
- **Tier 4:** Parent-Child Expansion (schema validation, size ratios, is_parent exclusion)

---

## 4. Files in This Run
| File | Description |
|------|-------------|
| \`REPORT.md\` | This summary |
| \`benchmark-v2-full.log\` | Full Tier 1-4 + HyDE + Judge output |
| \`beir-scifact.log\` | SciFact BEIR results |
| \`beir-nfcorpus.log\` | NFCorpus BEIR results |
| \`beir-fiqa.log\` | FiQA BEIR results |
| \`pipeline.log\` | End-to-end pipeline log |
| \`*.json\` | Machine-readable benchmark results |
REPORT_EOF

log "═══════════════════════════════════════════════════"
log "  ALL DONE ($(ts))"
log "═══════════════════════════════════════════════════"
log ""
log "CUSTOM CORPUS NDCG@10:"
log "  A.Vec=${VECTOR_NDCG}  B.FTS=${FTS_NDCG}  C.Hyb=${HYBRID_NDCG}"
log "  G.Full=${FULL_NDCG}  H.HyDE=${HYDE_NDCG}  K.Prod=${PROD_NDCG}"
log "  Judge=${JUDGE_AVG}/5  Composite=${COMPOSITE}"
log ""
log "BEIR NDCG@10:"
log "  SciFact=${SCIFACT_NDCG}  NFCorpus=${NFCORPUS_NDCG}  FiQA=${FIQA_NDCG}"
log ""
log "Report: $REPORT"

sms "memory-spark COMPLETE! Custom: Vec=${VECTOR_NDCG} Full=${FULL_NDCG} HyDE=${HYDE_NDCG} Prod=${PROD_NDCG} Judge=${JUDGE_AVG}/5. BEIR: Sci=${SCIFACT_NDCG} NF=${NFCORPUS_NDCG} FiQA=${FIQA_NDCG}. Report: evaluation/results/run-${TIMESTAMP}/REPORT.md"

log ""
log "Done. tmux session stays open for review."
