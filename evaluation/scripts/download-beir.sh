#!/usr/bin/env bash
# BEIR Dataset Downloader — all 19 publicly available datasets from TU Darmstadt
#
# Usage:
#   ./download-beir.sh              # All 19 datasets (~10.6 GB compressed)
#   ./download-beir.sh --small     # Just SciFact + NFCorpus + FiQA (~22 MB)
#   ./download-beir.sh --core      # The 15 datasets from the paper (~8.5 GB)
#   ./download-beir.sh scifact fiqa nq  # Specific datasets
#
# Note: robust04, bioasq, trec-news are NO LONGER available at the hosting URL.
# The TU Darmstadt server hosts 19 datasets total.

set -euo pipefail

BEIR_URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${DIR:-$SCRIPT_DIR/beir-datasets}"
mkdir -p "$DIR"

# ── Dataset definitions ────────────────────────────────────────────────────────

# The 15 datasets referenced in the memory-spark paper
PAPER_DATASETS=(
  "scifact"
  "nfcorpus"
  "fiqa"
  "nq"
  "hotpotqa"
  "arguana"
  "trec-covid-beir"   # renamed from trec-covid → trec-covid-beir
  "dbpedia-entity"     # renamed from dbpedia
  "fever"
  "climate-fever"
  "cqadupstack"
  "quora"
  "scidocs"
  "msmarco"
  "webis-touche2020"  # renamed from touche-2020
)

# Small/fast subset for CI and smoke tests
SMALL_DATASETS=("scifact" "nfcorpus" "fiqa")

# All 19 available datasets (paper 15 + 4 extras)
ALL_DATASETS=(
  "scifact"
  "nfcorpus"
  "fiqa"
  "nq"
  "hotpotqa"
  "arguana"
  "trec-covid-beir"
  "dbpedia-entity"
  "fever"
  "climate-fever"
  "cqadupstack"
  "quora"
  "scidocs"
  "msmarco"
  "webis-touche2020"
  "mrtydi"            # extra: multilingual (11 languages)
  "germanquad"        # extra: German QA
  "msmarco-v2"        # extra: MS MARCO v2 passage ranking
  "mmarco"            # extra: mMARCO (multilingual, smaller)
)

# ── Size check ─────────────────────────────────────────────────────────────────

check_space() {
  local required_gb="${1:-10}"
  local available_gb
  available_gb=$(df -BG "$DIR" | awk 'NR==2 {print int($4)}' | tr -d 'G')
  if (( available_gb < required_gb )); then
    echo "⚠️  Warning: ~${required_gb}GB needed, ~${available_gb}GB available"
    echo "   Run 'docker system prune -a' or remove old images to free space"
  fi
}

# ── Download a single dataset ──────────────────────────────────────────────────

download_dataset() {
  local ds="$1"
  local zip_path="$DIR/${ds}.zip"

  if [ -d "$DIR/$ds" ] && [ -f "$DIR/$ds/corpus.jsonl" ]; then
    local docs
    docs=$(wc -l < "$DIR/$ds/corpus.jsonl" 2>/dev/null || echo "?")
    echo "✅ $ds already downloaded ($docs docs)"
    return 0
  fi

  echo "⬇️  Downloading $ds..."
  if ! curl -f -L --progress-bar \
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/${ds}.zip" \
    -o "$zip_path" 2>/dev/null; then
    echo "❌ Failed to download $ds — removing partial file"
    rm -f "$zip_path"
    return 1
  fi

  echo "📦 Extracting $ds..."
  unzip -q -o "$zip_path" -d "$DIR/"

  # BEIR datasets extract to a subdirectory — flatten if needed
  local extracted_dir
  extracted_dir=$(find "$DIR" -maxdepth 1 -type d -name "${ds}*" | grep -v "\.zip$" | head -1)
  if [ -n "$extracted_dir" ] && [ "$extracted_dir" != "$DIR/$ds" ]; then
    if [ -d "$DIR/$ds" ]; then
      # Merge or replace — prefer the extracted content
      rm -rf "$DIR/${ds}_old"
      mv "$DIR/$ds" "$DIR/${ds}_old"
    fi
    mv "$extracted_dir" "$DIR/$ds"
  fi

  rm -f "$zip_path"

  if [ -f "$DIR/$ds/corpus.jsonl" ]; then
    local docs
    docs=$(wc -l < "$DIR/$ds/corpus.jsonl")
    local size
    size=$(du -sh "$DIR/$ds" | cut -f1)
    echo "✅ $ds ready: $docs docs, $size"
  else
    echo "⚠️  $ds extracted but corpus.jsonl not found — listing contents:"
    find "$DIR/$ds" -type f | head -10
  fi
}

# ── Main ───────────────────────────────────────────────────────────────────────

if [ "${1:-}" = "--small" ]; then
  DATASETS=("${SMALL_DATASETS[@]}")
  echo "📦 Downloading SMALL subset: ${SMALL_DATASETS[*]}"
  check_space 1
elif [ "${1:-}" = "--core" ]; then
  DATASETS=("${PAPER_DATASETS[@]}")
  echo "📦 Downloading PAPER CORE datasets (${#PAPER_DATASETS[@]}): ${PAPER_DATASETS[*]}"
  check_space 12
elif [ $# -gt 0 ]; then
  DATASETS=("$@")
  echo "📦 Downloading specified datasets: ${DATASETS[*]}"
else
  DATASETS=("${ALL_DATASETS[@]}")
  echo "📦 Downloading ALL 19 BEIR datasets (~10.6 GB compressed, ~35 GB extracted)"
  check_space 40
fi

echo ""

failed=0
for ds in "${DATASETS[@]}"; do
  download_dataset "$ds" || ((failed++))
done

echo ""
echo "═══════════════════════════════════════"
if [ $failed -eq 0 ]; then
  echo "✅ All datasets downloaded successfully"
else
  echo "⚠️  $failed dataset(s) failed — see above"
fi
echo ""
echo "Available datasets in $DIR:"
for d in "$DIR"/*/; do
  name=$(basename "$d")
  if [ -f "$d/corpus.jsonl" ]; then
    docs=$(wc -l < "$d/corpus.jsonl")
    size=$(du -sh "$d" | cut -f1)
    # Check for qrels
    if [ -f "$d/qrels/test.tsv" ]; then
      qrels=$(wc -l < "$d/qrels/test.tsv")
      echo "  $name: $docs docs, $size, $((qrels-1)) qrels"
    else
      echo "  $name: $docs docs, $size (⚠️ no qrels)"
    fi
  fi
done
