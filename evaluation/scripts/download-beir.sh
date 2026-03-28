#!/usr/bin/env bash
# Download BEIR datasets for evaluation.
# Usage: ./download-beir.sh [dataset...]
# Available: scifact nfcorpus fiqa nq hotpotqa
#
# Downloads from https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/
# NQ and HotpotQA are large (300MB+); use --small for just SciFact + NFCorpus + FiQA.

set -euo pipefail

BEIR_URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DIR="$(cd "$(dirname "$0")/.." && pwd)/beir-datasets"
mkdir -p "$DIR"

# Default: small datasets that download quickly
SMALL_DATASETS=("scifact" "nfcorpus" "fiqa")
LARGE_DATASETS=("nq" "hotpotqa")

if [ "${1:-}" = "--small" ]; then
  DATASETS=("${SMALL_DATASETS[@]}")
elif [ "${1:-}" = "--all" ]; then
  DATASETS=("${SMALL_DATASETS[@]}" "${LARGE_DATASETS[@]}")
elif [ $# -gt 0 ]; then
  DATASETS=("$@")
else
  DATASETS=("${SMALL_DATASETS[@]}")
fi

for ds in "${DATASETS[@]}"; do
  if [ -d "$DIR/$ds" ] && [ -f "$DIR/$ds/corpus.jsonl" ]; then
    echo "✅ $ds already downloaded"
    continue
  fi

  echo "⬇️  Downloading $ds..."
  wget -q --show-progress -O "$DIR/$ds.zip" "$BEIR_URL/$ds.zip" || {
    echo "❌ Failed to download $ds"
    continue
  }

  echo "📦 Extracting $ds..."
  unzip -q -o "$DIR/$ds.zip" -d "$DIR/"
  echo "✅ $ds ready ($(du -sh "$DIR/$ds" | cut -f1))"
done

echo ""
echo "Available datasets:"
for d in "$DIR"/*/; do
  name=$(basename "$d")
  if [ -f "$d/corpus.jsonl" ]; then
    docs=$(wc -l < "$d/corpus.jsonl")
    echo "  $name: $docs docs"
  fi
done
