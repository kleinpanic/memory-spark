# ═══════════════════════════════════════════════════════════════════════════════
# memory-spark eval harness — self-contained benchmark environment
# NO SECRETS — all config via environment variables
# GPU-NOTES: This image is CPU-only. For GPU inference, run this container
#            on a host with NVIDIA GPU + CUDA, or connect to a vLLM endpoint
#            via SPARK_EMBED_URL / SPARK_RERANK_URL env vars.
# ═══════════════════════════════════════════════════════════════════════════════

FROM node:22-slim

# ── Build arguments ────────────────────────────────────────────────────────────
# These are set at docker build time, NOT baked-in secrets
ARG DEBIAN_FRONTEND=noninteractive
ARG SPARK_EMBED_MODEL=nvidia/llama-embed-nemotron-8b
ARG SPARK_RERANK_MODEL=nvidia/llama-nemotron-rerank-1b-v2
ARG SPARK_HYDE_MODEL=nvidia/NVIDIA-Nemotron-Super-120B-A12B-NVFP4

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    unzip \
    git \
    # For potential Python data processing
    python3 \
    python3-pip \
    python3-venv \
    # For LaTeX compilation in the paper workflow
    texlive-latex-base \
    texlive-latex-extra \
    texlive-science \
    # Misc utilities
    bc \
    jq \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Node deps ─────────────────────────────────────────────────────────────────
# Copy only package files first (layer cache optimization)
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev 2>/dev/null || npm install --omit=dev

# ── App code ───────────────────────────────────────────────────────────────────
COPY src/ ./src/
COPY evaluation/ ./evaluation/
COPY index.ts ./
COPY tsconfig.json ./
COPY vite.config.ts ./

# ── BEIR datasets ─────────────────────────────────────────────────────────────
# NOTE: Datasets are NOT copied into the image by default.
# Mount them at runtime: docker run -v /path/to/beir-datasets:/data/beir-datasets
# Or build with: docker build --build-arg BEIR_DATA=/path/to/data
ENV BEIR_DATA_DIR=/data/beir-datasets
ENV EVAL_DB_DIR=/data/eval-lancedb
ENV RESULTS_DIR=/data/results

# ── vLLM endpoint configuration ────────────────────────────────────────────────
# Point to spark vLLM endpoints. If not set, embeddings use CPU fallback.
ENV SPARK_HOST=${SPARK_HOST:-localhost}
ENV SPARK_EMBED_URL=${SPARK_EMBED_URL:-http://localhost:18091/v1}
ENV SPARK_RERANK_URL=${SPARK_RERANK_URL:-http://localhost:18096/v1}
ENV SPARK_HYDE_URL=${SPARK_HYDE_URL:-http://localhost:18080/v1/chat/completions}

# ── Benchmark config ───────────────────────────────────────────────────────────
# comma-separated list of datasets to evaluate (default: scifact,nfcorpus,fiqa)
ENV BEIR_DATASETS=${BEIR_DATASETS:-scifact,nfcorpus,fiqa}
# comma-separated list of configs to test (A,B,C,D,E,F,G or 'all')
ENV BEIR_CONFIGS=${BEIR_CONFIGS:-all}
# Skip GPU-dependent configs (HyDE, full pipeline) if no GPU available
ENV SKIP_GPU_CONFIGS=${SKIP_GPU_CONFIGS:-false}
# Skip OCR in all RAG operations
ENV DISABLE_OCR=true

# ── Output dirs ────────────────────────────────────────────────────────────────
RUN mkdir -p "$BEIR_DATA_DIR" "$EVAL_DB_DIR" "$RESULTS_DIR"

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD node --check /app/index.ts || exit 1

# ── Entrypoint ─────────────────────────────────────────────────────────────────
COPY evaluation/harness/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
