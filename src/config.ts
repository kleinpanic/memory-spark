/**
 * memory-spark config schema + defaults
 *
 * All plugin config lives under:
 *   openclaw.json → plugins.entries["memory-spark"].config
 *
 * Providers fall back in order: spark → openai → gemini
 * This keeps the plugin portable for setups without a local DGX node.
 */

export type EmbedProvider = "spark" | "openai" | "gemini";
export type StorageBackend = "lancedb" | "sqlite-vec";

export interface SparkEndpoints {
  /** Embedding service — Qwen3-Embedding-4B or llama-embed-nemotron-8b */
  embed: string;         // default: http://dgx-spark.local:18091/v1
  /** Reranker — nvidia/llama-nemotron-rerank-1b-v2 */
  rerank: string;        // default: http://dgx-spark.local:18096/v1
  /** OCR fallback for scanned PDFs */
  ocr: string;           // default: http://dgx-spark.local:18097
  /** Named Entity Recognition for chunk tagging */
  ner: string;           // default: http://dgx-spark.local:18112
  /** Zero-shot classifier for auto-capture categorization */
  zeroShot: string;      // default: http://dgx-spark.local:18113
  /** Summarizer for large document pre-processing */
  summarizer: string;    // default: http://dgx-spark.local:18110
  /** STT for audio ingestion (parakeet) */
  stt: string;           // default: http://dgx-spark.local:18094
}

export interface EmbedConfig {
  /**
   * Preferred provider. Falls through chain on failure:
   *   spark → openai → gemini
   */
  provider: EmbedProvider;
  spark?: {
    baseUrl: string;   // SparkEndpoints.embed
    apiKey?: string;
    model: string;     // default: "Qwen/Qwen3-Embedding-4B"
  };
  openai?: {
    apiKey?: string;   // falls back to env OPENAI_API_KEY
    model: string;     // default: "text-embedding-3-small"
  };
  gemini?: {
    model: string;     // default: "gemini-embedding-001"
  };
}

export interface RerankConfig {
  enabled: boolean;
  spark?: {
    baseUrl: string;   // SparkEndpoints.rerank
    apiKey?: string;
    model: string;     // default: "nvidia/llama-nemotron-rerank-1b-v2"
  };
  /** Top-N results fed into reranker. Keep ≤50 for latency. */
  topN: number;        // default: 20
}

export interface AutoRecallConfig {
  enabled: boolean;
  /**
   * Agent IDs to enable auto-recall for.
   * Use ["*"] for all agents.
   */
  agents: string[];
  /** Max memory chunks to inject per turn */
  maxResults: number;       // default: 5
  /** Minimum similarity score (0–1) to include */
  minScore: number;         // default: 0.65
  /** How many recent conversation messages to use as recall query */
  queryMessageCount: number; // default: 4
}

export interface AutoCaptureConfig {
  enabled: boolean;
  /**
   * Agent IDs to capture from.
   * Use ["*"] for all agents.
   */
  agents: string[];
  /**
   * Zero-shot category labels to capture.
   * Anything not in this list (or "none") is discarded.
   */
  categories: string[];     // default: ["fact", "preference", "decision", "code-snippet"]
  /**
   * Minimum zero-shot confidence score to store (0–1).
   */
  minConfidence: number;    // default: 0.75
}

export interface WatchPath {
  /** Absolute or ~ path to watch */
  path: string;
  /** Which agents this content is scoped to (for per-agent search) */
  agents?: string[];
  /** Glob pattern for files to include */
  pattern?: string;         // default: "**/*.{md,txt,pdf,docx}"
}

export interface WatchConfig {
  enabled: boolean;
  paths: WatchPath[];
  /** File types to ingest */
  fileTypes: string[];      // default: ["md", "txt", "pdf", "docx"]
  /** Debounce ms before re-indexing a changed file */
  debounceMs: number;       // default: 2000
  /** Whether to do a full index pass on gateway start */
  indexOnBoot: boolean;     // default: true
}

export interface MigrateConfig {
  /**
   * If true, plugin will auto-detect old memory-core SQLite-vec DBs on
   * first boot and re-embed into LanceDB. Runs once, marks completion.
   */
  autoMigrateOnFirstBoot: boolean; // default: true
  /** Where to write migration status/progress */
  statusFile: string; // default: ~/.openclaw/runtime/state/memory-spark-migrate.json
}

export interface MemorySparkConfig {
  /** Primary storage backend. "lancedb" recommended; "sqlite-vec" for migration source. */
  backend: StorageBackend;
  /** Where LanceDB table files live */
  lancedbDir: string;       // default: ~/.openclaw/data/memory-spark/lancedb
  /** Where per-agent SQLite-vec DBs live (migration source) */
  sqliteVecDir: string;     // default: ~/.openclaw/memory
  embed: EmbedConfig;
  rerank: RerankConfig;
  autoRecall: AutoRecallConfig;
  autoCapture: AutoCaptureConfig;
  watch: WatchConfig;
  migrate: MigrateConfig;
  spark: SparkEndpoints;
}

export const DEFAULT_CONFIG: MemorySparkConfig = {
  backend: "lancedb",
  lancedbDir: "~/.openclaw/data/memory-spark/lancedb",
  sqliteVecDir: "~/.openclaw/memory",
  embed: {
    provider: "spark",
    spark: {
      baseUrl: "http://dgx-spark.local:18091/v1",
      model: "Qwen/Qwen3-Embedding-4B",
    },
    openai: {
      model: "text-embedding-3-small",
    },
    gemini: {
      model: "gemini-embedding-001",
    },
  },
  rerank: {
    enabled: true,
    spark: {
      baseUrl: "http://dgx-spark.local:18096/v1",
      model: "nvidia/llama-nemotron-rerank-1b-v2",
    },
    topN: 20,
  },
  autoRecall: {
    enabled: true,
    agents: ["main", "school", "research", "dev"],
    maxResults: 5,
    minScore: 0.65,
    queryMessageCount: 4,
  },
  autoCapture: {
    enabled: true,
    agents: ["main", "school", "research"],
    categories: ["fact", "preference", "decision", "code-snippet"],
    minConfidence: 0.75,
  },
  watch: {
    enabled: true,
    paths: [],
    fileTypes: ["md", "txt", "pdf", "docx"],
    debounceMs: 2000,
    indexOnBoot: true,
  },
  migrate: {
    autoMigrateOnFirstBoot: true,
    statusFile: "~/.openclaw/runtime/state/memory-spark-migrate.json",
  },
  spark: {
    embed:      "http://dgx-spark.local:18091/v1",
    rerank:     "http://dgx-spark.local:18096/v1",
    ocr:        "http://dgx-spark.local:18097",
    ner:        "http://dgx-spark.local:18112",
    zeroShot:   "http://dgx-spark.local:18113",
    summarizer: "http://dgx-spark.local:18110",
    stt:        "http://dgx-spark.local:18094",
  },
};

/** Merge user config over defaults, deep-merging nested objects */
export function resolveConfig(userConfig?: Partial<MemorySparkConfig>): MemorySparkConfig {
  // TODO: implement deep merge
  return { ...DEFAULT_CONFIG, ...(userConfig ?? {}) } as MemorySparkConfig;
}
