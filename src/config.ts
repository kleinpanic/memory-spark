/**
 * memory-spark config schema + defaults
 */

import os from "node:os";
import path from "node:path";

export type EmbedProviderId = "spark" | "openai" | "gemini";
// LanceDB is the only storage backend (sqlite-vec was removed in v0.3.0).
// Keeping the type alias for forward compatibility if we add new backends.
export type StorageBackendId = "lancedb";

export interface SparkEndpoints {
  embed: string;
  rerank: string;
  /** Legacy EasyOCR endpoint (kept for fallback) */
  ocr: string;
  /**
   * GLM-OCR endpoint — vLLM-served zai-org/GLM-OCR via OpenAI-compatible
   * chat completions with image_url vision input.
   * Model: zai-org/GLM-OCR (0.9B, #1 OmniDocBench V1.5)
   * Download: forge pull zai-org/GLM-OCR
   */
  glmOcr: string;
  ner: string;
  zeroShot: string;
  summarizer: string;
  stt: string;
}

export interface EmbedConfig {
  provider: EmbedProviderId;
  spark?: { baseUrl: string; apiKey?: string; model: string; dimensions?: number };
  openai?: { apiKey?: string; model: string };
  gemini?: { model: string };
}

export interface RerankConfig {
  enabled: boolean;
  spark?: { baseUrl: string; apiKey?: string; model: string };
  topN: number;
}

/** Full-text search configuration. All fields optional — defaults applied at use site. */
export interface FtsConfig {
  /** Whether FTS is enabled alongside vector search. Default: true */
  enabled?: boolean;
  /**
   * BM25 sigmoid midpoint — controls where FTS scores map to 0.5.
   * Typical BM25 scores for English text: 0–10, median ~2–4.
   * Calibrate by running representative FTS queries and observing score distribution.
   * Default: 3.0
   */
  sigmoidMidpoint?: number;
}

/** Chunking configuration for document ingestion. All fields optional — defaults applied at use site. */
export interface ChunkConfig {
  /** Maximum tokens per chunk (flat mode). Default: 400 */
  maxTokens?: number;
  /** Token overlap between consecutive chunks. Default: 50 */
  overlapTokens?: number;
  /** Minimum tokens for a chunk to be indexed. Default: 20 */
  minTokens?: number;
  /** Enable hierarchical parent-child chunking. Default: true */
  hierarchical?: boolean;
  /** Parent chunk size in tokens (hierarchical mode). Default: 2000 */
  parentMaxTokens?: number;
  /** Child chunk size in tokens (hierarchical mode). Default: 200 */
  childMaxTokens?: number;
  /** Child overlap tokens (hierarchical mode). Default: 25 */
  childOverlapTokens?: number;
}

/** Embed cache configuration. All fields optional — defaults applied at use site. */
export interface EmbedCacheConfig {
  /** Whether the query embed cache is enabled. Default: true */
  enabled?: boolean;
  /** Max cached query embeddings (LRU eviction). Default: 256 */
  maxSize?: number;
  /** Cache TTL in milliseconds. Default: 1800000 (30 min) */
  ttlMs?: number;
}

/** Search tuning configuration. All fields optional — defaults applied at use site. */
export interface SearchConfig {
  /** Refinement factor for ANN vector search. Higher = more precise, slower. Default: 20 */
  refineFactor?: number;
  /** Max write retries on commit conflict. Default: 3 */
  maxWriteRetries?: number;
  /** IVF_PQ partition count for vector index. Default: 10 */
  ivfPartitions?: number;
  /** IVF_PQ sub-vectors for vector index. Default: 64 */
  ivfSubVectors?: number;
}

/** HyDE (Hypothetical Document Embeddings) configuration */
export interface HydeConfig {
  /** Whether HyDE is enabled. Default: true */
  enabled: boolean;
  /** vLLM / OpenAI-compatible chat completions URL */
  llmUrl: string;
  /** Model name for the LLM */
  model: string;
  /** Max tokens for the hypothetical document. Default: 150 */
  maxTokens: number;
  /** Temperature for generation. Default: 0.7 */
  temperature: number;
  /** Timeout for the LLM call in ms. Default: 10000 */
  timeoutMs: number;
  /** Bearer token for auth (optional) */
  apiKey?: string;
}

/** Source and path weighting for recall scoring. All values are multipliers (1.0 = no change). */
export interface RecallWeights {
  /** Source-level weights */
  sources: {
    capture: number; // Default: 1.5
    memory: number; // Default: 1.0
    sessions: number; // Default: 0.5
    reference: number; // Default: 1.0
  };
  /** Path-level weights (applied on top of source weights) */
  paths: Record<string, number>;
  /** Glob-like path patterns → weights (e.g. "mistakes" matches any path containing "mistakes") */
  pathPatterns: Record<string, number>;
}

export interface AutoRecallConfig {
  enabled: boolean;
  agents: string[];
  /** Agents to exclude even when agents=["*"]. Takes precedence over agents. */
  ignoreAgents: string[];
  maxResults: number;
  minScore: number;
  queryMessageCount: number;
  /** Maximum tokens to inject as recalled memories. Default: 2000 */
  maxInjectionTokens: number;
  /** Source and path weighting for recall scoring. Fully configurable. */
  weights: RecallWeights;
  /**
   * MMR diversity lambda (0-1). Higher = more relevant, lower = more diverse.
   * 0.0 = max diversity, 1.0 = pure relevance ranking.
   * Default: 0.7
   */
  mmrLambda?: number;
  /**
   * Temporal decay configuration.
   * Formula: floor + (1 - floor) * exp(-rate * ageDays)
   */
  temporalDecay?: {
    /** Minimum score multiplier for old content. Default: 0.8 (old content keeps ≥80% of score) */
    floor?: number;
    /** Decay rate per day. Higher = faster decay. Default: 0.03 */
    rate?: number;
  };
  /**
   * Context dedup overlap threshold (0-1).
   * Chunks with >threshold token overlap with recent messages/LCM summaries are dropped.
   * Lower = more aggressive dedup. Default: 0.4
   */
  dedupOverlapThreshold?: number;
  /**
   * Overfetch multiplier for initial search before filtering/reranking.
   * Higher = better recall at cost of more compute. Default: 4
   */
  overfetchMultiplier?: number;
  /**
   * Whether to use FTS (full-text search) alongside vector search.
   * When false, only vector search is used (simpler, faster, less precise for keyword queries).
   * Default: true
   */
  ftsEnabled?: boolean;
}

export interface AutoCaptureConfig {
  enabled: boolean;
  agents: string[];
  /** Agents to exclude even when agents=["*"]. Takes precedence over agents. */
  ignoreAgents: string[];
  categories: string[];
  minConfidence: number;
  /** Minimum message length (chars) to consider for capture. Default: 30 */
  minMessageLength: number;
  /**
   * Whether to use the Spark zero-shot classifier (POST /v1/classify) for
   * categorizing captured messages. When false, falls back to a lightweight
   * keyword/regex heuristic that runs locally (no HTTP call).
   * Default: true (use Spark classifier when available).
   */
  useClassifier: boolean;
}

export interface WatchPath {
  path: string;
  agents?: string[];
  pattern?: string;
}

export interface WatchConfig {
  enabled: boolean;
  paths: WatchPath[];
  fileTypes: string[];
  debounceMs: number;
  indexOnBoot: boolean;
  // Glob-like patterns for files to skip (e.g. archive dirs, backups)
  excludePatterns: string[];
  /** Exact relative paths to skip (e.g. "memory/learnings.md") */
  excludePathsExact: string[];
  /** Whether to index session JSONL transcripts. Default: false (LCM handles sessions) */
  indexSessions: boolean;
}

export interface MigrateConfig {
  autoMigrateOnFirstBoot: boolean;
  statusFile: string;
}

export interface IngestConfig {
  /** Minimum quality score (0-1) for a chunk to be indexed. Default: 0.3 */
  minQuality: number;
  /**
   * Primary language for indexed content. Non-matching languages are penalized or excluded.
   * Set to "all" to disable language filtering entirely.
   * Default: "en"
   */
  language: string;
  /**
   * Non-Latin character ratio threshold (0-1). Chunks exceeding this ratio in characters
   * outside the configured language's script are excluded.
   * Default: 0.3 (30% non-Latin = excluded for language "en")
   */
  languageThreshold: number;
}

export interface ReferenceConfig {
  /** Whether reference library indexing is enabled. Default: true */
  enabled: boolean;
  /** Additional paths to index as reference material (e.g. textbooks, API docs) */
  paths: string[];
  /** Chunk size for reference docs (larger than knowledge chunks). Default: 800 */
  chunkSize: number;
  /**
   * Path prefix → tag mapping for organizing reference docs by topic.
   * e.g. { "InternalDocs/": "internal", "openclaw-docs/": "openclaw" }
   */
  tags: Record<string, string>;
}

export interface MemorySparkConfig {
  backend: StorageBackendId;
  lancedbDir: string;
  sqliteVecDir: string;
  embed: EmbedConfig;
  rerank: RerankConfig;
  autoRecall: AutoRecallConfig;
  autoCapture: AutoCaptureConfig;
  watch: WatchConfig;
  ingest: IngestConfig;
  migrate: MigrateConfig;
  spark: SparkEndpoints;
  /** Reference library configuration — additional docs indexed as "reference" content_type */
  reference: ReferenceConfig;
  /** Full-text search (BM25) tuning. Default: enabled with midpoint 3.0 */
  fts?: FtsConfig;
  /** Document chunking configuration. Default: 400 max tokens, 50 overlap, 20 min */
  chunk?: ChunkConfig;
  /** Embed query cache configuration. Default: enabled, 256 entries, 30m TTL */
  embedCache?: EmbedCacheConfig;
  /** Vector search and index tuning. Default: refineFactor 20, 3 retries, IVF_PQ(10, 64) */
  search?: SearchConfig;
  /** HyDE (Hypothetical Document Embeddings) — generate hypothetical docs for better retrieval */
  hyde?: HydeConfig;
  /** Override SPARK_HOST env var. Used to point at a different Spark node. */
  sparkHost?: string;
  /** Override SPARK_BEARER_TOKEN env var. Loaded from env/.env if not set. */
  sparkBearerToken?: string;
}

import fs from "node:fs";

/**
 * Load Spark bearer token — checks process.env first, then ~/.openclaw/.env.
 * process.env takes precedence so harness can override when running on Spark directly.
 */
function loadSparkToken(): string | undefined {
  // 1. Process environment (e.g. SPARK_BEARER_TOKEN=xxx npx tsx test-harness.ts)
  if (process.env["SPARK_BEARER_TOKEN"]) {
    return process.env["SPARK_BEARER_TOKEN"];
  }
  // 2. ~/.openclaw/.env file (default when running as OpenClaw plugin)
  try {
    const envPath = path.join(os.homedir(), ".openclaw", ".env");
    const content = fs.readFileSync(envPath, "utf-8");
    const match = content.match(/SPARK_BEARER_TOKEN=["']?([^"'\s\n]+)/); // eslint-disable-line sonarjs/duplicates-in-character-class -- " and ' are distinct chars
    return match?.[1];
  } catch {
    return undefined;
  }
}

/** Default Spark host fallback — used when neither config nor env specifies a host. */
const FALLBACK_SPARK_HOST = "localhost";

/**
 * Build a full default config using the resolved sparkHost and sparkToken.
 * Called inside resolveConfig() so user config overrides take effect.
 */
function buildDefaults(sparkHost: string, sparkToken: string | undefined): MemorySparkConfig {
  return {
    backend: "lancedb",
    lancedbDir: path.join(os.homedir(), ".openclaw", "data", "memory-spark", "lancedb"),
    sqliteVecDir: path.join(os.homedir(), ".openclaw", "memory"),
    embed: {
      provider: "spark",
      spark: {
        baseUrl: `http://${sparkHost}:18091/v1`,
        apiKey: sparkToken,
        model: "nvidia/llama-embed-nemotron-8b",
        dimensions: 4096,
      },
      openai: { model: "text-embedding-3-small" },
      gemini: { model: "gemini-embedding-001" },
    },
    rerank: {
      enabled: true,
      spark: {
        baseUrl: `http://${sparkHost}:18096/v1`,
        apiKey: sparkToken,
        model: "nvidia/llama-nemotron-rerank-1b-v2",
      },
      topN: 20,
    },
    autoRecall: {
      enabled: true,
      agents: ["*"],
      ignoreAgents: [],
      maxResults: 5,
      minScore: 0.75,
      queryMessageCount: 2,
      maxInjectionTokens: 2000,
      mmrLambda: 0.7,
      temporalDecay: { floor: 0.8, rate: 0.03 },
      dedupOverlapThreshold: 0.4,
      overfetchMultiplier: 4,
      ftsEnabled: true,
      weights: {
        sources: {
          capture: 1.5,
          memory: 1.0,
          sessions: 0.5,
          reference: 1.0,
        },
        paths: {
          "MEMORY.md": 1.4,
          "TOOLS.md": 1.3,
          "AGENTS.md": 1.2,
          "SOUL.md": 1.2,
          "USER.md": 1.3,
          "memory/learnings.md": 0.1,
        },
        pathPatterns: {
          mistakes: 1.6,
          "memory/archive/": 0.4,
        },
      },
    },
    autoCapture: {
      enabled: true,
      agents: ["*"],
      ignoreAgents: [],
      categories: ["fact", "preference", "decision", "code-snippet"],
      minConfidence: 0.6,
      minMessageLength: 30,
      useClassifier: true,
    },
    watch: {
      enabled: true,
      paths: [],
      fileTypes: ["md", "txt", "pdf", "docx"],
      debounceMs: 2000,
      indexOnBoot: true,
      excludePatterns: [
        "**/archive/**",
        "**/*.bak",
        "**/*-session-save.md",
        "**/zh-CN/**",
        "**/zh-TW/**",
        "**/ja/**",
        "**/ko/**",
        "**/fr/**",
        "**/de/**",
        "**/es/**",
        "**/pt-BR/**",
        "**/ru/**",
        "**/i18n/**",
        "**/locales/**",
        "**/locale/**",
        "**/translations/**",
        "**/translation/**",
      ],
      excludePathsExact: ["memory/learnings.md"],
      indexSessions: false,
    },
    ingest: {
      minQuality: 0.3,
      language: "en",
      languageThreshold: 0.3,
    },
    reference: {
      enabled: true,
      paths: (() => {
        // Auto-discover OpenClaw docs if available
        const docsPath = path.join(
          os.homedir(),
          ".local",
          "share",
          "npm",
          "lib",
          "node_modules",
          "openclaw",
          "docs",
        );
        try {
          fs.accessSync(docsPath);
          return [docsPath];
        } catch {
          return [];
        }
      })(),
      chunkSize: 800,
      tags: {},
    },
    hyde: {
      enabled: true,
      llmUrl: `http://${sparkHost}:18080/v1/chat/completions`,
      model: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
      maxTokens: 150,
      temperature: 0.7,
      timeoutMs: 10000,
      apiKey: sparkToken,
    },
    fts: {
      enabled: true,
      sigmoidMidpoint: 3.0,
    },
    chunk: {
      maxTokens: 400,
      overlapTokens: 50,
      minTokens: 20,
    },
    embedCache: {
      enabled: true,
      maxSize: 256,
      ttlMs: 30 * 60 * 1000, // 30 minutes
    },
    search: {
      refineFactor: 20,
      maxWriteRetries: 3,
      ivfPartitions: 10,
      ivfSubVectors: 64,
    },
    migrate: {
      autoMigrateOnFirstBoot: true,
      statusFile: path.join(
        os.homedir(),
        ".openclaw",
        "runtime",
        "state",
        "memory-spark-migrate.json",
      ),
    },
    spark: {
      embed: `http://${sparkHost}:18091/v1`,
      rerank: `http://${sparkHost}:18096/v1`,
      ocr: `http://${sparkHost}:18097`,
      glmOcr: `http://${sparkHost}:18080/v1`,
      ner: `http://${sparkHost}:18112`,
      zeroShot: `http://${sparkHost}:18113`,
      summarizer: `http://${sparkHost}:18110`,
      stt: `http://${sparkHost}:18094`,
    },
  };
}

/**
 * Exported for backward compat — snapshot using env-only resolution.
 * Prefer resolveConfig() for runtime use (it accepts user overrides).
 * @public — used by test harnesses and external scripts
 */
export const DEFAULT_CONFIG: MemorySparkConfig = buildDefaults(
  process.env["SPARK_HOST"] ?? FALLBACK_SPARK_HOST,
  loadSparkToken(),
);

/**
 * Check whether an agent should be processed by auto-recall/capture.
 * Wildcard ["*"] means all agents — unless they're in ignoreAgents.
 */
export function shouldProcessAgent(
  agentId: string,
  agents: string[],
  ignoreAgents: string[],
): boolean {
  if (ignoreAgents.includes(agentId)) return false;
  if (agents.includes("*")) return true;
  return agents.includes(agentId);
}

function expandHome(p: string): string {
  if (p.startsWith("~/") || p === "~") {
    return path.join(os.homedir(), p.slice(2));
  }
  return p;
}

/**
 * Deep merge user config over defaults.
 *
 * Resolution order for sparkHost / sparkBearerToken:
 *   1. userConfig.sparkHost / userConfig.sparkBearerToken  (openclaw.json plugin config)
 *   2. process.env.SPARK_HOST / SPARK_BEARER_TOKEN         (env var)
 *   3. ~/.openclaw/.env file                               (dotenv fallback)
 *   4. hardcoded fallback (localhost / undefined)
 */
export function resolveConfig(userConfig?: Partial<MemorySparkConfig>): MemorySparkConfig {
  // Resolve host and token with proper precedence
  const sparkHost = userConfig?.sparkHost ?? process.env["SPARK_HOST"] ?? FALLBACK_SPARK_HOST;
  const sparkToken = userConfig?.sparkBearerToken ?? loadSparkToken(); // checks process.env then ~/.openclaw/.env

  // Build defaults using the resolved host/token
  const defaults = buildDefaults(sparkHost, sparkToken);

  // MEMORY_SPARK_DATA_DIR env override — allows standalone dev/test without touching production.
  // When set, lancedbDir points at <dataDir>/lancedb/ instead of ~/.openclaw/data/memory-spark/lancedb/
  const dataDir = process.env["MEMORY_SPARK_DATA_DIR"];
  if (dataDir) {
    const resolved = path.isAbsolute(dataDir) ? dataDir : path.resolve(dataDir);
    defaults.lancedbDir = path.join(resolved, "lancedb");
  }

  if (!userConfig) return defaults;

  const merged: MemorySparkConfig = {
    backend: userConfig.backend ?? defaults.backend,
    lancedbDir: dataDir
      ? defaults.lancedbDir // env override takes precedence over user config
      : expandHome(userConfig.lancedbDir ?? defaults.lancedbDir),
    sqliteVecDir: expandHome(userConfig.sqliteVecDir ?? defaults.sqliteVecDir),
    embed: {
      ...defaults.embed,
      ...userConfig.embed,
      spark: { ...defaults.embed.spark!, ...userConfig.embed?.spark },
      openai: { ...defaults.embed.openai!, ...userConfig.embed?.openai },
      gemini: { ...defaults.embed.gemini!, ...userConfig.embed?.gemini },
    },
    rerank: {
      ...defaults.rerank,
      ...userConfig.rerank,
      spark: { ...defaults.rerank.spark!, ...userConfig.rerank?.spark },
    },
    autoRecall: {
      ...defaults.autoRecall,
      ...userConfig.autoRecall,
      temporalDecay: {
        ...defaults.autoRecall.temporalDecay,
        ...userConfig.autoRecall?.temporalDecay,
      },
      weights: {
        sources: {
          ...defaults.autoRecall.weights.sources,
          ...userConfig.autoRecall?.weights?.sources,
        },
        paths: { ...defaults.autoRecall.weights.paths, ...userConfig.autoRecall?.weights?.paths },
        pathPatterns: {
          ...defaults.autoRecall.weights.pathPatterns,
          ...userConfig.autoRecall?.weights?.pathPatterns,
        },
      },
    },
    autoCapture: { ...defaults.autoCapture, ...userConfig.autoCapture },
    watch: {
      ...defaults.watch,
      ...userConfig.watch,
      paths:
        userConfig.watch?.paths?.map((p) => ({
          ...p,
          path: expandHome(p.path),
        })) ?? defaults.watch.paths,
    },
    ingest: { ...defaults.ingest, ...userConfig.ingest },
    reference: {
      ...defaults.reference,
      ...userConfig.reference,
      paths: userConfig.reference?.paths?.map(expandHome) ?? defaults.reference.paths,
    },
    migrate: {
      ...defaults.migrate,
      ...userConfig.migrate,
      statusFile: expandHome(userConfig.migrate?.statusFile ?? defaults.migrate.statusFile),
    },
    hyde: { ...defaults.hyde!, ...userConfig.hyde },
    fts: { ...defaults.fts, ...userConfig.fts },
    chunk: { ...defaults.chunk, ...userConfig.chunk },
    embedCache: { ...defaults.embedCache, ...userConfig.embedCache },
    search: { ...defaults.search, ...userConfig.search },
    spark: { ...defaults.spark, ...userConfig.spark },
    // Carry through the resolved overrides so downstream can inspect them
    sparkHost: userConfig.sparkHost,
    sparkBearerToken: userConfig.sparkBearerToken,
  };

  return merged;
}
