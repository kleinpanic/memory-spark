/**
 * memory-spark config schema + defaults
 */

import os from "node:os";
import path from "node:path";

export type EmbedProviderId = "spark" | "openai" | "gemini";
export type StorageBackendId = "lancedb" | "sqlite-vec";

export interface SparkEndpoints {
  embed: string;
  rerank: string;
  ocr: string;
  ner: string;
  zeroShot: string;
  summarizer: string;
  stt: string;
}

export interface EmbedConfig {
  provider: EmbedProviderId;
  spark?: { baseUrl: string; apiKey?: string; model: string };
  openai?: { apiKey?: string; model: string };
  gemini?: { model: string };
}

export interface RerankConfig {
  enabled: boolean;
  spark?: { baseUrl: string; apiKey?: string; model: string };
  topN: number;
}

export interface AutoRecallConfig {
  enabled: boolean;
  agents: string[];
  maxResults: number;
  minScore: number;
  queryMessageCount: number;
}

export interface AutoCaptureConfig {
  enabled: boolean;
  agents: string[];
  categories: string[];
  minConfidence: number;
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
}

export interface MigrateConfig {
  autoMigrateOnFirstBoot: boolean;
  statusFile: string;
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
  migrate: MigrateConfig;
  spark: SparkEndpoints;
}

export const DEFAULT_CONFIG: MemorySparkConfig = {
  backend: "lancedb",
  lancedbDir: path.join(os.homedir(), ".openclaw", "data", "memory-spark", "lancedb"),
  sqliteVecDir: path.join(os.homedir(), ".openclaw", "memory"),
  embed: {
    provider: "spark",
    spark: {
      baseUrl: "http://dgx-spark.local:18091/v1",
      model: "Qwen/Qwen3-Embedding-4B",
    },
    openai: { model: "text-embedding-3-small" },
    gemini: { model: "gemini-embedding-001" },
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
    statusFile: path.join(os.homedir(), ".openclaw", "runtime", "state", "memory-spark-migrate.json"),
  },
  spark: {
    embed: "http://dgx-spark.local:18091/v1",
    rerank: "http://dgx-spark.local:18096/v1",
    ocr: "http://dgx-spark.local:18097",
    ner: "http://dgx-spark.local:18112",
    zeroShot: "http://dgx-spark.local:18113",
    summarizer: "http://dgx-spark.local:18110",
    stt: "http://dgx-spark.local:18094",
  },
};

function expandHome(p: string): string {
  if (p.startsWith("~/") || p === "~") {
    return path.join(os.homedir(), p.slice(2));
  }
  return p;
}

/** Deep merge user config over defaults */
export function resolveConfig(userConfig?: Partial<MemorySparkConfig>): MemorySparkConfig {
  if (!userConfig) return { ...DEFAULT_CONFIG };

  const merged: MemorySparkConfig = {
    backend: userConfig.backend ?? DEFAULT_CONFIG.backend,
    lancedbDir: expandHome(userConfig.lancedbDir ?? DEFAULT_CONFIG.lancedbDir),
    sqliteVecDir: expandHome(userConfig.sqliteVecDir ?? DEFAULT_CONFIG.sqliteVecDir),
    embed: {
      ...DEFAULT_CONFIG.embed,
      ...userConfig.embed,
      spark: { ...DEFAULT_CONFIG.embed.spark!, ...userConfig.embed?.spark },
      openai: { ...DEFAULT_CONFIG.embed.openai!, ...userConfig.embed?.openai },
      gemini: { ...DEFAULT_CONFIG.embed.gemini!, ...userConfig.embed?.gemini },
    },
    rerank: {
      ...DEFAULT_CONFIG.rerank,
      ...userConfig.rerank,
      spark: { ...DEFAULT_CONFIG.rerank.spark!, ...userConfig.rerank?.spark },
    },
    autoRecall: { ...DEFAULT_CONFIG.autoRecall, ...userConfig.autoRecall },
    autoCapture: { ...DEFAULT_CONFIG.autoCapture, ...userConfig.autoCapture },
    watch: {
      ...DEFAULT_CONFIG.watch,
      ...userConfig.watch,
      paths: userConfig.watch?.paths?.map((p) => ({
        ...p,
        path: expandHome(p.path),
      })) ?? DEFAULT_CONFIG.watch.paths,
    },
    migrate: {
      ...DEFAULT_CONFIG.migrate,
      ...userConfig.migrate,
      statusFile: expandHome(userConfig.migrate?.statusFile ?? DEFAULT_CONFIG.migrate.statusFile),
    },
    spark: { ...DEFAULT_CONFIG.spark, ...userConfig.spark },
  };

  return merged;
}
