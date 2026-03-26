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

export interface AutoRecallConfig {
  enabled: boolean;
  agents: string[];
  /** Agents to exclude even when agents=["*"]. Takes precedence over agents. */
  ignoreAgents: string[];
  maxResults: number;
  minScore: number;
  queryMessageCount: number;
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
  // 2. ~/.openclaw/.env file (default when running as OpenClaw plugin on user)
  try {
    const envPath = path.join(os.homedir(), ".openclaw", ".env");
    const content = fs.readFileSync(envPath, "utf-8");
    const match = content.match(/SPARK_BEARER_TOKEN=["']?([^"'\s\n]+)/);
    return match?.[1];
  } catch {
    return undefined;
  }
}

/** Default Spark host fallback — used when neither config nor env specifies a host. */
const FALLBACK_SPARK_HOST = "10.x.x.x";

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
      minScore: 0.65,
      queryMessageCount: 4,
    },
    autoCapture: {
      enabled: true,
      agents: ["*"],
      ignoreAgents: [],
      categories: ["fact", "preference", "decision", "code-snippet"],
      minConfidence: 0.75,
      minMessageLength: 30,
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
 *   4. hardcoded fallback (10.x.x.x / undefined)
 */
export function resolveConfig(userConfig?: Partial<MemorySparkConfig>): MemorySparkConfig {
  // Resolve host and token with proper precedence
  const sparkHost = userConfig?.sparkHost
    ?? process.env["SPARK_HOST"]
    ?? FALLBACK_SPARK_HOST;
  const sparkToken = userConfig?.sparkBearerToken
    ?? loadSparkToken(); // checks process.env then ~/.openclaw/.env

  // Build defaults using the resolved host/token
  const defaults = buildDefaults(sparkHost, sparkToken);

  if (!userConfig) return defaults;

  const merged: MemorySparkConfig = {
    backend: userConfig.backend ?? defaults.backend,
    lancedbDir: expandHome(userConfig.lancedbDir ?? defaults.lancedbDir),
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
    autoRecall: { ...defaults.autoRecall, ...userConfig.autoRecall },
    autoCapture: { ...defaults.autoCapture, ...userConfig.autoCapture },
    watch: {
      ...defaults.watch,
      ...userConfig.watch,
      paths: userConfig.watch?.paths?.map((p) => ({
        ...p,
        path: expandHome(p.path),
      })) ?? defaults.watch.paths,
    },
    migrate: {
      ...defaults.migrate,
      ...userConfig.migrate,
      statusFile: expandHome(userConfig.migrate?.statusFile ?? defaults.migrate.statusFile),
    },
    spark: { ...defaults.spark, ...userConfig.spark },
    // Carry through the resolved overrides so downstream can inspect them
    sparkHost: userConfig.sparkHost,
    sparkBearerToken: userConfig.sparkBearerToken,
  };

  return merged;
}
