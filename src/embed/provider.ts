/**
 * Embedding Provider
 *
 * Provider chain: spark → openai → gemini (fallback in order on failure)
 * All providers expose the same EmbedProvider interface.
 *
 * Spark endpoint is OpenAI-compatible (/v1/embeddings).
 * So "spark" is just "openai" with a custom baseUrl — no new protocol needed.
 *
 * Dimensions by model:
 *   Qwen/Qwen3-Embedding-4B         → 2560 dims
 *   nvidia/llama-embed-nemotron-8b  → 4096 dims (higher quality, slower)
 *   text-embedding-3-small          → 1536 dims (OpenAI fallback)
 *   gemini-embedding-001            → 3072 dims (Gemini fallback)
 *
 * IMPORTANT: All chunks in a table must share the same dims.
 * Migration must re-embed everything when switching providers.
 */

import type { EmbedConfig } from "../config.js";

export interface EmbedProvider {
  id: string;
  model: string;
  dims: number;
  /** Embed a single query (optimized for search queries, may use task_type="query") */
  embedQuery(text: string): Promise<number[]>;
  /** Embed a batch of document chunks */
  embedBatch(texts: string[]): Promise<number[][]>;
  /** Probe availability — resolves to true if reachable */
  probe(): Promise<boolean>;
}

/**
 * Create an EmbedProvider from config.
 * Tries providers in order until one probes successfully.
 */
export async function createEmbedProvider(cfg: EmbedConfig): Promise<EmbedProvider> {
  const chain: EmbedProvider[] = [];

  if (cfg.provider === "spark" && cfg.spark) {
    chain.push(createOpenAiCompatProvider({
      id: "spark",
      baseUrl: cfg.spark.baseUrl,
      apiKey: cfg.spark.apiKey ?? "none",
      model: cfg.spark.model,
    }));
  }

  if (cfg.openai) {
    chain.push(createOpenAiCompatProvider({
      id: "openai",
      baseUrl: "https://api.openai.com/v1",
      apiKey: cfg.openai.apiKey ?? process.env["OPENAI_API_KEY"] ?? "",
      model: cfg.openai.model,
    }));
  }

  if (cfg.gemini) {
    chain.push(createGeminiProvider({ model: cfg.gemini.model }));
  }

  for (const provider of chain) {
    try {
      const ok = await provider.probe();
      if (ok) return provider;
    } catch {
      // try next
    }
  }

  throw new Error(
    `memory-spark: no embedding provider available. Tried: ${chain.map((p) => p.id).join(", ")}`
  );
}

// ---------------------------------------------------------------------------
// OpenAI-compatible provider (covers Spark + standard OpenAI)
// ---------------------------------------------------------------------------

interface OpenAiCompatOptions {
  id: string;
  baseUrl: string;
  apiKey: string;
  model: string;
}

function createOpenAiCompatProvider(opts: OpenAiCompatOptions): EmbedProvider {
  // Model → dims lookup (extend as needed)
  const DIMS: Record<string, number> = {
    "Qwen/Qwen3-Embedding-4B": 2560,
    "nvidia/llama-embed-nemotron-8b": 4096,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
  };

  const dims = DIMS[opts.model] ?? 1536;

  async function request(input: string | string[]): Promise<number[][]> {
    // TODO: fetch(`${opts.baseUrl}/embeddings`, { method: "POST", ... })
    throw new Error(`OpenAiCompatProvider(${opts.id}).request() not yet implemented`);
  }

  return {
    id: opts.id,
    model: opts.model,
    dims,
    async embedQuery(text) {
      const results = await request(text);
      return results[0]!;
    },
    async embedBatch(texts) {
      // TODO: batch into groups of ≤2048 tokens (tiktoken check)
      return request(texts);
    },
    async probe() {
      // TODO: send a short probe embed, check 200
      return false;
    },
  };
}

// ---------------------------------------------------------------------------
// Gemini provider (fallback)
// ---------------------------------------------------------------------------

interface GeminiOptions {
  model: string;
}

function createGeminiProvider(_opts: GeminiOptions): EmbedProvider {
  return {
    id: "gemini",
    model: _opts.model,
    dims: 3072,
    async embedQuery(_text) {
      // TODO: use openclaw's existing gemini embedding client
      throw new Error("GeminiProvider.embedQuery() not yet implemented");
    },
    async embedBatch(_texts) {
      throw new Error("GeminiProvider.embedBatch() not yet implemented");
    },
    async probe() {
      return false;
    },
  };
}
