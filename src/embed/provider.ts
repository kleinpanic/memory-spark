/**
 * Embedding Provider — Spark → OpenAI → Gemini fallback chain.
 * All use OpenAI-compatible /v1/embeddings endpoint.
 */

import type { EmbedConfig } from "../config.js";

export interface EmbedProvider {
  id: string;
  model: string;
  dims: number;
  embedQuery(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
  probe(): Promise<boolean>;
}

const DIMS: Record<string, number> = {
  "Qwen/Qwen3-Embedding-4B": 2560,
  "nvidia/llama-embed-nemotron-8b": 4096,
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
  "text-embedding-ada-002": 1536,
  "gemini-embedding-001": 3072,
};

export async function createEmbedProvider(cfg: EmbedConfig): Promise<EmbedProvider> {
  const providers: Array<() => EmbedProvider> = [];

  if (cfg.spark) {
    providers.push(() =>
      makeOpenAiCompat(
        "spark",
        cfg.spark!.baseUrl,
        cfg.spark!.apiKey ?? "none",
        cfg.spark!.model,
        cfg.spark!.queryInstruction,
      ),
    );
  }

  if (cfg.openai) {
    const key = cfg.openai.apiKey ?? process.env["OPENAI_API_KEY"] ?? "";
    if (key) {
      providers.push(() =>
        makeOpenAiCompat("openai", "https://api.openai.com/v1", key, cfg.openai!.model),
      );
    }
  }

  if (cfg.gemini) {
    const key = process.env["GOOGLE_API_KEY"] ?? process.env["GEMINI_API_KEY"] ?? "";
    if (key) {
      providers.push(() => makeGemini(cfg.gemini!.model, key));
    }
  }

  // Try each provider in order
  for (const factory of providers) {
    const provider = factory();
    try {
      const ok = await provider.probe();
      if (ok) return provider;
    } catch {
      // Try next
    }
  }

  // If nothing probed, return the first one anyway (will fail at embed time with a clearer error)
  if (providers.length > 0) return providers[0]!();
  throw new Error("memory-spark: no embedding provider configured");
}

// ---------------------------------------------------------------------------
// OpenAI-compatible provider (covers Spark + OpenAI)
// ---------------------------------------------------------------------------

function makeOpenAiCompat(
  id: string,
  baseUrl: string,
  apiKey: string,
  model: string,
  queryInstruction?: string,
): EmbedProvider {
  const dims = DIMS[model] ?? 1536;

  async function embed(input: string | string[]): Promise<number[][]> {
    let resp: Response;
    try {
      resp = await fetch(`${baseUrl}/embeddings`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({ model, input }),
      });
    } catch (err: unknown) {
      // Unwrap Node fetch errors — they wrap the real cause (ECONNREFUSED, ETIMEDOUT, etc.)
      const cause = (err as { cause?: Error })?.cause;
      const code = (cause as { code?: string })?.code ?? "UNKNOWN";
      const detail = cause?.message ?? (err instanceof Error ? err.message : String(err));
      throw new Error(`Embed ${id} network error [${code}]: ${detail}`, { cause: err });
    }
    if (!resp.ok) {
      const body = await resp.text().catch(() => "");
      const status = resp.status;
      // Tag retryable vs fatal for the queue layer
      const err = new Error(`Embed ${id} failed (${status}): ${body.slice(0, 200)}`);
      (err as Error & { httpStatus: number }).httpStatus = status;
      throw err;
    }
    const data = (await resp.json()) as { data: Array<{ embedding: number[]; index: number }> };
    // Sort by index to preserve order
    return data.data.sort((a, b) => a.index - b.index).map((d) => d.embedding);
  }

  return {
    id,
    model,
    dims,
    async embedQuery(text) {
      // Instruction-aware models (e.g. Nemotron-8B) require queries to be prefixed
      // with a task instruction. Documents are embedded as raw text (no prefix).
      // This asymmetric encoding aligns query and document vectors into the correct
      // subspaces for retrieval. Without the prefix, query vectors land in document
      // space and retrieval quality degrades significantly.
      const input = queryInstruction
        ? `Instruct: ${queryInstruction}\nQuery: ${text}`
        : text;
      const results = await embed(input);
      return results[0]!;
    },
    async embedBatch(texts) {
      if (texts.length === 0) return [];
      // Batch in groups of 100 to avoid payload limits
      const results: number[][] = [];
      for (let i = 0; i < texts.length; i += 100) {
        const batch = texts.slice(i, i + 100);
        const vectors = await embed(batch);
        results.push(...vectors);
      }
      return results;
    },
    async probe() {
      try {
        const vectors = await embed("probe");
        return vectors.length === 1 && vectors[0]!.length > 0;
      } catch {
        return false;
      }
    },
  };
}

// ---------------------------------------------------------------------------
// Gemini provider (uses Google AI Generative API)
// ---------------------------------------------------------------------------

function makeGemini(model: string, apiKey: string): EmbedProvider {
  const dims = DIMS[model] ?? 3072;
  const baseUrl = "https://generativelanguage.googleapis.com/v1beta";

  return {
    id: "gemini",
    model,
    dims,
    async embedQuery(text) {
      const url = `${baseUrl}/models/${model}:embedContent?key=${apiKey}`;
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: `models/${model}`,
          content: { parts: [{ text }] },
          taskType: "RETRIEVAL_QUERY",
        }),
      });
      if (!resp.ok) throw new Error(`Gemini embed failed: ${resp.status}`);
      const data = (await resp.json()) as { embedding: { values: number[] } };
      return data.embedding.values;
    },
    async embedBatch(texts) {
      // Gemini batch embed
      const requests = texts.map((text) => ({
        model: `models/${model}`,
        content: { parts: [{ text }] },
        taskType: "RETRIEVAL_DOCUMENT",
      }));
      const url = `${baseUrl}/models/${model}:batchEmbedContents?key=${apiKey}`;
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ requests }),
      });
      if (!resp.ok) throw new Error(`Gemini batch embed failed: ${resp.status}`);
      const data = (await resp.json()) as { embeddings: Array<{ values: number[] }> };
      return data.embeddings.map((e) => e.values);
    },
    async probe() {
      try {
        const v = await this.embedQuery("probe");
        return v.length > 0;
      } catch {
        return false;
      }
    },
  };
}
