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
  /**
   * Embed a single text as a **document** (no instruction prefix).
   * Use this for HyDE hypothetical documents — they are pseudo-documents
   * and must be projected into document embedding space, not query space.
   *
   * For models without instruction prefixes, this is identical to embedQuery().
   */
  embedDocument(text: string): Promise<number[]>;
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
      if (process.env.MEMORY_SPARK_DEBUG) {
        console.debug(`[embed] probe ${provider.id}/${provider.model}: ${ok ? "OK" : "FAIL"}`);
      }
      if (ok) return provider;
    } catch (err) {
      if (process.env.MEMORY_SPARK_DEBUG) {
        console.debug(
          `[embed] probe ${provider.id}/${provider.model}: ERROR - ${err instanceof Error ? err.message : String(err)}`,
        );
      }
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

  // Warn if an instruction-aware model is used without a query instruction.
  // Without the prefix, queries embed into document space → retrieval quality degrades.
  const INSTRUCTION_AWARE_MODELS = [
    "llama-embed-nemotron",
    "e5-instruct",
    "instructor",
    "gte-qwen",
    "nomic-embed",
  ];
  if (!queryInstruction && INSTRUCTION_AWARE_MODELS.some((m) => model.toLowerCase().includes(m))) {
    console.warn(
      `[memory-spark] WARNING: Model "${model}" is instruction-aware but queryInstruction is empty. ` +
        `Queries will embed as raw text (document space) instead of query space. ` +
        `Set embed.spark.queryInstruction in config for optimal retrieval quality.`,
    );
  }

  async function embed(input: string | string[]): Promise<number[][]> {
    const batchSize = Array.isArray(input) ? input.length : 1;
    const verbose = process.env.VERBOSE || process.env.DEBUG_PIPELINE;

    if (verbose) {
      console.log(`[embed] ${id}/${model}: request batchSize=${batchSize}`);
    }

    let resp: Response;
    const t0embed = performance.now();
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
    const elapsedEmbed = (performance.now() - t0embed).toFixed(1);

    const results = data.data.sort((a, b) => a.index - b.index).map((d) => d.embedding);

    // Dimension verification
    const actualDims = results[0]?.length ?? 0;
    if (actualDims > 0 && actualDims !== dims) {
      console.warn(
        `[embed] ${id}/${model}: dimension mismatch! expected=${dims} actual=${actualDims}`,
      );
    }

    console.log(
      `[embed] ${id}/${model}: ${batchSize} texts → ${results.length} vectors (dims=${actualDims}) in ${elapsedEmbed}ms`,
    );

    // Sort by index to preserve order
    return results;
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
      const input = queryInstruction ? `Instruct: ${queryInstruction}\nQuery: ${text}` : text;
      if (process.env.VERBOSE || process.env.DEBUG_PIPELINE) {
        if (queryInstruction) {
          console.log(
            `[embed] embedQuery: instruction prefix applied — "${queryInstruction.slice(0, 60)}"`,
          );
        } else {
          console.log(`[embed] embedQuery: no instruction prefix (raw text)`);
        }
      }
      const results = await embed(input);
      return results[0]!;
    },
    async embedDocument(text) {
      // Always embed as raw text — no instruction prefix.
      // Used for HyDE hypothetical documents which must land in document space.
      if (process.env.VERBOSE || process.env.DEBUG_PIPELINE) {
        console.log(`[embed] embedDocument: no instruction prefix (document space, raw text)`);
      }
      const results = await embed(text);
      return results[0]!;
    },
    async embedBatch(texts) {
      if (texts.length === 0) return [];
      // Batch in groups of 100 to avoid payload limits
      const results: number[][] = [];
      const totalBatches = Math.ceil(texts.length / 100);
      console.log(`[embed] embedBatch: ${texts.length} texts → ${totalBatches} batch(es) of ≤100`);
      for (let i = 0; i < texts.length; i += 100) {
        const batch = texts.slice(i, i + 100);
        const batchIdx = Math.floor(i / 100) + 1;
        if (totalBatches > 1) {
          console.log(`[embed] embedBatch: batch ${batchIdx}/${totalBatches} size=${batch.length}`);
        }
        const vectors = await embed(batch);
        results.push(...vectors);
      }
      return results;
    },
    async probe() {
      // Retry once to handle transient failures (cold starts, brief 502s)
      for (let attempt = 1; attempt <= 2; attempt++) {
        try {
          const vectors = await embed("probe");
          return vectors.length === 1 && vectors[0]!.length > 0;
        } catch (err) {
          if (process.env.MEMORY_SPARK_DEBUG) {
            console.debug(
              `[embed] ${id} probe attempt ${attempt} error: ${err instanceof Error ? err.message : String(err)}`,
            );
          }
          if (attempt < 2) await new Promise((r) => setTimeout(r, 1000));
        }
      }
      return false;
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
    async embedDocument(text) {
      const url = `${baseUrl}/models/${model}:embedContent?key=${apiKey}`;
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: `models/${model}`,
          content: { parts: [{ text }] },
          taskType: "RETRIEVAL_DOCUMENT",
        }),
      });
      if (!resp.ok) throw new Error(`Gemini embed doc failed: ${resp.status}`);
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
