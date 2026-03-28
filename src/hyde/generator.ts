/**
 * HyDE — Hypothetical Document Embeddings
 *
 * Instead of embedding the raw user query, generate a hypothetical document
 * that would answer the query, then embed THAT. This bridges the semantic gap
 * between questions and answers — the hypothetical document lives in the same
 * embedding space as the stored knowledge.
 *
 * Reference: Gao et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
 *
 * Pipeline:
 *   1. User query → LLM generates hypothetical answer document
 *   2. Hypothetical document → Embedding model → Query vector
 *   3. Query vector → Vector search → Results
 *
 * FTS still uses the raw query (exact terms matter for keyword matching).
 */

export interface HydeConfig {
  /** Whether HyDE is enabled. Default: true */
  enabled: boolean;
  /** vLLM / OpenAI-compatible chat completions URL */
  llmUrl: string;
  /** Model name for the LLM. Default: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4" */
  model: string;
  /** Max tokens for the hypothetical document. Default: 150 */
  maxTokens: number;
  /** Temperature for generation. Default: 0.7 (some creativity helps diversity) */
  temperature: number;
  /** Timeout for the LLM call in ms. Default: 10000 */
  timeoutMs: number;
  /** Bearer token for auth (optional) */
  apiKey?: string;
}

export const HYDE_DEFAULTS: HydeConfig = {
  enabled: true,
  llmUrl: "http://10.99.1.1:18080/v1/chat/completions", // eslint-disable-line sonarjs/no-clear-text-protocols -- local network endpoint, not public
  model: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
  maxTokens: 150,
  temperature: 0.7,
  timeoutMs: 10000,
};

const HYDE_SYSTEM_PROMPT = `You are a technical documentation writer for an AI agent system called OpenClaw. Given a question, write a short document (2-4 sentences) that would directly answer it. Write as if you are the source document, not as if you are answering a question. Be specific and factual. Do not include phrases like "Based on..." or "According to...". Just state the information directly.`;

/**
 * Generate a hypothetical document that would answer the given query.
 * On any failure (timeout, error, empty response), returns null — caller
 * should fall back to embedding the raw query.
 */
export async function generateHypotheticalDocument(
  query: string,
  config: HydeConfig,
): Promise<string | null> {
  if (!config.enabled) return null;
  if (query.length < 10) return null; // Too short to generate meaningful hypothetical

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.timeoutMs);

  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (config.apiKey) headers["Authorization"] = `Bearer ${config.apiKey}`;

    const body = JSON.stringify({
      model: config.model,
      messages: [
        { role: "system", content: HYDE_SYSTEM_PROMPT },
        { role: "user", content: query },
      ],
      max_tokens: config.maxTokens,
      temperature: config.temperature,
      stream: false,
    });

    const response = await fetch(config.llmUrl, {
      method: "POST",
      headers,
      body,
      signal: controller.signal,
    });

    if (!response.ok) return null;

    const data = (await response.json()) as {
      choices?: Array<{ message?: { content?: string } }>;
    };

    const content = data.choices?.[0]?.message?.content?.trim();
    if (!content || content.length < 20) return null;

    // Strip think blocks (tag + content between tags) that leaked through
    const cleaned = content
      .replace(/<think>[\s\S]*?<\/think>/gi, "")
      .replace(/<\/?think>/gi, "")
      .replace(/^(Now|Let me|I'll|We need to)\b.*$/gm, "")
      .replace(/\n{2,}/g, "\n")
      .trim();

    return cleaned.length >= 20 ? cleaned : null;
  } catch {
    // Timeout, network error, parse error — all silently fall back
    return null;
  } finally {
    clearTimeout(timeout);
  }
}
