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
  /** Temperature for generation. Default: 0.3 (low to stay factual per Gao et al. 2022) */
  temperature: number;
  /** Timeout for the LLM call in ms. Default: 10000 */
  timeoutMs: number;
  /** Bearer token for auth (optional) */
  apiKey?: string;
}

export const HYDE_DEFAULTS: HydeConfig = {
  enabled: true,
  llmUrl: `http://${process.env.SPARK_HOST ?? "10.99.1.1"}:18080/v1/chat/completions`, // eslint-disable-line sonarjs/no-hardcoded-ip -- local network fallback, configurable via SPARK_HOST env
  model: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
  maxTokens: 150,
  temperature: 0.3,
  timeoutMs: 10000,
};

/**
 * HyDE system prompt — domain-agnostic by design.
 *
 * Per Gao et al. 2022, the hypothetical document should be written in the
 * style of a source document, NOT as a conversational answer. This ensures
 * the embedding lands in the same vector space as stored knowledge chunks.
 *
 * The prompt avoids mentioning specific domains (OpenClaw, Spark, etc.)
 * because the memory system stores diverse content: config docs, code notes,
 * personal preferences, school assignments, infrastructure docs, etc.
 */
const HYDE_SYSTEM_PROMPT = `Given a question or topic, write a short factual document (2-4 sentences) that would directly contain the answer. Write as if you are the source document itself — state facts directly. Do not write as if answering a question. Do not include phrases like "Based on...", "According to...", or "The answer is...". Just state the information.`;

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

    // Strip think blocks — handle both closed (<think>...</think>) and unclosed (<think>...) tags
    const cleaned = content
      .replace(/<think>[\s\S]*?<\/think>/gi, "")
      .replace(/<think>[\s\S]*/gi, "") // unclosed <think> — strip everything after it
      .replace(/<\/?think>/gi, "")
      .replace(/^(Now|Let me|I'll|We need to)\b.*$/gm, "")
      .replace(/\n{2,}/g, "\n")
      .trim();

    if (cleaned.length < 20) return null;

    // Quality gate — reject generic/unhelpful hypotheticals that would produce
    // poor document-space embeddings. Fall back to raw query embedding instead.
    const wordCount = cleaned.split(/\s+/).length;
    if (wordCount < 10) return null; // Too short to be a useful pseudo-document

    // Reject hedging/refusal patterns — these don't contain factual content
    const REJECTION_PATTERNS = [
      /\bi don'?t know\b/i,
      /\bcannot determine\b/i,
      /\bnot enough information\b/i,
      /\bunable to (answer|provide|determine)\b/i,
      /\bno information (available|provided)\b/i,
      /\bas an ai\b/i,
      /\bi'?m not sure\b/i,
      /\bI cannot\b/i,
    ];
    if (REJECTION_PATTERNS.some((p) => p.test(cleaned))) return null;

    return cleaned;
  } catch {
    // Timeout, network error, parse error — all silently fall back
    return null;
  } finally {
    clearTimeout(timeout);
  }
}
