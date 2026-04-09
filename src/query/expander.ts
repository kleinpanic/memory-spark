/**
 * Multi-Query Expansion — Phase 11B
 *
 * Generates alternative phrasings of a user query to break the retrieval ceiling.
 * Different query formulations activate different regions of the vector space,
 * surfacing documents that a single embedding would miss.
 *
 * Pipeline:
 *   1. User query → LLM generates N reformulations
 *   2. [original, r1, r2, ...rN] → Embed each → Vector search each
 *   3. Union results (dedupe by chunk ID, keep highest score)
 *   4. Reranker sorts the expanded candidate pool
 *
 * On any failure, gracefully degrades to [original] — no worse than current.
 */

export interface QueryExpansionConfig {
  /** Whether multi-query expansion is enabled. Default: true */
  enabled: boolean;
  /** vLLM / OpenAI-compatible chat completions URL */
  llmUrl: string;
  /** Model name for the LLM */
  model: string;
  /** Number of reformulations to generate. Default: 3 */
  numReformulations: number;
  /** Max tokens for generation. Default: 150 */
  maxTokens: number;
  /** Temperature for generation. Default: 0.7 (high for diversity) */
  temperature: number;
  /** Timeout for the LLM call in ms. Default: 15000 */
  timeoutMs: number;
  /** Bearer token for auth (optional) */
  apiKey?: string;
}

const DEFAULT_LLM_URL = `http://${process.env.SPARK_HOST ?? "127.0.0.1"}:18080/v1/chat/completions`;

export const QUERY_EXPANSION_DEFAULTS: QueryExpansionConfig = {
  enabled: true,
  llmUrl: DEFAULT_LLM_URL,
  model: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
  numReformulations: 3,
  maxTokens: 150,
  temperature: 0.7,
  timeoutMs: 15000,
  apiKey: process.env.SPARK_BEARER_TOKEN,
};

/**
 * Build the system prompt for query expansion.
 * Kept as a function so tests can verify the prompt includes the right N.
 */
export function buildExpansionPrompt(numReformulations: number): string {
  return [
    `Generate exactly ${numReformulations} alternative search queries for the given question or claim.`,
    `Each query should use different vocabulary, phrasing, or perspective while preserving the original meaning.`,
    `Strategies to use:`,
    `- Synonyms and domain-specific terminology`,
    `- Active vs. passive voice`,
    `- Different abstraction levels (specific ↔ general)`,
    `- Question form vs. declarative form`,
    `Return one query per line. No numbering, no explanations, no blank lines.`,
  ].join("\n");
}

/** Minimum query length to attempt expansion (too short = garbage in) */
const MIN_QUERY_LENGTH = 10;
/** Maximum length for a single reformulation line */
const MAX_REFORMULATION_LENGTH = 300;
/** Minimum length for a single reformulation line */
const MIN_REFORMULATION_LENGTH = 10;

/**
 * Parse LLM output into clean reformulation strings.
 * Exported for testing.
 */
export function parseReformulations(
  rawOutput: string,
  originalQuery: string,
  maxReformulations: number,
): string[] {
  const lines = rawOutput
    .split("\n")
    .map((line) => line.trim())
    // Strip common LLM formatting artifacts
    .map((line) => line.replace(/^\d+[.)]\s*/, "")) // "1. query" or "1) query"
    .map((line) => line.replace(/^[-•*]\s*/, "")) // "- query" or "• query"
    .map((line) => line.replace(/^["']|["']$/g, "")) // Strip surrounding quotes
    .map((line) => line.trim())
    .filter((line) => line.length >= MIN_REFORMULATION_LENGTH)
    .filter((line) => line.length <= MAX_REFORMULATION_LENGTH)
    // Reject lines that look like LLM meta-commentary
    .filter((line) => !/^(Note:|Here|These|I |The above|Alternative)/i.test(line))
    // Reject lines that are just the original query repeated
    .filter((line) => normalizeForComparison(line) !== normalizeForComparison(originalQuery));

  // Deduplicate near-identical reformulations
  const unique: string[] = [];
  for (const line of lines) {
    const normalized = normalizeForComparison(line);
    const isDupe = unique.some((existing) => normalizeForComparison(existing) === normalized);
    if (!isDupe) unique.push(line);
    if (unique.length >= maxReformulations) break;
  }

  return unique;
}

/**
 * Normalize text for comparison: lowercase, collapse whitespace, strip punctuation.
 */
function normalizeForComparison(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Generate alternative phrasings of a query using an LLM.
 *
 * Always returns an array starting with the original query.
 * On any failure, returns [original] — no worse than current pipeline.
 *
 * @returns [original, ...reformulations]
 */
export async function expandQuery(query: string, config: QueryExpansionConfig): Promise<string[]> {
  if (!config.enabled) return [query];
  if (query.length < MIN_QUERY_LENGTH) return [query];

  const verbose = Boolean(process.env.VERBOSE || process.env.DEBUG_PIPELINE);

  try {
    const result = await attemptExpansion(query, config, verbose);
    if (result.length === 0) {
      if (verbose)
        console.log(
          `[multi-query] expansion returned 0 valid reformulations — using original only`,
        );
      return [query];
    }
    if (verbose) {
      console.log(`[multi-query] generated ${result.length} reformulations:`);
      for (const r of result) console.log(`[multi-query]   → "${r.slice(0, 100)}"`);
    }
    return [query, ...result];
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`[multi-query] expansion failed: ${msg} — using original only`);
    return [query];
  }
}

/**
 * Internal: make the LLM call and parse the response.
 */
async function attemptExpansion(
  query: string,
  config: QueryExpansionConfig,
  verbose: boolean,
): Promise<string[]> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.timeoutMs);
  const t0 = performance.now();

  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (config.apiKey) headers["Authorization"] = `Bearer ${config.apiKey}`;

    const body = JSON.stringify({
      model: config.model,
      messages: [
        { role: "system", content: buildExpansionPrompt(config.numReformulations) },
        { role: "user", content: query },
      ],
      max_tokens: config.maxTokens,
      temperature: config.temperature,
      stream: false,
      // Disable thinking mode for models that support it (e.g. Nemotron-Super).
      // Expansion needs direct output, not reasoning traces.
      chat_template_kwargs: { enable_thinking: false },
    });

    const response = await fetch(config.llmUrl, {
      method: "POST",
      headers,
      body,
      signal: controller.signal,
    });

    if (!response.ok) {
      const errText = await response.text().catch(() => "");
      throw new Error(`LLM returned ${response.status}: ${errText.slice(0, 200)}`);
    }

    const data = (await response.json()) as {
      choices?: Array<{ message?: { content?: string } }>;
    };

    const rawContent = data?.choices?.[0]?.message?.content ?? "";
    const elapsed = Math.round(performance.now() - t0);

    if (verbose) {
      console.log(`[multi-query] LLM response in ${elapsed}ms, raw length=${rawContent.length}`);
      console.log(`[multi-query] raw output: "${rawContent.slice(0, 300).replace(/\n/g, "\\n")}"`);
    }

    if (!rawContent || rawContent.length < MIN_REFORMULATION_LENGTH) {
      console.log(
        `[multi-query] LLM returned empty/too-short content (${rawContent.length} chars) in ${elapsed}ms`,
      );
      return [];
    }

    // Check for LLM thinking/refusal artifacts
    if (
      rawContent.startsWith("<think>") ||
      rawContent.includes("I cannot") ||
      rawContent.includes("I'm sorry")
    ) {
      console.log(
        `[multi-query] LLM returned thinking trace or refusal in ${elapsed}ms — skipping`,
      );
      return [];
    }

    const reformulations = parseReformulations(rawContent, query, config.numReformulations);
    console.log(
      `[multi-query] ${reformulations.length}/${config.numReformulations} valid reformulations in ${elapsed}ms`,
    );
    return reformulations;
  } finally {
    clearTimeout(timeout);
  }
}
