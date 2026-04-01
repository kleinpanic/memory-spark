/**
 * Tests for Multi-Query Expansion (Phase 11B)
 *
 * Covers: parseReformulations, expandQuery, buildExpansionPrompt,
 * graceful degradation, quality gates, and edge cases.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  expandQuery,
  parseReformulations,
  buildExpansionPrompt,
  type QueryExpansionConfig,
  QUERY_EXPANSION_DEFAULTS,
} from "../src/query/expander.js";

// ── Test config (no real LLM calls) ──────────────────────────────────────

const TEST_CONFIG: QueryExpansionConfig = {
  ...QUERY_EXPANSION_DEFAULTS,
  enabled: true,
  llmUrl: "http://mock-llm:18080/v1/chat/completions",
  timeoutMs: 5000,
  apiKey: "test-token",
};

/** Helper: create a mock LLM response */
function mockLlmResponse(content: string, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => content,
    json: async () => ({
      choices: [{ message: { content } }],
    }),
  } as unknown as Response;
}

// ── parseReformulations ──────────────────────────────────────────────────

describe("parseReformulations", () => {
  const original = "What causes insulin resistance?";

  it("parses clean newline-separated output", () => {
    const raw = [
      "How does glucose metabolism become impaired?",
      "Mechanisms of cellular insulin signaling failure",
      "Factors leading to reduced insulin sensitivity",
    ].join("\n");

    const result = parseReformulations(raw, original, 3);
    expect(result).toHaveLength(3);
    expect(result[0]).toBe("How does glucose metabolism become impaired?");
    expect(result[1]).toBe("Mechanisms of cellular insulin signaling failure");
    expect(result[2]).toBe("Factors leading to reduced insulin sensitivity");
  });

  it("strips numbered prefixes (1. 2. 3.)", () => {
    const raw = [
      "1. How does glucose metabolism become impaired?",
      "2. Mechanisms of cellular insulin signaling failure",
      "3) Factors leading to reduced insulin sensitivity",
    ].join("\n");

    const result = parseReformulations(raw, original, 3);
    expect(result).toHaveLength(3);
    expect(result[0]).toBe("How does glucose metabolism become impaired?");
  });

  it("strips bullet prefixes (- • *)", () => {
    const raw = [
      "- How does glucose metabolism become impaired?",
      "• Mechanisms of cellular insulin signaling failure",
      "* Factors leading to reduced insulin sensitivity",
    ].join("\n");

    const result = parseReformulations(raw, original, 3);
    expect(result).toHaveLength(3);
    expect(result[0]).toBe("How does glucose metabolism become impaired?");
  });

  it("strips surrounding quotes", () => {
    const raw = [
      '"How does glucose metabolism become impaired?"',
      "'Mechanisms of cellular insulin signaling failure'",
    ].join("\n");

    const result = parseReformulations(raw, original, 3);
    expect(result).toHaveLength(2);
    expect(result[0]).toBe("How does glucose metabolism become impaired?");
  });

  it("filters empty and whitespace-only lines", () => {
    const raw = [
      "How does glucose metabolism become impaired?",
      "",
      "   ",
      "Mechanisms of cellular insulin signaling failure",
    ].join("\n");

    const result = parseReformulations(raw, original, 3);
    expect(result).toHaveLength(2);
  });

  it("rejects lines shorter than 10 chars", () => {
    const raw = [
      "short",
      "How does glucose metabolism become impaired?",
      "ok query?",
    ].join("\n");

    const result = parseReformulations(raw, original, 3);
    expect(result).toHaveLength(1);
    expect(result[0]).toBe("How does glucose metabolism become impaired?");
  });

  it("rejects lines longer than 300 chars", () => {
    const longLine = "A".repeat(301);
    const raw = [
      longLine,
      "How does glucose metabolism become impaired?",
    ].join("\n");

    const result = parseReformulations(raw, original, 3);
    expect(result).toHaveLength(1);
    expect(result[0]).toBe("How does glucose metabolism become impaired?");
  });

  it("rejects LLM meta-commentary lines", () => {
    const raw = [
      "Here are three alternative queries:",
      "How does glucose metabolism become impaired?",
      "Note: These are semantically equivalent queries.",
      "Mechanisms of cellular insulin signaling failure",
      "I hope these help with your search.",
    ].join("\n");

    const result = parseReformulations(raw, original, 3);
    expect(result).toHaveLength(2);
    expect(result[0]).toBe("How does glucose metabolism become impaired?");
    expect(result[1]).toBe("Mechanisms of cellular insulin signaling failure");
  });

  it("rejects lines that duplicate the original query", () => {
    const raw = [
      "What causes insulin resistance?",  // exact duplicate
      "How does glucose metabolism become impaired?",
      "what causes insulin resistance",    // case-insensitive duplicate
    ].join("\n");

    const result = parseReformulations(raw, original, 3);
    expect(result).toHaveLength(1);
    expect(result[0]).toBe("How does glucose metabolism become impaired?");
  });

  it("deduplicates near-identical reformulations", () => {
    const raw = [
      "How does glucose metabolism become impaired?",
      "How does glucose metabolism become impaired?",  // exact dupe
      "How does  glucose  metabolism  become  impaired?", // whitespace dupe
      "Mechanisms of cellular insulin signaling failure",
    ].join("\n");

    const result = parseReformulations(raw, original, 3);
    expect(result).toHaveLength(2);
  });

  it("respects maxReformulations limit", () => {
    const raw = [
      "Query reformulation one that is valid",
      "Query reformulation two that is valid",
      "Query reformulation three that is valid",
      "Query reformulation four that is valid",
      "Query reformulation five that is valid",
    ].join("\n");

    const result = parseReformulations(raw, original, 2);
    expect(result).toHaveLength(2);
  });

  it("handles completely empty input", () => {
    const result = parseReformulations("", original, 3);
    expect(result).toHaveLength(0);
  });

  it("handles input with only garbage", () => {
    const raw = [
      "ab",
      "",
      "   ",
      "Note: these are queries",
    ].join("\n");

    const result = parseReformulations(raw, original, 3);
    expect(result).toHaveLength(0);
  });
});

// ── buildExpansionPrompt ─────────────────────────────────────────────────

describe("buildExpansionPrompt", () => {
  it("includes the requested number of reformulations", () => {
    const prompt = buildExpansionPrompt(3);
    expect(prompt).toContain("exactly 3");
  });

  it("includes formatting instructions", () => {
    const prompt = buildExpansionPrompt(5);
    expect(prompt).toContain("exactly 5");
    expect(prompt).toContain("one query per line");
    expect(prompt).toContain("No numbering");
  });

  it("mentions diversity strategies", () => {
    const prompt = buildExpansionPrompt(3);
    expect(prompt).toContain("Synonym");
    expect(prompt).toContain("Active vs. passive");
  });
});

// ── expandQuery ──────────────────────────────────────────────────────────

describe("expandQuery", () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it("returns original + reformulations on success", async () => {
    const llmOutput = [
      "How does glucose metabolism become impaired?",
      "Mechanisms of cellular insulin signaling failure",
      "Factors leading to reduced insulin sensitivity",
    ].join("\n");

    fetchSpy.mockResolvedValueOnce(mockLlmResponse(llmOutput));

    const result = await expandQuery("What causes insulin resistance?", TEST_CONFIG);
    expect(result).toHaveLength(4);
    expect(result[0]).toBe("What causes insulin resistance?");
    expect(result[1]).toBe("How does glucose metabolism become impaired?");
  });

  it("always includes original query as first element", async () => {
    const llmOutput = "Alternative phrasing of the search query";
    fetchSpy.mockResolvedValueOnce(mockLlmResponse(llmOutput));

    const result = await expandQuery("What causes insulin resistance?", TEST_CONFIG);
    expect(result[0]).toBe("What causes insulin resistance?");
  });

  it("returns [original] when disabled", async () => {
    const config = { ...TEST_CONFIG, enabled: false };
    const result = await expandQuery("What causes insulin resistance?", config);
    expect(result).toEqual(["What causes insulin resistance?"]);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("returns [original] when query too short", async () => {
    const result = await expandQuery("short", TEST_CONFIG);
    expect(result).toEqual(["short"]);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("returns [original] on network error", async () => {
    fetchSpy.mockRejectedValueOnce(new Error("ECONNREFUSED"));

    const result = await expandQuery("What causes insulin resistance?", TEST_CONFIG);
    expect(result).toEqual(["What causes insulin resistance?"]);
  });

  it("returns [original] on HTTP error", async () => {
    fetchSpy.mockResolvedValueOnce(mockLlmResponse("Internal Server Error", 500));

    const result = await expandQuery("What causes insulin resistance?", TEST_CONFIG);
    expect(result).toEqual(["What causes insulin resistance?"]);
  });

  it("returns [original] on empty LLM response", async () => {
    fetchSpy.mockResolvedValueOnce(mockLlmResponse(""));

    const result = await expandQuery("What causes insulin resistance?", TEST_CONFIG);
    expect(result).toEqual(["What causes insulin resistance?"]);
  });

  it("returns [original] on LLM thinking trace", async () => {
    fetchSpy.mockResolvedValueOnce(mockLlmResponse("<think>Let me consider the query...</think>"));

    const result = await expandQuery("What causes insulin resistance?", TEST_CONFIG);
    expect(result).toEqual(["What causes insulin resistance?"]);
  });

  it("returns [original] on LLM refusal", async () => {
    fetchSpy.mockResolvedValueOnce(mockLlmResponse("I'm sorry, I cannot generate search queries."));

    const result = await expandQuery("What causes insulin resistance?", TEST_CONFIG);
    expect(result).toEqual(["What causes insulin resistance?"]);
  });

  it("returns [original] on timeout (abort)", async () => {
    const config = { ...TEST_CONFIG, timeoutMs: 1 };
    fetchSpy.mockImplementationOnce(
      () => new Promise((resolve) => setTimeout(() => resolve(mockLlmResponse("too late")), 100)),
    );

    const result = await expandQuery("What causes insulin resistance?", config);
    expect(result).toEqual(["What causes insulin resistance?"]);
  });

  it("sends correct request body to LLM", async () => {
    const llmOutput = "Alternative query reformulation here";
    fetchSpy.mockResolvedValueOnce(mockLlmResponse(llmOutput));

    await expandQuery("Test query for validation", TEST_CONFIG);

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, opts] = fetchSpy.mock.calls[0]! as [string, RequestInit];
    expect(url).toBe(TEST_CONFIG.llmUrl);

    const body = JSON.parse(opts.body as string);
    expect(body.model).toBe(TEST_CONFIG.model);
    expect(body.temperature).toBe(0.7);
    expect(body.max_tokens).toBe(150);
    expect(body.messages).toHaveLength(2);
    expect(body.messages[0].role).toBe("system");
    expect(body.messages[1].role).toBe("user");
    expect(body.messages[1].content).toBe("Test query for validation");
    // Thinking disabled for direct output
    expect(body.chat_template_kwargs).toEqual({ enable_thinking: false });
  });

  it("sends Bearer token when apiKey configured", async () => {
    fetchSpy.mockResolvedValueOnce(mockLlmResponse("Some reformulation of the query"));

    await expandQuery("Test query for auth check", TEST_CONFIG);

    const [, opts] = fetchSpy.mock.calls[0]! as [string, RequestInit];
    const headers = opts.headers as Record<string, string>;
    expect(headers["Authorization"]).toBe("Bearer test-token");
  });

  it("omits Authorization header when no apiKey", async () => {
    const config = { ...TEST_CONFIG, apiKey: undefined };
    fetchSpy.mockResolvedValueOnce(mockLlmResponse("Some reformulation of the query"));

    await expandQuery("Test query without auth", config);

    const [, opts] = fetchSpy.mock.calls[0]! as [string, RequestInit];
    const headers = opts.headers as Record<string, string>;
    expect(headers["Authorization"]).toBeUndefined();
  });

  it("filters garbage from mixed LLM output", async () => {
    const llmOutput = [
      "Here are the queries:",                          // meta-commentary → filtered
      "1. How does glucose metabolism become impaired?", // numbered → stripped
      "ab",                                              // too short → filtered
      "2. Mechanisms of insulin signaling failure in cells", // numbered → stripped
      "",                                                // empty → filtered
    ].join("\n");

    fetchSpy.mockResolvedValueOnce(mockLlmResponse(llmOutput));

    const result = await expandQuery("What causes insulin resistance?", TEST_CONFIG);
    expect(result).toHaveLength(3); // original + 2 valid
    expect(result[0]).toBe("What causes insulin resistance?");
    expect(result[1]).toBe("How does glucose metabolism become impaired?");
    expect(result[2]).toBe("Mechanisms of insulin signaling failure in cells");
  });

  it("handles malformed JSON response gracefully", async () => {
    const badResponse = {
      ok: true,
      status: 200,
      text: async () => "not json",
      json: async () => { throw new SyntaxError("Unexpected token"); },
    } as unknown as Response;

    fetchSpy.mockResolvedValueOnce(badResponse);

    const result = await expandQuery("What causes insulin resistance?", TEST_CONFIG);
    expect(result).toEqual(["What causes insulin resistance?"]);
  });

  it("handles response with missing choices array", async () => {
    const weirdResponse = {
      ok: true,
      status: 200,
      text: async () => "{}",
      json: async () => ({}),
    } as unknown as Response;

    fetchSpy.mockResolvedValueOnce(weirdResponse);

    const result = await expandQuery("What causes insulin resistance?", TEST_CONFIG);
    expect(result).toEqual(["What causes insulin resistance?"]);
  });

  it("respects numReformulations config", async () => {
    const config = { ...TEST_CONFIG, numReformulations: 2 };
    const llmOutput = [
      "Reformulation one of the search query",
      "Reformulation two of the search query",
      "Reformulation three of the search query",  // should be trimmed
    ].join("\n");

    fetchSpy.mockResolvedValueOnce(mockLlmResponse(llmOutput));

    const result = await expandQuery("What causes insulin resistance?", config);
    // original + 2 (not 3)
    expect(result).toHaveLength(3);
  });
});
