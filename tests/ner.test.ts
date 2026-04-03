/**
 * Tests for src/classify/ner.ts
 *
 * Mocks globalThis.fetch to simulate the Spark NER API
 * (POST /v1/extract) without requiring a live Spark endpoint.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { tagEntities } from "../src/classify/ner.js";
import type { MemorySparkConfig } from "../src/config.js";

// ---------------------------------------------------------------------------
// Minimal config factory
// ---------------------------------------------------------------------------

function makeConfig(overrides: { apiKey?: string; nerUrl?: string } = {}): MemorySparkConfig {
  return {
    spark: {
      ner: overrides.nerUrl ?? "http://spark-host:18112",
      zeroShot: "http://spark-host:18113",
      embed: "http://spark-host:18091/v1",
      rerank: "http://spark-host:18096/v1",
      ocr: "http://spark-host:18097",
      glmOcr: "http://spark-host:18080/v1",
      summarizer: "http://spark-host:18110",
      stt: "http://spark-host:18094",
    },
    embed: {
      provider: "spark",
      spark: {
        baseUrl: "http://spark-host:18091/v1",
        model: "nvidia/llama-embed-nemotron-8b",
        apiKey: overrides.apiKey,
      },
    },
  } as unknown as MemorySparkConfig;
}

// ---------------------------------------------------------------------------
// NER response fixture builder
// ---------------------------------------------------------------------------

type NerEntity = {
  entity_group: string;
  score: number;
  word: string;
  start: number;
  end: number;
};

function makeNerResponse(entities: NerEntity[]) {
  return {
    ok: true,
    json: async () => ({ entities, count: entities.length }),
  } as Response;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("tagEntities", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  // ── Entity extraction ─────────────────────────────────────────────────────

  it("extracts entity words from a successful response", async () => {
    (globalThis as any).fetch = vi.fn().mockResolvedValueOnce(
      makeNerResponse([
        { entity_group: "PER", score: 0.99, word: "Alice", start: 0, end: 5 },
        { entity_group: "ORG", score: 0.95, word: "OpenClaw", start: 10, end: 18 },
      ]),
    );

    const result = await tagEntities("Alice works at OpenClaw.", makeConfig());

    expect(result).toContain("Alice");
    expect(result).toContain("OpenClaw");
  });

  it("deduplicates repeated entity words", async () => {
    (globalThis as any).fetch = vi.fn().mockResolvedValueOnce(
      makeNerResponse([
        { entity_group: "PER", score: 0.99, word: "Alice", start: 0, end: 5 },
        { entity_group: "PER", score: 0.98, word: "Alice", start: 20, end: 25 },
      ]),
    );

    const result = await tagEntities("Alice said hello to Alice.", makeConfig());

    expect(result.filter((e) => e === "Alice")).toHaveLength(1);
  });

  it("strips leading ## from sub-word tokens", async () => {
    (globalThis as any).fetch = vi
      .fn()
      .mockResolvedValueOnce(
        makeNerResponse([{ entity_group: "ORG", score: 0.88, word: "##Corp", start: 5, end: 10 }]),
      );

    const result = await tagEntities("BigCorp is here.", makeConfig());

    expect(result).toContain("Corp");
    expect(result.some((e) => e.startsWith("##"))).toBe(false);
  });

  it("filters out entities with score <= 0.7", async () => {
    (globalThis as any).fetch = vi.fn().mockResolvedValueOnce(
      makeNerResponse([
        { entity_group: "PER", score: 0.65, word: "Charlie", start: 0, end: 7 },
        { entity_group: "LOC", score: 0.85, word: "London", start: 15, end: 21 },
      ]),
    );

    const result = await tagEntities("Charlie visited London.", makeConfig());

    expect(result).not.toContain("Charlie");
    expect(result).toContain("London");
  });

  it("filters out entity words with length <= 2", async () => {
    (globalThis as any).fetch = vi.fn().mockResolvedValueOnce(
      makeNerResponse([
        { entity_group: "ORG", score: 0.95, word: "UN", start: 0, end: 2 },
        { entity_group: "PER", score: 0.97, word: "IBM", start: 5, end: 8 },
      ]),
    );

    const result = await tagEntities("UN and IBM.", makeConfig());

    // "UN" has length 2 → filtered; "IBM" has length 3 → kept
    expect(result).not.toContain("UN");
    expect(result).toContain("IBM");
  });

  it("trims whitespace from entity words", async () => {
    (globalThis as any).fetch = vi
      .fn()
      .mockResolvedValueOnce(
        makeNerResponse([{ entity_group: "PER", score: 0.92, word: "  Bob  ", start: 0, end: 7 }]),
      );

    const result = await tagEntities("Bob was here.", makeConfig());

    expect(result).toContain("Bob");
    expect(result.some((e) => e !== e.trim())).toBe(false);
  });

  // ── Empty results ─────────────────────────────────────────────────────────

  it("returns an empty array when API returns no entities", async () => {
    (globalThis as any).fetch = vi.fn().mockResolvedValueOnce(makeNerResponse([]));

    const result = await tagEntities("Nothing to tag here.", makeConfig());

    expect(result).toEqual([]);
  });

  it("returns an empty array when all entities have low scores", async () => {
    (globalThis as any).fetch = vi.fn().mockResolvedValueOnce(
      makeNerResponse([
        { entity_group: "PER", score: 0.5, word: "Dave", start: 0, end: 4 },
        { entity_group: "ORG", score: 0.6, word: "Acme", start: 8, end: 12 },
      ]),
    );

    const result = await tagEntities("Dave at Acme.", makeConfig());

    expect(result).toEqual([]);
  });

  // ── API error handling ────────────────────────────────────────────────────

  it("returns [] when API responds with non-OK status (does not throw)", async () => {
    (globalThis as any).fetch = vi.fn().mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => ({}),
    } as unknown as Response);

    await expect(tagEntities("Some text.", makeConfig())).resolves.toEqual([]);
  });

  it("returns [] when fetch rejects (network error, does not throw)", async () => {
    (globalThis as any).fetch = vi.fn().mockRejectedValueOnce(new Error("ECONNREFUSED"));

    await expect(tagEntities("Some text.", makeConfig())).resolves.toEqual([]);
  });

  it("returns [] when fetch times out (AbortError, does not throw)", async () => {
    const abortErr = new DOMException("The operation was aborted.", "AbortError");
    (globalThis as any).fetch = vi.fn().mockRejectedValueOnce(abortErr);

    await expect(tagEntities("Some text.", makeConfig())).resolves.toEqual([]);
  });

  it("returns [] when response JSON is malformed (does not throw)", async () => {
    (globalThis as any).fetch = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => {
        throw new SyntaxError("Unexpected token");
      },
    } as unknown as Response);

    await expect(tagEntities("Some text.", makeConfig())).resolves.toEqual([]);
  });

  // ── Auth header ───────────────────────────────────────────────────────────

  it("includes Authorization header when apiKey is provided", async () => {
    (globalThis as any).fetch = vi.fn().mockResolvedValueOnce(makeNerResponse([]));

    await tagEntities("Test text.", makeConfig({ apiKey: "my-secret-key" }));

    const [, init] = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0] as [
      string,
      RequestInit,
    ];
    expect((init.headers as Record<string, string>)["Authorization"]).toBe("Bearer my-secret-key");
  });

  it("does NOT include Authorization header when apiKey is absent", async () => {
    (globalThis as any).fetch = vi.fn().mockResolvedValueOnce(makeNerResponse([]));

    await tagEntities("Test text.", makeConfig({ apiKey: undefined }));

    const [, init] = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0] as [
      string,
      RequestInit,
    ];
    expect((init.headers as Record<string, string>)["Authorization"]).toBeUndefined();
  });

  // ── URL and request body ──────────────────────────────────────────────────

  it("calls the correct NER endpoint URL", async () => {
    (globalThis as any).fetch = vi.fn().mockResolvedValueOnce(makeNerResponse([]));

    await tagEntities("Test.", makeConfig({ nerUrl: "http://custom-ner:19001" }));

    const [url] = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0] as [
      string,
    ];
    expect(url).toBe("http://custom-ner:19001/v1/extract");
  });

  it("sends text in the POST body", async () => {
    (globalThis as any).fetch = vi.fn().mockResolvedValueOnce(makeNerResponse([]));

    await tagEntities("Hello NER.", makeConfig());

    const [, init] = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0] as [
      string,
      RequestInit,
    ];
    const body = JSON.parse(init.body as string) as { text: string };
    expect(body.text).toBe("Hello NER.");
  });

  it("truncates text to 2000 chars before sending", async () => {
    (globalThis as any).fetch = vi.fn().mockResolvedValueOnce(makeNerResponse([]));

    const longText = "z".repeat(3000);
    await tagEntities(longText, makeConfig());

    const [, init] = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0] as [
      string,
      RequestInit,
    ];
    const body = JSON.parse(init.body as string) as { text: string };
    expect(body.text.length).toBe(2000);
  });
});
