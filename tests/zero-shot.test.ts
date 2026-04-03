/**
 * Tests for src/classify/zero-shot.ts
 *
 * Mocks globalThis.fetch to simulate the Spark zero-shot classifier API
 * (POST /v1/classify) without requiring a live Spark endpoint.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { classifyForCapture } from "../src/classify/zero-shot.js";
import type { MemorySparkConfig } from "../src/config.js";

// ---------------------------------------------------------------------------
// Minimal config factory
// ---------------------------------------------------------------------------

function makeConfig(overrides: { apiKey?: string; zeroShotUrl?: string } = {}): MemorySparkConfig {
  return {
    spark: {
      zeroShot: overrides.zeroShotUrl ?? "http://spark-host:18113",
      ner: "http://spark-host:18112",
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
    // The rest of the fields are not used by zero-shot but are required by the type.
  } as unknown as MemorySparkConfig;
}

// ---------------------------------------------------------------------------
// Mock helpers
// ---------------------------------------------------------------------------

function mockFetchOk(labels: string[], scores: number[]) {
  (globalThis as any).fetch = vi.fn().mockResolvedValueOnce({
    ok: true,
    json: async () => ({ labels, scores }),
  } as Response);
}

function mockFetchError(status = 500) {
  (globalThis as any).fetch = vi.fn().mockResolvedValueOnce({
    ok: false,
    status,
    json: async () => ({}),
  } as unknown as Response);
}

function mockFetchReject(err: Error = new Error("Network failure")) {
  (globalThis as any).fetch = vi.fn().mockRejectedValueOnce(err);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("classifyForCapture", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  // ── Success path ──────────────────────────────────────────────────────────

  it("returns the top label and score when API succeeds", async () => {
    mockFetchOk(["fact", "preference", "decision", "code-snippet"], [0.92, 0.05, 0.02, 0.01]);

    const result = await classifyForCapture("The sky is blue.", makeConfig());

    expect(result.label).toBe("fact");
    expect(result.score).toBeCloseTo(0.92);
  });

  it("returns preference label when it ranks first", async () => {
    mockFetchOk(["preference", "fact", "decision", "code-snippet"], [0.88, 0.07, 0.03, 0.02]);

    const result = await classifyForCapture("I prefer dark mode.", makeConfig());

    expect(result.label).toBe("preference");
    expect(result.score).toBeCloseTo(0.88);
  });

  it("returns code-snippet label for code text", async () => {
    mockFetchOk(["code-snippet", "fact", "preference", "decision"], [0.91, 0.05, 0.03, 0.01]);

    const result = await classifyForCapture("const x = 42;", makeConfig());

    expect(result.label).toBe("code-snippet");
    expect(result.score).toBeCloseTo(0.91);
  });

  // ── Score thresholding ────────────────────────────────────────────────────

  it("returns label=none when top score is below default minConfidence (0.75)", async () => {
    mockFetchOk(["fact", "preference"], [0.6, 0.4]);

    const result = await classifyForCapture("Some text.", makeConfig());

    expect(result.label).toBe("none");
    expect(result.score).toBeCloseTo(0.6);
  });

  it("returns label=none when top score is below a custom minConfidence", async () => {
    mockFetchOk(["fact", "preference"], [0.8, 0.2]);

    // Pass a higher threshold than the returned score
    const result = await classifyForCapture("Some text.", makeConfig(), 0.9);

    expect(result.label).toBe("none");
    expect(result.score).toBeCloseTo(0.8);
  });

  it("returns the label when score equals minConfidence exactly", async () => {
    mockFetchOk(["decision", "fact"], [0.75, 0.25]);

    const result = await classifyForCapture("We decided to use TypeScript.", makeConfig(), 0.75);

    expect(result.label).toBe("decision");
  });

  // ── Fallback / error handling ─────────────────────────────────────────────

  it("returns {label: 'none', score: 0} when API returns non-OK status", async () => {
    mockFetchError(503);

    const result = await classifyForCapture("Some text.", makeConfig());

    expect(result.label).toBe("none");
    expect(result.score).toBe(0);
  });

  it("does not throw when fetch rejects (network error)", async () => {
    mockFetchReject();

    await expect(classifyForCapture("Some text.", makeConfig())).resolves.toEqual({
      label: "none",
      score: 0,
    });
  });

  it("does not throw when fetch rejects with AbortError (timeout)", async () => {
    const abortErr = new DOMException("The operation was aborted.", "AbortError");
    mockFetchReject(abortErr);

    await expect(classifyForCapture("Some text.", makeConfig())).resolves.toEqual({
      label: "none",
      score: 0,
    });
  });

  // ── Auth header ───────────────────────────────────────────────────────────

  it("includes Authorization header when apiKey is provided", async () => {
    mockFetchOk(["fact"], [0.95]);

    await classifyForCapture("Some text.", makeConfig({ apiKey: "test-token-123" }));

    const [, init] = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0] as [
      string,
      RequestInit,
    ];
    expect((init.headers as Record<string, string>)["Authorization"]).toBe("Bearer test-token-123");
  });

  it("does NOT include Authorization header when apiKey is absent", async () => {
    mockFetchOk(["fact"], [0.95]);

    await classifyForCapture("Some text.", makeConfig({ apiKey: undefined }));

    const [, init] = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0] as [
      string,
      RequestInit,
    ];
    expect((init.headers as Record<string, string>)["Authorization"]).toBeUndefined();
  });

  // ── URL construction ──────────────────────────────────────────────────────

  it("calls the correct endpoint URL", async () => {
    mockFetchOk(["fact"], [0.95]);

    await classifyForCapture("Some text.", makeConfig({ zeroShotUrl: "http://custom:19000" }));

    const [url] = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0] as [
      string,
    ];
    expect(url).toBe("http://custom:19000/v1/classify");
  });

  // ── Request body ──────────────────────────────────────────────────────────

  it("sends text and labels in the request body", async () => {
    mockFetchOk(["fact"], [0.95]);

    await classifyForCapture("Hello world.", makeConfig());

    const [, init] = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0] as [
      string,
      RequestInit,
    ];
    const body = JSON.parse(init.body as string) as { text: string; labels: string[] };
    expect(body.text).toBe("Hello world.");
    expect(body.labels).toContain("fact");
    expect(body.labels).toContain("preference");
    expect(body.labels).toContain("decision");
    expect(body.labels).toContain("code-snippet");
    // "none" is filtered out from the request labels
    expect(body.labels).not.toContain("none");
  });

  it("truncates text longer than 2000 chars before sending", async () => {
    mockFetchOk(["fact"], [0.95]);

    const longText = "a".repeat(3000);
    await classifyForCapture(longText, makeConfig());

    const [, init] = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0] as [
      string,
      RequestInit,
    ];
    const body = JSON.parse(init.body as string) as { text: string };
    expect(body.text.length).toBe(2000);
  });
});
