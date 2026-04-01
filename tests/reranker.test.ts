/* eslint-disable @typescript-eslint/no-explicit-any */
// @ts-nocheck — Node 22's globalThis.fetch has strict `preconnect` typing
// that conflicts with vitest's vi.fn() mock. Tests run correctly via vitest.
/**
 * Reranker Unit & Integration Tests (Phase 10B)
 *
 * Tests the unified reranker pipeline including:
 * - Query normalization (declarative → interrogative)
 * - Logit recovery from sigmoid-compressed scores
 * - Score blending with alpha interpolation
 * - alphaOverride per-call behavior
 * - Spread guard logic
 * - Error handling and fallback
 * - End-to-end pipeline validation
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  blendScores,
  recoverLogit,
  isQuestion,
  normalizeQueryForReranker,
  createReranker,
  type Reranker,
  type RerankOptions,
} from "../src/rerank/reranker.js";
import type { SearchResult } from "../src/storage/backend.js";

// ── Test Helpers ──────────────────────────────────────────────────────────

function makeResult(id: string, score: number, text = `Document ${id}`): SearchResult {
  return {
    chunk: {
      id,
      text,
      path: `/test/${id}`,
      source: "memory" as const,
      agent_id: "test",
      start_line: 0,
      end_line: 0,
      vector: [],
      updated_at: new Date().toISOString(),
    },
    score,
    snippet: text,
    vector: new Array(4096).fill(0.1) as number[],
  };
}

function makeRerankResult(index: number, relevanceScore: number) {
  return { index, relevance_score: relevanceScore };
}

// ── recoverLogit ──────────────────────────────────────────────────────────

describe("recoverLogit", () => {
  it("recovers logit=0 from sigmoid=0.5", () => {
    const logit = recoverLogit(0.5);
    expect(logit).toBeCloseTo(0, 5);
  });

  it("recovers positive logit from sigmoid > 0.5", () => {
    const logit = recoverLogit(0.9);
    expect(logit).toBeCloseTo(Math.log(0.9 / 0.1), 5); // ≈ 2.197
    expect(logit).toBeGreaterThan(0);
  });

  it("recovers negative logit from sigmoid < 0.5", () => {
    const logit = recoverLogit(0.2);
    expect(logit).toBeCloseTo(Math.log(0.2 / 0.8), 5); // ≈ -1.386
    expect(logit).toBeLessThan(0);
  });

  it("clamps at boundaries to avoid ±Infinity", () => {
    expect(Number.isFinite(recoverLogit(0))).toBe(true);
    expect(Number.isFinite(recoverLogit(1))).toBe(true);
    expect(recoverLogit(0)).toBeLessThan(-10);
    expect(recoverLogit(1)).toBeGreaterThan(10);
  });

  it("is monotonically increasing", () => {
    const values = [0.1, 0.3, 0.5, 0.7, 0.9];
    const logits = values.map(recoverLogit);
    for (let i = 1; i < logits.length; i++) {
      expect(logits[i]!).toBeGreaterThan(logits[i - 1]!);
    }
  });

  it("handles typical Nemotron compressed range (0.83–1.0)", () => {
    const low = recoverLogit(0.83);
    const high = recoverLogit(0.99);
    const spread = high - low;
    // Logit spread should be meaningful even from a narrow sigmoid range
    expect(spread).toBeGreaterThan(2);
  });
});

// ── isQuestion ────────────────────────────────────────────────────────────

describe("isQuestion", () => {
  it("detects questions ending with ?", () => {
    expect(isQuestion("What is memory?")).toBe(true);
    expect(isQuestion("Is this relevant?")).toBe(true);
    expect(isQuestion("Really?")).toBe(true);
  });

  it("detects questions starting with question words", () => {
    expect(isQuestion("How does chunking work")).toBe(true);
    expect(isQuestion("What are embeddings")).toBe(true);
    expect(isQuestion("Why is the sky blue")).toBe(true);
    expect(isQuestion("Who discovered DNA")).toBe(true);
    expect(isQuestion("When was it published")).toBe(true);
    expect(isQuestion("Where is the data stored")).toBe(true);
    expect(isQuestion("Which model is best")).toBe(true);
  });

  it("detects auxiliary verb questions", () => {
    expect(isQuestion("Is it true that X")).toBe(true);
    expect(isQuestion("Are stem cells pluripotent")).toBe(true);
    expect(isQuestion("Does the model support batching")).toBe(true);
    expect(isQuestion("Can this handle 4096 dimensions")).toBe(true);
    expect(isQuestion("Has this been validated")).toBe(true);
    expect(isQuestion("Should we use RRF")).toBe(true);
    expect(isQuestion("Would this improve recall")).toBe(true);
  });

  it("rejects declarative statements", () => {
    expect(isQuestion("Stem cells can differentiate")).toBe(false);
    expect(isQuestion("The hippocampus is important for memory")).toBe(false);
    expect(isQuestion("0-dimensional biomaterials show inductive properties")).toBe(false);
    expect(isQuestion("BRCA1 mutations increase cancer risk")).toBe(false);
  });

  it("handles edge cases", () => {
    expect(isQuestion("")).toBe(false);
    expect(isQuestion("   ")).toBe(false);
    expect(isQuestion("?")).toBe(true);
    expect(isQuestion("A")).toBe(false);
  });
});

// ── normalizeQueryForReranker ──────────────────────────────────────────

describe("normalizeQueryForReranker", () => {
  it("returns questions unchanged", () => {
    const q = "What is the role of BRCA1?";
    expect(normalizeQueryForReranker(q)).toBe(q);
  });

  it("converts declarative claims to interrogative", () => {
    expect(normalizeQueryForReranker("Stem cells can differentiate")).toBe(
      "Is it true that stem cells can differentiate?",
    );
  });

  it("strips trailing period before converting", () => {
    expect(normalizeQueryForReranker("The hippocampus is important.")).toBe(
      "Is it true that the hippocampus is important?",
    );
  });

  it("lowercases the first character", () => {
    expect(normalizeQueryForReranker("BRCA1 mutations increase cancer risk")).toBe(
      "Is it true that bRCA1 mutations increase cancer risk?",
    );
  });

  it("handles already-normalized questions (no double-prefix)", () => {
    const q = "Is it true that stem cells can differentiate?";
    expect(normalizeQueryForReranker(q)).toBe(q);
  });

  it("handles auxiliary-verb questions", () => {
    const q = "Does chemotherapy affect immune response";
    expect(normalizeQueryForReranker(q)).toBe(q);
  });
});

// ── blendScores ───────────────────────────────────────────────────────────

describe("blendScores", () => {
  const pool = [
    makeResult("a", 0.9),
    makeResult("b", 0.7),
    makeResult("c", 0.5),
  ];

  describe("alpha=0 (pure reranker)", () => {
    it("returns results ordered by reranker logit scores", () => {
      const rerankResults = [
        makeRerankResult(2, 0.99), // c: highest reranker
        makeRerankResult(0, 0.85), // a: mid reranker
        makeRerankResult(1, 0.80), // b: lowest reranker
      ];
      const blended = blendScores(pool, rerankResults, 0);
      expect(blended[0]!.chunk.id).toBe("c");
      expect(blended[1]!.chunk.id).toBe("a");
      expect(blended[2]!.chunk.id).toBe("b");
    });

    it("uses recovered logits for better discrimination", () => {
      // Tight sigmoid scores but different logits
      const rerankResults = [
        makeRerankResult(0, 0.95), // a: logit ≈ 2.94
        makeRerankResult(1, 0.90), // b: logit ≈ 2.20
        makeRerankResult(2, 0.85), // c: logit ≈ 1.73
      ];
      const blended = blendScores(pool, rerankResults, 0);
      // Order should match reranker order
      expect(blended[0]!.chunk.id).toBe("a");
      expect(blended[1]!.chunk.id).toBe("b");
      expect(blended[2]!.chunk.id).toBe("c");
      // Scores should be spread across [0, 1]
      expect(blended[0]!.score).toBeCloseTo(1.0, 1);
      expect(blended[2]!.score).toBeCloseTo(0.0, 1);
    });
  });

  describe("alpha=1.0 (ignore reranker)", () => {
    it("preserves original ordering", () => {
      const rerankResults = [
        makeRerankResult(2, 0.99), // c: highest reranker — but ignored
        makeRerankResult(0, 0.10), // a: lowest reranker — but ignored
        makeRerankResult(1, 0.50), // b
      ];
      const blended = blendScores(pool, rerankResults, 1.0);
      // Original order: a(0.9) > b(0.7) > c(0.5)
      expect(blended[0]!.chunk.id).toBe("a");
      expect(blended[1]!.chunk.id).toBe("b");
      expect(blended[2]!.chunk.id).toBe("c");
    });
  });

  describe("alpha=0.5 (equal blend)", () => {
    it("prevents catastrophic reranker demotion", () => {
      // a is the best vector result, reranker demotes it
      const rerankResults = [
        makeRerankResult(0, 0.30), // a: reranker hates it
        makeRerankResult(1, 0.90), // b: reranker loves it
        makeRerankResult(2, 0.60), // c: mid
      ];
      const blended = blendScores(pool, rerankResults, 0.5);
      // a should stay near the top due to 50% vector weight
      const aIdx = blended.findIndex((r) => r.chunk.id === "a");
      expect(aIdx).toBeLessThanOrEqual(1); // a shouldn't drop below #2
    });
  });

  describe("alpha=0.3 (reranker-biased)", () => {
    it("allows reranker to reorder when correct", () => {
      // c is actually the best document, reranker knows it
      const rerankResults = [
        makeRerankResult(2, 0.99), // c: reranker's top pick
        makeRerankResult(0, 0.80), // a: mid
        makeRerankResult(1, 0.60), // b: lowest
      ];
      const blended = blendScores(pool, rerankResults, 0.3);
      // With 70% reranker weight, c should be promoted
      expect(blended[0]!.chunk.id).toBe("c");
    });

    it("limits damage when reranker is wrong", () => {
      // a is truly best (high vector), reranker wrongly promotes c
      const rerankResults = [
        makeRerankResult(2, 0.99), // c: reranker wrongly picks garbage
        makeRerankResult(1, 0.50), // b: mid
        makeRerankResult(0, 0.10), // a: reranker wrongly demotes best
      ];
      const blendedPure = blendScores(pool, rerankResults, 0);
      const blended03 = blendScores(pool, rerankResults, 0.3);
      const blended05 = blendScores(pool, rerankResults, 0.5);

      // Pure reranker: c wins (wrong)
      expect(blendedPure[0]!.chunk.id).toBe("c");

      // With α=0.3, a should rank higher than in pure reranker
      const aRankPure = blendedPure.findIndex((r) => r.chunk.id === "a");
      const aRank03 = blended03.findIndex((r) => r.chunk.id === "a");
      const aRank05 = blended05.findIndex((r) => r.chunk.id === "a");

      expect(aRank05).toBeLessThanOrEqual(aRank03);
      expect(aRank03).toBeLessThanOrEqual(aRankPure);
    });
  });

  describe("edge cases", () => {
    it("handles empty inputs", () => {
      expect(blendScores([], [], 0.3)).toEqual([]);
    });

    it("handles single candidate", () => {
      const single = [makeResult("only", 0.8)];
      const rerank = [makeRerankResult(0, 0.9)];
      const blended = blendScores(single, rerank, 0.5);
      expect(blended).toHaveLength(1);
      expect(blended[0]!.chunk.id).toBe("only");
    });

    it("handles identical reranker scores (no discrimination)", () => {
      const rerankResults = [
        makeRerankResult(0, 0.9),
        makeRerankResult(1, 0.9),
        makeRerankResult(2, 0.9),
      ];
      const blended = blendScores(pool, rerankResults, 0.3);
      expect(blended).toHaveLength(3);
      // When reranker has no signal, original order should dominate
      expect(blended[0]!.chunk.id).toBe("a");
    });

    it("handles identical original scores", () => {
      const equalPool = [
        makeResult("a", 0.5),
        makeResult("b", 0.5),
        makeResult("c", 0.5),
      ];
      const rerankResults = [
        makeRerankResult(2, 0.99),
        makeRerankResult(0, 0.50),
        makeRerankResult(1, 0.10),
      ];
      const blended = blendScores(equalPool, rerankResults, 0.3);
      // When original scores are tied, reranker should fully determine order
      expect(blended[0]!.chunk.id).toBe("c");
    });
  });

  describe("score normalization properties", () => {
    it("all blended scores are in [0, 1]", () => {
      const rerankResults = [
        makeRerankResult(0, 0.99),
        makeRerankResult(1, 0.50),
        makeRerankResult(2, 0.01),
      ];
      for (const alpha of [0, 0.1, 0.3, 0.5, 0.7, 1.0]) {
        const blended = blendScores(pool, rerankResults, alpha);
        for (const r of blended) {
          expect(r.score).toBeGreaterThanOrEqual(0);
          expect(r.score).toBeLessThanOrEqual(1);
        }
      }
    });

    it("alpha=0 and alpha=1 are boundary cases", () => {
      const rerankResults = [
        makeRerankResult(0, 0.99),
        makeRerankResult(1, 0.50),
        makeRerankResult(2, 0.01),
      ];
      const pure = blendScores(pool, rerankResults, 0);
      const orig = blendScores(pool, rerankResults, 1.0);

      // Pure reranker should order by reranker scores
      expect(pure[0]!.chunk.id).toBe("a"); // highest reranker
      expect(pure[2]!.chunk.id).toBe("c"); // lowest reranker

      // Pure original should order by vector scores
      expect(orig[0]!.chunk.id).toBe("a"); // highest vector (0.9)
      expect(orig[2]!.chunk.id).toBe("c"); // lowest vector (0.5)
    });

    it("blended scores change monotonically with alpha", () => {
      const rerankResults = [
        makeRerankResult(0, 0.99),
        makeRerankResult(1, 0.50),
        makeRerankResult(2, 0.01),
      ];
      // For doc "a" (high vector, high reranker), score should be stable
      // For doc "c" (low vector, low reranker), score should also be stable
      // The interesting case is when signals disagree
      const disagreePool = [
        makeResult("a", 0.9), // high vector
        makeResult("b", 0.1), // low vector
      ];
      const disagreeRerank = [
        makeRerankResult(1, 0.99), // b: high reranker (disagrees with vector)
        makeRerankResult(0, 0.01), // a: low reranker
      ];

      const at0 = blendScores(disagreePool, disagreeRerank, 0);
      const at03 = blendScores(disagreePool, disagreeRerank, 0.3);
      const at05 = blendScores(disagreePool, disagreeRerank, 0.5);
      const at1 = blendScores(disagreePool, disagreeRerank, 1.0);

      // At alpha=0, b wins (reranker)
      expect(at0[0]!.chunk.id).toBe("b");
      // At alpha=1, a wins (vector)
      expect(at1[0]!.chunk.id).toBe("a");
      // In between, there should be a crossover point
      const aScore03 = at03.find((r) => r.chunk.id === "a")!.score;
      const aScore05 = at05.find((r) => r.chunk.id === "a")!.score;
      // More alpha = more vector weight = higher score for a
      expect(aScore05).toBeGreaterThan(aScore03);
    });
  });
});

// ── alphaOverride Integration ─────────────────────────────────────────────

describe("alphaOverride (Phase 10B)", () => {
  // These tests use mock fetch to verify the reranker respects alphaOverride

  let originalFetch: typeof globalThis.fetch;

  const mockRerankResponse = (results: Array<{ index: number; relevance_score: number }>) => {
    return vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ results }),
    } as unknown as Response);
  };

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("uses config blendAlpha when no override provided", async () => {
    const rerankResults = [
      { index: 0, relevance_score: 0.95 },
      { index: 1, relevance_score: 0.85 },
    ];
    const fetchMock = mockRerankResponse(rerankResults);
    // First call = probe, second call = rerank
    globalThis.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ results: [{ index: 0, relevance_score: 0.5 }] }) } as unknown as Response)
      .mockImplementation(fetchMock);

    const reranker = await createReranker({
      enabled: true,
      spark: { baseUrl: "http://mock:18096", model: "test-model" },
      topN: 5,
      scoreBlendAlpha: 0.3,
    });

    const candidates = [makeResult("a", 0.9), makeResult("b", 0.7)];
    const result = await reranker.rerank("test query", candidates, 2);

    expect(result).toHaveLength(2);
    // Check that the fetch was called with top_n = pool.length (because alpha > 0)
    const rerankCall = fetchMock.mock.calls[0];
    const body = JSON.parse(rerankCall[1].body as string);
    expect(body.top_n).toBe(2); // pool.length = 2
  });

  it("overrides config alpha with per-call alphaOverride", async () => {
    // Need 3+ candidates so min-max normalization produces different spreads
    const rerankResults = [
      { index: 0, relevance_score: 0.95 },
      { index: 1, relevance_score: 0.85 },
      { index: 2, relevance_score: 0.50 },
    ];
    const fetchMock = mockRerankResponse(rerankResults);
    globalThis.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ results: [{ index: 0, relevance_score: 0.5 }] }) } as unknown as Response)
      .mockImplementation(fetchMock);

    // Config alpha = 0 (pure reranker), but we override to 0.5
    const reranker = await createReranker({
      enabled: true,
      spark: { baseUrl: "http://mock:18096", model: "test-model" },
      topN: 5,
      scoreBlendAlpha: 0,
    });

    // Candidates where vector and reranker disagree — a is best vector, c is worst
    const candidates = [makeResult("a", 0.9), makeResult("b", 0.5), makeResult("c", 0.1)];
    const resultNoOverride = await reranker.rerank("test query", candidates, 3);
    const resultWithOverride = await reranker.rerank("test query", candidates, 3, { alphaOverride: 0.5 });

    // The middle candidate (b) should have a different blended score
    // because alpha=0 ignores vector, alpha=0.5 considers vector
    const bScoreNoOverride = resultNoOverride.find((r) => r.chunk.id === "b")!.score;
    const bScoreWithOverride = resultWithOverride.find((r) => r.chunk.id === "b")!.score;
    expect(bScoreNoOverride).not.toBeCloseTo(bScoreWithOverride, 2);
  });

  it("alphaOverride=0 uses pure reranker even when config alpha > 0", async () => {
    const rerankResults = [
      { index: 1, relevance_score: 0.99 }, // b: highest reranker
      { index: 0, relevance_score: 0.30 }, // a: lowest reranker
    ];
    const fetchMock = mockRerankResponse(rerankResults);
    globalThis.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ results: [{ index: 0, relevance_score: 0.5 }] }) } as unknown as Response)
      .mockImplementation(fetchMock);

    const reranker = await createReranker({
      enabled: true,
      spark: { baseUrl: "http://mock:18096", model: "test-model" },
      topN: 5,
      scoreBlendAlpha: 0.5, // config says blend
    });

    const candidates = [makeResult("a", 0.9), makeResult("b", 0.3)];
    const result = await reranker.rerank("test query", candidates, 2, { alphaOverride: 0 });

    // With pure reranker (alpha=0), b should win
    expect(result[0]!.chunk.id).toBe("b");
  });

  it("normalizes queries regardless of alpha", async () => {
    const rerankResults = [
      { index: 0, relevance_score: 0.95 },
      { index: 1, relevance_score: 0.85 },
    ];
    const fetchMock = mockRerankResponse(rerankResults);
    globalThis.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ results: [{ index: 0, relevance_score: 0.5 }] }) } as unknown as Response)
      .mockImplementation(fetchMock);

    const reranker = await createReranker({
      enabled: true,
      spark: { baseUrl: "http://mock:18096", model: "test-model" },
      topN: 5,
      scoreBlendAlpha: 0.5,
    });

    const candidates = [makeResult("a", 0.9), makeResult("b", 0.7)];

    // Declarative claim — should be normalized
    await reranker.rerank("Stem cells can differentiate", candidates, 2);
    const call = fetchMock.mock.calls[0];
    const body = JSON.parse(call[1].body as string);
    expect(body.query).toBe("Is it true that stem cells can differentiate?");
  });

  it("normalizes queries even with alphaOverride=0", async () => {
    const rerankResults = [
      { index: 0, relevance_score: 0.95 },
    ];
    const fetchMock = mockRerankResponse(rerankResults);
    globalThis.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ results: [{ index: 0, relevance_score: 0.5 }] }) } as unknown as Response)
      .mockImplementation(fetchMock);

    const reranker = await createReranker({
      enabled: true,
      spark: { baseUrl: "http://mock:18096", model: "test-model" },
      topN: 5,
      scoreBlendAlpha: 0.5,
    });

    const candidates = [makeResult("a", 0.9)];

    await reranker.rerank("BRCA1 mutations increase cancer risk", candidates, 1, { alphaOverride: 0 });
    const call = fetchMock.mock.calls[0];
    const body = JSON.parse(call[1].body as string);
    expect(body.query).toBe("Is it true that bRCA1 mutations increase cancer risk?");
  });
});

// ── Error Handling ────────────────────────────────────────────────────────

describe("reranker error handling (Phase 10B)", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("falls back to input order on HTTP error", async () => {
    globalThis.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ results: [{ index: 0, relevance_score: 0.5 }] }) } as unknown as Response) // probe
      .mockResolvedValue({ ok: false, status: 503, statusText: "Service Unavailable", text: async () => "overloaded" } as unknown as Response);

    const reranker = await createReranker({
      enabled: true,
      spark: { baseUrl: "http://mock:18096", model: "test-model" },
      topN: 5,
    });

    const candidates = [makeResult("a", 0.9), makeResult("b", 0.7), makeResult("c", 0.5)];
    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    const result = await reranker.rerank("test", candidates, 2);

    // Should return original candidates, sliced to topN
    expect(result).toHaveLength(2);
    expect(result[0]!.chunk.id).toBe("a");
    expect(result[1]!.chunk.id).toBe("b");

    // Should log the error
    expect(errSpy).toHaveBeenCalledWith(expect.stringContaining("[reranker] ERROR: 503"));
    errSpy.mockRestore();
  });

  it("logs error body when available", async () => {
    globalThis.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ results: [{ index: 0, relevance_score: 0.5 }] }) } as unknown as Response) // probe
      .mockResolvedValue({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        text: async () => '{"error": "model not loaded"}',
      } as unknown as Response);

    const reranker = await createReranker({
      enabled: true,
      spark: { baseUrl: "http://mock:18096", model: "test-model" },
      topN: 5,
    });

    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    await reranker.rerank("test", [makeResult("a", 0.9)], 1);

    expect(errSpy).toHaveBeenCalledWith(expect.stringContaining("model not loaded"));
    errSpy.mockRestore();
  });

  it("handles body read failure gracefully", async () => {
    globalThis.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ results: [{ index: 0, relevance_score: 0.5 }] }) } as unknown as Response) // probe
      .mockResolvedValue({
        ok: false,
        status: 502,
        statusText: "Bad Gateway",
        text: async () => { throw new Error("socket hang up"); },
      } as unknown as Response);

    const reranker = await createReranker({
      enabled: true,
      spark: { baseUrl: "http://mock:18096", model: "test-model" },
      topN: 5,
    });

    const errSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const result = await reranker.rerank("test", [makeResult("a", 0.9)], 1);

    // Should still fall back gracefully
    expect(result).toHaveLength(1);
    expect(errSpy).toHaveBeenCalledWith(expect.stringContaining("[reranker] ERROR: 502"));
    errSpy.mockRestore();
  });

  it("returns empty array for empty candidates", async () => {
    globalThis.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ results: [{ index: 0, relevance_score: 0.5 }] }) } as unknown as Response);

    const reranker = await createReranker({
      enabled: true,
      spark: { baseUrl: "http://mock:18096", model: "test-model" },
      topN: 5,
    });

    const result = await reranker.rerank("test", [], 5);
    expect(result).toEqual([]);
  });
});

// ── Passthrough Reranker ──────────────────────────────────────────────────

describe("passthroughReranker", () => {
  it("returns candidates unchanged when reranker disabled", async () => {
    const reranker = await createReranker({
      enabled: false,
      topN: 5,
    });

    const candidates = [makeResult("a", 0.9), makeResult("b", 0.7), makeResult("c", 0.5)];
    const result = await reranker.rerank("test", candidates, 2);

    expect(result).toHaveLength(2);
    expect(result[0]!.chunk.id).toBe("a");
    expect(result[1]!.chunk.id).toBe("b");
  });

  it("accepts alphaOverride without error", async () => {
    const reranker = await createReranker({
      enabled: false,
      topN: 5,
    });

    const candidates = [makeResult("a", 0.9)];
    const result = await reranker.rerank("test", candidates, 1, { alphaOverride: 0.5 });
    expect(result).toHaveLength(1);
  });

  it("probe returns true", async () => {
    const reranker = await createReranker({
      enabled: false,
      topN: 5,
    });

    expect(await reranker.probe()).toBe(true);
  });
});

// ── Spread Guard ──────────────────────────────────────────────────────────

describe("spread guard", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("falls back to input order when logit spread is too narrow", async () => {
    // All reranker scores nearly identical → logit spread ≈ 0
    const tightResults = [
      { index: 0, relevance_score: 0.950 },
      { index: 1, relevance_score: 0.951 },
      { index: 2, relevance_score: 0.952 },
    ];
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ results: tightResults }),
    } as unknown as Response);
    globalThis.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ results: [{ index: 0, relevance_score: 0.5 }] }) } as unknown as Response) // probe
      .mockImplementation(fetchMock);

    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    const reranker = await createReranker({
      enabled: true,
      spark: { baseUrl: "http://mock:18096", model: "test-model", minScoreSpread: 0.5 },
      topN: 5,
    });

    const candidates = [makeResult("a", 0.9), makeResult("b", 0.7), makeResult("c", 0.5)];
    const result = await reranker.rerank("test query?", candidates, 3);

    // Should fall back to original order
    expect(result[0]!.chunk.id).toBe("a");
    expect(result[1]!.chunk.id).toBe("b");
    expect(result[2]!.chunk.id).toBe("c");

    // Check spread guard log
    expect(logSpy).toHaveBeenCalledWith(expect.stringContaining("spread guard"));
    logSpy.mockRestore();
  });
});

// ── End-to-End Consistency ────────────────────────────────────────────────

describe("end-to-end: alphaOverride produces same results as config alpha", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => {
    originalFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("config alpha=0.3 matches alphaOverride=0.3 on a zero-config reranker", async () => {
    const rerankResults = [
      { index: 0, relevance_score: 0.95 },
      { index: 1, relevance_score: 0.70 },
      { index: 2, relevance_score: 0.40 },
    ];

    // Reranker #1: config alpha = 0.3
    const fetchMock1 = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ results: rerankResults }),
    } as unknown as Response);
    globalThis.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ results: [{ index: 0, relevance_score: 0.5 }] }) } as unknown as Response)
      .mockImplementation(fetchMock1);

    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    const reranker1 = await createReranker({
      enabled: true,
      spark: { baseUrl: "http://mock:18096", model: "test" },
      topN: 5,
      scoreBlendAlpha: 0.3,
    });

    const candidates = [makeResult("a", 0.9), makeResult("b", 0.6), makeResult("c", 0.3)];
    const result1 = await reranker1.rerank("How does DNA repair work?", candidates, 3);

    // Reranker #2: config alpha = 0, override to 0.3
    const fetchMock2 = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ results: rerankResults }),
    } as unknown as Response);
    globalThis.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ results: [{ index: 0, relevance_score: 0.5 }] }) } as unknown as Response)
      .mockImplementation(fetchMock2);

    const reranker2 = await createReranker({
      enabled: true,
      spark: { baseUrl: "http://mock:18096", model: "test" },
      topN: 5,
      scoreBlendAlpha: 0,
    });

    const result2 = await reranker2.rerank("How does DNA repair work?", candidates, 3, { alphaOverride: 0.3 });

    // Results should be identical
    expect(result1).toHaveLength(result2.length);
    for (let i = 0; i < result1.length; i++) {
      expect(result1[i]!.chunk.id).toBe(result2[i]!.chunk.id);
      expect(result1[i]!.score).toBeCloseTo(result2[i]!.score, 10);
    }

    logSpy.mockRestore();
  });
});

// ── Regression: Phase 10A Logit Recovery ──────────────────────────────────

describe("logit recovery integration", () => {
  it("transforms tight sigmoid band into wide logit range", () => {
    // Typical Nemotron compressed scores
    const sigmoids = [0.83, 0.87, 0.91, 0.95, 0.99];
    const logits = sigmoids.map(recoverLogit);

    // Sigmoid range: 0.16
    const sigmoidSpread = Math.max(...sigmoids) - Math.min(...sigmoids);
    expect(sigmoidSpread).toBeCloseTo(0.16, 1);

    // Logit range should be much wider
    const logitSpread = Math.max(...logits) - Math.min(...logits);
    expect(logitSpread).toBeGreaterThan(3); // Much better discrimination
  });

  it("logit recovery preserves ordering", () => {
    const sigmoids = [0.83, 0.87, 0.91, 0.95, 0.99];
    const logits = sigmoids.map(recoverLogit);

    for (let i = 1; i < logits.length; i++) {
      expect(logits[i]!).toBeGreaterThan(logits[i - 1]!);
    }
  });

  it("blendScores uses logits not raw sigmoids for reranker signal", () => {
    const pool = [
      makeResult("a", 0.9),
      makeResult("b", 0.5),
    ];

    // Tight sigmoid scores — if we used raw values, discrimination would be poor
    const rerankResults = [
      makeRerankResult(1, 0.98), // b: slightly higher sigmoid
      makeRerankResult(0, 0.95), // a: slightly lower sigmoid
    ];

    // With alpha=0 (pure reranker), logit recovery should give clear discrimination
    const blended = blendScores(pool, rerankResults, 0);
    expect(blended[0]!.chunk.id).toBe("b"); // higher logit wins
    // Score spread should be meaningful, not crushed
    const spread = blended[0]!.score - blended[1]!.score;
    expect(spread).toBeGreaterThan(0.3);
  });
});
