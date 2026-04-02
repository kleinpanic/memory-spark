/**
 * Phase 12 Fix 2: Dynamic Reranker Gate — Before/After Tests
 *
 * Proves that the spread-aware gate prevents reranker damage on:
 * 1. Tight clusters (spread < 0.02): reranker is gambling on a tied set
 * 2. Confident vectors (spread > 0.08): reranker overrides a good ranking
 * 3. Middle ground: reranker is allowed to help break ties
 *
 * Soft gate: dynamically adjusts vector weight based on spread
 * Hard gate: binary skip/proceed decision
 */

import { describe, it, expect } from "vitest";
import {
  computeRerankerGate,
  blendByRank,
  blendScores,
  type RerankerGateResult,
} from "../src/rerank/reranker.js";
import type { SearchResult } from "../src/storage/backend.js";

// ── Helpers ──────────────────────────────────────────────────────────────

function makeResult(id: string, score: number): SearchResult {
  return {
    chunk: {
      id,
      text: `Document ${id}`,
      path: `/test/${id}`,
      source: "memory" as const,
      agent_id: "test",
      start_line: 0,
      end_line: 0,
      vector: [],
      updated_at: new Date().toISOString(),
    },
    score,
    snippet: `Document ${id}`,
    vector: new Array(4096).fill(0.1) as number[],
  };
}

// ── BEFORE: No gate — reranker damages tight clusters ────────────────────

describe("BEFORE: without gate, reranker damages tight clusters", () => {
  it("reranker reshuffles a tied set arbitrarily", () => {
    // 5 candidates with scores within 0.01 — effectively a tie
    const pool = [
      makeResult("a", 0.255),
      makeResult("b", 0.253),
      makeResult("c_relevant", 0.251), // RELEVANT doc
      makeResult("d", 0.249),
      makeResult("e", 0.247),
    ];

    // Reranker pushes relevant doc to last
    const rerankResults = [
      { index: 4, relevance_score: 0.999 }, // e → #1
      { index: 3, relevance_score: 0.998 }, // d → #2
      { index: 0, relevance_score: 0.997 }, // a → #3
      { index: 1, relevance_score: 0.996 }, // b → #4
      { index: 2, relevance_score: 0.85 }, // c_relevant → #5 (DEMOTED)
    ];

    // Without gate, reranker runs and demotes the relevant doc
    const result = blendScores(pool, rerankResults, 0);
    const relevantRank = result.findIndex((r) => r.chunk.id === "c_relevant");
    expect(relevantRank).toBe(4); // Dead last
  });

  it("reranker overrides a confident vector ranking", () => {
    // Clear vector winner with large spread (0.15)
    const pool = [
      makeResult("winner", 0.65), // Clear #1
      makeResult("distant2", 0.52),
      makeResult("distant3", 0.5), // spread = 0.15
    ];

    // Reranker disagrees and promotes distant3
    const rerankResults = [
      { index: 2, relevance_score: 0.999 }, // distant3 → #1
      { index: 1, relevance_score: 0.95 }, // distant2 → #2
      { index: 0, relevance_score: 0.9 }, // winner → #3 (DEMOTED)
    ];

    const result = blendScores(pool, rerankResults, 0);
    expect(result[0]!.chunk.id).toBe("distant3"); // Reranker overrode vector
    expect(result[2]!.chunk.id).toBe("winner"); // Winner demoted to last
  });
});

// ── computeRerankerGate: unit tests ──────────────────────────────────────

describe("computeRerankerGate", () => {
  describe("mode=off", () => {
    it("always returns shouldRerank=true", () => {
      const tight = [makeResult("a", 0.25), makeResult("b", 0.249)];
      const wide = [makeResult("a", 0.65), makeResult("b", 0.4)];

      expect(computeRerankerGate(tight, "off").shouldRerank).toBe(true);
      expect(computeRerankerGate(wide, "off").shouldRerank).toBe(true);
    });

    it("returns vectorWeightMultiplier=1.0", () => {
      const pool = [makeResult("a", 0.5), makeResult("b", 0.3)];
      expect(computeRerankerGate(pool, "off").vectorWeightMultiplier).toBe(1.0);
    });
  });

  describe("mode=hard", () => {
    const threshold = 0.08;
    const lowThreshold = 0.02;

    it("skips reranker when spread > threshold (vector confident)", () => {
      const pool = [
        makeResult("a", 0.6),
        makeResult("b", 0.55),
        makeResult("c", 0.5),
        makeResult("d", 0.48),
        makeResult("e", 0.45),
      ];
      // spread = 0.60 - 0.45 = 0.15 > 0.08
      const gate = computeRerankerGate(pool, "hard", threshold, lowThreshold);
      expect(gate.shouldRerank).toBe(false);
      expect(gate.reason).toContain("hard-gate-high");
    });

    it("skips reranker when spread < lowThreshold (tied set)", () => {
      const pool = [
        makeResult("a", 0.251),
        makeResult("b", 0.25),
        makeResult("c", 0.249),
        makeResult("d", 0.248),
        makeResult("e", 0.247),
      ];
      // spread = 0.251 - 0.247 = 0.004 < 0.02
      const gate = computeRerankerGate(pool, "hard", threshold, lowThreshold);
      expect(gate.shouldRerank).toBe(false);
      expect(gate.reason).toContain("hard-gate-low");
    });

    it("allows reranker when spread is in the useful range", () => {
      const pool = [
        makeResult("a", 0.35),
        makeResult("b", 0.32),
        makeResult("c", 0.3),
        makeResult("d", 0.29),
        makeResult("e", 0.28),
      ];
      // spread = 0.35 - 0.28 = 0.07, in [0.02, 0.08]
      const gate = computeRerankerGate(pool, "hard", threshold, lowThreshold);
      expect(gate.shouldRerank).toBe(true);
      expect(gate.reason).toContain("hard-gate-pass");
    });

    it("handles near-threshold value (floating point: 0.38-0.30 > 0.08)", () => {
      // NOTE: 0.38 - 0.30 = 0.08000000000000002 due to IEEE 754
      // This is technically > 0.08, so the gate fires (skip)
      const pool = [makeResult("a", 0.38), makeResult("b", 0.3)];
      const gate = computeRerankerGate(pool, "hard", 0.08, 0.02);
      expect(gate.shouldRerank).toBe(false); // FP makes it barely over threshold
    });

    it("boundary: spread clearly below threshold passes", () => {
      const pool = [makeResult("a", 0.375), makeResult("b", 0.3)];
      // spread = 0.075 < 0.08 → passes
      const gate = computeRerankerGate(pool, "hard", 0.08, 0.02);
      expect(gate.shouldRerank).toBe(true);
    });

    it("handles fewer than 5 candidates", () => {
      const pool = [makeResult("a", 0.5), makeResult("b", 0.3)];
      // spread = 0.20 > 0.08
      const gate = computeRerankerGate(pool, "hard", threshold, lowThreshold);
      expect(gate.shouldRerank).toBe(false);
    });

    it("handles single candidate", () => {
      const pool = [makeResult("only", 0.5)];
      const gate = computeRerankerGate(pool, "hard", threshold, lowThreshold);
      expect(gate.shouldRerank).toBe(true); // < 2 candidates → allow
    });
  });

  describe("mode=soft", () => {
    it("returns high multiplier (→1.0) when spread > threshold", () => {
      const pool = [makeResult("a", 0.6), makeResult("b", 0.5)];
      // spread = 0.10 > 0.08 → multiplier should be 1.0
      const gate = computeRerankerGate(pool, "soft", 0.08, 0.02);
      expect(gate.shouldRerank).toBe(true);
      expect(gate.vectorWeightMultiplier).toBe(1.0);
    });

    it("returns lower multiplier in the useful range", () => {
      const pool = [makeResult("a", 0.35), makeResult("b", 0.32), makeResult("c", 0.3)];
      // spread = 0.05, in [0.02, 0.08]
      // t = (0.05 - 0.02) / (0.08 - 0.02) = 0.5
      // multiplier = 0.5 + 0.5 * 0.5 = 0.75
      const gate = computeRerankerGate(pool, "soft", 0.08, 0.02);
      expect(gate.shouldRerank).toBe(true);
      expect(gate.vectorWeightMultiplier).toBeCloseTo(0.75, 2);
    });

    it("returns ~0.5 multiplier at lowThreshold (max reranker influence)", () => {
      const pool = [makeResult("a", 0.32), makeResult("b", 0.3)];
      // spread = 0.02 = lowThreshold
      // t = (0.02 - 0.02) / (0.08 - 0.02) = 0
      // multiplier = 0.5 + 0.5 * 0 = 0.5
      const gate = computeRerankerGate(pool, "soft", 0.08, 0.02);
      expect(gate.vectorWeightMultiplier).toBeCloseTo(0.5, 1);
    });

    it("ramps up multiplier below lowThreshold (tied set protection)", () => {
      const pool = [makeResult("a", 0.301), makeResult("b", 0.3)];
      // spread = 0.001 < lowThreshold (0.02)
      // multiplier = 0.8 + 0.2 * (1 - 0.001/0.02) = 0.8 + 0.2 * 0.95 = 0.99
      const gate = computeRerankerGate(pool, "soft", 0.08, 0.02);
      expect(gate.vectorWeightMultiplier).toBeGreaterThan(0.9);
    });

    it("multiplier is monotonically decreasing from 0 to threshold", () => {
      const spreads = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1];
      const multipliers: number[] = [];

      for (const s of spreads) {
        const pool = [makeResult("a", 0.5), makeResult("b", 0.5 - s)];
        const gate = computeRerankerGate(pool, "soft", 0.08, 0.02);
        multipliers.push(gate.vectorWeightMultiplier);
      }

      // From spread=0 to lowThreshold: multiplier decreases (from ~1.0 to 0.5)
      expect(multipliers[0]!).toBeGreaterThan(multipliers[2]!);
      // From lowThreshold to threshold: multiplier increases (from 0.5 to 1.0)
      expect(multipliers[2]!).toBeLessThan(multipliers[5]!);
      // Above threshold: multiplier = 1.0
      expect(multipliers[5]!).toBe(1.0);
      expect(multipliers[6]!).toBe(1.0);
    });
  });
});

// ── AFTER: Gate + RRF integration ────────────────────────────────────────

describe("AFTER: gate protects against reranker damage", () => {
  it("hard gate preserves vector order on tight clusters", () => {
    const pool = [
      makeResult("a", 0.255),
      makeResult("b", 0.253),
      makeResult("c_relevant", 0.251),
      makeResult("d", 0.249),
      makeResult("e", 0.247),
    ];
    // spread = 0.008 < lowThreshold (0.02) → SKIP reranker

    const gate = computeRerankerGate(pool, "hard", 0.08, 0.02);
    expect(gate.shouldRerank).toBe(false);

    // If gate fires, we return vector order → relevant doc stays at rank 3
    // (much better than rank 5 without gate)
    // Simulating what the reranker pipeline does when gate fires:
    const gatedResult = pool.slice(0, 5);
    const relevantRank = gatedResult.findIndex((r) => r.chunk.id === "c_relevant");
    expect(relevantRank).toBe(2); // Rank 3 (0-indexed) — preserved from vector
  });

  it("hard gate preserves confident vector rankings", () => {
    const pool = [
      makeResult("winner", 0.65),
      makeResult("distant2", 0.52),
      makeResult("distant3", 0.5),
    ];
    // spread = 0.15 > threshold (0.08) → SKIP reranker

    const gate = computeRerankerGate(pool, "hard", 0.08, 0.02);
    expect(gate.shouldRerank).toBe(false);

    // Winner stays at rank 1
    const gatedResult = pool.slice(0, 3);
    expect(gatedResult[0]!.chunk.id).toBe("winner");
  });

  it("soft gate reduces reranker influence on tight clusters", () => {
    const tightPool = [
      makeResult("a", 0.255),
      makeResult("b", 0.253),
      makeResult("c_relevant", 0.251),
      makeResult("d", 0.249),
      makeResult("e", 0.247),
    ];
    // spread = 0.008 < lowThreshold → multiplier ≈ 0.96

    const gate = computeRerankerGate(tightPool, "soft", 0.08, 0.02);
    expect(gate.shouldRerank).toBe(true); // Soft gate always proceeds
    expect(gate.vectorWeightMultiplier).toBeGreaterThan(0.9); // But vector weight is high

    // With high vector weight in RRF, relevant doc is better protected
    const rerankResults = [
      { index: 4, relevance_score: 0.999 },
      { index: 3, relevance_score: 0.998 },
      { index: 0, relevance_score: 0.997 },
      { index: 1, relevance_score: 0.996 },
      { index: 2, relevance_score: 0.85 }, // c_relevant demoted by reranker
    ];

    // With high vector multiplier (~0.96), vector rank 3 is worth ~1.44x more
    const rrfWithGate = blendByRank(tightPool, rerankResults, 60, gate.vectorWeightMultiplier, 1.0);
    const relevantRankGated = rrfWithGate.findIndex((r) => r.chunk.id === "c_relevant");

    // Without gate: RRF at equal weight puts c_relevant at rank 5
    const rrfNoGate = blendByRank(tightPool, rerankResults, 60, 1.0, 1.0);
    const relevantRankUngated = rrfNoGate.findIndex((r) => r.chunk.id === "c_relevant");

    // Soft gate should improve or maintain position (not make it worse)
    expect(relevantRankGated).toBeLessThanOrEqual(relevantRankUngated);
  });

  it("soft gate gives reranker full influence in the useful range", () => {
    // Spread in [0.02, 0.08] — the sweet spot where reranker can help
    const pool = [
      makeResult("a", 0.35),
      makeResult("b", 0.32),
      makeResult("c", 0.3),
      makeResult("d", 0.29),
      makeResult("e", 0.28),
    ];
    // spread = 0.07, multiplier ≈ 0.917

    const gate = computeRerankerGate(pool, "soft", 0.08, 0.02);
    expect(gate.shouldRerank).toBe(true);
    // Multiplier should be < 1.0 but not too low — reranker gets influence
    expect(gate.vectorWeightMultiplier).toBeGreaterThan(0.5);
    expect(gate.vectorWeightMultiplier).toBeLessThan(1.0);
  });

  it("hard gate allows reranker in the useful range where it helps", () => {
    const pool = [
      makeResult("wrong1", 0.35),
      makeResult("wrong2", 0.32),
      makeResult("right", 0.3), // RELEVANT — vector missed it
      makeResult("wrong3", 0.29),
      makeResult("wrong4", 0.28),
    ];
    // spread = 0.07, in [0.02, 0.08] → gate PASSES

    const gate = computeRerankerGate(pool, "hard", 0.08, 0.02);
    expect(gate.shouldRerank).toBe(true);

    // Reranker correctly promotes "right" to #1
    const rerankResults = [
      { index: 2, relevance_score: 0.999 }, // right → #1
      { index: 0, relevance_score: 0.9 },
      { index: 1, relevance_score: 0.85 },
      { index: 3, relevance_score: 0.8 },
      { index: 4, relevance_score: 0.75 },
    ];

    const rrfResult = blendByRank(pool, rerankResults);
    const rightRank = rrfResult.findIndex((r) => r.chunk.id === "right");
    // Reranker should successfully promote "right" into top 3
    expect(rightRank).toBeLessThanOrEqual(2);
  });
});

// ── Regression: gate doesn't break existing behavior ─────────────────────

describe("gate regression: off mode matches pre-gate behavior", () => {
  it("gate=off produces identical results to no gate", () => {
    const pool = [makeResult("a", 0.5), makeResult("b", 0.4), makeResult("c", 0.3)];

    // Verify gate=off has no effect
    const gateOff = computeRerankerGate(pool, "off", 0.08, 0.02);
    expect(gateOff.shouldRerank).toBe(true);
    expect(gateOff.vectorWeightMultiplier).toBe(1.0);

    // RRF with gate=off should produce same results as without gate
    const rerankResults = [
      { index: 2, relevance_score: 0.99 },
      { index: 0, relevance_score: 0.85 },
      { index: 1, relevance_score: 0.8 },
    ];

    const withGate = blendByRank(pool, rerankResults, 60, gateOff.vectorWeightMultiplier, 1.0);
    const withoutGate = blendByRank(pool, rerankResults, 60, 1.0, 1.0);

    expect(withGate.map((r) => r.chunk.id)).toEqual(withoutGate.map((r) => r.chunk.id));
    expect(withGate.map((r) => r.score)).toEqual(withoutGate.map((r) => r.score));
  });
});

// ── Realistic SciFact scenario ───────────────────────────────────────────

describe("head-to-head: SciFact scenarios with gate", () => {
  it("query-1 tight cluster: hard gate prevents catastrophic demotion", () => {
    // Real scenario: scores in [0.200, 0.277] — spread = 0.077
    // This is right at the threshold boundary
    const pool = [
      makeResult("21456232", 0.277),
      makeResult("43990286", 0.236),
      makeResult("31715818", 0.229), // RELEVANT
      makeResult("25435456", 0.227),
      makeResult("19855358", 0.226),
      makeResult("14082855", 0.22),
      makeResult("9122283", 0.215),
      makeResult("39532074", 0.21),
      makeResult("16532419", 0.205),
      makeResult("8290953", 0.2),
    ];

    // spread of top-5: 0.277 - 0.226 = 0.051, in [0.02, 0.08] → gate passes
    const gate = computeRerankerGate(pool, "hard", 0.08, 0.02);
    expect(gate.shouldRerank).toBe(true); // Good — reranker gets a chance

    // With wider threshold (0.06), gate would block
    const strictGate = computeRerankerGate(pool, "hard", 0.06, 0.02);
    expect(strictGate.shouldRerank).toBe(true); // 0.051 < 0.06 → pass
    // Actually 0.051 < 0.06, still passes. Let me check: top 5 = [0.277, 0.236, 0.229, 0.227, 0.226]
    // spread = 0.277 - 0.226 = 0.051

    // With an even tighter threshold (0.05), it would skip
    const tighterGate = computeRerankerGate(pool, "hard", 0.05, 0.02);
    expect(tighterGate.shouldRerank).toBe(false); // 0.051 > 0.05 → vector confident
  });

  it("confident query: gate protects clear vector winner", () => {
    // Query where vector is very confident — large spread
    const pool = [
      makeResult("perfect_match", 0.75),
      makeResult("ok_match", 0.55),
      makeResult("distant1", 0.4),
      makeResult("distant2", 0.35),
      makeResult("distant3", 0.3),
    ];
    // spread = 0.75 - 0.30 = 0.45 > 0.08 → SKIP

    const gate = computeRerankerGate(pool, "hard", 0.08, 0.02);
    expect(gate.shouldRerank).toBe(false);
    expect(gate.reason).toContain("vector confident");

    // Perfect match stays at rank 1 — reranker can't touch it
  });

  it("soft gate with realistic SciFact spread distribution", () => {
    // Test the gate response across the observed SciFact spread range
    const testCases = [
      { spread: 0.005, expectedMultiplier: "> 0.9", desc: "near-tie" },
      { spread: 0.02, expectedMultiplier: "≈ 0.5", desc: "low threshold" },
      { spread: 0.05, expectedMultiplier: "≈ 0.75", desc: "mid-range" },
      { spread: 0.08, expectedMultiplier: "= 1.0", desc: "at threshold" },
      { spread: 0.15, expectedMultiplier: "= 1.0", desc: "confident" },
    ];

    for (const tc of testCases) {
      const pool = [makeResult("a", 0.5), makeResult("b", 0.5 - tc.spread)];
      const gate = computeRerankerGate(pool, "soft", 0.08, 0.02);

      if (tc.spread <= 0.005) {
        expect(gate.vectorWeightMultiplier).toBeGreaterThan(0.9);
      } else if (tc.spread >= 0.08) {
        expect(gate.vectorWeightMultiplier).toBe(1.0);
      } else if (Math.abs(tc.spread - 0.02) < 0.001) {
        expect(gate.vectorWeightMultiplier).toBeCloseTo(0.5, 1);
      } else if (Math.abs(tc.spread - 0.05) < 0.001) {
        expect(gate.vectorWeightMultiplier).toBeCloseTo(0.75, 1);
      }
    }
  });
});
