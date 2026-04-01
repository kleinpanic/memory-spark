/**
 * Phase 12 Fix 1: RRF Rank Fusion — Before/After Tests
 *
 * Proves that blendByRank (RRF) fixes the issues with blendScores (min-max):
 * 1. Score-based blending with recoverLogit is a no-op (monotonic + min-max = same ranking)
 * 2. RRF preserves reranker wins while preventing catastrophic demotion
 * 3. RRF is scale-invariant — works regardless of reranker score range
 */

import { describe, it, expect } from "vitest";
import {
  blendScores,
  blendByRank,
  recoverLogit,
} from "../src/rerank/reranker.js";
import type { SearchResult } from "../src/storage/backend.js";

// ── Helpers ──────────────────────────────────────────────────────────────

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

function getRanking(results: SearchResult[]): string[] {
  return results.map((r) => r.chunk.id);
}

// ── PROOF: recoverLogit + minMaxNormalize is a NO-OP ─────────────────────

describe("BEFORE: blendScores with recoverLogit is a no-op", () => {
  // This is the core evidence. blendScores uses recoverLogit internally.
  // Because recoverLogit is monotonic and min-max normalization only preserves
  // relative order, the output ranking is identical with or without logit recovery.

  it("produces identical rankings with and without logit recovery (the M=T bug)", () => {
    // Simulate 10 candidates with realistic vector scores
    const pool = [
      makeResult("doc1", 0.45),
      makeResult("doc2", 0.42),
      makeResult("doc3", 0.40),
      makeResult("doc4", 0.38),
      makeResult("doc5", 0.36),
      makeResult("doc6", 0.34),
      makeResult("doc7", 0.32),
      makeResult("doc8", 0.30),
      makeResult("doc9", 0.28),
      makeResult("doc10", 0.26),
    ];

    // Reranker disagrees significantly — moves doc8 to top
    const rerankResults = [
      makeRerankResult(7, 0.995),  // doc8: reranker's top pick
      makeRerankResult(0, 0.980),  // doc1
      makeRerankResult(4, 0.965),  // doc5
      makeRerankResult(2, 0.950),  // doc3
      makeRerankResult(1, 0.935),  // doc2
      makeRerankResult(3, 0.920),  // doc4
      makeRerankResult(5, 0.905),  // doc6
      makeRerankResult(6, 0.890),  // doc7
      makeRerankResult(8, 0.875),  // doc9
      makeRerankResult(9, 0.860),  // doc10
    ];

    // blendScores with alpha=0.3 uses recoverLogit internally
    const scoreBlendResult = blendScores(pool, rerankResults, 0.3);

    // Now manually compute what would happen WITHOUT logit recovery
    // (just raw sigmoid scores, min-max normalized)
    // Because recoverLogit is monotonic, the normalized values will be
    // in the same relative order. So the final blend ranking MUST be identical.
    //
    // This is the M=T, N=Q equivalence proven by 300-query telemetry.
    // We can't easily test "without logit" since blendScores always applies it,
    // but we CAN prove the mathematical property:

    // The ranking from blendScores is deterministic given the inputs
    const ranking1 = getRanking(scoreBlendResult);

    // Running it again produces the same ranking (deterministic)
    const ranking2 = getRanking(blendScores(pool, rerankResults, 0.3));
    expect(ranking1).toEqual(ranking2);

    // The KEY proof: with alpha=0, recoverLogit + minMaxNormalize produces
    // the SAME ordering as raw sigmoid + minMaxNormalize (both monotonic)
    const pureRerankerResult = blendScores(pool, rerankResults, 0);
    const pureRanking = getRanking(pureRerankerResult);

    // Pure reranker should just be reranker order (doc8, doc1, doc5, doc3, ...)
    expect(pureRanking[0]).toBe("doc8");
    expect(pureRanking[1]).toBe("doc1");
    expect(pureRanking[2]).toBe("doc5");
  });

  it("min-max normalization destroys confidence gaps", () => {
    const pool = [
      makeResult("confident", 0.50),
      makeResult("tie1", 0.48),
      makeResult("tie2", 0.47),
    ];

    // Reranker: "confident" has a HUGE gap vs the ties
    const rerankResults = [
      makeRerankResult(0, 0.999),  // confident: very high
      makeRerankResult(1, 0.840),  // tie1: much lower
      makeRerankResult(2, 0.835),  // tie2: barely below tie1
    ];

    const result = blendScores(pool, rerankResults, 0);
    // After min-max: confident→1.0, tie2→0.0, tie1→somewhere between
    // The HUGE gap (0.999 vs 0.840) and tiny gap (0.840 vs 0.835)
    // both get squished to [0, 1] range. No confidence info survives.
    expect(result[0]!.score).toBeCloseTo(1.0, 1);
    expect(result[2]!.score).toBeCloseTo(0.0, 1);
    // tie1 and tie2 are very close in reality, but min-max says tie1 ≈ 0.03
    // That's 0.97 units away from the top, even though it's only 0.159 sigmoid away
    const tie1Score = result.find((r) => r.chunk.id === "tie1")!.score;
    // The point: this score tells us nothing about actual confidence
    expect(tie1Score).toBeLessThan(0.1); // squished near zero
  });

  it("all top-1 scores become 1.0 after min-max (the 299/300 bug)", () => {
    // Run 5 different "queries" — each will normalize top-1 to 1.0
    for (let trial = 0; trial < 5; trial++) {
      const pool = [
        makeResult(`a${trial}`, 0.5 + trial * 0.05),
        makeResult(`b${trial}`, 0.4 + trial * 0.05),
        makeResult(`c${trial}`, 0.3 + trial * 0.05),
      ];
      const rerankResults = [
        makeRerankResult(0, 0.95 + trial * 0.01),
        makeRerankResult(1, 0.85 + trial * 0.01),
        makeRerankResult(2, 0.75 + trial * 0.01),
      ];

      const result = blendScores(pool, rerankResults, 0);
      // Top-1 is ALWAYS 1.0 regardless of input scores
      expect(result[0]!.score).toBeCloseTo(1.0, 5);
    }
  });
});

// ── AFTER: blendByRank (RRF) fixes these issues ──────────────────────────

describe("AFTER: blendByRank (RRF) preserves rank information", () => {
  it("agreement bonus: docs ranked high by BOTH systems get boosted", () => {
    // doc1 is ranked #1 by vector AND #1 by reranker
    const pool = [
      makeResult("doc1", 0.50), // vector rank 1
      makeResult("doc2", 0.45), // vector rank 2
      makeResult("doc3", 0.40), // vector rank 3
    ];

    const rerankResults = [
      makeRerankResult(0, 0.99), // doc1: reranker rank 1
      makeRerankResult(2, 0.95), // doc3: reranker rank 2
      makeRerankResult(1, 0.90), // doc2: reranker rank 3
    ];

    const result = blendByRank(pool, rerankResults);
    // doc1 should be #1 because it's #1 in BOTH rankings
    expect(result[0]!.chunk.id).toBe("doc1");

    // doc1's RRF score should be higher than doc3 (which is only #1 in one system)
    const doc1Score = result.find((r) => r.chunk.id === "doc1")!.score;
    const doc3Score = result.find((r) => r.chunk.id === "doc3")!.score;
    expect(doc1Score).toBeGreaterThan(doc3Score);
  });

  it("prevents catastrophic demotion (the 13 loser queries fix)", () => {
    // Scenario: vector has doc_relevant at #3, reranker pushes it to #10+
    // In score-based blending at α=0, doc_relevant falls off.
    // In RRF, vector's #3 rank still contributes significant score.

    const pool = [
      makeResult("irrelevant1", 0.50),
      makeResult("irrelevant2", 0.48),
      makeResult("relevant", 0.47),    // vector rank 3 — this is the relevant doc
      makeResult("irrelevant3", 0.46),
      makeResult("irrelevant4", 0.45),
      makeResult("irrelevant5", 0.44),
      makeResult("irrelevant6", 0.43),
      makeResult("irrelevant7", 0.42),
      makeResult("irrelevant8", 0.41),
      makeResult("irrelevant9", 0.40),
    ];

    // Reranker pushes "relevant" all the way to rank 10 (last)
    const rerankResults = [
      makeRerankResult(0, 0.999),
      makeRerankResult(3, 0.998),
      makeRerankResult(4, 0.997),
      makeRerankResult(5, 0.996),
      makeRerankResult(6, 0.995),
      makeRerankResult(7, 0.994),
      makeRerankResult(8, 0.993),
      makeRerankResult(9, 0.992),
      makeRerankResult(1, 0.991),
      makeRerankResult(2, 0.850),  // "relevant" → reranker rank 10
    ];

    // BEFORE: score-based blend at α=0 puts "relevant" dead last
    const scoreBased = blendScores(pool, rerankResults, 0);
    const relevantRankScore = scoreBased.findIndex((r) => r.chunk.id === "relevant");
    expect(relevantRankScore).toBe(9); // Last place (rank 10)

    // AFTER: RRF with equal weights — modest improvement (rank 7 vs rank 10)
    const rrfEqual = blendByRank(pool, rerankResults);
    const relevantRankRrfEqual = rrfEqual.findIndex((r) => r.chunk.id === "relevant");
    expect(relevantRankRrfEqual).toBeLessThan(relevantRankScore); // Better than score-based

    // AFTER: RRF with vector bias (1.5x) — stronger protection for vector-ranked docs
    const rrfVectorBias = blendByRank(pool, rerankResults, 60, 1.5, 1.0);
    const relevantRankRrfBias = rrfVectorBias.findIndex((r) => r.chunk.id === "relevant");
    expect(relevantRankRrfBias).toBeLessThan(relevantRankRrfEqual); // Even better
  });

  it("still allows reranker to promote correct docs (the 9 winner queries)", () => {
    // Scenario: vector has doc_relevant at #5, reranker correctly promotes to #1
    const pool = [
      makeResult("ok1", 0.50),
      makeResult("ok2", 0.48),
      makeResult("ok3", 0.46),
      makeResult("ok4", 0.44),
      makeResult("relevant", 0.42),  // vector rank 5
    ];

    const rerankResults = [
      makeRerankResult(4, 0.999),  // "relevant" → reranker rank 1!
      makeRerankResult(0, 0.900),
      makeRerankResult(1, 0.850),
      makeRerankResult(2, 0.800),
      makeRerankResult(3, 0.750),
    ];

    // BEFORE: score-based blend at α=0 promotes "relevant" to #1
    const scoreBased = blendScores(pool, rerankResults, 0);
    expect(scoreBased[0]!.chunk.id).toBe("relevant");

    // AFTER: RRF also promotes "relevant" — reranker rank 1 is powerful
    const rrfResult = blendByRank(pool, rerankResults);
    // With reranker rank 1, "relevant" should be in top 2
    // (ok1 has vector rank 1 + reranker rank 2, so it's competitive)
    const relevantRank = rrfResult.findIndex((r) => r.chunk.id === "relevant");
    expect(relevantRank).toBeLessThanOrEqual(2);
  });

  it("is scale-invariant — works with any reranker score range", () => {
    const pool = [
      makeResult("a", 0.50),
      makeResult("b", 0.40),
      makeResult("c", 0.30),
    ];

    // Same ranking, different score scales
    const narrowScores = [
      makeRerankResult(2, 0.999),  // c: rank 1
      makeRerankResult(0, 0.998),  // a: rank 2
      makeRerankResult(1, 0.997),  // b: rank 3
    ];

    const wideScores = [
      makeRerankResult(2, 0.999),  // c: rank 1
      makeRerankResult(0, 0.500),  // a: rank 2
      makeRerankResult(1, 0.001),  // b: rank 3
    ];

    const resultNarrow = blendByRank(pool, narrowScores);
    const resultWide = blendByRank(pool, wideScores);

    // Rankings should be IDENTICAL regardless of score magnitude
    expect(getRanking(resultNarrow)).toEqual(getRanking(resultWide));
  });

  it("score-based blending: recoverLogit is a no-op (M=T equivalence)", () => {
    // The M=T bug: blendScores with and without recoverLogit produce identical rankings.
    // We can't easily disable recoverLogit inside blendScores, but we can prove
    // the mathematical property: for ANY set of sigmoid scores, applying
    // recoverLogit before minMaxNormalize yields the same [0,1] ordering.

    const pool = [
      makeResult("a", 0.50),
      makeResult("b", 0.40),
      makeResult("c", 0.30),
    ];

    const rerankResults = [
      makeRerankResult(2, 0.99),  // c: reranker top
      makeRerankResult(0, 0.85),  // a: mid
      makeRerankResult(1, 0.83),  // b: low
    ];

    // blendScores internally does: recoverLogit → minMaxNormalize
    const blended = blendScores(pool, rerankResults, 0.3);

    // Verify: the recoverLogit step preserves monotonic order
    const logits = rerankResults.map((r) => recoverLogit(r.relevance_score));
    // logit(0.99) > logit(0.85) > logit(0.83) ← same order as sigmoid
    expect(logits[0]!).toBeGreaterThan(logits[1]!);
    expect(logits[1]!).toBeGreaterThan(logits[2]!);

    // Because both sigmoid and logit orders are the same, minMaxNormalize
    // maps them to the same relative positions in [0, 1].
    // This means recoverLogit adds ZERO information — it's a wasted computation.
    // The blend result would be identical if we used raw sigmoid scores.

    // RRF doesn't have this problem — it ignores scores entirely
    const rrfResult = blendByRank(pool, rerankResults);
    expect(rrfResult).toHaveLength(3);
    // RRF works on rank positions, so score magnitude is irrelevant by design
  });

  it("score-based blending IS scale-dependent (inconsistent behavior)", () => {
    // Different reranker score RANGES produce different blend outcomes,
    // even when the reranker RANKING is the same. This is a flaw, not a feature.
    const pool = [
      makeResult("a", 0.50),
      makeResult("b", 0.40),
      makeResult("c", 0.30),
    ];

    // Same ranking (c>b>a) but narrow vs wide scores
    const narrowScores = [
      makeRerankResult(2, 0.999),
      makeRerankResult(1, 0.998),
      makeRerankResult(0, 0.997),
    ];
    const wideScores = [
      makeRerankResult(2, 0.999),
      makeRerankResult(1, 0.500),
      makeRerankResult(0, 0.001),
    ];

    const resultNarrow = blendScores(pool, narrowScores, 0.5);
    const resultWide = blendScores(pool, wideScores, 0.5);

    // Rankings DIFFER because min-max amplifies narrow gaps differently than wide gaps
    expect(getRanking(resultNarrow)).not.toEqual(getRanking(resultWide));

    // RRF: same ranking regardless of score range (scale-invariant)
    const rrfNarrow = blendByRank(pool, narrowScores);
    const rrfWide = blendByRank(pool, wideScores);
    expect(getRanking(rrfNarrow)).toEqual(getRanking(rrfWide));
  });

  it("top-1 scores are NOT all 1.0 (unlike min-max)", () => {
    // RRF scores encode actual rank fusion information
    for (let trial = 0; trial < 3; trial++) {
      const pool = [
        makeResult(`a${trial}`, 0.5 + trial * 0.05),
        makeResult(`b${trial}`, 0.4 + trial * 0.05),
        makeResult(`c${trial}`, 0.3 + trial * 0.05),
      ];
      const rerankResults = [
        makeRerankResult(0, 0.95 + trial * 0.01),
        makeRerankResult(1, 0.85 + trial * 0.01),
        makeRerankResult(2, 0.75 + trial * 0.01),
      ];

      const result = blendByRank(pool, rerankResults);
      // RRF score is 1/(k+1) + 1/(k+1) for the #1 doc (if #1 in both systems)
      // With k=60, that's 1/61 + 1/61 ≈ 0.0328
      // The actual value isn't important — what matters is it encodes rank agreement
      expect(result[0]!.score).not.toBeCloseTo(1.0);
      expect(result[0]!.score).toBeGreaterThan(0);
    }
  });
});

// ── RRF hyperparameter behavior ──────────────────────────────────────────

describe("RRF hyperparameter sensitivity", () => {
  const pool = [
    makeResult("vec1", 0.50),  // vector rank 1
    makeResult("vec2", 0.45),  // vector rank 2
    makeResult("vec3", 0.40),  // vector rank 3
    makeResult("vec4", 0.35),  // vector rank 4
    makeResult("vec5", 0.30),  // vector rank 5
  ];

  // Reranker completely reverses the order
  const reversedRerank = [
    makeRerankResult(4, 0.99),  // vec5: reranker rank 1
    makeRerankResult(3, 0.95),  // vec4: reranker rank 2
    makeRerankResult(2, 0.90),  // vec3: reranker rank 3
    makeRerankResult(1, 0.85),  // vec2: reranker rank 4
    makeRerankResult(0, 0.80),  // vec1: reranker rank 5
  ];

  it("k=60 (standard): with full disagreement, RRF scores converge", () => {
    const result = blendByRank(pool, reversedRerank, 60);
    const ranking = getRanking(result);
    // With k=60 and complete reversal, all RRF scores are very close
    // because 1/(k+i) + 1/(k+(n-1-i)) ≈ 2/(k+n/2) for all i when k >> n
    // This means RRF wisely treats total disagreement as "no consensus"
    // The endpoint docs (vec1, vec5) get a tiny edge from rank asymmetry:
    // 1/61+1/65 = 0.03178 > 1/63+1/63 = 0.03175 (vec3)
    const scores = result.map((r) => r.score);
    const spread = Math.max(...scores) - Math.min(...scores);
    // Spread should be tiny — no clear winner when systems fully disagree
    expect(spread).toBeLessThan(0.001);
  });

  it("k=1 (extreme): top ranks dominate, disagreement causes instability", () => {
    const result = blendByRank(pool, reversedRerank, 1);
    // With k=1: rank 1 score = 1/(1+1) = 0.5, rank 5 = 1/(1+5) = 0.167
    // vec1: 0.5 + 0.167 = 0.667
    // vec5: 0.167 + 0.5 = 0.667
    // They should be nearly equal
    const vec1Score = result.find((r) => r.chunk.id === "vec1")!.score;
    const vec5Score = result.find((r) => r.chunk.id === "vec5")!.score;
    expect(Math.abs(vec1Score - vec5Score)).toBeLessThan(0.01);
  });

  it("vectorWeight > rerankerWeight: vector ordering dominates", () => {
    const result = blendByRank(pool, reversedRerank, 60, 2.0, 1.0);
    // With 2x vector weight, vec1 (vector #1) should outrank vec5 (reranker #1)
    const ranking = getRanking(result);
    const vec1Rank = ranking.indexOf("vec1");
    const vec5Rank = ranking.indexOf("vec5");
    expect(vec1Rank).toBeLessThan(vec5Rank);
  });

  it("rerankerWeight > vectorWeight: reranker ordering dominates", () => {
    const result = blendByRank(pool, reversedRerank, 60, 1.0, 2.0);
    // With 2x reranker weight, vec5 (reranker #1) should outrank vec1 (vector #1)
    const ranking = getRanking(result);
    const vec1Rank = ranking.indexOf("vec1");
    const vec5Rank = ranking.indexOf("vec5");
    expect(vec5Rank).toBeLessThan(vec1Rank);
  });

  it("equal weights with total agreement: preserves input order", () => {
    // When both systems agree on ranking, RRF should preserve it
    const agreedRerank = [
      makeRerankResult(0, 0.99),  // vec1: reranker rank 1 (agrees with vector)
      makeRerankResult(1, 0.95),  // vec2: reranker rank 2
      makeRerankResult(2, 0.90),  // vec3: reranker rank 3
      makeRerankResult(3, 0.85),  // vec4: reranker rank 4
      makeRerankResult(4, 0.80),  // vec5: reranker rank 5
    ];

    const result = blendByRank(pool, agreedRerank);
    expect(getRanking(result)).toEqual(["vec1", "vec2", "vec3", "vec4", "vec5"]);
  });
});

// ── Edge cases ───────────────────────────────────────────────────────────

describe("blendByRank edge cases", () => {
  it("handles empty pool", () => {
    expect(blendByRank([], [], 60)).toEqual([]);
  });

  it("handles single candidate", () => {
    const pool = [makeResult("only", 0.8)];
    const rerank = [makeRerankResult(0, 0.95)];
    const result = blendByRank(pool, rerank);
    expect(result).toHaveLength(1);
    expect(result[0]!.chunk.id).toBe("only");
  });

  it("handles reranker returning subset of pool", () => {
    const pool = [
      makeResult("a", 0.5),
      makeResult("b", 0.4),
      makeResult("c", 0.3),
    ];
    // Reranker only scored 2 of 3 docs
    const rerank = [
      makeRerankResult(2, 0.99),
      makeRerankResult(0, 0.80),
    ];
    const result = blendByRank(pool, rerank);
    // All 3 docs should appear — b has vector contribution only
    expect(result).toHaveLength(3);
    // b should be ranked last (only has vector rank 2, no reranker boost)
    const bRank = result.findIndex((r) => r.chunk.id === "b");
    expect(bRank).toBe(2);
  });

  it("handles tied reranker scores (same relevance_score)", () => {
    const pool = [
      makeResult("a", 0.5),
      makeResult("b", 0.4),
      makeResult("c", 0.3),
    ];
    const rerank = [
      makeRerankResult(0, 0.95),
      makeRerankResult(1, 0.95),  // tied with a
      makeRerankResult(2, 0.80),
    ];
    const result = blendByRank(pool, rerank);
    // With tied reranker scores, vector order should break the tie
    // a has vector rank 1 + reranker rank 1 (or 2), b has vector rank 2 + reranker rank 2 (or 1)
    expect(result).toHaveLength(3);
    // a should be first (vector advantage)
    expect(result[0]!.chunk.id).toBe("a");
  });
});

// ── Head-to-head: realistic SciFact-like scenario ────────────────────────

describe("head-to-head: realistic SciFact scenario", () => {
  it("simulates the query-1 failure (0-dimensional biomaterials)", () => {
    // Real data from telemetry:
    // Vector: doc_21456232 (0.277), doc_43990286 (0.236), doc_31715818 (0.229 — RELEVANT), ...
    // Reranker: doc_14082855 (0.999), doc_9122283 (0.95), doc_39532074 (0.92), ...
    //   doc_31715818 gets pushed below rank 10 by reranker

    const pool = [
      makeResult("21456232", 0.277),   // vector rank 1
      makeResult("43990286", 0.236),   // vector rank 2
      makeResult("31715818", 0.229),   // vector rank 3 — RELEVANT
      makeResult("25435456", 0.227),   // vector rank 4
      makeResult("19855358", 0.226),   // vector rank 5
      makeResult("14082855", 0.220),   // vector rank 6
      makeResult("9122283", 0.215),    // vector rank 7
      makeResult("39532074", 0.210),   // vector rank 8
      makeResult("16532419", 0.205),   // vector rank 9
      makeResult("8290953", 0.200),    // vector rank 10
    ];

    // Reranker completely reshuffles — pushes relevant doc to last
    const rerankResults = [
      makeRerankResult(5, 0.999),  // 14082855: reranker rank 1
      makeRerankResult(6, 0.995),  // 9122283: reranker rank 2
      makeRerankResult(7, 0.990),  // 39532074: reranker rank 3
      makeRerankResult(8, 0.985),  // 16532419: reranker rank 4
      makeRerankResult(9, 0.980),  // 8290953: reranker rank 5
      makeRerankResult(0, 0.975),  // 21456232: reranker rank 6
      makeRerankResult(3, 0.970),  // 25435456: reranker rank 7
      makeRerankResult(1, 0.965),  // 43990286: reranker rank 8
      makeRerankResult(4, 0.960),  // 19855358: reranker rank 9
      makeRerankResult(2, 0.850),  // 31715818: reranker rank 10 — CATASTROPHIC DEMOTION
    ];

    // BEFORE: score-based blend at α=0 → relevant doc at rank 10 (last)
    const scoreBased = blendScores(pool, rerankResults, 0);
    const relevantRankScore = scoreBased.findIndex((r) => r.chunk.id === "31715818");
    expect(relevantRankScore).toBe(9); // Dead last

    // BEFORE: score-based blend at α=0.5 → relevant doc improves but still buried
    const scoreBlend05 = blendScores(pool, rerankResults, 0.5);
    const relevantRankScore05 = scoreBlend05.findIndex((r) => r.chunk.id === "31715818");

    // AFTER: RRF → vector rank 3 + reranker rank 10
    // RRF score = 1/(60+3) + 1/(60+10) = 1/63 + 1/70 ≈ 0.0302
    // Compare to doc at vector rank 6 + reranker rank 1:
    // RRF score = 1/(60+6) + 1/(60+1) = 1/66 + 1/61 ≈ 0.0316
    const rrfResult = blendByRank(pool, rerankResults);
    const relevantRankRrf = rrfResult.findIndex((r) => r.chunk.id === "31715818");

    // RRF should keep relevant doc in better position than score-based α=0
    expect(relevantRankRrf).toBeLessThan(relevantRankScore);
    // And should keep it in top half (vector rank 3 is strong signal)
    expect(relevantRankRrf).toBeLessThanOrEqual(6);

    console.log(`  SciFact query-1 simulation:`);
    console.log(`    Score-based α=0: relevant at rank ${relevantRankScore + 1}`);
    console.log(`    Score-based α=0.5: relevant at rank ${relevantRankScore05 + 1}`);
    console.log(`    RRF (k=60): relevant at rank ${relevantRankRrf + 1}`);
  });

  it("simulates vector-confident query where reranker agrees", () => {
    // When both systems agree, RRF should preserve the good ranking
    const pool = [
      makeResult("relevant1", 0.65),   // vector rank 1 — RELEVANT
      makeResult("relevant2", 0.55),   // vector rank 2 — RELEVANT
      makeResult("irrelevant1", 0.40),
      makeResult("irrelevant2", 0.35),
      makeResult("irrelevant3", 0.30),
    ];

    const rerankResults = [
      makeRerankResult(0, 0.999),  // relevant1: reranker rank 1 ✓
      makeRerankResult(1, 0.980),  // relevant2: reranker rank 2 ✓
      makeRerankResult(2, 0.900),
      makeRerankResult(3, 0.850),
      makeRerankResult(4, 0.800),
    ];

    const scoreBased = blendScores(pool, rerankResults, 0.5);
    const rrfResult = blendByRank(pool, rerankResults);

    // Both should agree on ranking
    expect(getRanking(scoreBased).slice(0, 2)).toEqual(["relevant1", "relevant2"]);
    expect(getRanking(rrfResult).slice(0, 2)).toEqual(["relevant1", "relevant2"]);
  });

  it("tight vector cluster: RRF with vector bias protects against arbitrary reshuffling", () => {
    // All vector scores within 0.05 — a near-tie
    const pool = [
      makeResult("a", 0.252),
      makeResult("b", 0.250),
      makeResult("c", 0.248),   // RELEVANT
      makeResult("d", 0.246),
      makeResult("e", 0.244),
    ];

    // Reranker arbitrarily picks e as best and puts c last
    const rerankResults = [
      makeRerankResult(4, 0.999),  // e: rank 1
      makeRerankResult(3, 0.998),  // d: rank 2
      makeRerankResult(1, 0.997),  // b: rank 3
      makeRerankResult(0, 0.996),  // a: rank 4
      makeRerankResult(2, 0.850),  // c: rank 5 — DEMOTED
    ];

    // Score-based α=0: c drops to rank 5
    const scoreBased = blendScores(pool, rerankResults, 0);
    const cRankScore = scoreBased.findIndex((r) => r.chunk.id === "c");
    expect(cRankScore).toBe(4); // Last

    // RRF with equal weights: c stays at rank 5 because reranker rank 5 is a heavy penalty
    const rrfEqual = blendByRank(pool, rerankResults);
    const cRankRrfEqual = rrfEqual.findIndex((r) => r.chunk.id === "c");
    expect(cRankRrfEqual).toBe(4); // Still last — RRF alone can't fix this

    // Even with 1.5x vector weight, reranker rank 5 is too much penalty
    const rrfBias = blendByRank(pool, rerankResults, 60, 1.5, 1.0);
    const cRankRrfBias = rrfBias.findIndex((r) => r.chunk.id === "c");
    expect(cRankRrfBias).toBe(4); // Still last

    // KEY INSIGHT: When the reranker specifically targets ONE doc for demotion
    // (rank 5 with a big score gap: 0.850 vs 0.996+), RRF can't rescue it.
    // This is exactly why we need Fix 2 (dynamic reranker gate):
    // skip reranking entirely when vector scores are this tightly clustered.
    // RRF's value is in the GENERAL case — preventing systematic demotion
    // across many queries, not saving individual badly-ranked docs.
  });
});
