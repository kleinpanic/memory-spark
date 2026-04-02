/**
 * Information Retrieval Metrics — BEIR-compatible implementation.
 *
 * Implements the standard IR evaluation metrics as defined in:
 * - BEIR: A Heterogeneous Benchmark for Zero-Shot Evaluation of Information Retrieval Models
 *   (Thakur et al., 2021) https://arxiv.org/abs/2104.08663
 * - RAGAS: Automated Evaluation of Retrieval Augmented Generation
 *   (Es et al., 2023) https://arxiv.org/abs/2309.15217
 *
 * All metrics follow pytrec_eval conventions for compatibility with published baselines.
 */

/** Relevance judgments: queryId → { docId → relevanceScore } */
export type Qrels = Record<string, Record<string, number>>;
/** Retrieval results: queryId → { docId → retrievalScore } */
export type Results = Record<string, Record<string, number>>;

/**
 * Normalized Discounted Cumulative Gain at k (NDCG@k)
 * Primary metric in BEIR. Accounts for position-dependent relevance.
 */
export function ndcgAtK(qrels: Qrels, results: Results, k: number): Record<string, number> {
  const scores: Record<string, number> = {};
  for (const queryId of Object.keys(qrels)) {
    const queryQrels = qrels[queryId]!;
    const queryResults = results[queryId] ?? {};
    const ranked = Object.entries(queryResults)
      .sort(([, a], [, b]) => b - a)
      .slice(0, k);

    let dcg = 0;
    for (let i = 0; i < ranked.length; i++) {
      const rel = queryQrels[ranked[i]![0]] ?? 0;
      dcg += (Math.pow(2, rel) - 1) / Math.log2(i + 2);
    }

    const idealRels = Object.values(queryQrels)
      .sort((a, b) => b - a)
      .slice(0, k);
    let idcg = 0;
    for (let i = 0; i < idealRels.length; i++) {
      idcg += (Math.pow(2, idealRels[i]!) - 1) / Math.log2(i + 2);
    }

    scores[queryId] = idcg === 0 ? 0 : dcg / idcg;
  }
  return scores;
}

/**
 * Mean Reciprocal Rank at k (MRR@k)
 */
export function mrrAtK(qrels: Qrels, results: Results, k: number): Record<string, number> {
  const scores: Record<string, number> = {};
  for (const queryId of Object.keys(qrels)) {
    const queryQrels = qrels[queryId]!;
    const queryResults = results[queryId] ?? {};
    const ranked = Object.entries(queryResults)
      .sort(([, a], [, b]) => b - a)
      .slice(0, k);

    let rr = 0;
    for (let i = 0; i < ranked.length; i++) {
      if ((queryQrels[ranked[i]![0]] ?? 0) > 0) {
        rr = 1 / (i + 1);
        break;
      }
    }
    scores[queryId] = rr;
  }
  return scores;
}

/**
 * Recall at k (Recall@k)
 */
export function recallAtK(qrels: Qrels, results: Results, k: number): Record<string, number> {
  const scores: Record<string, number> = {};
  for (const queryId of Object.keys(qrels)) {
    const queryQrels = qrels[queryId]!;
    const queryResults = results[queryId] ?? {};
    const ranked = Object.entries(queryResults)
      .sort(([, a], [, b]) => b - a)
      .slice(0, k);

    const totalRelevant = Object.values(queryQrels).filter((r) => r > 0).length;
    if (totalRelevant === 0) {
      scores[queryId] = 0;
      continue;
    }

    let retrieved = 0;
    for (const [docId] of ranked) {
      if ((queryQrels[docId] ?? 0) > 0) retrieved++;
    }
    scores[queryId] = retrieved / totalRelevant;
  }
  return scores;
}

/**
 * Mean Average Precision at k (MAP@k)
 */
export function mapAtK(qrels: Qrels, results: Results, k: number): Record<string, number> {
  const scores: Record<string, number> = {};
  for (const queryId of Object.keys(qrels)) {
    const queryQrels = qrels[queryId]!;
    const queryResults = results[queryId] ?? {};
    const ranked = Object.entries(queryResults)
      .sort(([, a], [, b]) => b - a)
      .slice(0, k);

    const totalRelevant = Object.values(queryQrels).filter((r) => r > 0).length;
    if (totalRelevant === 0) {
      scores[queryId] = 0;
      continue;
    }

    let ap = 0;
    let relevantSoFar = 0;
    for (let i = 0; i < ranked.length; i++) {
      if ((queryQrels[ranked[i]![0]] ?? 0) > 0) {
        relevantSoFar++;
        ap += relevantSoFar / (i + 1);
      }
    }
    // BEIR uses pytrec_eval which divides by totalRelevant (not min(R,k)).
    // A 2026-03-27 "fix" changed this to min(R,k) but that was a regression:
    // it inflates MAP ~3.8x on high-R datasets like NFCorpus (avg 38 relevant/query).
    // Reverted 2026-04-02 after audit confirmed pytrec_eval uses R.
    scores[queryId] = ap / totalRelevant;
  }
  return scores;
}

/**
 * Precision at k (P@k)
 */
export function precisionAtK(qrels: Qrels, results: Results, k: number): Record<string, number> {
  const scores: Record<string, number> = {};
  for (const queryId of Object.keys(qrels)) {
    const queryQrels = qrels[queryId]!;
    const queryResults = results[queryId] ?? {};
    const ranked = Object.entries(queryResults)
      .sort(([, a], [, b]) => b - a)
      .slice(0, k);

    let relevant = 0;
    for (const [docId] of ranked) {
      if ((queryQrels[docId] ?? 0) > 0) relevant++;
    }
    // Bug fix (2026-03-27): was `relevant / ranked.length`, which inflates P@k
    // when fewer than k results are returned. P@10 with 3 results, 2 relevant
    // was giving 0.67 instead of correct 0.20.
    scores[queryId] = relevant / k;
  }
  return scores;
}

export function mean(scores: Record<string, number>): number {
  const vals = Object.values(scores);
  if (vals.length === 0) return 0;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

/** Standard BEIR evaluation at multiple k values */
export function evaluateBEIR(
  qrels: Qrels,
  results: Results,
  kValues: number[] = [1, 3, 5, 10],
): {
  ndcg: Record<string, number>;
  mrr: Record<string, number>;
  recall: Record<string, number>;
  map: Record<string, number>;
  precision: Record<string, number>;
} {
  const out: Record<string, Record<string, number>> = {
    ndcg: {},
    mrr: {},
    recall: {},
    map: {},
    precision: {},
  };
  for (const k of kValues) {
    out.ndcg![`@${k}`] = mean(ndcgAtK(qrels, results, k));
    out.mrr![`@${k}`] = mean(mrrAtK(qrels, results, k));
    out.recall![`@${k}`] = mean(recallAtK(qrels, results, k));
    out.map![`@${k}`] = mean(mapAtK(qrels, results, k));
    out.precision![`@${k}`] = mean(precisionAtK(qrels, results, k));
  }
  return out as ReturnType<typeof evaluateBEIR>;
}

export function formatBEIRResults(results: ReturnType<typeof evaluateBEIR>): string {
  const lines: string[] = [];
  lines.push("┌──────────┬──────────┬──────────┬──────────┬──────────┐");
  lines.push("│ Metric   │   @1     │   @3     │   @5     │   @10    │");
  lines.push("├──────────┼──────────┼──────────┼──────────┼──────────┤");
  for (const [name, values] of Object.entries(results)) {
    const cells = ["@1", "@3", "@5", "@10"].map((k) => (values[k] ?? 0).toFixed(4).padStart(8));
    lines.push(`│ ${name.padEnd(8)} │${cells.join(" │")} │`);
  }
  lines.push("└──────────┴──────────┴──────────┴──────────┴──────────┘");
  return lines.join("\n");
}
