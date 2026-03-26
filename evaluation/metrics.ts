/**
 * Information Retrieval Evaluation Metrics
 *
 * Implements standard IR metrics following BEIR (Thakur et al., 2021):
 * - NDCG@K (Normalized Discounted Cumulative Gain)
 * - MRR (Mean Reciprocal Rank)
 * - MAP@K (Mean Average Precision)
 * - Recall@K
 * - Precision@K
 *
 * All metrics use graded relevance (0-3) compatible with BEIR evaluation.
 */

export interface RetrievedDoc {
  path: string;
  text: string;
  score: number;
}

export interface RelevanceJudgment {
  path_contains: string;
  snippet_contains?: string;
  grade: number; // 0-3
}

export interface QueryResult {
  queryId: string;
  category: string;
  query: string;
  retrieved: RetrievedDoc[];
  relevant: RelevanceJudgment[];
  latencyMs: number;
}

export interface MetricResult {
  mean: number;
  std: number;
  ci95: [number, number];
  values: number[];
}

export interface EvalResults {
  runId: string;
  timestamp: string;
  config: Record<string, boolean | string | number>;
  metrics: {
    ndcg_at_1: MetricResult;
    ndcg_at_5: MetricResult;
    ndcg_at_10: MetricResult;
    mrr: MetricResult;
    map_at_10: MetricResult;
    recall_at_1: MetricResult;
    recall_at_3: MetricResult;
    recall_at_5: MetricResult;
    recall_at_10: MetricResult;
    precision_at_5: MetricResult;
    mean_latency_ms: number;
    p50_latency_ms: number;
    p95_latency_ms: number;
    p99_latency_ms: number;
  };
  perCategory: Record<string, {
    ndcg_at_10: number;
    mrr: number;
    recall_at_5: number;
    count: number;
  }>;
  perQuery: QueryResult[];
}

/**
 * Match a retrieved document against relevance judgments.
 * Returns the highest matching grade (0 if no match).
 */
function gradeDocument(doc: RetrievedDoc, judgments: RelevanceJudgment[]): number {
  let maxGrade = 0;
  for (const j of judgments) {
    const pathMatch = doc.path.toLowerCase().includes(j.path_contains.toLowerCase());
    if (!pathMatch) continue;

    if (j.snippet_contains) {
      const snippetMatch = doc.text.toLowerCase().includes(j.snippet_contains.toLowerCase());
      if (!snippetMatch) continue;
    }

    maxGrade = Math.max(maxGrade, j.grade);
  }
  return maxGrade;
}

/**
 * Discounted Cumulative Gain at K
 */
function dcgAtK(grades: number[], k: number): number {
  let dcg = 0;
  for (let i = 0; i < Math.min(grades.length, k); i++) {
    dcg += (Math.pow(2, grades[i]) - 1) / Math.log2(i + 2);
  }
  return dcg;
}

/**
 * Normalized DCG at K
 */
export function ndcgAtK(retrieved: RetrievedDoc[], judgments: RelevanceJudgment[], k: number): number {
  const grades = retrieved.slice(0, k).map(doc => gradeDocument(doc, judgments));

  // Ideal ranking: sort all possible grades descending
  const idealGrades = judgments.map(j => j.grade).sort((a, b) => b - a).slice(0, k);

  const dcg = dcgAtK(grades, k);
  const idcg = dcgAtK(idealGrades, k);

  return idcg === 0 ? 0 : dcg / idcg;
}

/**
 * Reciprocal Rank (1/rank of first relevant result)
 */
export function reciprocalRank(retrieved: RetrievedDoc[], judgments: RelevanceJudgment[]): number {
  for (let i = 0; i < retrieved.length; i++) {
    if (gradeDocument(retrieved[i], judgments) > 0) {
      return 1 / (i + 1);
    }
  }
  return 0;
}

/**
 * Average Precision at K
 */
export function averagePrecisionAtK(retrieved: RetrievedDoc[], judgments: RelevanceJudgment[], k: number): number {
  let numRelevant = 0;
  let sumPrecision = 0;
  const totalRelevant = judgments.filter(j => j.grade > 0).length;

  for (let i = 0; i < Math.min(retrieved.length, k); i++) {
    if (gradeDocument(retrieved[i], judgments) > 0) {
      numRelevant++;
      sumPrecision += numRelevant / (i + 1);
    }
  }

  return totalRelevant === 0 ? 0 : sumPrecision / totalRelevant;
}

/**
 * Recall at K: fraction of relevant docs found in top-K
 */
export function recallAtK(retrieved: RetrievedDoc[], judgments: RelevanceJudgment[], k: number): number {
  const totalRelevant = judgments.filter(j => j.grade > 0).length;
  if (totalRelevant === 0) return 0;

  let found = 0;
  const topK = retrieved.slice(0, k);

  // For each judgment, check if ANY top-K doc matches
  for (const j of judgments) {
    if (j.grade === 0) continue;
    const matched = topK.some(doc => {
      const pathMatch = doc.path.toLowerCase().includes(j.path_contains.toLowerCase());
      if (!pathMatch) return false;
      if (j.snippet_contains) {
        return doc.text.toLowerCase().includes(j.snippet_contains.toLowerCase());
      }
      return true;
    });
    if (matched) found++;
  }

  return found / totalRelevant;
}

/**
 * Precision at K
 */
export function precisionAtK(retrieved: RetrievedDoc[], judgments: RelevanceJudgment[], k: number): number {
  const topK = retrieved.slice(0, k);
  if (topK.length === 0) return 0;

  let relevant = 0;
  for (const doc of topK) {
    if (gradeDocument(doc, judgments) > 0) relevant++;
  }

  return relevant / topK.length;
}

/**
 * Compute aggregate metric from per-query values
 */
export function aggregate(values: number[]): MetricResult {
  if (values.length === 0) {
    return { mean: 0, std: 0, ci95: [0, 0], values: [] };
  }

  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
  const std = Math.sqrt(variance);

  // 95% CI using t-distribution approximation (z=1.96 for large n)
  const se = std / Math.sqrt(values.length);
  const ci95: [number, number] = [
    Math.max(0, mean - 1.96 * se),
    Math.min(1, mean + 1.96 * se)
  ];

  return { mean, std, ci95, values };
}

/**
 * Compute percentile from sorted array
 */
export function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

/**
 * Compile all metrics from query results
 */
export function compileResults(
  queryResults: QueryResult[],
  config: Record<string, boolean | string | number>
): EvalResults {
  const ndcg1: number[] = [];
  const ndcg5: number[] = [];
  const ndcg10: number[] = [];
  const mrr: number[] = [];
  const map10: number[] = [];
  const recall1: number[] = [];
  const recall3: number[] = [];
  const recall5: number[] = [];
  const recall10: number[] = [];
  const prec5: number[] = [];
  const latencies: number[] = [];

  const catMetrics: Record<string, { ndcg10: number[]; mrr: number[]; recall5: number[] }> = {};

  for (const qr of queryResults) {
    ndcg1.push(ndcgAtK(qr.retrieved, qr.relevant, 1));
    ndcg5.push(ndcgAtK(qr.retrieved, qr.relevant, 5));
    ndcg10.push(ndcgAtK(qr.retrieved, qr.relevant, 10));
    mrr.push(reciprocalRank(qr.retrieved, qr.relevant));
    map10.push(averagePrecisionAtK(qr.retrieved, qr.relevant, 10));
    recall1.push(recallAtK(qr.retrieved, qr.relevant, 1));
    recall3.push(recallAtK(qr.retrieved, qr.relevant, 3));
    recall5.push(recallAtK(qr.retrieved, qr.relevant, 5));
    recall10.push(recallAtK(qr.retrieved, qr.relevant, 10));
    prec5.push(precisionAtK(qr.retrieved, qr.relevant, 5));
    latencies.push(qr.latencyMs);

    // Per-category
    if (!catMetrics[qr.category]) {
      catMetrics[qr.category] = { ndcg10: [], mrr: [], recall5: [] };
    }
    catMetrics[qr.category].ndcg10.push(ndcgAtK(qr.retrieved, qr.relevant, 10));
    catMetrics[qr.category].mrr.push(reciprocalRank(qr.retrieved, qr.relevant));
    catMetrics[qr.category].recall5.push(recallAtK(qr.retrieved, qr.relevant, 5));
  }

  const sortedLatencies = [...latencies].sort((a, b) => a - b);

  const perCategory: EvalResults["perCategory"] = {};
  for (const [cat, m] of Object.entries(catMetrics)) {
    perCategory[cat] = {
      ndcg_at_10: m.ndcg10.reduce((a, b) => a + b, 0) / m.ndcg10.length,
      mrr: m.mrr.reduce((a, b) => a + b, 0) / m.mrr.length,
      recall_at_5: m.recall5.reduce((a, b) => a + b, 0) / m.recall5.length,
      count: m.ndcg10.length,
    };
  }

  return {
    runId: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    config,
    metrics: {
      ndcg_at_1: aggregate(ndcg1),
      ndcg_at_5: aggregate(ndcg5),
      ndcg_at_10: aggregate(ndcg10),
      mrr: aggregate(mrr),
      map_at_10: aggregate(map10),
      recall_at_1: aggregate(recall1),
      recall_at_3: aggregate(recall3),
      recall_at_5: aggregate(recall5),
      recall_at_10: aggregate(recall10),
      precision_at_5: aggregate(prec5),
      mean_latency_ms: latencies.reduce((a, b) => a + b, 0) / latencies.length,
      p50_latency_ms: percentile(sortedLatencies, 50),
      p95_latency_ms: percentile(sortedLatencies, 95),
      p99_latency_ms: percentile(sortedLatencies, 99),
    },
    perCategory,
    perQuery: queryResults,
  };
}
