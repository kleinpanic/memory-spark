/**
 * NER — Spark 18112 entity tagging. Returns [] on failure (non-fatal).
 */

import type { MemorySparkConfig } from "../config.js";

interface NerResult {
  entity_group: string;
  score: number;
  word: string;
  start: number;
  end: number;
}

export async function tagEntities(text: string, cfg: MemorySparkConfig): Promise<string[]> {
  try {
    const resp = await fetch(cfg.spark.ner, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inputs: text.slice(0, 2000) }),
      signal: AbortSignal.timeout(5000),
    });
    if (!resp.ok) return [];

    const results = await resp.json() as NerResult[];
    // Deduplicate and filter short words
    const entities = new Set<string>();
    for (const r of results) {
      const word = r.word.trim().replace(/^##/, "");
      if (word.length > 2 && r.score > 0.7) {
        entities.add(word);
      }
    }
    return Array.from(entities);
  } catch {
    return [];
  }
}
