/**
 * NER — Spark 18112 entity tagging.
 * Endpoint: POST /v1/extract with {"text": "..."}
 * Returns [] on failure (non-fatal).
 */

import type { MemorySparkConfig } from "../config.js";

interface NerResponse {
  entities: Array<{
    entity_group: string;
    score: number;
    word: string;
    start: number;
    end: number;
  }>;
  count: number;
}

export async function tagEntities(text: string, cfg: MemorySparkConfig): Promise<string[]> {
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (cfg.embed.spark?.apiKey) {
      headers["Authorization"] = `Bearer ${cfg.embed.spark.apiKey}`;
    }
    const resp = await fetch(`${cfg.spark.ner}/v1/extract`, {
      method: "POST",
      headers,
      body: JSON.stringify({ text: text.slice(0, 2000) }),
      signal: AbortSignal.timeout(10000),
    });
    if (!resp.ok) return [];

    const data = await resp.json() as NerResponse;
    const entities = new Set<string>();
    for (const e of data.entities) {
      const word = e.word.trim().replace(/^##/, "");
      if (word.length > 2 && e.score > 0.7) {
        entities.add(word);
      }
    }
    return Array.from(entities);
  } catch {
    return [];
  }
}
