/**
 * Named Entity Recognition (NER)
 *
 * Tags entities in text chunks using Spark 18112 (dslim/bert-base-NER).
 * Results stored as chunk metadata for faceted search / filtering.
 *
 * Entity types extracted: PER (person), ORG (org), LOC (location), MISC
 * We normalize these to simple string tags: ["Klein", "OpenClaw", "Spark", ...]
 *
 * Used in two places:
 *   1. Ingest pipeline — tag each chunk from ingested files
 *   2. Auto-capture — tag captured memory facts
 *
 * API: Spark NER is a HuggingFace token-classification endpoint.
 * POST { inputs: "text here" }
 * Returns: [{ entity_group: "PER", score: 0.99, word: "Klein", ... }]
 *
 * Best-effort: if Spark NER is unavailable, returns empty array (non-fatal).
 */

import type { MemorySparkConfig } from "../config.js";

/**
 * Extract entity tags from text using Spark NER.
 * Returns array of entity strings (deduplicated, normalized).
 * Never throws — returns [] on failure.
 */
export async function tagEntities(text: string, cfg: MemorySparkConfig): Promise<string[]> {
  // TODO:
  // const resp = await fetch(`${cfg.spark.ner}`, {
  //   method: "POST",
  //   headers: { "Content-Type": "application/json" },
  //   body: JSON.stringify({ inputs: text }),
  // });
  // const entities = await resp.json() as NerResult[];
  // return [...new Set(entities.map(e => e.word).filter(w => w.length > 2))];
  return [];
}
