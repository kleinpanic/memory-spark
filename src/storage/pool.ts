/**
 * Pool routing — determines which logical pool a MemoryChunk belongs to.
 *
 * Pools are logical sections within the single LanceDB table.
 * The `pool` column enables efficient WHERE-based filtering without
 * needing multiple tables (per LanceDB best practices).
 *
 * Pool types:
 *   - agent_memory     — per-agent workspace files, captures
 *   - agent_tools      — per-agent tool definitions
 *   - agent_mistakes   — per-agent mistakes (private to agent)
 *   - shared_knowledge — cross-agent facts, infrastructure docs
 *   - shared_mistakes  — cross-agent mistakes (visible to all)
 *   - shared_rules     — global rules & preferences (always injected)
 *   - reference_library — PDFs, documentation (tool-call only)
 *   - reference_code   — code examples (tool-call only)
 *
 * @module storage/pool
 */

import type { MemoryChunk } from "./backend.js";

/** All valid pool values */
export const POOL_VALUES = [
  "agent_memory",
  "agent_tools",
  "agent_mistakes",
  "shared_knowledge",
  "shared_mistakes",
  "shared_rules",
  "reference_library",
  "reference_code",
] as const;

export type PoolValue = (typeof POOL_VALUES)[number];

/** Pools that are auto-injected during recall */
export const AUTO_INJECT_POOLS: PoolValue[] = [
  "agent_memory",
  "agent_tools",
  "agent_mistakes",
  "shared_knowledge",
  "shared_mistakes",
  "shared_rules",
];

/** Pools that are NEVER auto-injected (tool-call only) */
export const REFERENCE_POOLS: PoolValue[] = ["reference_library", "reference_code"];

/**
 * Pools that are always injected regardless of relevance score.
 * @public — pool API constant for consumers
 */
export const ALWAYS_INJECT_POOLS: PoolValue[] = ["shared_rules"];

/**
 * Determine which logical pool a chunk belongs to based on content_type, path, and category.
 *
 * Routing rules (in priority order):
 * 1. Explicit `pool` already set on chunk → use it
 * 2. content_type === "tool" or TOOLS.md path → "agent_tools"
 * 3. MISTAKES.md path or content_type === "mistake" → "agent_mistakes" (per-agent default)
 * 4. content_type === "rule" or "preference" → "shared_rules"
 * 5. content_type === "reference" → "reference_library"
 * 6. content_type === "reference_code" → "reference_code"
 * 7. Everything else → "agent_memory"
 */
export function resolvePool(chunk: Partial<MemoryChunk>): PoolValue {
  // Explicit pool overrides auto-routing
  if (chunk.pool && POOL_VALUES.includes(chunk.pool as PoolValue)) {
    return chunk.pool as PoolValue;
  }

  const contentType = chunk.content_type ?? "knowledge";
  const pathLower = (chunk.path ?? "").toLowerCase();
  const basename = pathLower.split("/").pop() ?? "";

  // Tool definitions → agent_tools
  if (contentType === "tool" || basename === "tools.md" || basename.startsWith("tools-")) {
    return "agent_tools";
  }

  // Mistakes → agent_mistakes (per-agent by default)
  // Promotion to shared_mistakes is an explicit action
  if (basename === "mistakes.md" || pathLower.includes("mistakes/") || contentType === "mistake") {
    return "agent_mistakes";
  }

  // Rules and preferences → shared_rules
  if (contentType === "rule" || contentType === "preference") {
    return "shared_rules";
  }

  // Reference documents → reference_library
  if (contentType === "reference") {
    return "reference_library";
  }

  // Code references → reference_code
  if (contentType === "reference_code") {
    return "reference_code";
  }

  // Default: agent's own memory
  return "agent_memory";
}

/**
 * Check if a pool should be auto-injected during recall.
 * Reference pools are NEVER auto-injected.
 */
export function isAutoInjectPool(pool: string): boolean {
  return AUTO_INJECT_POOLS.includes(pool as PoolValue);
}

/**
 * Check if a pool should ALWAYS be injected regardless of relevance score.
 */
export function isAlwaysInjectPool(pool: string): boolean {
  return ALWAYS_INJECT_POOLS.includes(pool as PoolValue);
}
