/**
 * Embedding Cache — LRU in-memory cache for query embeddings.
 *
 * Why: Auto-recall fires on every agent turn. If an agent has a multi-turn
 * conversation about the same topic, the query text is often similar or
 * identical. Caching avoids redundant Spark round-trips (~150ms each).
 *
 * Cache is query-only (not document embeddings). Document embeddings are
 * one-shot during indexing and don't benefit from caching.
 *
 * TTL prevents stale cache entries from surviving too long if the embedding
 * model changes (though dims-lock.ts prevents actual model swaps).
 */

import crypto from "node:crypto";

export interface EmbedCacheConfig {
  /** Whether the cache is enabled. Default: true */
  enabled: boolean;
  /** Maximum number of cached embeddings. Default: 256 */
  maxSize: number;
  /** TTL in milliseconds. Default: 30 minutes */
  ttlMs: number;
}

const DEFAULT_CACHE_CONFIG: EmbedCacheConfig = {
  enabled: true,
  maxSize: 256,
  ttlMs: 30 * 60 * 1000, // 30 minutes
};

interface CacheEntry {
  vector: number[];
  usedAt: number;
  createdAt: number;
}

/**
 * Normalize query text for cache key generation.
 * Strips whitespace variance so "  hello  world  " matches "hello world".
 */
function normalizeForKey(text: string): string {
  return text.trim().replace(/\s+/g, " ").toLowerCase();
}

function hashKey(text: string): string {
  return crypto.createHash("sha256").update(normalizeForKey(text)).digest("hex").slice(0, 16);
}

export class EmbedCache {
  private cache = new Map<string, CacheEntry>();
  private cfg: EmbedCacheConfig;
  private hits = 0;
  private misses = 0;

  constructor(cfg?: Partial<EmbedCacheConfig>) {
    this.cfg = { ...DEFAULT_CACHE_CONFIG, ...cfg };
  }

  get(text: string): number[] | undefined {
    if (!this.cfg.enabled) {
      this.misses++;
      return undefined;
    }

    const key = hashKey(text);
    const entry = this.cache.get(key);

    if (!entry) {
      this.misses++;
      return undefined;
    }

    // TTL check
    if (Date.now() - entry.createdAt > this.cfg.ttlMs) {
      this.cache.delete(key);
      this.misses++;
      return undefined;
    }

    // LRU touch
    entry.usedAt = Date.now();
    this.hits++;
    return entry.vector;
  }

  set(text: string, vector: number[]): void {
    if (!this.cfg.enabled) return;

    const key = hashKey(text);
    const now = Date.now();

    // Evict if at capacity
    if (this.cache.size >= this.cfg.maxSize && !this.cache.has(key)) {
      this.evictLRU();
    }

    this.cache.set(key, { vector, usedAt: now, createdAt: now });
  }

  /** Cache stats for memory_health tool */
  stats(): { size: number; maxSize: number; hits: number; misses: number; hitRate: string } {
    const total = this.hits + this.misses;
    const hitRate = total > 0 ? ((this.hits / total) * 100).toFixed(1) + "%" : "N/A";
    return {
      size: this.cache.size,
      maxSize: this.cfg.maxSize,
      hits: this.hits,
      misses: this.misses,
      hitRate,
    };
  }

  /** Clear all cached embeddings */
  clear(): void {
    this.cache.clear();
    this.hits = 0;
    this.misses = 0;
  }

  private evictLRU(): void {
    let oldest: string | undefined;
    let oldestTime = Infinity;

    for (const [key, entry] of this.cache) {
      if (entry.usedAt < oldestTime) {
        oldestTime = entry.usedAt;
        oldest = key;
      }
    }

    if (oldest) this.cache.delete(oldest);
  }
}
