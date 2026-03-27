/**
 * Cached Embed Provider — wraps any EmbedProvider/EmbedQueue with an LRU cache.
 * Only caches embedQuery (recall queries). embedBatch (indexing) bypasses cache.
 */

import { EmbedCache, type EmbedCacheConfig } from "./cache.js";

/** Minimal embed interface (covers both EmbedProvider and EmbedQueue) */
export interface EmbedLike {
  embedQuery(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
}

export interface CachedEmbedProvider extends EmbedLike {
  /** Expose cache stats for health/debug tools */
  cacheStats(): ReturnType<EmbedCache["stats"]>;
  /** Clear the cache */
  cacheClear(): void;
}

/**
 * Wrap an embed provider (or queue) with query caching.
 * Document embeddings (embedBatch) are NOT cached — they're one-shot during indexing.
 */
export function withCache(inner: EmbedLike, cfg?: Partial<EmbedCacheConfig>): CachedEmbedProvider {
  const cache = new EmbedCache(cfg);

  return {
    async embedQuery(text: string): Promise<number[]> {
      const cached = cache.get(text);
      if (cached) return cached;

      const vector = await inner.embedQuery(text);
      cache.set(text, vector);
      return vector;
    },

    async embedBatch(texts: string[]): Promise<number[][]> {
      // Indexing — no cache, go straight to provider
      return inner.embedBatch(texts);
    },

    cacheStats() {
      return cache.stats();
    },

    cacheClear() {
      cache.clear();
    },
  };
}
