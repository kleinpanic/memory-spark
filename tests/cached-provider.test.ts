/**
 * Tests for src/embed/cached-provider.ts
 * Verifies caching behavior: hits, misses, disabled, TTL expiry, maxSize eviction.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { withCache, type EmbedLike } from "../src/embed/cached-provider.js";

/** Build a mock EmbedLike whose embedQuery returns [idx, idx, idx] per call */
function makeMockProvider(): EmbedLike & { callCount: number } {
  let callCount = 0;
  return {
    get callCount() {
      return callCount;
    },
    async embedQuery(text: string): Promise<number[]> {
      callCount++;
      return [callCount, callCount, callCount];
    },
    async embedDocument(text: string): Promise<number[]> {
      callCount++;
      return [callCount, callCount, callCount];
    },
    async embedBatch(texts: string[]): Promise<number[][]> {
      callCount++;
      return texts.map((_, i) => [callCount + i, callCount + i, callCount + i]);
    },
  };
}

describe("withCache — cache hit", () => {
  it("returns cached vector on second call without calling inner provider again", async () => {
    const inner = makeMockProvider();
    const provider = withCache(inner);

    const first = await provider.embedQuery("hello world");
    const second = await provider.embedQuery("hello world");

    expect(first).toEqual(second);
    expect(inner.callCount).toBe(1); // Only one real call
  });

  it("whitespace-normalizes cache key (leading/trailing/extra spaces)", async () => {
    const inner = makeMockProvider();
    const provider = withCache(inner);

    const a = await provider.embedQuery("  hello   world  ");
    const b = await provider.embedQuery("hello world");

    expect(a).toEqual(b);
    expect(inner.callCount).toBe(1);
  });

  it("case-normalizes cache key", async () => {
    const inner = makeMockProvider();
    const provider = withCache(inner);

    const a = await provider.embedQuery("Hello World");
    const b = await provider.embedQuery("hello world");

    expect(a).toEqual(b);
    expect(inner.callCount).toBe(1);
  });
});

describe("withCache — cache miss", () => {
  it("calls provider on first query and stores result", async () => {
    const inner = makeMockProvider();
    const provider = withCache(inner);

    const result = await provider.embedQuery("unique query");

    expect(result).toEqual([1, 1, 1]);
    expect(inner.callCount).toBe(1);
  });

  it("different queries get different cached results", async () => {
    const inner = makeMockProvider();
    const provider = withCache(inner);

    const a = await provider.embedQuery("query alpha");
    const b = await provider.embedQuery("query beta");

    expect(a).not.toEqual(b);
    expect(inner.callCount).toBe(2);
  });

  it("cacheStats reflects hits and misses", async () => {
    const inner = makeMockProvider();
    const provider = withCache(inner);

    await provider.embedQuery("x");
    await provider.embedQuery("x"); // hit
    await provider.embedQuery("y"); // miss

    const stats = provider.cacheStats();
    expect(stats.hits).toBe(1);
    expect(stats.misses).toBe(2);
    expect(stats.size).toBe(2);
  });
});

describe("withCache — cache disabled", () => {
  it("bypasses cache entirely when enabled=false", async () => {
    const inner = makeMockProvider();
    const provider = withCache(inner, { enabled: false });

    await provider.embedQuery("hello");
    await provider.embedQuery("hello");
    await provider.embedQuery("hello");

    expect(inner.callCount).toBe(3); // Every call goes to provider
  });

  it("cacheStats size remains 0 when disabled", async () => {
    const inner = makeMockProvider();
    const provider = withCache(inner, { enabled: false });

    await provider.embedQuery("test");

    expect(provider.cacheStats().size).toBe(0);
  });
});

describe("withCache — TTL expiry", () => {
  it("re-fetches from provider after TTL expires", async () => {
    vi.useFakeTimers();

    const inner = makeMockProvider();
    const provider = withCache(inner, { ttlMs: 1000 }); // 1 second TTL

    const first = await provider.embedQuery("query");
    expect(inner.callCount).toBe(1);

    // Advance time past TTL
    vi.advanceTimersByTime(1500);

    const second = await provider.embedQuery("query");
    expect(inner.callCount).toBe(2); // Must re-fetch
    expect(second).not.toEqual(first); // New vector from provider

    vi.useRealTimers();
  });

  it("does NOT re-fetch before TTL expires", async () => {
    vi.useFakeTimers();

    const inner = makeMockProvider();
    const provider = withCache(inner, { ttlMs: 5000 });

    await provider.embedQuery("stable query");
    vi.advanceTimersByTime(4000); // Still within TTL
    await provider.embedQuery("stable query");

    expect(inner.callCount).toBe(1);

    vi.useRealTimers();
  });
});

describe("withCache — maxSize eviction", () => {
  it("evicts LRU entry when cache is full", async () => {
    vi.useFakeTimers();

    const inner = makeMockProvider();
    const provider = withCache(inner, { maxSize: 2 });

    // Fill cache: A at t=0, B at t=1
    vi.setSystemTime(0);
    await provider.embedQuery("A");
    vi.setSystemTime(1);
    await provider.embedQuery("B");

    // Touch A at t=2 to make B the LRU
    vi.setSystemTime(2);
    await provider.embedQuery("A"); // cache hit, refreshes A's usedAt

    // Add C — should evict B (oldest usedAt)
    vi.setSystemTime(3);
    await provider.embedQuery("C");

    expect(provider.cacheStats().size).toBe(2);

    // B should now be a cache miss (evicted)
    const callsBefore = inner.callCount;
    await provider.embedQuery("B");
    expect(inner.callCount).toBe(callsBefore + 1);

    vi.useRealTimers();
  });

  it("does not grow beyond maxSize", async () => {
    const inner = makeMockProvider();
    const provider = withCache(inner, { maxSize: 3 });

    await provider.embedQuery("one");
    await provider.embedQuery("two");
    await provider.embedQuery("three");
    await provider.embedQuery("four"); // triggers eviction

    expect(provider.cacheStats().size).toBe(3);
  });
});

describe("withCache — embedDocument and embedBatch bypass cache", () => {
  it("embedDocument always calls inner provider", async () => {
    const inner = makeMockProvider();
    const provider = withCache(inner);

    await provider.embedDocument("doc text");
    await provider.embedDocument("doc text");

    expect(inner.callCount).toBe(2); // No caching for documents
  });

  it("embedBatch always calls inner provider", async () => {
    const inner = makeMockProvider();
    const provider = withCache(inner);

    await provider.embedBatch(["a", "b"]);
    await provider.embedBatch(["a", "b"]);

    expect(inner.callCount).toBe(2);
  });
});

describe("withCache — cacheClear", () => {
  it("clears cache so next call is a miss", async () => {
    const inner = makeMockProvider();
    const provider = withCache(inner);

    await provider.embedQuery("hello");
    provider.cacheClear();

    const statsBefore = inner.callCount;
    await provider.embedQuery("hello");

    expect(inner.callCount).toBe(statsBefore + 1);
    expect(provider.cacheStats().size).toBe(1);
  });

  it("resets hit/miss counters on clear", async () => {
    const inner = makeMockProvider();
    const provider = withCache(inner);

    await provider.embedQuery("a");
    await provider.embedQuery("a"); // hit
    provider.cacheClear();

    const stats = provider.cacheStats();
    expect(stats.hits).toBe(0);
    expect(stats.misses).toBe(0);
  });
});
