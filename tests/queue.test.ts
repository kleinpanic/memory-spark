import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { EmbedQueue } from "../src/embed/queue.js";
import type { EmbedProvider } from "../src/embed/provider.js";

// ---------------------------------------------------------------------------
// Mock provider factory
// ---------------------------------------------------------------------------

function makeProvider(overrides: Partial<EmbedProvider> = {}): EmbedProvider {
  return {
    id: "mock",
    model: "mock-model",
    dims: 128,
    embedQuery: vi.fn().mockResolvedValue([0.1, 0.2, 0.3]),
    embedDocument: vi.fn().mockResolvedValue([0.4, 0.5, 0.6]),
    embedBatch: vi.fn().mockResolvedValue([
      [0.1, 0.2],
      [0.3, 0.4],
    ]),
    probe: vi.fn().mockResolvedValue(true),
    ...overrides,
  };
}

function makeLogger() {
  return {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Flush all timers and pending microtasks in a loop until stable. */
async function flushAll() {
  for (let i = 0; i < 20; i++) {
    await Promise.resolve();
  }
}

// ---------------------------------------------------------------------------
// 1. Basic routing — embedQuery vs embedDocument
// ---------------------------------------------------------------------------

describe("EmbedQueue — routing", () => {
  let provider: EmbedProvider;
  let queue: EmbedQueue;

  beforeEach(() => {
    provider = makeProvider();
    queue = new EmbedQueue(provider, { maxRetries: 0 });
  });

  it("embedQuery delegates to provider.embedQuery", async () => {
    const result = await queue.embedQuery("hello world");
    expect(provider.embedQuery).toHaveBeenCalledWith("hello world");
    expect(provider.embedDocument).not.toHaveBeenCalled();
    expect(result).toEqual([0.1, 0.2, 0.3]);
  });

  it("embedDocument delegates to provider.embedDocument", async () => {
    const result = await queue.embedDocument("some document");
    expect(provider.embedDocument).toHaveBeenCalledWith("some document");
    expect(provider.embedQuery).not.toHaveBeenCalled();
    expect(result).toEqual([0.4, 0.5, 0.6]);
  });

  it("exposes provider metadata (model, dims, id)", () => {
    expect(queue.model).toBe("mock-model");
    expect(queue.dims).toBe(128);
    expect(queue.id).toBe("mock");
  });
});

// ---------------------------------------------------------------------------
// 2. Serialization — concurrency=1 means sequential
// ---------------------------------------------------------------------------

describe("EmbedQueue — serialization (concurrency=1)", () => {
  it("processes items sequentially, not in parallel", async () => {
    const order: string[] = [];
    let resolveFirst!: () => void;

    const provider = makeProvider({
      embedQuery: vi.fn().mockImplementation((text: string) => {
        order.push(`start:${text}`);
        return new Promise<number[]>((resolve) => {
          if (text === "first") {
            resolveFirst = () => {
              order.push(`end:${text}`);
              resolve([1]);
            };
          } else {
            order.push(`end:${text}`);
            resolve([2]);
          }
        });
      }),
    });

    const queue = new EmbedQueue(provider, { concurrency: 1, maxRetries: 0 });

    const p1 = queue.embedQuery("first");
    const p2 = queue.embedQuery("second");

    // At this point, "first" is started but not resolved; "second" is waiting
    await flushAll();
    expect(order).toContain("start:first");
    expect(order).not.toContain("start:second");

    // Resolve first
    resolveFirst();
    await p1;
    await p2;

    expect(order).toEqual(["start:first", "end:first", "start:second", "end:second"]);
  });

  it("tracks active/queued counts correctly", async () => {
    let resolveEmbed!: () => void;
    const provider = makeProvider({
      embedQuery: vi.fn().mockImplementation(
        () =>
          new Promise<number[]>((resolve) => {
            resolveEmbed = () => resolve([1]);
          }),
      ),
    });

    const queue = new EmbedQueue(provider, { concurrency: 1, maxRetries: 0 });

    const p1 = queue.embedQuery("a");
    const p2 = queue.embedQuery("b");
    await flushAll();

    expect(queue.stats.active).toBe(1);
    expect(queue.stats.queued).toBe(1);

    resolveEmbed();
    await flushAll();

    // p1 done; now b is active
    expect(queue.stats.queued).toBe(0);

    resolveEmbed();
    await Promise.all([p1, p2]);

    expect(queue.stats.active).toBe(0);
    expect(queue.stats.processed).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// 3. Retry logic with exponential backoff
// ---------------------------------------------------------------------------

describe("EmbedQueue — retry / backoff", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("retries up to maxRetries times on failure", async () => {
    const provider = makeProvider({
      embedQuery: vi
        .fn()
        .mockRejectedValueOnce(new Error("fail1"))
        .mockRejectedValueOnce(new Error("fail2"))
        .mockRejectedValueOnce(new Error("fail3"))
        .mockResolvedValue([1, 2, 3]),
    });

    const queue = new EmbedQueue(provider, {
      maxRetries: 3,
      baseDelayMs: 100,
      maxDelayMs: 10000,
      timeoutMs: 60000,
      // Keep circuit threshold high so it doesn't open
      unhealthyThreshold: 99,
    });

    const promise = queue.embedQuery("test");

    // Advance timers for each retry delay
    await vi.runAllTimersAsync();

    const result = await promise;
    expect(result).toEqual([1, 2, 3]);
    expect(provider.embedQuery).toHaveBeenCalledTimes(4); // 1 original + 3 retries
  });

  it("uses exponential backoff: delay doubles each retry", async () => {
    // Track calls in order to verify backoff
    const callTimestamps: number[] = [];
    let fakeNow = 0;
    vi.setSystemTime(fakeNow);

    const provider = makeProvider({
      embedQuery: vi.fn().mockImplementation(async () => {
        callTimestamps.push(fakeNow);
        if (callTimestamps.length <= 2) throw new Error("fail");
        return [1];
      }),
    });

    const queue = new EmbedQueue(provider, {
      maxRetries: 3,
      baseDelayMs: 2000,
      maxDelayMs: 30000,
      timeoutMs: 120_000,
      unhealthyThreshold: 99,
    });

    const promise = queue.embedQuery("test");
    promise.catch(() => {});

    // Tick forward to trigger first retry (2000ms)
    fakeNow = 2000;
    vi.setSystemTime(fakeNow);
    await vi.advanceTimersByTimeAsync(2000);

    // Tick forward to trigger second retry (4000ms more)
    fakeNow = 6000;
    vi.setSystemTime(fakeNow);
    await vi.advanceTimersByTimeAsync(4000);

    await promise;

    // call 0 is immediate; gap to call 1 = ~2000; gap to call 2 = ~4000
    expect(callTimestamps.length).toBe(3);
    const gap1 = callTimestamps[1]! - callTimestamps[0]!;
    const gap2 = callTimestamps[2]! - callTimestamps[1]!;
    expect(gap1).toBe(2000);
    expect(gap2).toBe(4000);
  });

  it("rejects after retries exhausted", async () => {
    const provider = makeProvider({
      embedQuery: vi.fn().mockRejectedValue(new Error("always fails")),
    });

    const queue = new EmbedQueue(provider, {
      maxRetries: 2,
      baseDelayMs: 100,
      unhealthyThreshold: 99,
    });

    const promise = queue.embedQuery("test");
    promise.catch(() => {}); // prevent unhandled rejection while timers run
    await vi.runAllTimersAsync();

    await expect(promise).rejects.toThrow("always fails");
    expect(provider.embedQuery).toHaveBeenCalledTimes(3); // 1 + 2 retries
  });
});

// ---------------------------------------------------------------------------
// 4. Circuit breaker — threshold triggers OPEN
// ---------------------------------------------------------------------------

describe("EmbedQueue — circuit breaker (threshold)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("opens circuit after unhealthyThreshold consecutive failures", async () => {
    const provider = makeProvider({
      embedQuery: vi.fn().mockRejectedValue(new Error("down")),
    });

    const logger = makeLogger();
    const queue = new EmbedQueue(
      provider,
      {
        maxRetries: 0,
        unhealthyThreshold: 3,
        unhealthyCooldownMs: 60_000,
      },
      logger,
    );

    // Fire 3 requests — each fails once (maxRetries=0, no retry)
    // Attach .catch to prevent unhandled rejection when circuit opens mid-batch
    const promises = [
      queue.embedQuery("a").catch(() => {}),
      queue.embedQuery("b").catch(() => {}),
      queue.embedQuery("c").catch(() => {}),
    ];

    await vi.runAllTimersAsync();
    await Promise.allSettled(promises);

    // On the 3rd failure, circuit should open
    expect(queue.stats.circuitState).toBe("OPEN");
    expect(queue.stats.consecutiveFailures).toBeGreaterThanOrEqual(3);
  });

  it("rejects immediately with 'Circuit breaker OPEN' when circuit is open", async () => {
    const provider = makeProvider({
      embedQuery: vi.fn().mockRejectedValue(new Error("server down")),
    });

    const queue = new EmbedQueue(provider, {
      maxRetries: 0,
      unhealthyThreshold: 2,
      unhealthyCooldownMs: 60_000,
    });

    // Trip the circuit
    await Promise.allSettled([queue.embedQuery("x"), queue.embedQuery("y")]);
    await vi.runAllTimersAsync();

    expect(queue.stats.circuitState).toBe("OPEN");

    // Next request should be rejected with circuit-open error
    const after = queue.embedQuery("z");
    after.catch(() => {}); // prevent unhandled rejection while timers run
    await vi.runAllTimersAsync();

    await expect(after).rejects.toThrow(/Circuit breaker OPEN/);
  });

  it("resets consecutive failures on success", async () => {
    const provider = makeProvider({
      embedQuery: vi.fn().mockRejectedValueOnce(new Error("fail")).mockResolvedValue([1, 2]),
    });

    const queue = new EmbedQueue(provider, {
      maxRetries: 1,
      baseDelayMs: 100,
      unhealthyThreshold: 5,
    });

    const promise = queue.embedQuery("test");
    await vi.runAllTimersAsync();
    const result = await promise;
    expect(result).toEqual([1, 2]);
    expect(queue.stats.consecutiveFailures).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// 5. Circuit breaker — cooldown and HALF_OPEN → CLOSED recovery
// ---------------------------------------------------------------------------

describe("EmbedQueue — circuit breaker (cooldown / recovery)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("transitions OPEN → HALF_OPEN after cooldown period", async () => {
    const provider = makeProvider({
      embedQuery: vi.fn().mockRejectedValue(new Error("down")),
    });

    const cooldownMs = 5_000;
    const queue = new EmbedQueue(provider, {
      maxRetries: 0,
      unhealthyThreshold: 2,
      unhealthyCooldownMs: cooldownMs,
    });

    await Promise.allSettled([queue.embedQuery("a"), queue.embedQuery("b")]);
    await vi.runAllTimersAsync();

    expect(queue.stats.circuitState).toBe("OPEN");

    // Advance past cooldown
    vi.advanceTimersByTime(cooldownMs + 1);

    // isHealthy() checks elapsed time and transitions
    expect(queue.isHealthy()).toBe(true);
    expect(queue.stats.circuitState).toBe("HALF_OPEN");
  });

  it("closes circuit on success in HALF_OPEN state", async () => {
    const provider = makeProvider({
      embedQuery: vi
        .fn()
        .mockRejectedValueOnce(new Error("fail"))
        .mockRejectedValueOnce(new Error("fail"))
        .mockResolvedValue([9, 9, 9]),
    });

    const logger = makeLogger();
    const cooldownMs = 1_000;
    const queue = new EmbedQueue(
      provider,
      {
        maxRetries: 0,
        unhealthyThreshold: 2,
        unhealthyCooldownMs: cooldownMs,
      },
      logger,
    );

    // Trip the circuit
    await Promise.allSettled([queue.embedQuery("a"), queue.embedQuery("b")]);
    await vi.runAllTimersAsync();

    expect(queue.stats.circuitState).toBe("OPEN");

    // Advance past cooldown → HALF_OPEN
    vi.advanceTimersByTime(cooldownMs + 100);

    // Now send a request that succeeds → should close circuit
    const promise = queue.embedQuery("recovery");
    await vi.runAllTimersAsync();
    const result = await promise;

    expect(result).toEqual([9, 9, 9]);
    expect(queue.stats.circuitState).toBe("CLOSED");
    expect(logger.info).toHaveBeenCalledWith(expect.stringContaining("CIRCUIT CLOSED"));
  });

  it("re-opens circuit on failure in HALF_OPEN state with doubled timeout", async () => {
    const provider = makeProvider({
      embedQuery: vi.fn().mockRejectedValue(new Error("still down")),
    });

    const cooldownMs = 2_000;
    const queue = new EmbedQueue(provider, {
      maxRetries: 0,
      unhealthyThreshold: 2,
      unhealthyCooldownMs: cooldownMs,
      circuitBreaker: { multiplier: 2 },
    });

    // Trip the circuit
    await Promise.allSettled([queue.embedQuery("a"), queue.embedQuery("b")]);
    await vi.runAllTimersAsync();
    expect(queue.stats.circuitState).toBe("OPEN");

    // Advance past cooldown → HALF_OPEN
    vi.advanceTimersByTime(cooldownMs + 100);
    expect(queue.isHealthy()).toBe(true); // transitions to HALF_OPEN

    // Fail in HALF_OPEN → back to OPEN with doubled timeout
    await Promise.allSettled([queue.embedQuery("test")]);
    await vi.runAllTimersAsync();

    expect(queue.stats.circuitState).toBe("OPEN");
    // nextRetryMs should be approximately doubled cooldown (capped at maxResetMs)
    expect(queue.stats.nextRetryMs).toBeGreaterThan(cooldownMs);
  });

  it("fires recovery callbacks on circuit close", async () => {
    const provider = makeProvider({
      embedQuery: vi
        .fn()
        .mockRejectedValueOnce(new Error("fail"))
        .mockRejectedValueOnce(new Error("fail"))
        .mockResolvedValue([1]),
    });

    const queue = new EmbedQueue(provider, {
      maxRetries: 0,
      unhealthyThreshold: 2,
      unhealthyCooldownMs: 500,
    });

    const recoveryFn = vi.fn();
    queue.onRecovery(recoveryFn);

    const settled = Promise.allSettled([queue.embedQuery("a"), queue.embedQuery("b")]);
    await vi.runAllTimersAsync();
    await settled;

    expect(queue.stats.circuitState).toBe("OPEN");

    // Advance past cooldown → HALF_OPEN
    vi.advanceTimersByTime(600);

    const p = queue.embedQuery("recover");
    await vi.runAllTimersAsync();
    await p;

    // _fireRecoveryCallbacks uses setImmediate; flush it with a microtask loop
    for (let i = 0; i < 10; i++) await Promise.resolve();
    expect(recoveryFn).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// 6. Timeout handling
// ---------------------------------------------------------------------------

describe("EmbedQueue — timeout", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("rejects with timeout error when provider hangs", async () => {
    const provider = makeProvider({
      embedQuery: vi.fn().mockImplementation(
        () =>
          new Promise(() => {
            /* never resolves */
          }),
      ),
    });

    const queue = new EmbedQueue(provider, {
      maxRetries: 0,
      timeoutMs: 3000,
      unhealthyThreshold: 99,
    });

    const promise = queue.embedQuery("slow");

    // Advance past the timeout
    vi.advanceTimersByTime(3001);
    await flushAll();

    await expect(promise).rejects.toThrow("Embed timeout (3000ms)");
  });

  it("does not time out when provider responds within timeoutMs", async () => {
    let resolve!: (v: number[]) => void;
    const provider = makeProvider({
      embedQuery: vi.fn().mockImplementation(
        () =>
          new Promise<number[]>((res) => {
            resolve = res;
          }),
      ),
    });

    const queue = new EmbedQueue(provider, {
      maxRetries: 0,
      timeoutMs: 5000,
    });

    const promise = queue.embedQuery("fast enough");
    vi.advanceTimersByTime(4000);
    resolve([7, 8, 9]);
    await flushAll();

    await expect(promise).resolves.toEqual([7, 8, 9]);
  });
});

// ---------------------------------------------------------------------------
// 7. Error propagation when retries exhausted
// ---------------------------------------------------------------------------

describe("EmbedQueue — error propagation", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("propagates the original Error type after exhaustion", async () => {
    const originalError = new Error("network gone");
    const provider = makeProvider({
      embedQuery: vi.fn().mockRejectedValue(originalError),
    });

    const queue = new EmbedQueue(provider, {
      maxRetries: 1,
      baseDelayMs: 100,
      unhealthyThreshold: 99,
    });

    const p = queue.embedQuery("fail");
    p.catch(() => {}); // prevent unhandled rejection while timers run
    await vi.runAllTimersAsync();

    await expect(p).rejects.toThrow("network gone");
  });

  it("wraps non-Error rejections into Error objects", async () => {
    const provider = makeProvider({
      embedQuery: vi.fn().mockRejectedValue("string error"),
    });

    const queue = new EmbedQueue(provider, {
      maxRetries: 0,
      unhealthyThreshold: 99,
    });

    const p = queue.embedQuery("fail");
    p.catch(() => {}); // prevent unhandled rejection while timers run
    await vi.runAllTimersAsync();

    await expect(p).rejects.toBeInstanceOf(Error);
  });

  it("does not retry on fatal HTTP 401", async () => {
    const err = Object.assign(new Error("Unauthorized"), { httpStatus: 401 });
    const provider = makeProvider({
      embedQuery: vi.fn().mockRejectedValue(err),
    });

    const queue = new EmbedQueue(provider, { maxRetries: 3, unhealthyThreshold: 99 });

    const p = queue.embedQuery("test");
    p.catch(() => {}); // prevent unhandled rejection while timers run
    await vi.runAllTimersAsync();

    await expect(p).rejects.toThrow("Unauthorized");
    // Should only be called once — no retries on 401
    expect(provider.embedQuery).toHaveBeenCalledTimes(1);
  });

  it("does not retry on fatal HTTP 403", async () => {
    const err = Object.assign(new Error("Forbidden"), { httpStatus: 403 });
    const provider = makeProvider({
      embedQuery: vi.fn().mockRejectedValue(err),
    });

    const queue = new EmbedQueue(provider, { maxRetries: 3, unhealthyThreshold: 99 });

    const p = queue.embedQuery("test");
    p.catch(() => {}); // prevent unhandled rejection while timers run
    await vi.runAllTimersAsync();

    await expect(p).rejects.toThrow("Forbidden");
    expect(provider.embedQuery).toHaveBeenCalledTimes(1);
  });

  it("increments failed stat counter after exhaustion", async () => {
    const provider = makeProvider({
      embedQuery: vi.fn().mockRejectedValue(new Error("fail")),
    });

    const queue = new EmbedQueue(provider, {
      maxRetries: 1,
      baseDelayMs: 50,
      unhealthyThreshold: 99,
    });

    const p = queue.embedQuery("a");
    p.catch(() => {});
    await vi.runAllTimersAsync();
    await Promise.allSettled([p]);

    expect(queue.stats.failed).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// 8. Queue drain behavior
// ---------------------------------------------------------------------------

describe("EmbedQueue — drain", () => {
  it("rejects all queued items on drain()", async () => {
    let activeResolve!: () => void;
    const provider = makeProvider({
      embedQuery: vi.fn().mockImplementation((text: string) => {
        if (text === "active") {
          return new Promise<number[]>((res) => {
            activeResolve = () => res([1]);
          });
        }
        return new Promise<number[]>(() => {
          /* queued, never resolves */
        });
      }),
    });

    const queue = new EmbedQueue(provider, { concurrency: 1, maxRetries: 0 });

    // First one starts (active), rest queue up
    const p1 = queue.embedQuery("active");
    const p2 = queue.embedQuery("waiting1");
    const p3 = queue.embedQuery("waiting2");

    await flushAll();

    // Drain — should reject p2 and p3
    queue.drain();

    await expect(p2).rejects.toThrow("EmbedQueue drained (shutdown)");
    await expect(p3).rejects.toThrow("EmbedQueue drained (shutdown)");

    // Queue should now be empty
    expect(queue.stats.queued).toBe(0);

    // Resolve active item — should still work
    activeResolve();
    await expect(p1).resolves.toEqual([1]);
  });

  it("drain on empty queue is a no-op", () => {
    const queue = new EmbedQueue(makeProvider(), { maxRetries: 0 });
    expect(() => queue.drain()).not.toThrow();
    expect(queue.stats.queued).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// 9. embedBatch — delegation and chunking
// ---------------------------------------------------------------------------

describe("EmbedQueue — embedBatch", () => {
  it("returns empty array for empty input", async () => {
    const provider = makeProvider();
    const queue = new EmbedQueue(provider);
    const result = await queue.embedBatch([]);
    expect(result).toEqual([]);
    expect(provider.embedBatch).not.toHaveBeenCalled();
  });

  it("delegates to provider.embedBatch for a small batch", async () => {
    const vectors = [
      [1, 2],
      [3, 4],
      [5, 6],
    ];
    const provider = makeProvider({
      embedBatch: vi.fn().mockResolvedValue(vectors),
    });

    const queue = new EmbedQueue(provider, { maxRetries: 0 });
    const result = await queue.embedBatch(["a", "b", "c"]);

    expect(provider.embedBatch).toHaveBeenCalledWith(["a", "b", "c"]);
    expect(result).toEqual(vectors);
  });

  it("chunks large batches into groups of 8", async () => {
    // 20 texts → ceil(20/8) = 3 chunks: [8, 8, 4]
    const texts = Array.from({ length: 20 }, (_, i) => `text${i}`);
    const provider = makeProvider({
      embedBatch: vi
        .fn()
        .mockImplementation((batch: string[]) => Promise.resolve(batch.map(() => [0.1, 0.2]))),
    });

    const queue = new EmbedQueue(provider, { maxRetries: 0 });
    const results = await queue.embedBatch(texts);

    expect(provider.embedBatch).toHaveBeenCalledTimes(3);
    expect(provider.embedBatch).toHaveBeenNthCalledWith(1, texts.slice(0, 8));
    expect(provider.embedBatch).toHaveBeenNthCalledWith(2, texts.slice(8, 16));
    expect(provider.embedBatch).toHaveBeenNthCalledWith(3, texts.slice(16, 20));

    expect(results).toHaveLength(20);
    expect(results.every((v) => Array.isArray(v))).toBe(true);
  });

  it("exactly 8 texts uses a single batch call", async () => {
    const texts = Array.from({ length: 8 }, (_, i) => `t${i}`);
    const provider = makeProvider({
      embedBatch: vi.fn().mockResolvedValue(texts.map(() => [1])),
    });

    const queue = new EmbedQueue(provider, { maxRetries: 0 });
    await queue.embedBatch(texts);

    expect(provider.embedBatch).toHaveBeenCalledTimes(1);
    expect(provider.embedBatch).toHaveBeenCalledWith(texts);
  });
});

// ---------------------------------------------------------------------------
// 10. Stats and failed item tracking
// ---------------------------------------------------------------------------

describe("EmbedQueue — stats / failed item tracking", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("tracks processed count correctly", async () => {
    const provider = makeProvider();
    const queue = new EmbedQueue(provider, { maxRetries: 0 });

    await queue.embedQuery("a");
    await queue.embedQuery("b");
    await queue.embedDocument("c");

    expect(queue.stats.processed).toBe(3);
  });

  it("tracks failed items map when trackFailedItems=true", async () => {
    const provider = makeProvider({
      embedQuery: vi.fn().mockRejectedValue(new Error("boom")),
    });

    const queue = new EmbedQueue(provider, {
      maxRetries: 0,
      trackFailedItems: true,
      unhealthyThreshold: 99,
    });

    await Promise.allSettled([queue.embedQuery("failing text")]);
    await vi.runAllTimersAsync();

    const failed = queue.getFailedItems();
    expect(failed).toHaveLength(1);
    expect(failed[0]!.input).toBe("failing text");
    expect(failed[0]!.lastError).toBe("boom");
  });

  it("clearFailedItems empties the failed map", async () => {
    const provider = makeProvider({
      embedQuery: vi.fn().mockRejectedValue(new Error("fail")),
    });

    const queue = new EmbedQueue(provider, {
      maxRetries: 0,
      trackFailedItems: true,
      unhealthyThreshold: 99,
    });

    await Promise.allSettled([queue.embedQuery("x")]);
    await vi.runAllTimersAsync();

    expect(queue.getFailedItems()).toHaveLength(1);
    queue.clearFailedItems();
    expect(queue.getFailedItems()).toHaveLength(0);
  });

  it("circuitBreakerState returns correct state and nextRetryMs", async () => {
    const provider = makeProvider({
      embedQuery: vi.fn().mockRejectedValue(new Error("fail")),
    });

    const queue = new EmbedQueue(provider, {
      maxRetries: 0,
      unhealthyThreshold: 1,
      unhealthyCooldownMs: 10_000,
    });

    // Closed initially
    expect(queue.circuitBreakerState.state).toBe("CLOSED");
    expect(queue.circuitBreakerState.nextRetryMs).toBeNull();

    await Promise.allSettled([queue.embedQuery("a")]);
    await vi.runAllTimersAsync();

    // Now open
    expect(queue.circuitBreakerState.state).toBe("OPEN");
    expect(queue.circuitBreakerState.nextRetryMs).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// 11. Backwards compat — deprecated unhealthyThreshold / unhealthyCooldownMs
// ---------------------------------------------------------------------------

describe("EmbedQueue — backwards compat config fields", () => {
  it("unhealthyThreshold maps to circuitBreaker.failureThreshold", () => {
    const provider = makeProvider();
    const queue = new EmbedQueue(provider, {
      unhealthyThreshold: 7,
    });
    // Access via stats which reflect internal state
    expect(queue.stats.circuitState).toBe("CLOSED");
    // No direct way to read threshold from outside; verify it doesn't crash
    // and that the queue behaves with the threshold set
    expect(queue.isHealthy()).toBe(true);
  });

  it("unhealthyCooldownMs maps to circuitBreaker.initialResetMs", async () => {
    vi.useFakeTimers();

    const provider = makeProvider({
      embedQuery: vi.fn().mockRejectedValue(new Error("down")),
    });

    const queue = new EmbedQueue(provider, {
      maxRetries: 0,
      unhealthyThreshold: 1,
      unhealthyCooldownMs: 3_000,
    });

    await Promise.allSettled([queue.embedQuery("a")]);
    await vi.runAllTimersAsync();

    expect(queue.stats.circuitState).toBe("OPEN");

    vi.advanceTimersByTime(3_001);
    // Should now be HALF_OPEN after cooldown
    expect(queue.isHealthy()).toBe(true);
    expect(queue.stats.circuitState).toBe("HALF_OPEN");

    vi.useRealTimers();
  });
});
