import { EmbedQueue } from "../src/embed/queue.js";
import type { EmbedProvider } from "../src/embed/provider.js";

// Mock embed provider that fails N times then succeeds
function createMockProvider(failCount: number): EmbedProvider {
  let calls = 0;
  let currentFailures = 0;
  return {
    id: "mock",
    model: "mock-model",
    dims: 384,
    embedDocument: async (text: string) => {
      if (currentFailures < failCount) {
        currentFailures++;
        throw new Error("Simulated failure");
      }
      return Array(1024).fill(0.1);
    },
    embedQuery: async (text: string) => {
      calls++;
      if (calls <= failCount) {
        throw new Error(`Mock failure ${calls}`);
      }
      return new Array(384).fill(0);
    },
    embedBatch: async (texts: string[]) => {
      calls++;
      if (calls <= failCount) {
        throw new Error(`Mock failure ${calls}`);
      }
      return texts.map(() => new Array(384).fill(0));
    },
    probe: async () => true, // Mock probe always succeeds
  };
}

const logger = {
  info: (m: string) => console.log(`  [INFO] ${m}`),
  warn: (m: string) => console.log(`  [WARN] ${m}`),
  error: (m: string) => console.log(`  [ERR]  ${m}`),
};

async function test(name: string, fn: () => Promise<void>) {
  console.log(`\n=== ${name} ===`);
  try {
    await fn();
    console.log(`  ✅ PASS`);
  } catch (err) {
    console.log(`  ❌ FAIL: ${err}`);
    process.exitCode = 1;
  }
}

async function main() {
  // Test 1: CLOSED → OPEN transition on threshold
  await test("1. CLOSED → OPEN on threshold", async () => {
    const provider = createMockProvider(10); // Always fail
    const queue = new EmbedQueue(
      provider,
      {
        maxRetries: 0,
        circuitBreaker: { failureThreshold: 3, initialResetMs: 1000, maxResetMs: 60000, multiplier: 2 },
      },
      logger
    );

    // Trigger failures
    for (let i = 0; i < 5; i++) {
      try {
        await queue.embedQuery(`test ${i}`);
      } catch {}
    }

    const state = queue.circuitBreakerState;
    if (state.state !== "OPEN") {
      throw new Error(`Expected OPEN, got ${state.state}`);
    }
    console.log(`    State: ${state.state}, next retry: ${state.nextRetryMs}ms`);
  });

  // Test 2: Failed item tracking
  await test("2. Failed item tracking", async () => {
    const provider = createMockProvider(10);
    const queue = new EmbedQueue(
      provider,
      { maxRetries: 0, circuitBreaker: { failureThreshold: 3, initialResetMs: 1000, maxResetMs: 60000, multiplier: 2 }, trackFailedItems: true },
      logger
    );

    // Trigger failures
    try { await queue.embedQuery("test input"); } catch {}

    const failed = queue.getFailedItems();
    if (failed.length === 0) {
      throw new Error("Expected failed items");
    }
    console.log(`    Failed items: ${failed.length}`);
    console.log(`    First: ${failed[0]?.input}`);
  });

  // Test 3: Stats include circuit state
  await test("3. Stats include circuit state", async () => {
    const provider = createMockProvider(10);
    const queue = new EmbedQueue(
      provider,
      { maxRetries: 0, circuitBreaker: { failureThreshold: 3, initialResetMs: 1000, maxResetMs: 60000, multiplier: 2 } },
      logger
    );

    // Trigger failures to open circuit
    for (let i = 0; i < 5; i++) {
      try { await queue.embedQuery(`test ${i}`); } catch {}
    }

    const stats = queue.stats;
    if (stats.circuitState !== "OPEN") {
      throw new Error(`Expected OPEN in stats, got ${stats.circuitState}`);
    }
    console.log(`    Stats: circuitState=${stats.circuitState}, failedItemsCount=${stats.failedItemsCount}`);
  });

  console.log("\n✅ All circuit breaker tests passed!");
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
