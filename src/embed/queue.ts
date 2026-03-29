/**
 * EmbedQueue — serialized embed requests with retry, backoff, and circuit breaker.
 *
 * Prevents hammering the embed server. All embed calls go through this queue.
 * Single-concurrency by default (Spark embed is single-worker uvicorn).
 * Retries with exponential backoff on failure.
 * Circuit breaker pattern: CLOSED → OPEN → HALF_OPEN → CLOSED
 * Tracks failed items for retry on recovery.
 */

import type { EmbedProvider } from "./provider.js";

export interface QueueLogger {
  info: (m: string) => void;
  warn: (m: string) => void;
  error: (m: string) => void;
}

/** Circuit breaker states */
export type CircuitState = "CLOSED" | "OPEN" | "HALF_OPEN";

/** Circuit breaker configuration */
export interface CircuitBreakerConfig {
  /** Failures before opening circuit (default: 5) */
  failureThreshold: number;
  /** Initial reset timeout in ms (default: 5 min) */
  initialResetMs: number;
  /** Max reset timeout cap in ms (default: 30 min) */
  maxResetMs: number;
  /** Multiplier for backoff (default: 2) */
  multiplier: number;
}

/** Failed item for retry tracking */
export interface FailedItem {
  /** Text or batch that failed */
  input: string | string[];
  /** When it failed */
  failedAt: number;
  /** How many times retried */
  retries: number;
  /** Last error message */
  lastError: string;
}

export interface EmbedQueueConfig {
  /** Max concurrent embed requests (default: 1 for single-worker servers) */
  concurrency?: number;
  /** Max retries per request before giving up */
  maxRetries?: number;
  /** Base delay for exponential backoff in ms (default: 2000) */
  baseDelayMs?: number;
  /** Max delay cap in ms (default: 30000) */
  maxDelayMs?: number;
  /** Request timeout in ms (default: 30000) */
  timeoutMs?: number;
  /** @deprecated Use circuitBreaker.failureThreshold instead */
  unhealthyThreshold?: number;
  /** @deprecated Use circuitBreaker.initialResetMs instead */
  unhealthyCooldownMs?: number;
  /** Circuit breaker configuration */
  circuitBreaker?: Partial<CircuitBreakerConfig>;
  /** Enable failed item tracking for retry on recovery */
  trackFailedItems?: boolean;
}

interface QueueItem<T> {
  fn: () => Promise<T>;
  resolve: (v: T) => void;
  reject: (e: Error) => void;
  retries: number;
  /** Input text(s) for failed item tracking */
  input?: string | string[];
}

const DEFAULT_CIRCUIT_BREAKER: CircuitBreakerConfig = {
  failureThreshold: 5,
  initialResetMs: 5 * 60_000,  // 5 min
  maxResetMs: 30 * 60_000,     // 30 min
  multiplier: 2,
};

export class EmbedQueue {
  private provider: EmbedProvider;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- heterogeneous queue items
  private queue: Array<QueueItem<any>> = [];
  private active = 0;
  private concurrency: number;
  private maxRetries: number;
  private baseDelayMs: number;
  private maxDelayMs: number;
  private timeoutMs: number;
  private logger: QueueLogger;
  private _processed = 0;
  private _failed = 0;

  // Circuit breaker state
  private circuitState: CircuitState = "CLOSED";
  private circuitOpenedAt: number | null = null;
  private currentResetMs: number;
  private circuitBreaker: CircuitBreakerConfig;
  private consecutiveFailures = 0;

  // Failed item tracking
  private trackFailedItems: boolean;
  private failedItems: Map<string, FailedItem> = new Map();

  // Recovery callbacks
  private _wasUnhealthy = false;
  private _recoveryCallbacks: Array<() => void> = [];

  constructor(provider: EmbedProvider, cfg?: EmbedQueueConfig, logger?: QueueLogger) {
    this.provider = provider;
    this.concurrency = cfg?.concurrency ?? 1;
    this.maxRetries = cfg?.maxRetries ?? 3;
    this.baseDelayMs = cfg?.baseDelayMs ?? 2000;
    this.maxDelayMs = cfg?.maxDelayMs ?? 30000;
    this.timeoutMs = cfg?.timeoutMs ?? 30000;
    this.logger = logger ?? { info: console.log, warn: console.warn, error: console.error };

    // Circuit breaker config (with backwards compat for old fields)
    this.circuitBreaker = {
      ...DEFAULT_CIRCUIT_BREAKER,
      ...cfg?.circuitBreaker,
    };
    // Backwards compat: if old fields are set, use them
    if (cfg?.unhealthyThreshold) {
      this.circuitBreaker.failureThreshold = cfg.unhealthyThreshold;
    }
    if (cfg?.unhealthyCooldownMs) {
      this.circuitBreaker.initialResetMs = cfg.unhealthyCooldownMs;
    }

    this.currentResetMs = this.circuitBreaker.initialResetMs;
    this.trackFailedItems = cfg?.trackFailedItems ?? true;
  }

  /** Embed a single query (for search). Goes through queue. */
  async embedQuery(text: string): Promise<number[]> {
    return this.enqueue(
      async () => {
        const result = await this.withTimeout(this.provider.embedQuery(text));
        return result;
      },
      text
    );
  }

  /** Embed a batch of texts (for indexing). Serialized through queue one batch at a time. */
  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    // Break into small batches to avoid overwhelming the server
    const BATCH_SIZE = 8;
    const allResults: number[][] = [];
    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batch = texts.slice(i, i + BATCH_SIZE);
      const results = await this.enqueue(
        async () => {
          return this.withTimeout(this.provider.embedBatch(batch));
        },
        batch
      );
      allResults.push(...results);
    }
    return allResults;
  }

  /** Get queue stats */
  get stats() {
    return {
      queued: this.queue.length,
      active: this.active,
      processed: this._processed,
      failed: this._failed,
      circuitState: this.circuitState,
      consecutiveFailures: this.consecutiveFailures,
      failedItemsCount: this.failedItems.size,
      nextRetryMs: this.getNextRetryMs(),
    };
  }

  /** Get circuit breaker state */
  get circuitBreakerState(): { state: CircuitState; nextRetryMs: number | null } {
    const nextRetry = this.getNextRetryMs();
    return {
      state: this.circuitState,
      nextRetryMs: nextRetry > 0 ? nextRetry : null,
    };
  }

  /** Get failed items for external retry */
  getFailedItems(): FailedItem[] {
    return [...this.failedItems.values()];
  }

  /** Clear failed items (after manual retry) */
  clearFailedItems(): void {
    this.failedItems.clear();
  }

  /** Check if the embed service is considered healthy */
  isHealthy(): boolean {
    switch (this.circuitState) {
      case "CLOSED":
        return true;
      case "OPEN":
        // Check if reset timeout has elapsed
        if (Date.now() - (this.circuitOpenedAt ?? 0) >= this.currentResetMs) {
          this.circuitState = "HALF_OPEN";
          this.logger.info("memory-spark queue: [CIRCUIT HALF_OPEN] Testing connection...");
          return true;
        }
        return false;
      case "HALF_OPEN":
        return true; // Allow one test request
    }
  }

  /** The embed provider's model/dims info */
  get model() {
    return this.provider.model;
  }
  get dims() {
    return this.provider.dims;
  }
  get id() {
    return this.provider.id;
  }

  /** Drain remaining queue items with errors (for shutdown) */
  drain(): void {
    for (const item of this.queue) {
      item.reject(new Error("EmbedQueue drained (shutdown)"));
    }
    this.queue = [];
  }

  /**
   * Register a callback to be fired once when the queue recovers from an
   * unhealthy state (i.e., Spark comes back online mid-session).
   */
  onRecovery(fn: () => void): void {
    this._recoveryCallbacks.push(fn);
  }

  // --- internals ---

  private getNextRetryMs(): number {
    if (this.circuitState === "CLOSED") return 0;
    if (this.circuitOpenedAt === null) return 0;
    const elapsed = Date.now() - this.circuitOpenedAt;
    return Math.max(0, this.currentResetMs - elapsed);
  }

  private enqueue<T>(fn: () => Promise<T>, input?: string | string[]): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.queue.push({ fn, resolve, reject, retries: 0, input });
      this.process();
    });
  }

  private async process(): Promise<void> {
    while (this.queue.length > 0 && this.active < this.concurrency) {
      // Check circuit breaker
      if (!this.isHealthy()) {
        const waitMs = this.getNextRetryMs();
        this.logger.warn(
          `memory-spark queue: [CIRCUIT OPEN] Spark unreachable, next retry in ${Math.round(waitMs / 1000)}s`
        );
        // Schedule next check
        setTimeout(() => this.process(), Math.min(waitMs + 1000, this.currentResetMs));
        return;
      }

      const item = this.queue.shift()!;
      this.active++;

      try {
        const result = await item.fn();
        this.handleSuccess(item);
        item.resolve(result);
      } catch (err) {
        this.handleFailure(item, err);
      } finally {
        this.active--;
      }

      // Continue processing
      this.process();
    }
  }

  private handleSuccess(item: QueueItem<unknown>): void {
    this.consecutiveFailures = 0;
    this._processed++;

    // If we were unhealthy and just succeeded, circuit is now CLOSED
    if (this.circuitState === "HALF_OPEN") {
      this.circuitState = "CLOSED";
      this.currentResetMs = this.circuitBreaker.initialResetMs;
      this.logger.info("memory-spark queue: [CIRCUIT CLOSED] Spark recovered");
      this._fireRecoveryCallbacks();
    }

    // Remove from failed items if it was tracked
    if (item.input && this.trackFailedItems) {
      const key = this.getFailedItemKey(item.input);
      this.failedItems.delete(key);
    }

    this._wasUnhealthy = false;
  }

  private handleFailure(item: QueueItem<unknown>, err: unknown): void {
    this.consecutiveFailures++;
    const errMsg = err instanceof Error ? err.message : String(err);
    const httpStatus = (err as { httpStatus?: number }).httpStatus;

    // Track failed item
    if (item.input && this.trackFailedItems) {
      const key = this.getFailedItemKey(item.input);
      const existing = this.failedItems.get(key);
      this.failedItems.set(key, {
        input: item.input,
        failedAt: Date.now(),
        retries: (existing?.retries ?? 0) + 1,
        lastError: errMsg,
      });
    }

    // Fatal errors — don't retry
    const isFatal = httpStatus === 401 || httpStatus === 403;
    if (isFatal) {
      this._failed++;
      this.logger.error(
        `memory-spark queue: FATAL ${httpStatus} — not retrying (check auth/token): ${errMsg}`
      );
      item.reject(err instanceof Error ? err : new Error(errMsg));
      return;
    }

    // Check if we should open the circuit
    if (this.circuitState === "HALF_OPEN") {
      // Failed in HALF_OPEN → back to OPEN with doubled timeout
      this.currentResetMs = Math.min(
        this.currentResetMs * this.circuitBreaker.multiplier,
        this.circuitBreaker.maxResetMs
      );
      this.circuitState = "OPEN";
      this.circuitOpenedAt = Date.now();
      this.logger.error(
        `memory-spark queue: [CIRCUIT OPEN] HALF_OPEN failed, next retry in ${this.currentResetMs / 60000}min: ${errMsg}`
      );
    } else if (this.consecutiveFailures >= this.circuitBreaker.failureThreshold) {
      // Threshold reached → open circuit
      this.circuitState = "OPEN";
      this.circuitOpenedAt = Date.now();
      this.currentResetMs = this.circuitBreaker.initialResetMs;
      this._wasUnhealthy = true;
      this.logger.error(
        `memory-spark queue: [CIRCUIT OPEN] ${this.consecutiveFailures} consecutive failures, parking for ${this.currentResetMs / 60000}min`
      );
    }

    // Retry logic
    if (item.retries < this.maxRetries && this.circuitState !== "OPEN") {
      item.retries++;
      const multiplier = httpStatus === 429 ? 3 : 1;
      const delay = Math.min(
        this.baseDelayMs * Math.pow(2, item.retries - 1) * multiplier,
        this.maxDelayMs
      );
      const tag =
        httpStatus === 429 ? "RATE LIMITED" : httpStatus ? `HTTP ${httpStatus}` : "network";
      this.logger.warn(
        `memory-spark queue: retry ${item.retries}/${this.maxRetries} [${tag}] in ${delay}ms: ${errMsg}`
      );
      setTimeout(() => {
        this.queue.unshift(item);
        this.process();
      }, delay);
    } else if (this.circuitState === "OPEN") {
      // Circuit is open — don't reject, just leave in failed items for later retry
      this._failed++;
      this.logger.error(
        `memory-spark queue: circuit OPEN, deferring item (${this.failedItems.size} failed items pending)`
      );
      item.reject(new Error(`Circuit breaker OPEN: ${errMsg}`));
    } else {
      // Max retries exceeded
      this._failed++;
      this.logger.error(
        `memory-spark queue: gave up after ${this.maxRetries} retries: ${errMsg}`
      );
      item.reject(err instanceof Error ? err : new Error(errMsg));
    }
  }

  private getFailedItemKey(input: string | string[]): string {
    if (Array.isArray(input)) {
      return input.join("|");
    }
    return input;
  }

  private _fireRecoveryCallbacks(): void {
    const cbs = this._recoveryCallbacks.slice();
    setImmediate(() => {
      for (const cb of cbs) {
        try {
          cb();
        } catch {
          /* swallow callback errors */
        }
      }
    });
  }

  private withTimeout<T>(promise: Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error(`Embed timeout (${this.timeoutMs}ms)`)),
        this.timeoutMs
      );
      promise
        .then((v) => {
          clearTimeout(timer);
          resolve(v);
        })
        .catch((e) => {
          clearTimeout(timer);
          reject(e);
        });
    });
  }
}
