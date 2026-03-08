/**
 * EmbedQueue — serialized embed requests with retry, backoff, and health monitoring.
 * 
 * Prevents hammering the embed server. All embed calls go through this queue.
 * Single-concurrency by default (Spark embed is single-worker uvicorn).
 * Retries with exponential backoff on failure.
 * Tracks health state and pauses if the service is down.
 */

import type { EmbedProvider } from "./provider.js";

export interface QueueLogger {
  info: (m: string) => void;
  warn: (m: string) => void;
  error: (m: string) => void;
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
  /** Consecutive failures before marking unhealthy (default: 3) */
  unhealthyThreshold?: number;
  /** Cooldown before retrying after going unhealthy in ms (default: 60000) */
  unhealthyCooldownMs?: number;
}

interface QueueItem<T> {
  fn: () => Promise<T>;
  resolve: (v: T) => void;
  reject: (e: Error) => void;
  retries: number;
}

export class EmbedQueue {
  private provider: EmbedProvider;
  private queue: Array<QueueItem<any>> = [];
  private active = 0;
  private concurrency: number;
  private maxRetries: number;
  private baseDelayMs: number;
  private maxDelayMs: number;
  private timeoutMs: number;
  private unhealthyThreshold: number;
  private unhealthyCooldownMs: number;
  private consecutiveFailures = 0;
  private unhealthySince: number | null = null;
  private _wasUnhealthy = false;
  private _recoveryCallbacks: Array<() => void> = [];
  private logger: QueueLogger;
  private _processed = 0;
  private _failed = 0;

  constructor(provider: EmbedProvider, cfg?: EmbedQueueConfig, logger?: QueueLogger) {
    this.provider = provider;
    this.concurrency = cfg?.concurrency ?? 1;
    this.maxRetries = cfg?.maxRetries ?? 3;
    this.baseDelayMs = cfg?.baseDelayMs ?? 2000;
    this.maxDelayMs = cfg?.maxDelayMs ?? 30000;
    this.timeoutMs = cfg?.timeoutMs ?? 30000;
    this.unhealthyThreshold = cfg?.unhealthyThreshold ?? 3;
    this.unhealthyCooldownMs = cfg?.unhealthyCooldownMs ?? 60000;
    this.logger = logger ?? { info: console.log, warn: console.warn, error: console.error };
  }

  /** Embed a single query (for search). Goes through queue. */
  async embedQuery(text: string): Promise<number[]> {
    return this.enqueue(async () => {
      const result = await this.withTimeout(this.provider.embedQuery(text));
      return result;
    });
  }

  /** Embed a batch of texts (for indexing). Serialized through queue one batch at a time. */
  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    // Break into small batches to avoid overwhelming the server
    const BATCH_SIZE = 8;
    const allResults: number[][] = [];
    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batch = texts.slice(i, i + BATCH_SIZE);
      const results = await this.enqueue(async () => {
        return this.withTimeout(this.provider.embedBatch(batch));
      });
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
      healthy: this.isHealthy(),
      consecutiveFailures: this.consecutiveFailures,
    };
  }

  /** Check if the embed service is considered healthy */
  isHealthy(): boolean {
    if (this.unhealthySince === null) return true;
    // Check if cooldown has elapsed
    if (Date.now() - this.unhealthySince >= this.unhealthyCooldownMs) {
      this.logger.info("memory-spark queue: cooldown elapsed, marking healthy for retry");
      this.unhealthySince = null;
      this.consecutiveFailures = 0;
      return true;
    }
    return false;
  }

  /** The embed provider's model/dims info */
  get model() { return this.provider.model; }
  get dims() { return this.provider.dims; }
  get id() { return this.provider.id; }

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
   * The callback is called asynchronously via setImmediate.
   */
  onRecovery(fn: () => void): void {
    this._recoveryCallbacks.push(fn);
  }

  // --- internals ---

  private enqueue<T>(fn: () => Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.queue.push({ fn, resolve, reject, retries: 0 });
      this.process();
    });
  }

  private async process(): Promise<void> {
    while (this.queue.length > 0 && this.active < this.concurrency) {
      if (!this.isHealthy()) {
        const waitMs = this.unhealthyCooldownMs - (Date.now() - (this.unhealthySince ?? Date.now()));
        this.logger.warn(`memory-spark queue: unhealthy, waiting ${Math.round(waitMs / 1000)}s cooldown`);
        setTimeout(() => this.process(), Math.min(waitMs + 1000, this.unhealthyCooldownMs));
        return;
      }

      const item = this.queue.shift()!;
      this.active++;

      try {
        const result = await item.fn();
        const recovering = this._wasUnhealthy;
        this.consecutiveFailures = 0;
        this._processed++;
        item.resolve(result);
        // Fire recovery callbacks once after returning to healthy
        if (recovering) {
          this._wasUnhealthy = false;
          const cbs = this._recoveryCallbacks.slice();
          setImmediate(() => {
            for (const cb of cbs) {
              try { cb(); } catch {}
            }
          });
        }
      } catch (err) {
        this.consecutiveFailures++;
        const errMsg = err instanceof Error ? err.message : String(err);

        if (item.retries < this.maxRetries) {
          // Retry with exponential backoff
          item.retries++;
          const delay = Math.min(
            this.baseDelayMs * Math.pow(2, item.retries - 1),
            this.maxDelayMs
          );
          this.logger.warn(
            `memory-spark queue: retry ${item.retries}/${this.maxRetries} in ${delay}ms: ${errMsg}`
          );
          setTimeout(() => {
            this.queue.unshift(item); // Re-add to front of queue
            this.process();
          }, delay);
        } else {
          // Max retries exceeded
          this._failed++;
          this.logger.error(`memory-spark queue: gave up after ${this.maxRetries} retries: ${errMsg}`);
          item.reject(err instanceof Error ? err : new Error(errMsg));
        }

        // Check if we should mark unhealthy
        if (this.consecutiveFailures >= this.unhealthyThreshold) {
          this.unhealthySince = Date.now();
          this._wasUnhealthy = true;
          this.logger.error(
            `memory-spark queue: ${this.consecutiveFailures} consecutive failures — marking UNHEALTHY for ${this.unhealthyCooldownMs / 1000}s`
          );
        }
      } finally {
        this.active--;
      }

      // Continue processing
      this.process();
    }
  }

  private withTimeout<T>(promise: Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`Embed timeout (${this.timeoutMs}ms)`)), this.timeoutMs);
      promise
        .then((v) => { clearTimeout(timer); resolve(v); })
        .catch((e) => { clearTimeout(timer); reject(e); });
    });
  }
}
