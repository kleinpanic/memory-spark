/**
 * Tests for src/auto/capture.ts — createAutoCaptureHandler
 */
import { describe, it, expect, vi, beforeEach, type Mock } from "vitest";

// ── Mock all heavy dependencies before importing the module under test ──

vi.mock("../src/classify/zero-shot.js", () => ({
  classifyForCapture: vi.fn(),
}));

vi.mock("../src/classify/heuristic.js", () => ({
  heuristicClassify: vi.fn(),
}));

vi.mock("../src/classify/ner.js", () => ({
  tagEntities: vi.fn().mockResolvedValue([]),
}));

vi.mock("../src/classify/quality.js", () => ({
  scoreChunkQuality: vi.fn().mockReturnValue({ score: 0.8 }),
}));

vi.mock("../src/config.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/config.js")>();
  return {
    ...actual,
    shouldProcessAgent: vi.fn().mockReturnValue(true),
  };
});

vi.mock("../src/security.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/security.js")>();
  return {
    ...actual,
    looksLikePromptInjection: vi.fn().mockReturnValue(false),
  };
});

// ── Now import ──

import { createAutoCaptureHandler } from "../src/auto/capture.js";
import { classifyForCapture } from "../src/classify/zero-shot.js";
import { heuristicClassify } from "../src/classify/heuristic.js";
import { scoreChunkQuality } from "../src/classify/quality.js";
import { shouldProcessAgent } from "../src/config.js";
import { looksLikePromptInjection } from "../src/security.js";
import type { AutoCaptureDeps } from "../src/auto/capture.js";
import type { StorageBackend, MemoryChunk } from "../src/storage/backend.js";
import type { EmbedProvider } from "../src/embed/provider.js";

// ── Helpers ────────────────────────────────────────────────────────────────

function makeBackend(overrides: Partial<StorageBackend> = {}): StorageBackend {
  return {
    upsert: vi.fn().mockResolvedValue(undefined),
    vectorSearch: vi.fn().mockResolvedValue([]),
    search: vi.fn().mockResolvedValue([]),
    delete: vi.fn().mockResolvedValue(undefined),
    listPaths: vi.fn().mockResolvedValue([]),
    getByPath: vi.fn().mockResolvedValue([]),
    ...overrides,
  } as unknown as StorageBackend;
}

function makeEmbed(overrides: Partial<EmbedProvider> = {}): EmbedProvider {
  return {
    embedQuery: vi.fn().mockResolvedValue(new Array(384).fill(0.1)),
    embedBatch: vi.fn().mockResolvedValue([]),
    ...overrides,
  } as unknown as EmbedProvider;
}

function makeDeps(overrides: Partial<AutoCaptureDeps> = {}): AutoCaptureDeps {
  return {
    cfg: {
      enabled: true,
      categories: ["decision", "preference", "fact", "code-snippet"],
      minConfidence: 0.5,
      minMessageLength: 30,
      useClassifier: true,
      agents: ["*"],
      ignoreAgents: [],
    },
    globalCfg: {} as any,
    backend: makeBackend(),
    embed: makeEmbed(),
    ...overrides,
  };
}

function makeEvent(
  messages: unknown[],
  success = true,
): { messages: unknown[]; success: boolean } {
  return { messages, success };
}

function userMsg(text: string) {
  return { role: "user", content: text };
}

function assistantMsg(text: string) {
  return { role: "assistant", content: text };
}

// A clearly long-enough user message with a decision pattern so heuristic catches it
const DECISION_MSG = "We decided to use PostgreSQL for the main database going forward.";
const LONG_FACT_MSG =
  "The server runs on port 8080 and is deployed to the production cluster at 10.0.0.1.";

// ── Tests ──────────────────────────────────────────────────────────────────

describe("createAutoCaptureHandler", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Default mocks: zero-shot returns a good decision result, heuristic too
    (classifyForCapture as Mock).mockResolvedValue({ label: "decision", score: 0.8 });
    (heuristicClassify as Mock).mockReturnValue({ label: "decision", score: 0.7 });
    (scoreChunkQuality as Mock).mockReturnValue({ score: 0.8 });
    (looksLikePromptInjection as Mock).mockReturnValue(false);
    (shouldProcessAgent as Mock).mockReturnValue(true);
  });

  // ── 1. createAutoCaptureHandler returns a function ─────────────────────

  describe("factory", () => {
    it("returns a function (the handler)", () => {
      const handler = createAutoCaptureHandler(makeDeps());
      expect(typeof handler).toBe("function");
    });

    it("returned handler is async", () => {
      const handler = createAutoCaptureHandler(makeDeps());
      const result = handler(makeEvent([userMsg(DECISION_MSG)]), { agentId: "dev" });
      expect(result).toBeInstanceOf(Promise);
      return result; // let it settle
    });
  });

  // ── 2. Skips when disabled or event failed ─────────────────────────────

  describe("early exits", () => {
    it("does nothing when cfg.enabled is false", async () => {
      const deps = makeDeps({ cfg: { ...makeDeps().cfg, enabled: false } });
      const backend = deps.backend as any;
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect(backend.upsert).not.toHaveBeenCalled();
    });

    it("does nothing when event.success is false", async () => {
      const deps = makeDeps();
      const backend = deps.backend as any;
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)], false),
        { agentId: "dev" },
      );
      expect(backend.upsert).not.toHaveBeenCalled();
    });

    it("does nothing when no messages are provided", async () => {
      const deps = makeDeps();
      await createAutoCaptureHandler(deps)(makeEvent([]), { agentId: "dev" });
      expect((deps.backend as any).upsert).not.toHaveBeenCalled();
    });
  });

  // ── 3. Message filtering: role ─────────────────────────────────────────

  describe("message role filtering", () => {
    it("captures user messages", async () => {
      const deps = makeDeps();
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).toHaveBeenCalledOnce();
    });

    it("skips system messages", async () => {
      const deps = makeDeps();
      const sysMsg = {
        role: "system",
        content: "We decided to use PostgreSQL for the main database going forward.",
      };
      await createAutoCaptureHandler(deps)(makeEvent([sysMsg]), { agentId: "dev" });
      expect((deps.backend as any).upsert).not.toHaveBeenCalled();
    });

    it("captures assistant messages with decision pattern", async () => {
      const deps = makeDeps();
      const text =
        "We decided to use Redis as the caching layer. The conclusion is that Memcached is insufficient.";
      await createAutoCaptureHandler(deps)(
        makeEvent([assistantMsg(text)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).toHaveBeenCalledOnce();
    });

    it("captures assistant messages with fact/infrastructure pattern", async () => {
      const deps = makeDeps();
      const text = "The server runs at port 9090 and is configured as read-only replica.";
      await createAutoCaptureHandler(deps)(
        makeEvent([assistantMsg(text)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).toHaveBeenCalledOnce();
    });

    it("skips assistant messages without decision/fact patterns", async () => {
      const deps = makeDeps();
      // Generic assistant reply — no decision/fact keywords
      const text =
        "Sure! I can help you with that. Let me take a look at your current setup and suggest some options.";
      await createAutoCaptureHandler(deps)(
        makeEvent([assistantMsg(text)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).not.toHaveBeenCalled();
    });
  });

  // ── 4. Short message filtering ─────────────────────────────────────────

  describe("minMessageLength filtering", () => {
    it("skips messages shorter than minMessageLength (30 chars)", async () => {
      const deps = makeDeps();
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg("too short")]), // < 30 chars
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).not.toHaveBeenCalled();
    });

    it("captures messages at or above minMessageLength", async () => {
      const deps = makeDeps();
      // Exactly 30 chars wouldn't pass extractCaptureMessages (uses minLen from cfg)
      // Use something clearly over threshold
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).toHaveBeenCalled();
    });

    it("respects custom minMessageLength config", async () => {
      const deps = makeDeps({
        cfg: { ...makeDeps().cfg, minMessageLength: 200 },
      });
      // DECISION_MSG is ~65 chars — below 200
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).not.toHaveBeenCalled();
    });
  });

  // ── 5. MAX_CAPTURES_PER_TURN = 3 ──────────────────────────────────────

  describe("MAX_CAPTURES_PER_TURN", () => {
    it("captures at most 3 messages per turn", async () => {
      const deps = makeDeps();
      const messages = [
        userMsg("We decided to use PostgreSQL as the primary relational database for this project."),
        userMsg("We prefer TypeScript over JavaScript for all new services in this repository."),
        userMsg("We decided to deploy everything on Kubernetes with a GitOps workflow approach."),
        userMsg("We chose React over Vue for the frontend application given team familiarity."),
        userMsg("We decided the CI/CD pipeline should run tests on every single pull request."),
      ];
      await createAutoCaptureHandler(deps)(makeEvent(messages), { agentId: "dev" });
      expect((deps.backend as any).upsert).toHaveBeenCalledTimes(3);
    });

    it("captures fewer than 3 if fewer qualifying messages", async () => {
      const deps = makeDeps();
      const messages = [
        userMsg("We decided to use PostgreSQL as the primary relational database."),
        userMsg("We prefer TypeScript over JavaScript for all new services here."),
      ];
      await createAutoCaptureHandler(deps)(makeEvent(messages), { agentId: "dev" });
      expect((deps.backend as any).upsert).toHaveBeenCalledTimes(2);
    });
  });

  // ── 6. Deduplication: similarity > 0.92 ────────────────────────────────

  describe("deduplication", () => {
    it("skips capture when an existing memory scores >= DEDUP_THRESHOLD (0.92)", async () => {
      const deps = makeDeps({
        backend: makeBackend({
          vectorSearch: vi.fn().mockResolvedValue([
            { id: "abc", score: 0.95, text: DECISION_MSG },
          ]),
        }),
      });
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).not.toHaveBeenCalled();
    });

    it("captures when existing memory scores just below threshold (0.91)", async () => {
      const deps = makeDeps({
        backend: makeBackend({
          vectorSearch: vi.fn().mockResolvedValue([
            { id: "abc", score: 0.91, text: DECISION_MSG },
          ]),
        }),
      });
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).toHaveBeenCalledOnce();
    });

    it("captures when no existing memories found", async () => {
      const deps = makeDeps({
        backend: makeBackend({
          vectorSearch: vi.fn().mockResolvedValue([]),
        }),
      });
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).toHaveBeenCalledOnce();
    });

    it("deduplicates identical messages within the same turn (extractCaptureMessages)", async () => {
      const deps = makeDeps();
      // Same message twice — extractCaptureMessages deduplicates via seen set
      const messages = [userMsg(DECISION_MSG), userMsg(DECISION_MSG)];
      await createAutoCaptureHandler(deps)(makeEvent(messages), { agentId: "dev" });
      expect((deps.backend as any).upsert).toHaveBeenCalledTimes(1);
    });
  });

  // ── 7. Prompt injection detection ─────────────────────────────────────

  describe("prompt injection detection", () => {
    it("skips message that looks like prompt injection", async () => {
      (looksLikePromptInjection as Mock).mockReturnValue(true);
      const deps = makeDeps();
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).not.toHaveBeenCalled();
    });

    it("captures message when injection check returns false", async () => {
      (looksLikePromptInjection as Mock).mockReturnValue(false);
      const deps = makeDeps();
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).toHaveBeenCalledOnce();
    });

    it("calls looksLikePromptInjection with the message text", async () => {
      const deps = makeDeps();
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect(looksLikePromptInjection).toHaveBeenCalledWith(DECISION_MSG);
    });
  });

  // ── 8. shouldProcessAgent filtering ────────────────────────────────────

  describe("shouldProcessAgent filtering", () => {
    it("skips capture when shouldProcessAgent returns false", async () => {
      (shouldProcessAgent as Mock).mockReturnValue(false);
      const deps = makeDeps();
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).not.toHaveBeenCalled();
    });

    it("passes agentId, agents, and ignoreAgents to shouldProcessAgent", async () => {
      const deps = makeDeps({
        cfg: {
          ...makeDeps().cfg,
          agents: ["dev", "main"],
          ignoreAgents: ["immune"],
        },
      });
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect(shouldProcessAgent).toHaveBeenCalledWith("dev", ["dev", "main"], ["immune"]);
    });

    it("uses 'unknown' as agentId when ctx.agentId is not provided", async () => {
      const deps = makeDeps();
      await createAutoCaptureHandler(deps)(makeEvent([userMsg(DECISION_MSG)]), {});
      expect(shouldProcessAgent).toHaveBeenCalledWith("unknown", ["*"], []);
    });
  });

  // ── 9. Error handling: non-fatal ───────────────────────────────────────

  describe("error handling", () => {
    it("does not throw when backend.upsert fails", async () => {
      const deps = makeDeps({
        backend: makeBackend({
          upsert: vi.fn().mockRejectedValue(new Error("DB connection lost")),
        }),
      });
      await expect(
        createAutoCaptureHandler(deps)(makeEvent([userMsg(DECISION_MSG)]), { agentId: "dev" }),
      ).resolves.toBeUndefined();
    });

    it("does not throw when embed.embedQuery fails", async () => {
      const deps = makeDeps({
        embed: makeEmbed({
          embedQuery: vi.fn().mockRejectedValue(new Error("Spark offline")),
        }),
      });
      await expect(
        createAutoCaptureHandler(deps)(makeEvent([userMsg(DECISION_MSG)]), { agentId: "dev" }),
      ).resolves.toBeUndefined();
    });

    it("does not throw when classifyForCapture fails", async () => {
      (classifyForCapture as Mock).mockRejectedValue(new Error("classifier down"));
      const deps = makeDeps();
      await expect(
        createAutoCaptureHandler(deps)(makeEvent([userMsg(DECISION_MSG)]), { agentId: "dev" }),
      ).resolves.toBeUndefined();
    });

    it("continues processing subsequent messages after a single capture failure", async () => {
      let callCount = 0;
      const deps = makeDeps({
        embed: makeEmbed({
          embedQuery: vi.fn().mockImplementation(() => {
            callCount++;
            if (callCount === 1) return Promise.reject(new Error("first embed fails"));
            return Promise.resolve(new Array(384).fill(0.1));
          }),
        }),
      });
      const messages = [
        userMsg("We decided to use PostgreSQL as the primary relational database for this project."),
        userMsg("We prefer TypeScript over JavaScript for all new services in this repository."),
      ];
      await createAutoCaptureHandler(deps)(makeEvent(messages), { agentId: "dev" });
      // First embed fails → no upsert, but second should succeed
      expect((deps.backend as any).upsert).toHaveBeenCalledTimes(1);
    });

    it("logs a warning (console.warn) on capture error", async () => {
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
      const deps = makeDeps({
        backend: makeBackend({
          upsert: vi.fn().mockRejectedValue(new Error("DB error")),
        }),
      });
      await createAutoCaptureHandler(deps)(makeEvent([userMsg(DECISION_MSG)]), { agentId: "dev" });
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining("[memory-spark]"),
        expect.any(String),
      );
      warnSpy.mockRestore();
    });
  });

  // ── 10. Quality scoring integration ───────────────────────────────────

  describe("quality scoring", () => {
    it("calls scoreChunkQuality with the message text", async () => {
      const deps = makeDeps();
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect(scoreChunkQuality).toHaveBeenCalledWith(
        DECISION_MSG,
        expect.stringContaining("capture/dev"),
        "capture",
      );
    });

    it("skips capture when quality score is below 0.3", async () => {
      (scoreChunkQuality as Mock).mockReturnValue({ score: 0.29 });
      const deps = makeDeps();
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).not.toHaveBeenCalled();
    });

    it("captures when quality score is exactly 0.3 (boundary)", async () => {
      (scoreChunkQuality as Mock).mockReturnValue({ score: 0.3 });
      const deps = makeDeps();
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).toHaveBeenCalledOnce();
    });
  });

  // ── 11. Heuristic vs zero-shot classifier ─────────────────────────────

  describe("classifier fallback (heuristic vs zero-shot)", () => {
    it("uses heuristicClassify when useClassifier is false", async () => {
      const deps = makeDeps({
        cfg: { ...makeDeps().cfg, useClassifier: false },
      });
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect(classifyForCapture).not.toHaveBeenCalled();
      expect(heuristicClassify).toHaveBeenCalledWith(DECISION_MSG);
    });

    it("uses classifyForCapture (zero-shot) when useClassifier is true", async () => {
      const deps = makeDeps({ cfg: { ...makeDeps().cfg, useClassifier: true } });
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect(classifyForCapture).toHaveBeenCalled();
    });

    it("falls back to heuristicClassify when zero-shot returns 'none'", async () => {
      (classifyForCapture as Mock).mockResolvedValue({ label: "none", score: 0 });
      (heuristicClassify as Mock).mockReturnValue({ label: "decision", score: 0.7 });
      const deps = makeDeps();
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect(heuristicClassify).toHaveBeenCalledWith(DECISION_MSG);
      expect((deps.backend as any).upsert).toHaveBeenCalledOnce();
    });

    it("skips capture when both zero-shot AND heuristic return 'none'", async () => {
      (classifyForCapture as Mock).mockResolvedValue({ label: "none", score: 0 });
      (heuristicClassify as Mock).mockReturnValue({ label: "none", score: 0 });
      const deps = makeDeps();
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).not.toHaveBeenCalled();
    });

    it("skips when category is not in cfg.categories", async () => {
      (classifyForCapture as Mock).mockResolvedValue({ label: "other", score: 0.9 });
      (heuristicClassify as Mock).mockReturnValue({ label: "other", score: 0.9 });
      const deps = makeDeps({
        cfg: { ...makeDeps().cfg, categories: ["decision", "preference"] },
      });
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).not.toHaveBeenCalled();
    });

    it("applies lower effective confidence threshold for heuristic-only results (<=0.70)", async () => {
      // Heuristic scores cap at 0.70 — a score of 0.62 should still pass with 0.6 threshold
      (classifyForCapture as Mock).mockResolvedValue({ label: "none", score: 0 });
      (heuristicClassify as Mock).mockReturnValue({ label: "fact", score: 0.65 });
      const deps = makeDeps({
        cfg: { ...makeDeps().cfg, minConfidence: 0.75 }, // would fail normal threshold
      });
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(LONG_FACT_MSG)]),
        { agentId: "dev" },
      );
      // 0.65 >= 0.6 effective heuristic threshold → should capture
      expect((deps.backend as any).upsert).toHaveBeenCalledOnce();
    });

    it("skips when heuristic score is below the effective heuristic threshold (0.6)", async () => {
      (classifyForCapture as Mock).mockResolvedValue({ label: "none", score: 0 });
      (heuristicClassify as Mock).mockReturnValue({ label: "fact", score: 0.55 });
      const deps = makeDeps({
        cfg: { ...makeDeps().cfg, minConfidence: 0.75 },
      });
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(LONG_FACT_MSG)]),
        { agentId: "dev" },
      );
      expect((deps.backend as any).upsert).not.toHaveBeenCalled();
    });
  });

  // ── 12. Captured chunk shape ───────────────────────────────────────────

  describe("chunk shape / storage", () => {
    it("stores a chunk with expected fields", async () => {
      const deps = makeDeps();
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      const upsertCall = (deps.backend as any).upsert.mock.calls[0][0] as MemoryChunk[];
      expect(upsertCall).toHaveLength(1);
      const chunk = upsertCall[0]!;
      expect(chunk.source).toBe("capture");
      expect(chunk.agent_id).toBe("dev");
      expect(chunk.text).toBe(DECISION_MSG);
      expect(chunk.content_type).toBe("knowledge");
      expect(chunk.pool).toBe("agent_memory");
      expect(typeof chunk.id).toBe("string");
      expect(chunk.id.length).toBeGreaterThan(0);
      expect(chunk.path).toMatch(/^capture\/dev\//);
      expect(chunk.category).toBe("decision");
      expect(typeof chunk.confidence).toBe("number");
    });

    it("includes the embed vector in the stored chunk", async () => {
      const fakeVector = new Array(384).fill(0.5);
      const deps = makeDeps({
        embed: makeEmbed({
          embedQuery: vi.fn().mockResolvedValue(fakeVector),
        }),
      });
      await createAutoCaptureHandler(deps)(
        makeEvent([userMsg(DECISION_MSG)]),
        { agentId: "dev" },
      );
      const chunk = (deps.backend as any).upsert.mock.calls[0][0][0] as MemoryChunk;
      expect(chunk.vector).toEqual(fakeVector);
    });
  });
});
