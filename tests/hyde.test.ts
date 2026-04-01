/**
 * HyDE (Hypothetical Document Embeddings) — unit tests.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  generateHypotheticalDocument,
  type HydeConfig,
  HYDE_DEFAULTS,
} from "../src/hyde/generator.js";

const mockConfig: HydeConfig = {
  ...HYDE_DEFAULTS,
  llmUrl: "http://localhost:9999/v1/chat/completions",
  timeoutMs: 5000,
};

describe("HyDE Generator", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns null when disabled", async () => {
    const result = await generateHypotheticalDocument("What is OpenClaw?", {
      ...mockConfig,
      enabled: false,
    });
    expect(result).toBeNull();
  });

  it("returns null for very short queries", async () => {
    const result = await generateHypotheticalDocument("hi", mockConfig);
    expect(result).toBeNull();
  });

  it("generates hypothetical document from LLM response", async () => {
    const mockResponse = {
      choices: [
        {
          message: {
            content:
              "OpenClaw is a multi-agent AI orchestration platform. It coordinates multiple specialized agents through a gateway daemon, managing tools, memory, and inter-agent communication.",
          },
        },
      ],
    };

    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    } as unknown as Response);

    const result = await generateHypotheticalDocument("What is OpenClaw?", mockConfig);
    expect(result).toBeTruthy();
    expect(result!.length).toBeGreaterThan(20);
    expect(result).toContain("OpenClaw");
  });

  it("returns null on HTTP error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: false,
      status: 500,
    } as unknown as Response);

    const result = await generateHypotheticalDocument("What is OpenClaw?", mockConfig);
    expect(result).toBeNull();
  });

  it("returns null on empty content", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ choices: [{ message: { content: "" } }] }),
    } as unknown as Response);

    const result = await generateHypotheticalDocument("What is OpenClaw?", mockConfig);
    expect(result).toBeNull();
  });

  it("returns null on network error", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValueOnce(new Error("Network timeout"));

    const result = await generateHypotheticalDocument("What is OpenClaw?", mockConfig);
    expect(result).toBeNull();
  });

  it("strips think tags from response", async () => {
    const mockResponse = {
      choices: [
        {
          message: {
            content:
              "<think>Let me write a document about this system and its architecture.</think>OpenClaw is a gateway daemon that orchestrates multiple specialized AI agents on Linux. It manages tools, memory, sessions, and inter-agent communication through a unified configuration system.",
          },
        },
      ],
    };

    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    } as unknown as Response);

    const result = await generateHypotheticalDocument("What is OpenClaw?", mockConfig);
    expect(result).toBeTruthy();
    expect(result).not.toContain("<think>");
    expect(result).not.toContain("</think>");
    expect(result).toContain("OpenClaw");
  });

  it("strips plan narration from response", async () => {
    const mockResponse = {
      choices: [
        {
          message: {
            content:
              "Now let me write about this.\nOpenClaw is a multi-agent AI orchestration platform running on Linux. It coordinates multiple specialized agents through a gateway daemon for tools, memory, and inter-agent communication.",
          },
        },
      ],
    };

    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    } as unknown as Response);

    const result = await generateHypotheticalDocument("What is OpenClaw?", mockConfig);
    expect(result).toBeTruthy();
    expect(result).not.toMatch(/^Now let me/);
  });

  // ── Quality Gate Tests (Phase 3C) ─────────────────────────────────

  it("rejects 'I don't know' responses (quality gate)", async () => {
    const mockResponse = {
      choices: [
        {
          message: {
            content:
              "I don't know the answer to that question. There is not enough information available to provide a response.",
          },
        },
      ],
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    } as unknown as Response);

    const result = await generateHypotheticalDocument("What is the flux capacitor?", mockConfig);
    expect(result).toBeNull();
  });

  it("rejects 'cannot determine' responses (quality gate)", async () => {
    const mockResponse = {
      choices: [
        {
          message: {
            content:
              "We cannot determine the exact configuration from the available information. The system may be running a different version than expected.",
          },
        },
      ],
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    } as unknown as Response);

    const result = await generateHypotheticalDocument("What version is installed?", mockConfig);
    expect(result).toBeNull();
  });

  it("rejects 'as an AI' responses (quality gate)", async () => {
    const mockResponse = {
      choices: [
        {
          message: {
            content:
              "As an AI language model, I don't have access to specific system configurations or real-time data about your setup.",
          },
        },
      ],
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    } as unknown as Response);

    const result = await generateHypotheticalDocument("What is my timezone?", mockConfig);
    expect(result).toBeNull();
  });

  it("rejects too-short hypotheticals (< 10 words)", async () => {
    const mockResponse = {
      choices: [{ message: { content: "OpenClaw is a platform for agents. That's it." } }],
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    } as unknown as Response);

    const result = await generateHypotheticalDocument("What is OpenClaw?", mockConfig);
    expect(result).toBeNull();
  });

  it("accepts valid factual hypotheticals (passes quality gate)", async () => {
    const mockResponse = {
      choices: [
        {
          message: {
            content:
              "Klein's timezone is America/New_York (Eastern Time). The OpenClaw system is configured to use this timezone for all scheduling, cron jobs, and timestamp display.",
          },
        },
      ],
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    } as unknown as Response);

    const result = await generateHypotheticalDocument("What is Klein's timezone?", mockConfig);
    expect(result).toBeTruthy();
    expect(result).toContain("America/New_York");
  });

  it("rejects 'unable to provide' hedging (quality gate)", async () => {
    const mockResponse = {
      choices: [
        {
          message: {
            content:
              "The system is unable to provide specific configuration details without access to the actual deployment environment and runtime state.",
          },
        },
      ],
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    } as unknown as Response);

    const result = await generateHypotheticalDocument("What port is the gateway on?", mockConfig);
    expect(result).toBeNull();
  });

  it("sends auth header when apiKey is set", async () => {
    const mockResponse = {
      choices: [
        {
          message: {
            content: "OpenClaw runs on Debian Linux as a systemd service with multiple agents.",
          },
        },
      ],
    };

    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    } as unknown as Response);

    await generateHypotheticalDocument("What is OpenClaw?", {
      ...mockConfig,
      apiKey: "test-token-123",
    });

    expect(global.fetch).toHaveBeenCalledWith(
      mockConfig.llmUrl,
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer test-token-123",
        }),
      }),
    );
  });
});
