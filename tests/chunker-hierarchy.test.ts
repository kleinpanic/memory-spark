/**
 * Parent-Child Hierarchical Chunking — unit tests.
 */

import { describe, it, expect } from "vitest";
import {
  chunkDocument,
  chunkDocumentHierarchical,
  estimateTokens,
} from "../src/embed/chunker.js";

describe("Hierarchical Chunking", () => {
  const sampleMarkdown = `# OpenClaw Architecture

## Gateway

The gateway daemon is the central process that coordinates all agents. It manages sessions, routes messages, and handles tool execution. The gateway runs as a systemd service on the host machine.

Configuration is stored in openclaw.json which defines agents, models, plugins, and routing rules. Changes require a restart via oc-restart.

### Session Management

Sessions track conversations between users and agents. Each session has a unique key and maintains message history. Sessions can be isolated (cron/heartbeat), interactive (Discord/Telegram), or sub-agent (spawned by other agents).

The session system supports compaction via LCM (Lossless Context Management) which summarizes old messages to save token budget while preserving retrievable detail.

## Agents

OpenClaw supports multiple specialized agents:

- **meta** — configuration architect, maintains bootstrap files
- **dev** — software development, code reviews, PRs  
- **main** — general assistant, daily driver
- **school** — academic work, Canvas LMS integration
- **immune** — security audits, health checks
- **research** — deep research tasks
- **ghost** — auto-reply handler
- **recovery** — disaster recovery

Each agent has its own workspace directory with AGENTS.md, SOUL.md, MEMORY.md, and other configuration files.

## Memory System

The memory system uses LanceDB for vector storage with 4096-dimensional Nvidia Nemotron embeddings. It supports:

1. **Auto-recall** — memories are injected into prompts via before_prompt_build hooks
2. **Auto-capture** — important facts from conversations are automatically stored
3. **Source weighting** — different sources get different priority multipliers
4. **Temporal decay** — recent memories are boosted with an exponential decay floor of 0.8
5. **MMR diversity** — results are diversified using Maximum Marginal Relevance

### Pools

Memory chunks are organized into logical pools:
- agent_memory — per-agent workspace files
- agent_tools — tool definitions
- shared_knowledge — cross-agent facts
- shared_mistakes — error patterns with 1.6x boost
- reference_library — external docs (tool-call only)

## Cron and Heartbeats

Agents run on cron schedules. Heartbeats are periodic check-ins where agents audit their domain and report findings. The meta agent runs every 4 hours, immune every 6 hours.

Cron jobs use OpenClaw's built-in scheduler which supports "at" (one-shot), "every" (recurring), and "cron" (expression) schedules.`;

  it("produces parent and child chunks", () => {
    const result = chunkDocumentHierarchical(
      { text: sampleMarkdown, path: "test.md", source: "memory", ext: "md" },
      { parentMaxTokens: 500, childMaxTokens: 100 },
    );

    expect(result.length).toBeGreaterThan(0);

    for (const group of result) {
      expect(group.parent.id).toBeTruthy();
      expect(group.parent.text.length).toBeGreaterThan(0);
      expect(group.children.length).toBeGreaterThan(0);

      for (const child of group.children) {
        expect(child.parentId).toBe(group.parent.id);
        expect(child.text.length).toBeGreaterThan(0);
      }
    }
  });

  it("children are smaller than parents", () => {
    const result = chunkDocumentHierarchical(
      { text: sampleMarkdown, path: "test.md", source: "memory", ext: "md" },
      { parentMaxTokens: 1000, childMaxTokens: 200 },
    );

    for (const group of result) {
      const parentTokens = estimateTokens(group.parent.text);
      for (const child of group.children) {
        const childTokens = estimateTokens(child.text);
        // Children should generally be smaller than their parent
        // (allow some tolerance for edge cases with overlap)
        expect(childTokens).toBeLessThanOrEqual(parentTokens + 50);
      }
    }
  });

  it("parent IDs are unique", () => {
    const result = chunkDocumentHierarchical(
      { text: sampleMarkdown, path: "test.md", source: "memory", ext: "md" },
    );

    const ids = result.map((r) => r.parent.id);
    const uniqueIds = new Set(ids);
    expect(uniqueIds.size).toBe(ids.length);
  });

  it("all children reference valid parent IDs", () => {
    const result = chunkDocumentHierarchical(
      { text: sampleMarkdown, path: "test.md", source: "memory", ext: "md" },
    );

    const parentIds = new Set(result.map((r) => r.parent.id));

    for (const group of result) {
      for (const child of group.children) {
        expect(parentIds.has(child.parentId)).toBe(true);
      }
    }
  });

  it("preserves parentHeading in children", () => {
    const result = chunkDocumentHierarchical(
      { text: sampleMarkdown, path: "test.md", source: "memory", ext: "md" },
    );

    // At least some children should have a parentHeading from the markdown
    const allChildren = result.flatMap((r) => r.children);
    const withHeadings = allChildren.filter((c) => c.parentHeading);
    expect(withHeadings.length).toBeGreaterThan(0);
  });

  it("returns empty array for empty input", () => {
    const result = chunkDocumentHierarchical(
      { text: "", path: "empty.md", source: "memory", ext: "md" },
    );
    expect(result).toEqual([]);
  });

  it("returns empty array for whitespace-only input", () => {
    const result = chunkDocumentHierarchical(
      { text: "   \n\n   ", path: "empty.md", source: "memory", ext: "md" },
    );
    expect(result).toEqual([]);
  });

  it("existing chunkDocument still works (backward compat)", () => {
    const flat = chunkDocument(
      { text: sampleMarkdown, path: "test.md", source: "memory", ext: "md" },
      { maxTokens: 400 },
    );

    expect(flat.length).toBeGreaterThan(0);
    for (const chunk of flat) {
      expect(chunk.text).toBeTruthy();
      expect(chunk.startLine).toBeGreaterThan(0);
    }
  });

  it("produces more children than parents for large documents", () => {
    const result = chunkDocumentHierarchical(
      { text: sampleMarkdown, path: "test.md", source: "memory", ext: "md" },
      { parentMaxTokens: 500, childMaxTokens: 100 },
    );

    const totalParents = result.length;
    const totalChildren = result.reduce((sum, group) => sum + group.children.length, 0);

    // With 500-token parents and 100-token children, should have more children than parents
    expect(totalChildren).toBeGreaterThan(totalParents);
  });

  it("handles small documents (single parent)", () => {
    const small = "# OpenClaw Gateway\n\nThe gateway daemon is the central process that coordinates all agents in the system. It manages sessions, routes messages between users and agents, and handles tool execution. Configuration is stored in openclaw.json.";
    const result = chunkDocumentHierarchical(
      { text: small, path: "small.md", source: "memory", ext: "md" },
      { parentMaxTokens: 2000, childMaxTokens: 200, childMinTokens: 10 },
    );

    // Small doc should produce 1 parent with at least 1 child
    expect(result.length).toBeGreaterThanOrEqual(1);
    expect(result[0]!.children.length).toBeGreaterThanOrEqual(1);
  });
});
