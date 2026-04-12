#!/usr/bin/env npx tsx
/**
 * run-agentic-benchmark.ts — Agentic RAG Evaluation
 * 
 * Evaluates memory-spark by spawning an OpenClaw agent session and letting it
 * use memory_search to answer BEIR queries end-to-end.
 * 
 * Usage:
 *   SPARK_HOST=10.99.1.1 SPARK_BEARER_TOKEN=<token> \
 *   BEIR_DATA_DIR=/path/to/beir-datasets \
 *   npx tsx evaluation/scripts/run-agentic-benchmark.ts \
 *     --datasets scifact,nfcorpus \
 *     --sample 50 \
 *     --output ./agentic-results
 */

import fs from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";
import { resolveConfig } from "../../src/config.js";
import { createEmbedProvider } from "../../src/embed/provider.js";
import { EmbedQueue } from "../../src/embed/queue.js";
import { LanceDBBackend } from "../../src/storage/lancedb.js";
import { MemoryChunk } from "../../src/storage/backend.js";
import type { Qrels } from "../metrics.js";

// ── Types ─────────────────────────────────────────────────────────────────────

interface BeirQuery { _id: string; text: string }
interface AgenticResult {
  queryId: string;
  query: string;
  retrievedPaths: string[];
  agentAnswer: string;
  retrievalScore: number;
  answerScore: number;
  latencyMs: number;
}
interface DatasetResult {
  dataset: string;
  corpusSize: number;
  queryCount: number;
  sampleSize: number;
  results: AgenticResult[];
  avgRetrievalScore: number;
  avgAnswerScore: number;
  avgLatencyMs: number;
  timestamp: string;
}

// ── Gateway HTTP Client ────────────────────────────────────────────────────────

async function createGatewayClient(token: string) {
  async function invoke(tool: string, action: string, args: Record<string, unknown>, sessionKey?: string): Promise<unknown> {
    const body: Record<string, unknown> = { tool, action, args };
    if (sessionKey) body.sessionKey = sessionKey;
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        const res = await fetch("https://127.0.0.1:18789/tools/invoke", {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(30000),
        });
        const data = await res.json() as { ok: boolean; result?: unknown; error?: { message: string } };
        if (!res.ok || !data.ok) throw new Error(data.error?.message ?? `HTTP ${res.status}`);
        return data.result;
      } catch (err) {
        if (attempt === 3) throw err;
        await new Promise(r => setTimeout(r, 2000 * attempt));
      }
    }
    throw new Error("unreachable");
  }

  return {
    invoke,
    sessions: {
      async spawn(agentId: string, task: string, model?: string) {
        const result = await invoke("sessions_spawn", "spawn", {
          task,
          agentId,
          mode: "session",
          runtime: "subagent",
          lightContext: false,
          ...(model ? { model } : {}),
        }) as { sessionKey: string };
        return result.sessionKey;
      },
      async send(sessionKey: string, message: string, timeoutSeconds = 120) {
        const result = await invoke("sessions_send", "send", {
          sessionKey,
          message,
          timeoutSeconds,
        });
        return result as { message?: string };
      },
      async list(limit = 5) {
        return invoke("sessions_list", "list", { limit }) as Promise<{ sessions: Array<{ sessionKey: string; agentId: string }> }>;
      },
    },
  };
}

// ── LLM-as-Judge ──────────────────────────────────────────────────────────────

async function scoreWithJudge(
  query: string,
  answer: string,
  relevantDocTexts: string[],
  llmUrl: string,
  bearerToken: string,
): Promise<number> {
  const docs = relevantDocTexts.slice(0, 5).join("\n");
  const prompt = `Score this RAG answer: 0.0 (wrong) to 1.0 (perfect).

Query: "${query}"
Answer: "${answer}"
Retrieved docs contain: ${docs.slice(0, 200)}

Respond ONLY with a number between 0.0 and 1.0. No explanation.`;

  try {
    const res = await fetch(llmUrl, {
      method: "POST",
      headers: { "Authorization": `Bearer ${bearerToken}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "nvidia/Nemotron-3-Super-120B-A12B-NVFP4",
        messages: [{ role: "user", content: prompt }],
        max_tokens: 8,
        temperature: 0,
      }),
      signal: AbortSignal.timeout(30000),
    });
    const data = await res.json() as { choices?: Array<{ message?: { content?: string } }> };
    const text = data.choices?.[0]?.message?.content ?? "0";
    return Math.max(0, Math.min(1, parseFloat(text.trim()) || 0));
  } catch {
    return 0;
  }
}

// ── Dataset Loading ────────────────────────────────────────────────────────────

async function loadQueries(dataset: string, beirDataDir: string, sampleSize?: number): Promise<BeirQuery[]> {
  const filePath = path.join(beirDataDir, dataset, "queries.jsonl");
  const content = await fs.readFile(filePath, "utf-8");
  const queries = content.trim().split("\n").filter(Boolean).map(line => JSON.parse(line) as BeirQuery);
  if (sampleSize && sampleSize < queries.length) {
    // Deterministic sample based on dataset name
    const seed = dataset.split("").reduce((a, c) => a + c.charCodeAt(0), 0);
    const sampled: BeirQuery[] = [];
    let state = seed;
    for (const q of queries) {
      state = (state * 1664525 + 1013904223) & 0x7fffffff;
      if (sampled.length < sampleSize && (state % queries.length) < sampleSize) sampled.push(q);
    }
    return sampled;
  }
  return queries;
}

async function loadQrels(dataset: string, beirDataDir: string): Promise<Qrels> {
  const filePath = path.join(beirDataDir, dataset, "qrels", "test.tsv");
  const content = await fs.readFile(filePath, "utf-8");
  const qrels: Qrels = {};
  for (const line of content.trim().split("\n").slice(1)) {
    const [queryId, , docId, rel] = line.split("\t");
    if (!qrels[queryId]) qrels[queryId] = {};
    qrels[queryId][docId] = parseInt(rel, 10);
  }
  return qrels;
}

async function loadCorpusForIngest(
  dataset: string,
  beirDataDir: string,
): Promise<Array<{ id: string; text: string; metadata: Record<string, unknown> }>> {
  const corpusPath = path.join(beirDataDir, dataset, "corpus.jsonl");
  const content = await fs.readFile(corpusPath, "utf-8");
  return content.trim().split("\n").filter(Boolean).map(line => {
    const doc = JSON.parse(line) as { _id: string; title?: string; text?: string };
    return {
      id: doc._id,
      text: (doc.title ? doc.title + " " : "") + (doc.text ?? ""),
      metadata: { path: `beir/${dataset}/${doc._id}`, source: dataset },
    };
  });
}

// ── Ingest corpus into storage ─────────────────────────────────────────────────

async function ingestCorpus(
  storage: LanceDBBackend,
  embedQueue: EmbedQueue,
  records: Array<{ id: string; text: string; metadata: Record<string, unknown> }>,
  dataset: string,
): Promise<void> {
  const total = records.length;
  const batchSize = 100;
  let indexed = 0;

  for (let i = 0; i < total; i += batchSize) {
    const batch = records.slice(i, i + batchSize);
    const texts = batch.map(r => r.text);

    try {
      const vectors = await embedQueue.embedBatch(texts);
      const chunks: MemoryChunk[] = batch.map((r, idx) => ({
        id: r.id,
        path: r.metadata.path as string,
        source: "ingest" as const,
        agent_id: "eval",
        start_line: 0,
        end_line: 0,
        text: r.text,
        vector: vectors[idx],
        updated_at: new Date().toISOString(),
        content_type: "knowledge",
        pool: "agent_memory",
      }));

      await storage.upsert(chunks);
      indexed += batch.length;
    } catch (err) {
      console.error(`  [${dataset}] Batch error at ${i}: ${err instanceof Error ? err.message : String(err)}`);
    }

    if ((i / batchSize) % 20 === 0) {
      console.log(`  [${dataset}] Indexed ${indexed}/${total} (${((indexed / total) * 100).toFixed(0)}%)`);
    }
  }
  console.log(`  [${dataset}] ✅ Indexed ${indexed}/${total}`);
}

// ── Agentic query loop ─────────────────────────────────────────────────────────

async function runAgenticQuery(
  sessionKey: string,
  client: Awaited<ReturnType<typeof createGatewayClient>>,
  query: string,
  queryId: string,
  qrels: Qrels,
  llmUrl: string,
  bearerToken: string,
): Promise<AgenticResult> {
  const start = Date.now();

  const prompt = `You are a RAG benchmark agent. Use memory_search to answer the query.

QUERY: "${query}"

RULES:
1. Call memory_search with the query text (try 2-3 different phrasings to find good results)
2. Read the most relevant results using memory_get if needed
3. Provide a factual answer based ONLY on retrieved documents
4. End your response with:
   ANSWER_END: <your answer>
   RETRIEVED_IDS: <doc IDs from the path field of search results, comma-separated>`;

  let rawResponse = "";
  try {
    const result = await client.sessions.send(sessionKey, prompt, 180);
    rawResponse = typeof result === "object" && result !== null && "message" in result
      ? String(result.message)
      : JSON.stringify(result);
  } catch (err) {
    rawResponse = `[SESSION_ERROR: ${err instanceof Error ? err.message : String(err)}]`;
  }

  const latencyMs = Date.now() - start;

  // Parse structured response
  const answerMatch = rawResponse.match(/ANSWER_END:\s*(.*?)(?:\n|$)/is);
  const idsMatch = rawResponse.match(/RETRIEVED_IDS:\s*(.*?)(?:\n|$)/is);

  const agentAnswer = answerMatch?.[1]?.trim() ?? rawResponse.slice(0, 400);
  const retrievedIds = idsMatch?.[1]?.split(",").map(s => s.trim()).filter(Boolean) ?? [];

  // Compute retrieval score: what % of highly-relevant docs did agent retrieve?
  const highlyRelevant = Object.entries(qrels[queryId] ?? {})
    .filter(([, rel]) => rel >= 2)
    .map(([docId]) => docId);
  const retrieved = retrievedIds.filter(id =>
    highlyRelevant.some(hr => id.includes(hr) || hr.includes(id))
  );
  const retrievalScore = highlyRelevant.length > 0
    ? retrieved.length / highlyRelevant.length
    : 0;

  // Score answer quality with LLM judge
  const relevantTexts = highlyRelevant.slice(0, 3).map(id => `doc:${id}`);
  const answerScore = await scoreWithJudge(query, agentAnswer, relevantTexts, llmUrl, bearerToken);

  return {
    queryId,
    query,
    retrievedPaths: retrievedIds,
    agentAnswer,
    retrievalScore,
    answerScore,
    latencyMs,
  };
}

// ── Main ───────────────────────────────────────────────────────────────────────

const { values: args } = parseArgs({
  options: {
    datasets:  { type: "string", default: "scifact" },
    sample:    { type: "string", default: "" },
    output:    { type: "string", default: "./agentic-results" },
    "db-dir":  { type: "string", default: "" },
  },
});

const datasetList = args.datasets!.split(",").map(d => d.trim());
const outputDir = args.output!;
const beirDataDir = process.env.BEIR_DATA_DIR ?? "/data/beir-datasets";
const sparkHost = process.env.SPARK_HOST ?? "10.99.1.1";
const sparkToken = process.env.SPARK_BEARER_TOKEN ?? "";
const llmUrl = `http://${sparkHost}:18080/v1/chat/completions`;

// Get gateway token
let gateToken = process.env.OPENCLAW_GATEWAY_TOKEN ?? "";
if (!gateToken) {
  try {
    const cfgJson = await fs.readFile(path.join(process.env.HOME ?? "", ".openclaw/openclaw.json"), "utf-8");
    gateToken = JSON.parse(cfgJson).gateway?.auth?.token ?? "";
  } catch { /* ignore */ }
}

// Resolve config
const cfg = await resolveConfig();
if (args["db-dir"]) cfg.storage.dbPath = args["db-dir"]!;
cfg.storage.ocrEnabled = false;

// Init storage + embedder
const storage = new LanceDBBackend({ ...cfg.storage, ocrEnabled: false });
await storage.ready;
const embedProvider = await createEmbedProvider(cfg.embed);
const embedQueue = new EmbedQueue(embedProvider, { maxConcurrency: 8, maxRetries: 2 });

// Gateway client
const client = await createGatewayClient(gateToken);

// Pre-flight
console.log("\n🔍 Pre-flight checks...");
try {
  const { sessions } = await client.sessions.list(1);
  console.log(`✅ Gateway reachable (${sessions.length} active sessions)`);
} catch (err) {
  console.error(`❌ Gateway unreachable: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
}

// Per-dataset sample sizes
const sampleSizes = args.sample
  ? Object.fromEntries(args.sample.split(",").map(s => { const [k, v] = s.split(":"); return [k.trim(), parseInt(v)]; }))
  : {};

await fs.mkdir(outputDir, { recursive: true });

for (const dataset of datasetList) {
  console.log(`\n📂 ${dataset}`);
  
  const [allQueries, qrels, corpus] = await Promise.all([
    loadQueries(dataset, beirDataDir),
    loadQrels(dataset, beirDataDir),
    loadCorpusForIngest(dataset, beirDataDir),
  ]);

  const sampleSize = sampleSizes[dataset] ?? sampleSizes["default"] ?? allQueries.length;
  const queries = allQueries.slice(0, sampleSize);
  console.log(`   ${allQueries.length} total queries, sampling ${queries.length}`);
  console.log(`   ${corpus.length} corpus docs to ingest...`);

  // Ingest corpus
  await ingestCorpus(storage, embedQueue, corpus, dataset);

  // Spawn agent session for this dataset
  const setupPrompt = `You are a RAG benchmark agent. You have access to:
- memory_search(query, maxResults) — search the knowledge base
- memory_get(path, from?, lines?) — read specific sections of indexed files

Your task: Answer user queries accurately using ONLY the retrieved information.
Be thorough — try multiple search phrasings to find the best documents.
Always end responses with ANSWER_END: <your answer> and RETRIEVED_IDS: <doc IDs>.`;

  let sessionKey: string;
  try {
    sessionKey = await client.sessions.spawn("main", setupPrompt);
    console.log(`   🧵 Session: ${sessionKey}`);
  } catch (err) {
    console.error(`   ❌ Session spawn failed: ${err instanceof Error ? err.message : String(err)}`);
    continue;
  }

  const results: AgenticResult[] = [];

  for (let i = 0; i < queries.length; i++) {
    process.stdout.write(`\r   Query ${i + 1}/${queries.length}...`);
    try {
      const r = await runAgenticQuery(sessionKey, client, queries[i].text, queries[i]._id, qrels, llmUrl, sparkToken);
      results.push(r);
    } catch (err) {
      console.error(`\n   ❌ Error: ${err instanceof Error ? err.message : String(err)}`);
      results.push({ queryId: queries[i]._id, query: queries[i].text, retrievedPaths: [], agentAnswer: `[ERROR]`, retrievalScore: 0, answerScore: 0, latencyMs: 0 });
    }
  }
  console.log();

  const avgRetrieval = results.reduce((s, r) => s + r.retrievalScore, 0) / results.length;
  const avgAnswer = results.reduce((s, r) => s + r.answerScore, 0) / results.length;
  const avgLatency = results.reduce((s, r) => s + r.latencyMs, 0) / results.length;

  console.log(`   ✅ Retrieval: ${avgRetrieval.toFixed(3)} | Answer: ${avgAnswer.toFixed(3)} | ${avgLatency}ms/q`);

  const result: DatasetResult = {
    dataset,
    corpusSize: allQueries.length,
    queryCount: allQueries.length,
    sampleSize,
    results,
    avgRetrievalScore: avgRetrieval,
    avgAnswerScore: avgAnswer,
    avgLatencyMs: avgLatency,
    timestamp: new Date().toISOString(),
  };

  const outPath = path.join(outputDir, `${dataset}-agentic-results.json`);
  await fs.writeFile(outPath, JSON.stringify(result, null, 2));
  console.log(`   📄 → ${outPath}`);
}

console.log("\n🎉 Agentic benchmark complete!");
