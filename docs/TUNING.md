# Tuning Guide

## Reranker Gate

The dynamic reranker gate is the most impactful tunable in the pipeline. It controls when the cross-encoder fires.

### Hard Gate (recommended — production default)

```json
{
  "rerankerGate": "hard",
  "rerankerGateThreshold": 0.08,
  "rerankerGateLowThreshold": 0.02
}
```

**How it works:**
- Compute the "spread" of the top-5 vector scores (max − min)
- If spread > 0.08: vector is confident → skip reranker
- If spread < 0.02: scores are tied → skip reranker (it's gambling)
- If spread 0.02–0.08: ambiguous → fire reranker

**Tuning:**
- Raise `rerankerGateThreshold` (e.g., 0.10) to fire the reranker more often. Good if your reranker is accurate and latency isn't a concern.
- Lower it (e.g., 0.06) to trust vector more. Good if the reranker isn't helping much on your data.
- The `rerankerGateLowThreshold` rarely needs tuning — 0.02 is a good floor.

### Soft Gate

```json
{
  "rerankerGate": "soft"
}
```

Dynamically scales the vector weight in RRF based on spread. Higher spread → higher vector trust → less reranker influence. Provides smoother transitions than the hard gate but adds latency since the reranker always fires.

### Off

```json
{
  "rerankerGate": "off"
}
```

Reranker fires on every query. Use only if you want maximum reranker influence and don't mind the latency.

## RRF Parameters

### k (Smoothing Constant)

```json
{
  "rrfK": 60
}
```

- Default: 60 (standard, used by Elasticsearch/Azure)
- Lower (20–30): Top ranks have more influence. Sharp cutoff — good for "first result matters" use cases.
- Higher (60–100): Smoother, more democratic ranking. Good for diverse result sets.

### Vector/Reranker Weights

```json
{
  "rrfVectorWeight": 1.0,
  "rrfRerankerWeight": 1.0
}
```

Control relative influence of vector search vs. cross-encoder in RRF:
- Equal (1.0/1.0): Default. Both signals contribute equally.
- Vector-biased (1.5/1.0): Trust semantic search more. Good when your embeddings are well-aligned.
- Reranker-biased (1.0/1.5): Trust the cross-encoder more. Good for well-calibrated rerankers.

## Source Weights

Control how different content types are weighted in recall:

```json
{
  "weights": {
    "sources": {
      "capture": 1.5,
      "memory": 1.0,
      "sessions": 0.5,
      "reference": 1.0
    },
    "paths": {
      "MEMORY.md": 1.4,
      "MISTAKES.md": 1.6,
      "TOOLS.md": 1.3,
      "AGENTS.md": 1.2
    },
    "pathPatterns": {
      "mistakes": 1.6,
      "memory/archive/": 0.4
    }
  }
}
```

- Captures (extracted facts) get 1.5× because they're high-signal
- Sessions get 0.5× because they're noisy raw conversation
- Mistakes get 1.6× because learning from errors is critical
- Archives get 0.4× because they're old and less relevant

## Temporal Decay

```json
{
  "temporalDecay": {
    "floor": 0.8,
    "rate": 0.03
  }
}
```

Formula: `score *= floor + (1 - floor) * exp(-rate * ageDays)`

| Age | Decay Factor |
|-----|-------------|
| 0 days | 1.00 |
| 7 days | 0.96 |
| 30 days | 0.89 |
| 90 days | 0.81 |
| 365 days | 0.80 |

**Tuning:**
- Raise `floor` (e.g., 0.9) for more weight on old memories
- Lower `floor` (e.g., 0.6) to strongly prefer recent content
- Raise `rate` (e.g., 0.1) for faster decay

## MMR Lambda

```json
{
  "mmrLambda": 0.9
}
```

Controls relevance vs. diversity trade-off:
- λ = 1.0: Pure relevance (no diversity penalty)
- λ = 0.9: Mild diversity (default)
- λ = 0.7: Moderate diversity
- λ = 0.5: Equal relevance and diversity

Higher lambda is generally better for factual recall. Lower lambda helps when you want to surface diverse perspectives.

## Token Budget

```json
{
  "maxInjectionTokens": 2000
}
```

Maximum tokens injected into the agent's context per turn. Higher = more memories but more context usage. Adjust based on your model's context window and how much memory context you want.

## Auto-Capture

```json
{
  "autoCapture": {
    "enabled": true,
    "agents": ["main", "dev"],
    "minMessageLength": 30
  }
}
```

- Capture has a built-in dedup threshold of 0.92 cosine similarity — near-duplicates are skipped
- Max 3 captures per turn to prevent context pollution
- Only user messages and assistant messages with decision/fact patterns are captured
