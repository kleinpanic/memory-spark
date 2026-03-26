/**
 * Shared test configuration — no deployment-specific values.
 * All internal IPs, tokens, and paths are abstracted here.
 */
export const TEST_CONFIG = {
  /** Use localhost for all test endpoints — no real Spark needed for unit tests */
  sparkHost: "localhost",
  sparkPort: 8080,

  /** Dummy bearer token for tests that need auth headers */
  bearerToken: "test-bearer-token-do-not-use-in-production",

  /** Fake agent IDs for test isolation */
  agents: {
    primary: "test-agent",
    secondary: "test-agent-2",
    ignored: "test-ignored-agent",
  },

  /** Test workspace paths (relative to test runner) */
  paths: {
    fixtures: "tests/fixtures",
    sampleDocs: "tests/fixtures/sample-docs",
    groundTruth: "tests/fixtures/ground-truth.json",
  },
} as const;
