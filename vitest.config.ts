import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Test file patterns
    include: ["tests/**/*.test.ts", "tests/**/*.spec.ts"],
    // Exclude integration tests by default (need Spark)
    exclude: ["tests/integration.ts", "tests/harness.ts", "node_modules/**"],
    // Coverage configuration
    coverage: {
      provider: "v8",
      include: ["src/**/*.ts"],
      all: true, // Include files with zero coverage in reports — makes untested modules visible
      exclude: [
        "src/**/*.d.ts",
      ],
      reporter: ["text", "text-summary", "lcov", "json-summary"],
      reportsDirectory: "./coverage",
      // Thresholds — raised from placeholder 15% (audit 2026-04-02)
      // Target: 50%+ as missing test files are added (capture, queue, dims-lock, etc.)
      thresholds: {
        statements: 35,
        branches: 30,
        functions: 30,
        lines: 35,
      },
    },
    // Global test timeout
    testTimeout: 10_000,
    // Reporter
    reporters: ["verbose"],
    // Environment
    pool: "forks", // safer for LanceDB native bindings
  },
});
