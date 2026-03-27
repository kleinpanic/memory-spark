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
      exclude: [
        "src/**/*.d.ts",
        "src/storage/multi-table-backend.ts", // deleted
        "src/storage/table-manager.ts", // deleted
      ],
      reporter: ["text", "text-summary", "lcov", "json-summary"],
      reportsDirectory: "./coverage",
      // Unit test thresholds (pure logic only — no Spark/network deps)
      // Integration tests run separately and cover storage/embed/ingest
      thresholds: {
        statements: 15,
        branches: 18,
        functions: 12,
        lines: 15,
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
