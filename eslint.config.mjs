import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import unusedImports from "eslint-plugin-unused-imports";
import sonarjs from "eslint-plugin-sonarjs";
import importX from "eslint-plugin-import-x";

export default tseslint.config(
  // Base configs
  eslint.configs.recommended,
  ...tseslint.configs.recommended,

  // SonarJS — deeper bug detection (cognitive complexity, no duplicate strings, etc.)
  sonarjs.configs.recommended,

  // Global ignores
  {
    ignores: ["dist/", "node_modules/", "scripts/", "tests/fixtures/", "coverage/"],
  },

  // TypeScript source files
  {
    files: ["src/**/*.ts", "tests/**/*.ts", "evaluation/**/*.ts", "index.ts"],
    plugins: {
      "unused-imports": unusedImports,
      "import-x": importX,
    },
    rules: {
      // ── TypeScript ──────────────────────────────────────────
      "@typescript-eslint/no-unused-vars": "off", // Handled by unused-imports
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-non-null-assertion": "off",
      "@typescript-eslint/consistent-type-imports": [
        "warn",
        { prefer: "type-imports", fixStyle: "inline-type-imports" },
      ],

      // ── Unused imports — auto-fixable ───────────────────────
      "unused-imports/no-unused-imports": "error",
      "unused-imports/no-unused-vars": [
        "warn",
        {
          vars: "all",
          varsIgnorePattern: "^_",
          args: "after-used",
          argsIgnorePattern: "^_",
        },
      ],

      // ── Import ordering ─────────────────────────────────────
      "import-x/order": [
        "warn",
        {
          groups: [
            "builtin",    // node:fs, node:path
            "external",   // npm packages
            "internal",   // project aliases
            "parent",     // ../
            "sibling",    // ./
            "index",      // ./index
          ],
          "newlines-between": "always",
          alphabetize: { order: "asc", caseInsensitive: true },
        },
      ],
      "import-x/no-duplicates": "error",

      // ── SonarJS overrides ───────────────────────────────────
      // These are too aggressive for a project with lots of similar search/merge functions
      "sonarjs/cognitive-complexity": ["warn", 25], // default is 15
      "sonarjs/no-duplicate-string": "off", // too noisy for test files and config
      "sonarjs/no-nested-conditional": "warn",
      "sonarjs/todo-tag": "off", // TODOs are tracked in CHECKLIST.md and oc-tasks
      "sonarjs/no-commented-code": "warn", // downgrade from error to warning
      "sonarjs/hashing": "off", // md5 used for content dedup, not security

      // ── General ─────────────────────────────────────────────
      "no-console": "off",
      "prefer-const": "error",
      "no-var": "error",
      eqeqeq: ["error", "always", { null: "ignore" }],
    },
  },

  // Deep nesting is inherent to plugin SDK register() and chokidar event handlers
  {
    files: ["index.ts", "src/ingest/watcher.ts"],
    rules: {
      "sonarjs/no-nested-functions": "off",
      "sonarjs/no-unenclosed-multiline-block": "warn",
      // index.ts uses inline import() for dynamic types in tool execute blocks
      "@typescript-eslint/consistent-type-imports": "off",
    },
  },

  // Test-specific overrides (more lenient)
  {
    files: ["tests/**/*.ts", "evaluation/**/*.ts"],
    rules: {
      "sonarjs/cognitive-complexity": "off",
      "sonarjs/no-nested-conditional": "off",
      "@typescript-eslint/no-explicit-any": "off",
      // Tests often have long similar assertion blocks
      "sonarjs/no-identical-functions": "off",
    },
  },
);
