#!/usr/bin/env bash
# Local CI — runs the full quality pipeline.
# Same checks as GitHub Actions CI workflow.
#
# Usage:
#   ./scripts/ci.sh          # full CI (typecheck + lint + format + knip + test + coverage)
#   ./scripts/ci.sh --quick  # quick CI (typecheck + lint + test, no coverage)

set -euo pipefail
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

step() { echo -e "\n${YELLOW}▸ $1${NC}"; }
pass() { echo -e "${GREEN}  ✓ $1${NC}"; }
fail() { echo -e "${RED}  ✗ $1${NC}"; exit 1; }

QUICK=false
[[ "${1:-}" == "--quick" ]] && QUICK=true

# ── Step 1: TypeScript type check ──────────────────────────────────────────
step "TypeScript type check"
npx tsc --noEmit && pass "0 type errors" || fail "Type errors found"

# ── Step 2: ESLint ─────────────────────────────────────────────────────────
step "ESLint (src + index)"
npx eslint src/ index.ts --max-warnings 20 && pass "0 errors" || fail "Lint errors found"

# ── Step 3: Prettier format check ─────────────────────────────────────────
if [[ "$QUICK" == "false" ]]; then
  step "Prettier format check"
  npx prettier --check 'src/**/*.ts' 'tests/**/*.ts' 'evaluation/**/*.ts' 'index.ts' && pass "All files formatted" || fail "Formatting issues found (run: npm run format)"
fi

# ── Step 4: Knip dead code check ──────────────────────────────────────────
if [[ "$QUICK" == "false" ]]; then
  step "Knip dead code analysis"
  npx knip && pass "No dead code" || fail "Dead code found"
fi

# ── Step 5: Vitest ─────────────────────────────────────────────────────────
if [[ "$QUICK" == "false" ]]; then
  step "Vitest with coverage"
  npx vitest run --coverage && pass "All tests pass with coverage" || fail "Tests failed"
else
  step "Vitest (no coverage)"
  npx vitest run && pass "All tests pass" || fail "Tests failed"
fi

# ── Step 6: FTS+WHERE validation ───────────────────────────────────────────
if [[ "$QUICK" == "false" ]]; then
  step "FTS+WHERE LanceDB validation"
  npx tsx tests/fts-where-test.ts && pass "FTS+WHERE works" || fail "FTS+WHERE broken"
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
if [[ "$QUICK" == "true" ]]; then
  echo -e "${GREEN}✅ Quick CI passed (typecheck + lint + test)${NC}"
else
  echo -e "${GREEN}✅ Full CI passed (typecheck + lint + format + knip + test + coverage + FTS)${NC}"
fi
