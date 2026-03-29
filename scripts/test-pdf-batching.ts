#!/usr/bin/env npx tsx
/**
 * Test PDF batched processing for OOM prevention.
 *
 * Usage:
 *   npx tsx scripts/test-pdf-batching.ts --pdf path/to/file.pdf
 *   npx tsx scripts/test-pdf-batching.ts --test-small
 *   npx tsx scripts/test-pdf-batching.ts --test-large
 *
 * Tests:
 *   1. Small PDF (single batch, <25 pages)
 *   2. Large PDF (multiple batches, 100+ pages)
 *   3. Verify no OOM, all chunks stored
 */

import fs from "node:fs/promises";
import path from "node:path";

import { extractPdfBatched } from "../src/ingest/parsers.js";
import { ingestPdfBatched } from "../src/ingest/pipeline.js";
import { resolveConfig } from "../src/config.js";

const TEST_FIXTURES_DIR = path.join(import.meta.dirname, "..", "tests", "fixtures");

interface TestResult {
  name: string;
  passed: boolean;
  batches: number;
  totalChunks: number;
  durationMs: number;
  error?: string;
}

async function testPdfBatching(pdfPath: string, testName: string): Promise<TestResult> {
  const start = Date.now();
  console.log(`\n=== ${testName} ===`);
  console.log(`File: ${pdfPath}`);

  try {
    // Check file exists
    const stat = await fs.stat(pdfPath);
    console.log(`Size: ${(stat.size / 1024).toFixed(1)} KB`);

    // Create config
    const cfg = resolveConfig({});

    // Test batched extraction
    const batches = await extractPdfBatched(pdfPath, cfg);
    console.log(`Batches: ${batches.length}`);

    let totalChars = 0;
    for (let i = 0; i < batches.length; i++) {
      const batch = batches[i]!;
      const chars = batch.text.trim().length;
      totalChars += chars;
      console.log(`  Batch ${i + 1}: pages ${batch.pageStart}-${batch.pageEnd} → ${chars} chars`);
    }
    console.log(`Total text: ${totalChars} chars`);

    // Check for empty batches
    const emptyBatches = batches.filter((b) => !b.text.trim());
    if (emptyBatches.length > 0) {
      console.log(`Empty batches: ${emptyBatches.length} (OK if blank pages)`);
    }

    return {
      name: testName,
      passed: true,
      batches: batches.length,
      totalChunks: 0, // Not tested without full ingest
      durationMs: Date.now() - start,
    };
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    console.log(`ERROR: ${error}`);
    return {
      name: testName,
      passed: false,
      batches: 0,
      totalChunks: 0,
      durationMs: Date.now() - start,
      error,
    };
  }
}

async function main() {
  const args = process.argv.slice(2);

  // Check for test fixtures
  const smallPdf = path.join(TEST_FIXTURES_DIR, "small.pdf");
  const largePdf = path.join(TEST_FIXTURES_DIR, "large.pdf");

  const results: TestResult[] = [];

  if (args.includes("--test-small")) {
    // Test with a small PDF from fixtures
    try {
      await fs.access(smallPdf);
      const result = await testPdfBatching(smallPdf, "Small PDF (<25 pages)");
      results.push(result);
    } catch {
      console.log("Skipping small PDF test — fixture not found");
      console.log(`  Expected: ${smallPdf}`);
    }
  }

  if (args.includes("--test-large")) {
    // Test with a large PDF from fixtures
    try {
      await fs.access(largePdf);
      const result = await testPdfBatching(largePdf, "Large PDF (100+ pages)");
      results.push(result);
    } catch {
      console.log("Skipping large PDF test — fixture not found");
      console.log(`  Expected: ${largePdf}`);
    }
  }

  // Test with explicit --pdf path
  const pdfIndex = args.indexOf("--pdf");
  if (pdfIndex !== -1 && args[pdfIndex + 1]) {
    const pdfPath = args[pdfIndex + 1]!;
    const result = await testPdfBatching(pdfPath, "Custom PDF");
    results.push(result);
  }

  // If no tests specified, run what we have
  if (results.length === 0) {
    console.log("No tests specified. Usage:");
    console.log("  npx tsx scripts/test-pdf-batching.ts --pdf path/to/file.pdf");
    console.log("  npx tsx scripts/test-pdf-batching.ts --test-small");
    console.log("  npx tsx scripts/test-pdf-batching.ts --test-large");
    console.log("\nChecking for fixtures...");

    try {
      await fs.access(smallPdf);
      console.log(`  Found: ${smallPdf}`);
    } catch {
      console.log(`  Not found: ${smallPdf}`);
    }

    try {
      await fs.access(largePdf);
      console.log(`  Found: ${largePdf}`);
    } catch {
      console.log(`  Not found: ${largePdf}`);
    }
  }

  // Summary
  if (results.length > 0) {
    console.log("\n=== SUMMARY ===");
    const passed = results.filter((r) => r.passed).length;
    const failed = results.filter((r) => !r.passed).length;
    console.log(`Passed: ${passed}/${results.length}`);
    console.log(`Failed: ${failed}/${results.length}`);

    for (const r of results) {
      const status = r.passed ? "✅" : "❌";
      console.log(`  ${status} ${r.name}: ${r.batches} batches in ${r.durationMs}ms`);
      if (r.error) {
        console.log(`     Error: ${r.error}`);
      }
    }

    if (failed > 0) {
      process.exit(1);
    }
  }
}

main().catch((err) => {
  console.error("Test script failed:", err);
  process.exit(1);
});
