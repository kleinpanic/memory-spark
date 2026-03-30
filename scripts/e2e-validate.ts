#!/usr/bin/env npx tsx
/**
 * E2E Validation Script for Docker Test Harness
 * Tests: Plugin load, Spark embed, LanceDB write
 * 
 * Usage: npx tsx scripts/e2e-validate.ts
 */

import { LanceDBBackend } from '../src/storage/lancedb.js';
import { createEmbedProvider } from '../src/embed/provider.js';
import { resolveConfig } from '../src/config.js';
import type { MemoryChunk } from '../src/storage/backend.js';

async function main() {
  console.log('═══════════════════════════════════════════');
  console.log('  E2E Validation: Docker + Spark + LanceDB');
  console.log('═══════════════════════════════════════════\n');

  // 1. Test config resolution
  console.log('▶ [1/4] Testing config resolution...');
  const cfg = resolveConfig({
    lancedbDir: '/home/node/.openclaw/data/testDbBEIR/lancedb',
  });
  console.log(`  ✓ lancedbDir: ${cfg.lancedbDir}`);
  console.log(`  ✓ sparkHost: ${cfg.sparkHost}`);

  // 2. Test Spark embedding
  console.log('\n▶ [2/4] Testing Spark embedding...');
  const embed = await createEmbedProvider(cfg.embed);
  console.log(`  ✓ Provider: ${embed.id} / ${embed.model}`);
  console.log(`  ✓ Dimensions: ${embed.dims}`);
  
  const testText = 'This is a test embedding for validation.';
  const embedding = await embed.embedQuery(testText);
  console.log(`  ✓ Embedding generated: ${embedding.length} dims`);
  console.log(`  ✓ First 5 values: [${embedding.slice(0, 5).map((v: number) => v.toFixed(4)).join(', ')}]`);

  // 3. Test LanceDB backend
  console.log('\n▶ [3/4] Testing LanceDB backend...');
  const backend = new LanceDBBackend(cfg);
  await backend.open();
  console.log('  ✓ Backend opened');

  // 4. Test write + search
  console.log('\n▶ [4/4] Testing write + search...');
  const testChunk: MemoryChunk = {
    id: 'test-e2e-validation',
    path: '/test/e2e-validation.ts',
    source: 'ingest',
    agent_id: 'test',
    start_line: 1,
    end_line: 10,
    text: testText,
    vector: embedding,
    updated_at: new Date().toISOString(),
    pool: 'agent_memory', // Use valid pool from POOL_VALUES
    category: 'fact',
    quality_score: 1.0,
    token_count: 10,
  };

  await backend.upsert([testChunk]);
  console.log('  ✓ Test chunk written');

  const results = await backend.vectorSearch(embedding, {
    query: testText,
    agentId: 'test', // Filter by agent_id instead of pool
    maxResults: 1,
  });
  console.log(`  ✓ Search returned ${results.length} result(s)`);

  if (results.length > 0 && results[0].chunk.id === testChunk.id) {
    console.log('  ✓ Retrieved correct chunk');
  } else {
    throw new Error('Search did not return expected chunk');
  }

  // Cleanup
  await backend.deleteById([testChunk.id]);
  console.log('  ✓ Test chunk cleaned up');

  await backend.close();

  console.log('\n═══════════════════════════════════════════');
  console.log('  ✅ ALL E2E TESTS PASSED');
  console.log('═══════════════════════════════════════════');
}

main().catch(err => {
  console.error('\n❌ E2E VALIDATION FAILED:', err);
  process.exit(1);
});
