# Production Integration Checklist

## Prerequisites (must all pass before enabling)

- [ ] Pool reindex complete (2838 files, pool column verified)
- [ ] Benchmark NDCG@10 ≥ 0.50 (vector_only or hybrid_no_reranker)
- [ ] Tier 2 pipeline tests pass (garbage rejection, token budget, security)
- [ ] Tier 3 pool isolation tests pass (WHERE pool= queries succeed)
- [ ] BEIR SciFact baseline stable (≥ 0.75 vector-only)

## Klein Gateway (broklein)

### Current State
- Plugin symlinked: `~/.openclaw/extensions/memory-spark → ~/codeWS/TypeScript/memory-spark`
- Config: `plugins.entries.memory-spark.enabled: false`
- Load path: already in `plugins.load.paths`
- Memory slot: `plugins.slots.memory: "memory-spark"`
- Spark host: `10.99.1.1`

### Steps
1. Ensure `test-data/lancedb` benchmarks pass
2. **Move production index to correct location:**
   - Production LanceDB path: `~/.openclaw/data/memory-spark/lancedb/`
   - Run: `MEMORY_SPARK_DATA_DIR=~/.openclaw/data/memory-spark npx tsx tools/indexer.ts`
   - This creates a FRESH production index with pool columns from all 20 agent workspaces
3. **Enable plugin via oc-restart:**
   ```bash
   cat > /tmp/oc-staged.json << 'EOF'
   {
     "plugins": {
       "entries": {
         "memory-spark": {
           "enabled": true
         }
       }
     }
   }
   EOF
   oc-restart request --staged /tmp/oc-staged.json --reason "Enable memory-spark plugin (benchmarks passed)" --session-id "$OPENCLAW_SESSION_KEY"
   ```
4. After Klein approves: `oc-restart apply`
5. Post-restart verify:
   ```bash
   openclaw status | grep memory-spark
   # Should show: memory-spark (enabled, memory slot)
   ```

### Validation (after enable)
- Send a test message to any agent
- Check agent response includes `<relevant-memories>` XML block
- Verify no errors in gateway logs: `journalctl --user -u openclaw-gateway -n 50 | grep memory-spark`

## Nicholas Gateway

### Current State
- memory-spark was removed from Nicholas config during stress testing
- Config backup at: `~/.openclaw/memory-spark.disabled.json` on Nicholas
- Nicholas has Spark access via WireGuard (10.99.1.1)

### Steps
1. Wait for Klein gateway to be stable for 24h
2. Re-add memory-spark to Nicholas config:
   - Add to `plugins.allow`, `plugins.load.paths`, and `plugins.entries`
   - Nicholas needs its own index OR can point at a shared LanceDB
   - Recommended: separate index per gateway (each gateway indexes its own agent workspaces)
3. Run indexer on Nicholas: `MEMORY_SPARK_DATA_DIR=~/.openclaw/data/memory-spark npx tsx tools/indexer.ts`
4. Enable via Nicholas's oc-restart flow

## Rollback Plan

If memory-spark causes issues after enabling:
1. Disable immediately: `jq '.plugins.entries["memory-spark"].enabled = false' ~/.openclaw/openclaw.json > /tmp/oc-disable.json`
2. `oc-restart request --staged /tmp/oc-disable.json --reason "memory-spark causing issues — disabling"`
3. The plugin's `enabled: false` state is safe — it stays loaded but inactive

## Post-Enable Monitoring

Watch for:
- **Context window bloat**: `<relevant-memories>` XML exceeding 2000 tokens
- **Spark timeouts**: embed/rerank calls taking > 10s
- **Capture noise**: agent_memory pool growing with garbage (check `memory_index_status` tool)
- **LanceDB write conflicts**: concurrent writes from multiple agents (the writeLock mutex should handle this)
