import * as fs from "node:fs/promises";
import * as path from "node:path";
import { exec } from "node:child_process";
import { promisify } from "node:util";
import * as os from "node:os";

const execAsync = promisify(exec);

const HOME = os.homedir();
const OC_PKG = path.join(HOME, ".local/share/npm/lib/node_modules/openclaw/package.json");
const OC_DOCS = path.join(HOME, ".local/share/npm/lib/node_modules/openclaw/docs");
const GIT_DOCS = path.join(HOME, ".local/srcs/gitclones/openclaw/docs");
const KB_DIR = path.join(HOME, ".openclaw/workspace-meta/memory/knowledge-base/openclaw-docs");

async function main() {
  console.log("[sync-rag] Starting doc sync...");

  try {
    const pkgData = await fs.readFile(OC_PKG, "utf-8");
    const pkg = JSON.parse(pkgData);
    const currentVer = pkg.version;
    console.log(`[sync-rag] OpenClaw installed version: ${currentVer}`);

    const installDir = path.join(KB_DIR, `installed-v${currentVer}`);
    
    await fs.mkdir(KB_DIR, { recursive: true });

    let alreadyIndexed = false;
    try {
      const stats = await fs.stat(installDir);
      alreadyIndexed = stats.isDirectory();
    } catch {
      alreadyIndexed = false;
    }

    if (alreadyIndexed) {
      console.log(`[sync-rag] Already indexed: installed-v${currentVer}`);
    } else {
      console.log(`[sync-rag] New version detected! Copying docs for v${currentVer}...`);
      await fs.mkdir(installDir, { recursive: true });
      await execAsync(`rsync -a --include="*.md" --include="*/" --exclude="*" "${OC_DOCS}/" "${installDir}/"`);
      console.log(`[sync-rag] ✅ Copied docs to installed-v${currentVer}/`);

      // Clean up old versions (keep newest 2)
      const dirs = await fs.readdir(KB_DIR);
      const installDirs = dirs.filter(d => d.startsWith("installed-v"));
      installDirs.sort(); // Sorting lexically is fine for now, or maybe sort by stat time if needed
      if (installDirs.length > 2) {
        const toRemove = installDirs.slice(0, installDirs.length - 2);
        for (const dir of toRemove) {
          console.log(`[sync-rag] Removing old version: ${dir}`);
          await fs.rm(path.join(KB_DIR, dir), { recursive: true, force: true });
        }
      }
    }

    // Refresh git-latest
    let gitHash = "unknown";
    try {
      const gitStat = await fs.stat(GIT_DOCS);
      if (gitStat.isDirectory()) {
        console.log(`[sync-rag] Syncing git-latest...`);
        const latestDir = path.join(KB_DIR, "git-latest");
        await fs.mkdir(latestDir, { recursive: true });
        await execAsync(`rsync -a --delete --include="*.md" --include="*/" --exclude="*" "${GIT_DOCS}/" "${latestDir}/"`);
        
        try {
          const { stdout } = await execAsync(`git -C "${GIT_DOCS}/.." rev-parse --short HEAD`);
          gitHash = stdout.trim();
        } catch { /* git hash extraction is optional */ }
        
        await fs.writeFile(path.join(latestDir, "_version.md"), `# OpenClaw git-latest as of ${new Date().toISOString()} (commit: ${gitHash})\n`);
        console.log(`[sync-rag] ✅ git-latest synced (commit: ${gitHash})`);
      }
    } catch {
      console.log(`[sync-rag] No git clone docs found at ${GIT_DOCS} — skipping git-latest sync`);
    }

    const versionsMd = `# OpenClaw Docs in RAG

| Directory | Version | Notes |
|-----------|---------|-------|
| installed-v${currentVer}/ | ${currentVer} | **Currently running** — matches production gateway |
| git-latest/ | ${gitHash} | Latest from git — may have unreleased features |

## Important
The **installed version (${currentVer})** is what the gateway is actually running.
Klein has custom modifications — do NOT assume git-latest matches production behavior.
`;
    await fs.writeFile(path.join(KB_DIR, "VERSIONS.md"), versionsMd);
    console.log("[sync-rag] Done.");
  } catch (err) {
    console.error("[sync-rag] Error:", err);
  }
}

main();
