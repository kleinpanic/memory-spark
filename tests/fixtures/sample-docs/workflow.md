# Workflow Documentation

## Task Management
All work is tracked through `oc-tasks`. Before starting multi-step work:
1. Create a task: `oc-tasks add "description" -a agent -p priority`
2. Dispatch it: `oc-tasks dispatch <id>`
3. Log progress: `oc-tasks comment <id> "progress"`
4. Complete: `oc-tasks done <id>`

## Config Changes
Config edits follow a strict protocol:
1. Schema lookup: `gateway config.schema.lookup <path>`
2. Stage changes to `/tmp/oc-staged.json`
3. Request approval: `oc-restart request --staged /tmp/oc-staged.json`
4. Wait for human approval
5. Apply only after approval

## Memory Architecture
- `MEMORY.md` — long-term agent memory index
- `memory/` — daily notes and archives
- `mistakes/` — error tracking with detailed write-ups
- `MISTAKES.md` — lean index of all mistakes
