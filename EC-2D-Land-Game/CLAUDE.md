# AI Instructions — EC-2D-Land (Electronic Consciousness game)

Full method + rationale: [`plan/README.md`](plan/README.md).

## This project uses a THREE-ROLE workflow — read before doing anything

Work moves through three **separate sessions**, each a distinct role. The model that writes
code never certifies its own work.

1. **PLANNER** — cuts one single-session phase out of the backlog into `plan/phases/`; keeps
   `plan/plan-index.md` and `plan/architecture.md` current. Writes no code.
2. **IMPLEMENTER** — implements exactly one planned phase, touching only its listed files; ends
   at `awaiting verification`. **Never self-certifies `✅ complete`.**
3. **VERIFIER** — fresh context, ideally a different model; does **adversarial analysis**:
   reproduces every pasted output, diffs `plan/`, attacks the invariants, spot-checks ticked
   items. Only role that sets `✅ complete`; owns `plan/defects.md`.

Full rules and the role→model guidance are in [`plan/working-agreement.md`](plan/working-agreement.md).
Do not deviate from the lifecycle — it exists because coding models mark things done that
aren't, so honesty and scope control must be mechanical, not voluntary.

## Where to look

| What | Where |
|------|-------|
| The method + why it works + how to run each session | [`plan/README.md`](plan/README.md) |
| Rules for all three roles (the working agreement) | [`plan/working-agreement.md`](plan/working-agreement.md) |
| Stack, in-memory data model, config, and the invariants every phase must preserve | [`plan/architecture.md`](plan/architecture.md) |
| Phase table, statuses, ordering | [`plan/plan-index.md`](plan/plan-index.md) |
| Per-phase task files (written by the planner) | [`plan/phases/`](plan/phases/) |
| Defect register (verifier-owned) | [`plan/defects.md`](plan/defects.md) |
| Session changelog (every session appends, newest on top) | `ai-updates.md` (create on first session) |

**Scope note:** this `plan/` covers only the `EC-2D-Land-Game/` subdirectory of the
`Electronic-Consciousness` repo — the rest of the repo is an unrelated markdown thesis. Run all
commands below from `EC-2D-Land-Game/`.

## Session start (all roles)

1. Read `plan/working-agreement.md` for your role, then `plan/plan-index.md`.
2. Load **only** what your role needs (implementers: your one phase file + `plan/architecture.md`
   — nothing else to start).
3. Commit a baseline **before** any edits; one commit per session at the end.
4. Append your session to `ai-updates.md` with the **pasted** output of every verification
   command you ran. A claim without reproducible pasted output counts as NOT DONE.
