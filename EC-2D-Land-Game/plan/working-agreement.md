# Working Agreement — Three-Role Coding-Model Workflow

This project is built by **three separate sessions, each a distinct role and (recommended) a
distinct model.** The roles never run concurrently in the same working tree; every session
commits its own edits before the next begins. This structure exists because coding models mark
work "done" that isn't: honesty and scope control have to be **mechanical**, not voluntary. The
model that writes code never certifies its own work.

## Role → model mapping (guidance — adjust to the models you have)

This workflow is model-count-agnostic. What matters is capability and independence, not which
specific models are on hand.

- **Match the model to the phase.** A mid-capability coding model handles a well-scoped phase
  (≤ ~3 files, concrete acceptance criteria) reliably. Reserve a large, long-context model for
  greenfield/architecture work and for authoring precise phase files from a big spec.
- **The implementer is the strongest coding model appropriate to the phase's difficulty;** the
  planner can share that model (separate session, separate context).
- **The verifier should be a DIFFERENT model from the implementer whenever you have one** — a
  second model catches blind spots the implementer's model can't see in its own output. That
  independence is the whole point. Escalate a genuinely uncertain finding to your strongest model.
- **With only one model, still run three separate fresh sessions.** The verifier's independence
  then comes entirely from fresh context + the adversarial procedure below — lean harder on the
  mechanical gate, and distrust any "this looks fine" instinct (a same-model session wrote it).

| Role | Pick |
|------|------|
| **Planner** | A coding model with enough context to hold the spec; may share with Implementer |
| **Implementer** | Strongest coding model appropriate to the phase's difficulty |
| **Verifier** | A different model from the Implementer when available; otherwise a fresh session of the same model |

## The loop

```
PLANNER (writes/updates ONE phase file)
   → IMPLEMENTER (builds exactly that phase → awaiting verification)
      → VERIFIER (adversarial → ✅ complete, OR ⚠️ failed + files a defect)
         → (on failure) back to PLANNER (remediation phase) or IMPLEMENTER
```

One phase moves through all three roles before the next starts. Never pipeline multiple phases
through one role.

---

## Role 1 — PLANNER

**Owns:** `plan/plan-index.md`, `plan/architecture.md`, `plan/phases/*.md`. **Never writes app
code, never writes tests, never ticks an implementation checkbox.**

1. `git add -A && git commit -m "pre-plan <phase>: baseline"`; paste `git log --oneline -1`.
2. Read `plan/architecture.md` + the relevant part of the overall spec for the slice you are cutting.
3. Write **one** `plan/phases/phase-NN-<slug>.md` that carves a single-session slice. A phase is
   correctly sized when:
   - It touches **≤ 3 files** (split service work from wiring/UX into separate phases).
   - Every checklist item has a **machine-checkable acceptance command** — a `grep`, a named test,
     a parse check — no manual-only "looks right" items.
   - Canonical examples are written as **exact-equality assertions** the implementer must
     reproduce (`assert out == "..."`), never substring checks.
   - It names any **exact signatures/schema** the implementer should copy, not invent.
   - It lists the **invariants from `architecture.md`** the phase must not break.
   - It follows the shape of the sample phase file in `phases/` (Step 0 baseline, checkpoints
     where service+wiring split, named tests, a **Files** list, Step Z close-out, and an
     **Acceptance (verification session)** block).
4. Set the phase's row in `plan/plan-index.md` to `planned — ready to implement`.
5. Log a one-paragraph entry in `ai-updates.md`: what phase you cut and its blast radius (grep the
   codebase + `plan/phases/` for every symbol the phase touches; list the hits).
6. `git add -A && git commit -m "plan <phase>: <slug>"`.

**Planner may NOT** set a phase to `awaiting verification` or `✅ complete`, and may not edit
`plan/defects.md` (that is the verifier's).

## Role 2 — IMPLEMENTER

**Reads only: this file, the ONE phase file, and `plan/architecture.md`. Nothing else to start.**

1. `git add -A && git commit -m "pre <phase>: baseline"`; paste `git log --oneline -1`.
2. Run the test suite — record the baseline count in `ai-updates.md` (your final count must exceed
   it). If `git status` shows dirty files you did not create, STOP and report — don't commit them.
3. Implement **only** this phase. Touch **only** the files in its **Files** list. No refactors
   beyond the task, no renames, no dependency changes unless the phase says so.
4. **Plan files are READ-ONLY except two edits:** flip `[ ]`→`[x]` (children before parents — never
   tick a parent over an unticked child) and set this phase's Status to `awaiting verification`.
   NEVER reword or delete a checklist item. `git diff -- plan/` is checked in verification; any
   other change fails the session.
5. **A checkbox is ticked only after running its acceptance command and pasting the real output
   into `ai-updates.md`.** A claim without reproducible pasted output counts as NOT DONE. "Remove X,
   replace with Y" is two checkboxes. "Replace all call sites" means a repo-wide grep proves zero
   remain — run it, paste it.
6. Append the session summary + all pasted outputs to `ai-updates.md`.
7. `git add -A && git commit -m "<phase>: <one-line summary>"` — ONE commit per session so the
   verifier can diff exactly what you did.
8. Set the phase Status (phase file + index) to `awaiting verification`. **You NEVER set
   `✅ complete` — that is not your call.** You never edit `plan/defects.md`.

## Role 3 — VERIFIER (adversarial — fresh context; a different model from the implementer when available)

Your job is **to find where the implementer's claims are false.** Assume they are until you have
reproduced them. Owns `plan/defects.md` exclusively.

1. Fresh context. `git log --oneline` to find the implementer's baseline and work commits.
2. **Mechanical gate (all must pass):**
   - Test suite green; final count exceeds the pasted baseline.
   - Run **every** acceptance command in the phase file yourself; compare byte-for-byte against the
     outputs pasted in `ai-updates.md`. A paste that could not have come from the shown command is
     **fabrication** and fails the session outright.
   - `git diff <baseline>..HEAD -- plan/` shows ONLY checkbox flips + the Status line. Any reworded
     or removed checklist item fails the session.
   - `git status` clean of stray files; only the phase's Files list (+ `ai-updates.md`, plan files)
     changed.
3. **Adversarial analysis** — spot-check at least 3 ticked items at random against the code
   (including one sub-checkbox and one "test exists" claim), then attack the project's invariants
   (see `architecture.md` → Invariants, and `defects.md` → watch-list). A test that only passes
   because a fake deviates from the real dependency's behavior is a wrong test, not a pass. If a
   finding is genuinely uncertain, escalate that one item to a deep-dive session on your strongest
   model, or flag it in `ai-updates.md` for a human — never wave it through because you are unsure.
4. **Verdict — you are the ONLY authority for ✅:**
   - All good → set the phase Status (file + index) to `✅ complete (verified <date>)`.
   - Anything false → set Status to `⚠️ verification failed`, write the findings in `ai-updates.md`,
     and file each as a defect in `plan/defects.md` with the next D-number (severity, exact
     location, reproduction). The planner then cuts a remediation phase.
5. `git add -A && git commit -m "verify <phase>: <pass|fail>"`.

---

## Standing engineering rules (apply to every implementation phase)

1. **One phase per session. Never start the next.** Stop at a checkpoint if context runs low.
2. **Secrets from env only.** Never write a key/token/password into any file.
3. **Parameterized queries only.** Never interpolate input into a query string.
4. **Schema changes are additive and idempotent** (`CREATE ... IF NOT EXISTS` / `ADD COLUMN IF NOT
   EXISTS`); never destructively rewrite an existing migration.
5. **Everything testable offline.** No live external service required to run the tests; fake it.
   A live-only test is not acceptance.
6. **Delete means delete** — no deprecated wrappers, commented-out code, or unused imports left
   behind. Check touched files for now-unused imports and zero-caller functions.
7. **Update `ai-updates.md` every session** (newest on top): date, role, phase, files touched,
   decisions, and the pasted outputs of every verification command you ran.
