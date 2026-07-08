# Getting the best out of coding models — a worked setup

A standalone, copy-me template showing how to structure a software project so that
local/hosted **coding models** produce reliable, verifiable work instead of confident-looking
slop. It is model-agnostic: it works with one model or several, small or large.

The core idea, learned the hard way: a coding model succeeds on **one narrow, tightly
specified, machine-checkable task at a time**, and fails on open-ended ones. So you don't hand
it "build the app" — you hand it "implement exactly this phase, touch only these files, and
make this exact command print this exact output." And the model that writes the code is **not**
allowed to certify its own work, because it will cheerfully mark things done that aren't.

## The three roles

Every unit of work (a "phase") moves through three **separate sessions**, each a distinct role:

1. **Planner** — carves the next single-session phase out of the overall spec into one phase
   file: exact files to touch (≤ ~3), exact signatures, and an acceptance command for every
   checklist item. Writes no code.
2. **Implementer** — implements *only* that one phase, touching *only* its listed files, and
   ends at `awaiting verification`. **Never marks its own work complete.**
3. **Verifier** — a fresh session (a *different model* if you have one) that does
   **adversarial analysis**: it assumes the implementer's claims are false until it reproduces
   them, re-runs every acceptance command, diffs the plan files, attacks the project's
   invariants, and is the **only** role allowed to mark a phase `✅ complete`.

```
Planner (writes one phase file)
  → Implementer (builds it → "awaiting verification")
    → Verifier (adversarial → "✅ complete"  OR  "⚠️ failed" + files a defect)
       → on failure, back to Planner (remediation phase) or Implementer
```

One phase goes through all three roles before the next phase starts. Don't pipeline several
phases through one role.

## Why the split works

- **Narrow phases** keep the task inside what a mid-sized model can actually hold and finish.
- **Machine-checkable acceptance** (a `grep`, a named test, a parse check) removes "looks
  right" judgment calls that invite false completion.
- **A separate verifier** breaks the conflict of interest — the writer never grades its own
  homework. When the verifier is a *different* model, it also catches blind spots specific to
  the implementer's model.
- **Plan files are read-only to the implementer** except ticking a box or setting a status, so
  scope can't quietly creep and checklist items can't be reworded into weaker claims.

## Picking models for the roles (capability, not brand)

The workflow is model-count-agnostic. Match capability to the job:

| Role | Pick |
|------|------|
| **Planner** | A coding model with enough context to hold the spec and emit precise phase files. May share a model with the Implementer. |
| **Implementer** | The strongest coding model appropriate to the phase's difficulty. Long-context models shine on greenfield/architecture; a mid model is fine for a well-scoped phase. |
| **Verifier** | A **different** model from the Implementer when you have one — independence is the point. With only one model, still run it as a **separate fresh session** and lean harder on the mechanical checks. |

Rule of thumb: reserve your biggest model for greenfield/architecture and for authoring phase
files from a large spec; a smaller model can implement a phase that's already been sliced thin
and fully specified.

## What's in this folder (the layout to copy)

```
example-usage/
├── README.md              ← this guide
├── CLAUDE.md              ← the entry instructions a project drops in (points at plan/)
└── plan/
    ├── working-agreement.md   ← the rules for all three roles (the heart of the method)
    ├── plan-index.md          ← phase table + statuses + ordering
    ├── architecture.md        ← one place for stack/schema/invariants (read by every session)
    ├── defects.md             ← verifier-owned register of anything found false
    └── phases/
        └── phase-01-*.md      ← one fully-worked sample phase, in the exact format to reuse
```

The files use a tiny sample project — a **URL-shortener API** — purely to make the phase
format concrete. Swap in your own architecture and phases; the *structure and rules* are the
reusable part.

## How to run it

1. **Start a Planner session.** It reads `plan/architecture.md` + the relevant part of your
   spec, writes `plan/phases/phase-NN-*.md`, and sets that phase to `planned` in the index.
2. **Start an Implementer session.** It reads the working agreement, that one phase file, and
   `architecture.md` — nothing else — implements, pastes its verification output into an
   updates log, commits, and sets the phase to `awaiting verification`.
3. **Start a Verifier session (fresh, ideally a different model).** It reproduces every pasted
   output, diffs `plan/`, spot-checks ticked items adversarially, and sets `✅ complete` or
   files a defect and sends it back.

Each session commits its own work so the next can diff exactly what changed.

> A fuller, real-world instance of this same method (dozens of phases, a live defect register,
> and the audit history that motivated the adversarial verifier) lives in the sibling
> `example-plan/` folder — look there when you want to see it at scale.
