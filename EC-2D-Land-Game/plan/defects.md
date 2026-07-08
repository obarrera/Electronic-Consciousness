# Defect Register — EC-2D-Land (Electronic Consciousness game)

> **Owned by the VERIFIER role exclusively.** Planners and implementers never edit this file —
> not even to mark a defect fixed (a completion claim is not the implementer's to certify). When
> a remediation phase closes a defect, the verifier that certifies that phase marks it
> `FIXED (Phase N)` here.

Statuses: **OPEN** / **FIXED (Phase N)** / **REGRESSED** (fixed, then partially undone).

## Verified implementation state

| Phase | Claimed | Verifier verdict |
|-------|---------|------------------|
| 1 | Crash-safety + population-integrity fixes — awaiting verification | No phase has reached verification yet. |

## Defect register

_(empty — no verifier pass has run yet)_

| ID | Severity | Status | Defect | Where |
|----|----------|--------|--------|-------|
| — | — | — | — | — |

## Invariant watch-list (what the verifier attacks every phase)

Mirror of the invariants in [`architecture.md`](architecture.md) — the checklist the verifier
runs against every phase's diff, filing a D-number on any violation:

- A duplicate `AI_Agent` reference in `ai_agents_2d`, or the list exceeding `MAX_AGENTS`.
- Any grid-placement `while True:` search with no attempt cap (a full grid should give up, not
  hang the process).
- Any loop over `ai_agents_2d` directly (not `ai_agents_2d[:]`) whose body can mutate the list.
- Any `self.energy +=` not clamped to 100.
- Any external resource load (audio, model/asset file) that can raise before the game loop
  starts, uncaught.
- `RecursiveEnvironment3D.objects` growing without bound across repeated layer/transcendence
  events at the same layer.

> This list is deliberately identical to `architecture.md`'s Invariants section — the verifier
> should not need to re-derive it from scratch.
