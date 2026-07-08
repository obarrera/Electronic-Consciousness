# Plan Index — EC-2D-Land (Electronic Consciousness game)

> The **index**: the phase table, statuses, and ordering. Details live in per-phase files under
> [`phases/`](phases/); architecture in [`architecture.md`](architecture.md); the defect register
> in [`defects.md`](defects.md); the three-role session rules in
> [`working-agreement.md`](working-agreement.md).

## How a phase moves (three roles, three sessions)

**Planner** cuts one single-session phase into `plan/phases/` → **Implementer** builds exactly
that phase and sets `awaiting verification` → **Verifier** (fresh context, ideally a different
model) does **adversarial analysis** and is the only role that sets `✅ complete`. Full rules +
role→model guidance: [`working-agreement.md`](working-agreement.md).

Statuses: `needs plan` → `planned — ready to implement` → `awaiting verification` →
`✅ complete (verified)` / `⚠️ verification failed`.

## Phase index

| Phase | Task | Status | Depends on |
|-------|------|--------|------------|
| 1 | [Crash-safety + population-integrity fixes](phases/phase-01-crash-safety-and-population-integrity.md) | awaiting verification | — |
| 2 | Neural-net wiring cleanup (dedupe imports, drop dead `create_model`, drop unused `matplotlib`) | needs plan | 1 |
| 3 | 3D world lifecycle fixes (`RecursiveEnvironment3D.layer` staleness, unbounded `objects` growth) | needs plan | 1 |
| 4 | Rendering + value hygiene (`EsotericSymbol` color scaling, `Element.get_color` default branch, `check_goal` energy clamp) | needs plan | 1 |
| 5 | Dead-code removal (orphan `move_goal`, dead `AIAgent3D.interact_with_objects`, unreachable `SolidShape3D` symbol branches) | needs plan | 1 |

Phases 2–5 are candidate slices proposed at scaffold time for a planner session to expand into
full phase files — narrow them further if any grows past ~3 files.

## Ordering

**1** ships first — it fixes the guaranteed crash (missing `AIAgent3D.die_and_rebirth`) and the
core population-integrity bug (duplicate agents in `ai_agents_2d`), so later phases aren't built
against known-broken foundations. **2–5** are independent of each other and of ordering beyond
depending on 1 (all touch code paths that assume the population list and 3D lifecycle are sane).

## Revision history

- **2026-07-07:** Scaffolded onto the existing codebase mid-review. Phase 1 written up from a
  prior code-review pass and implemented in the same session; phases 2–5 proposed as
  `needs plan` for a future planner session. Verification of Phase 1 by a fresh session is still
  outstanding.
