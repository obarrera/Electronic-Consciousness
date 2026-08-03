# Phase 2 — The player warms the cells

**Files:** EC-2D-Land.py (events + effects), lattice.py (small helpers ok)

## Why
"Who warms the cells?" — the player becomes the layer above: attention that
changes the lattice, per the book's prologue.

## Tasks
1. Left-click empty cell: WARM it — bloom food there (+ gentle gold particle
   ring, existing "birth" tone at low vol). Existing click-on-agent inspect
   stays (agent hit-test first).
2. Right-click (or SHIFT+click) empty cell: CHILL it — temporarily cold cell
   agents avoid (drains 1 energy on entry, fades after ~150 ticks, blue tint).
3. Budget so it stays a nudge, not a god-mode: warmth costs a regenerating
   "attention" meter (e.g. 3 charges, +1 per 100 ticks) shown as small dots in
   the HUD strip. Chill costs the same meter.
4. Learning coupling (with Phase 1): player-bloomed food counts in rewards —
   the player literally teaches.
5. Help bar + README controls table updated.

## Acceptance
- [ ] Manual: click warms (food appears, agents route toward it), SHIFT+click
      chills (agents avoid), meter depletes/regenerates, inspect still works.
- [ ] Autopilot unaffected (no synthetic clicks needed); ast + smoke gates.
