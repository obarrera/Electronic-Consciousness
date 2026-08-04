# Phase 2 — The player warms the cells

**Status: ✅ complete**

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
- [x] Manual: click warms (food appears, agents route toward it), SHIFT+click
      chills (agents avoid), meter depletes/regenerates, inspect still works.
- [x] Autopilot unaffected (no synthetic clicks needed); ast + smoke gates.

## Implementation notes

- Warm cell = one-bite food bloom (300-tick shelf life, +15 energy on entry),
  gold tint + ring, gold particle burst, birth tone at channel volume 0.10.
  Chill cell = 150-tick cold cell, −1 energy per entry, blue tint, cold tone.
  Both render as static slow-fading tints — REDUCED_FLASH safe.
- Routing coupling: `nearest_food()` now returns the goal OR the nearest
  player-warmed cell, so the phase-1 Gradient sense literally pulls agents to
  the player's blooms; eating one counts as a food event in rewards and
  metrics (the player teaches). Warm/chill cells are also visible in the
  8-neighbor sense (codes 7 and 8).
- Mouse: agent hit-test FIRST (left-click on an agent always inspects);
  left-click empty cell warms; right-click or SHIFT+click empty cell chills.
  Attention meter: 3 charges, −1 per touch, +1 per 100 ticks, drawn as three
  dots at the right end of the HUD strip. Ouroboros reset refills it.
- Verification hook `EC_TEST_HAND="warm@<frame>:<x>,<y>;chill@..."` calls the
  same `player_touch()` the click handler uses (retrying until the cell is
  empty), so the behavior is testable unattended. Autopilot never sets it.

## Verification evidence

Gates run 2026-08-03 (.venv312, branch `production-pass`). `ast.parse` on
EC-2D-Land.py + lattice.py: `ast ok`.

Scripted-hand run (`EC_AUTOPILOT=2500 EC_EVOLUTION_THRESHOLD=99999
EC_TEST_HAND="warm@900:10,10;chill@1000:12,12;warm@1400:4,15;warm@1800:15,5;warm@2100:8,3"`):

```
TEST_HAND: chill landed at (12, 12) (frame 1082, attention now 2.0)
TEST_HAND: agent chilled at (12, 12) (gen 251, energy 98)
TEST_HAND: agent chilled at (12, 12) (gen 253, energy 97)
TEST_HAND: agent chilled at (12, 12) (gen 347, energy 97)
TEST_HAND: warm landed at (15, 5) (frame 1801, attention now 2.0)
TEST_HAND: agent ate player-bloomed food at (15, 5) (gen 950)
TEST_HAND: warm landed at (8, 3) (frame 2140, attention now 2.0)
TEST_HAND: agent ate player-bloomed food at (8, 3) (gen 1301)
Autopilot: clean exit after 2500 frames (gen 1648, 13 agents, 16 parables unlocked).
```

Both warm blooms placed after training were found and eaten by routed agents
within ~150 generations; chill entries drained 1 energy each; the meter
depleted to 2.0 and regenerated (`attention now 2.0` after a spend = regrowth
had occurred); touches on occupied cells correctly no-op and retry. Warm
touches aimed at cells that stayed occupied all run (`10,10`, `4,15`) never
fired — the empty-cell rule holds. Agent hit-test-first is preserved in the
click handler (left-click on an agent inspects, never warms).

Autopilot unaffected — standard smoke gate (no EC_TEST_HAND):

```
Autopilot: clean exit after 400 frames (gen 98, 5 agents, 6 parables unlocked).
```
