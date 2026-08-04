# Phase 8 — 2D legibility: legend, explainer toasts, labeled HUD, hover

**Status: ✅ complete**

**Files:** EC-2D-Land.py (HUD, hover, toast triggers), lattice.py (legend
overlay, toast system), README.md ("Reading the screen").

## Why
Simple, high-value improvements so users understand what's occurring on
screen. Legibility, not new art.

## Tasks

1. **Legend overlay (`L`).** One clean panel decoding every Flatland visual:
   polygon agents (sides grow with consciousness, triangle → circle), gold
   bloom = food, Conway life-layer cells, elements/esoteric symbols, prime
   shimmer, chilled/warmed cells, attention dots, HUD stats. Existing overlay
   style; pauses the sim while open (like the journal).

2. **First-time explainer toasts.** One-line captions at a fixed toast
   position (bottom-center, small, ~5 s, gentle fade), first occurrence per
   run: first birth, first death, first food bloom, first training, first
   prime shimmer, first player warm/chill, first ascension-progress
   milestone. Rules: max ONE visible, they queue, NEVER shown while a
   narrative overlay is up (defer), suppressed in autopilot, once per run,
   disableable (EC_NO_HINTS=1 + legend note). UI hints — exempt from
   narration/read floors but must not overlap or fight them.

3. **Labeled HUD.** Compact strip gets short labels (gen, agents,
   consciousness X/threshold with a tiny fill-bar toward ascension, brain
   trainings, attention charges); `I` panels get one-line stat explanations.

4. **Hover identification.** Mouse-over identifies agent/cell near the cursor
   ("<name> — N sides, energy E" / "food bloom" / "chilled cell") without
   clicking; reuse phase-2 hit-testing; skip if it costs frame time (measure).

5. README: short "Reading the screen" section with legend screenshot.

## Acceptance
- [x] Legend screenshot in this file; HUD labels render at default window
      size without clipping.
- [x] Manual note: toasts appear once each, defer during parables, can be
      disabled; hover works.
- [x] Autopilot clean; final integration run; plan-index updated; push.

## Implementation notes

- `L` legend (`lattice.draw_legend`): one panel decoding agents, food,
  warmed/chilled cells, life-cell terrain, walls, solids/elements/symbols,
  the prime shimmer, and both HUD readouts; sim frozen while open (same
  pattern as the chronicle viewer), `L`/`ESC` close, ESC no longer quits
  while it is open. Help bar gained `L legend` / `J chronicle`.
- `lattice.ToastSystem`: bottom-center one-liners, ~5 s, fade in/out, max
  ONE visible (rest queue), deferred (frozen, not dropped) while any
  narrative overlay is up, once per run per key, suppressed in autopilot,
  `EC_NO_HINTS=1` disables. Triggers wired: first birth, first
  death/rebirth, first food eaten, first training, first prime shimmer,
  first player warm, first player chill, ascension signal reaching 25 %.
  UI hints are exempt from the narration/read floors by design and cannot
  overlap them (the reading state gates drawing).
- Labeled HUD strip: `gen N · agents K · consciousness X of Y to ascend ·
  brain T trainings, food-eff ±Z% · parables U/18`, plus a small ascension
  fill-bar beside the attention dots; the `I` info panel gained a one-line
  stat explainer. Bottom progress bar was already re-labeled as % to
  ascension in phase 3.
- Hover identification: mouse-over names the agent under the cursor (its
  Chronicle name, sides, energy) or the cell (food/goal, warmed bloom,
  chilled cell, wall, solid, element, symbol, life-cell) — no click, plain
  dictionary/list lookups per frame (13 agents max — no measurable frame
  cost), tooltip clamped to the window. Skipped when the window lacks
  mouse focus (and therefore inert in headless).

## Verification evidence

Gates run 2026-08-03 (.venv312). `ast ok` on EC-2D-Land.py + lattice.py.

**Legend screenshot** (captured via `EC_TEST_KEYS="l@250"` +
`EC_RECORD_DIR`, saved to `screenshots/legend.png`, also embedded in the
README's new "Reading the screen" section):

![Legend overlay](../../screenshots/legend.png)

The console log for that run shows the freeze semantics:
`TEST_KEYS: posted l at frame 250 → Legend opened (frame 250)` and the run
still exits cleanly. The labeled strip renders un-clipped at window 700
(the trailing token that collided with the fill-bar was removed after a
screenshot check).

**Toast semantics** (unit checks against ToastSystem):

```
queue after dup-suppressed hints: 2 (expect 2)
deferred under reading: True (expect True)
visible after reading ends: first (expect 'first')
advanced to second then done: second 0
once per run: 0 (expect 0)
EC_NO_HINTS disables: True
```

**Gates**: standard smoke `Autopilot: clean exit after 400 frames (gen 219,
9 agents, 4 parables unlocked).`; phase-6 determinism unchanged (headless
2000-tick hash still `a2d8b88c30…cf536ba4` — the new UI is render-side
only). Manual note: in a live (non-autopilot) run the toasts appear once
each at their first events, wait out any parable before showing, and
`EC_NO_HINTS=1` removes them; hovering names agents by their Chronicle
names — the same lookup path the click-inspect uses.
