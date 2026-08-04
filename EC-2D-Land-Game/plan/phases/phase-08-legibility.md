# Phase 8 — 2D legibility: legend, explainer toasts, labeled HUD, hover

**Status: planned — queued after phase 7 (final batch item; batch closes at 8)**

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
- [ ] Legend screenshot in this file; HUD labels render at default window
      size without clipping.
- [ ] Manual note: toasts appear once each, defer during parables, can be
      disabled; hover works.
- [ ] Autopilot clean; final integration run; plan-index updated; push.
