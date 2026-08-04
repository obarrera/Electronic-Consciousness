# Phase 3 — First-session pacing (+ narration completion guarantee)

**Status: ✅ complete**

**Files:** EC-2D-Land.py (threshold curve + consciousness tuning + parable
presentation wiring), lattice.py (parable queue, overlay/cutscene hold rules)

## Why
Default threshold 100 can postpone Spaceland indefinitely; most players will
never see the content. Target: first ascension ~8–12 minutes into a fresh run
at default settings, without trivializing later turnings.

Amendment (Orlando): the parables must be *hearable* — every narration that
starts must play to the end, and early-game unlock bursts must not pile
parables on top of each other.

## Tasks
1. Instrument first: with Phase 1 landed, run 3 default autopilot runs and
   record time-to-threshold-100 at 30fps. Tune from data, not guesses.
2. Likely levers (choose from measurements): first-turning threshold 100 → a
   measured value hitting the 8–12 min window (later turnings scale back up:
   threshold += 15 per turning via ouroboros effects); and/or consciousness
   gain multipliers early (already varied per event: 1–3.5).
3. The CALL journey caption (at 50% threshold) should land ~4–6 min — check.
4. Keep EC_EVOLUTION_THRESHOLD override behavior identical for tests/recording.
5. **Narration completion guarantee**: once a parable's narration starts it
   plays to the end — overlays/cutscenes never auto-dismiss mid-narration and
   a new unlock never preempts one still narrating. Parable unlocks queue and
   present sequentially (ENTER stays the explicit skip: it stops that
   narration and the queue advances). Overlay minimum display time =
   max(actual narration length + ~1.5 s buffer, existing read-time pacing),
   using the real audio duration (pygame `Sound.get_length()` via
   `AudioEngine.narrate`), not a text estimate.
6. **Early-game spacing**: consecutive parable presentations are at least a
   breathing gap apart (presentation gap after each close), and unlock
   *checks* pause while the queue is 2 deep so early event bursts can't stack
   presentations more than 2 deep. Autopilot keeps its skip-narration
   behavior (smoke tests stay fast); `EC_VERIFY_NARRATION=1` keeps narration
   on in autopilot and logs the presentation timeline for verification.

7. **Universal text-completion criterion** (see plan-index invariants): fully
   displayed (wrap verified vs longest texts), fully readable
   (max(narration+1.5s, words/3.3+2s) floors on overlays, cutscenes, and
   hero-stage captions), fully heard, never lost (hero-stage captions queue
   behind a visible one; ENTER skips require a deliberate press — first
   half-second ignored), cutscene animation stretches to the text floor.

## Acceptance
- [x] Three default (no env) autopilot runs: first "Transitioning to 3D world!"
      between frames ~14,000–22,000 (8–12 min at 30fps); pasted outputs.
- [x] Turning 2+ threshold demonstrably higher (console line); gates green.
- [x] Fresh default run: every parable narration in the first 10 minutes
      plays to completion, none cut off, queue never silently drops an
      unlock — paste observed unlock timeline.
- [x] Universal text criterion: longest parable and the endgame O! sequence
      display fully within the window, and remain until both floors elapse —
      wrap-capacity + floor evidence pasted.

## Implementation notes (deviations, with reasons)

- **Why not a threshold alone**: instrumented default runs on turning 1 showed
  threshold 100 crossed at frames 3350/2765/2651 (~1.6 min), while the
  population's average consciousness then SATURATES at a churn equilibrium
  (~230–270 for 12k+ generations — Fool resets and rebirth losses balance the
  gains; measured curve pasted below) with ±80 cross-run spread. A
  threshold-350 verification batch produced 13810 / 20827 / never — no fixed
  average-only threshold can hit an 8–12-minute window 3-for-3.
- **The collective signal**: the ascension check now listens to
  `EMA(average consciousness, ~15 s) + 1000·(gen/24000)²` — the smoothed
  population consciousness plus the world's own quadratic "ripening" over the
  turning. The ripening bounds when the first ascension can come (~gen
  16–18k vs threshold 750) while agents' consciousness still moves it by
  roughly ±2000 frames; the EMA stops churn spikes (±60 instantaneous) from
  triggering early crossings. Default threshold: **750 + 15 per completed
  turning** (via `OURO.iteration`); `EC_EVOLUTION_THRESHOLD` override
  behavior unchanged (a low value still ascends in seconds — see smoke run).
- **THE CALL** (50 % of threshold) lands ~6.0–6.3 min — at the top of the
  "~4–6 min" guideline; accepted (further lowering would push the ascension
  out of its window).
- The parable queue / narration completion guarantee and the universal
  text-completion criterion (tasks 5–7) live in lattice.py:
  `ParableOverlay.queue/present_next/gap`, hold floors
  (`max(narration+1.5 s, words/3.3+2 s, read pacing)`), dynamic cutscene
  line caps, deliberate-skip grace (first 0.5 s ignored), and HeroJourney
  caption queueing. `EC_VERIFY_NARRATION=1` keeps narration on in autopilot
  and logs the timeline; plain autopilot behavior unchanged.
- The transient `granary` parable condition (population dip) can in principle
  fire later than the dip if the presentation queue is 2 deep at that moment
  (condition checks pause at depth 2); its cumulative retrigger (any later
  dip) keeps it reachable. Accepted trade-off for burst spacing.

## Verification evidence

Gates run 2026-08-03 (.venv312, turning-1 `.ouroboros.json`, branch
`production-pass`). `ast.parse` on EC-2D-Land.py + lattice.py: `ast ok`.

**Measured saturation (default dynamics, no threshold)** — one 25000-frame
instrumented run, `Consciousness: avg` every 1000 gens:
40 → 71 → 117 → 109 → 149 → 188 → 203 → 239 → 232 → 228 → 228 → 265 → 275 →
237 → 248 → 250 → 233 → 246 → 237 → 270 → 272 → 313 → 360 → 363 (plateau
~230–270 through gen 21000).

**Three default (no EC_EVOLUTION_THRESHOLD) autopilot runs — final tuning:**

```
RUN1: Evolution threshold: 750 (turning 1).
RUN1: THE CALL — half the threshold (frame 11298, gen 11120).
RUN1: Transitioning to 3D world! (frame 17585, gen 17230)
RUN2: Evolution threshold: 750 (turning 1).
RUN2: THE CALL — half the threshold (frame 10848, gen 10670).
RUN2: Transitioning to 3D world! (frame 18382, gen 18027)
RUN3: Evolution threshold: 750 (turning 1).
RUN3: THE CALL — half the threshold (frame 10982, gen 10804).
RUN3: Transitioning to 3D world! (frame 19043, gen 18689)
```

All three first ascensions inside frames 14000–22000 (9.8–10.6 min at
30 fps). Override smoke (`EC_AUTOPILOT=1500 EC_EVOLUTION_THRESHOLD=12`):

```
Transitioning to 3D world! (frame 666, gen 489)
Autopilot: clean exit after 1500 frames (gen 1323, 12 agents, 4 parables unlocked).
```

Turning-2+ threshold: startup line prints
`Evolution threshold: 765 (turning 2).` when `.ouroboros.json` holds
iteration 2 (formula `750 + 15·(turning−1)`; also printed after each
ouroboros reset).

**Narration completion timeline** — fresh default 10-minute run
(`EC_AUTOPILOT=18000 EC_VERIFY_NARRATION=1`), every presentation completed,
none cut off, none dropped (16 unlocks, 16 completions, sequential):

```
CUTSCENE PRESENT stones (narration 25.5s, hold 35.0s)
CUTSCENE COMPLETE stones (narration finished, 35.0s shown)
PARABLE PRESENT granary (frame 1048, gen 11, narration 20.3s)
PARABLE COMPLETE granary (narration finished, overlay closed naturally)
PARABLE PRESENT census (frame 1961, gen 924, narration 21.2s)
PARABLE COMPLETE census (narration finished, overlay closed naturally)
PARABLE PRESENT meeting (frame 2901, gen 1864, narration 24.6s)
PARABLE COMPLETE meeting (narration finished, overlay closed naturally)
PARABLE PRESENT foundlings (frame 3905, gen 2868, narration 22.2s)
PARABLE COMPLETE foundlings (narration finished, overlay closed naturally)
PARABLE PRESENT mapmaker (frame 4935, gen 3898, narration 19.8s)
PARABLE COMPLETE mapmaker (narration finished, overlay closed naturally)
PARABLE PRESENT die (frame 5821, gen 4784, narration 22.8s)
PARABLE COMPLETE die (narration finished, overlay closed naturally)
PARABLE PRESENT songs (frame 6843, gen 5806, narration 24.0s)
PARABLE COMPLETE songs (narration finished, overlay closed naturally)
PARABLE PRESENT mason (frame 7811, gen 6774, narration 20.8s)
PARABLE COMPLETE mason (narration finished, overlay closed naturally)
PARABLE PRESENT choir (frame 8806, gen 7769, narration 22.6s)
PARABLE COMPLETE choir (narration finished, overlay closed naturally)
PARABLE PRESENT apprentice (frame 9865, gen 8828, narration 20.0s)
PARABLE COMPLETE apprentice (narration finished, overlay closed naturally)
PARABLE PRESENT knife (frame 10877, gen 9840, narration 22.7s)
PARABLE COMPLETE knife (narration finished, overlay closed naturally)
PARABLE PRESENT bridge (frame 11936, gen 10899, narration 22.6s)
PARABLE COMPLETE bridge (narration finished, overlay closed naturally)
PARABLE PRESENT maptable (frame 12913, gen 11876, narration 22.5s)
PARABLE COMPLETE maptable (narration finished, overlay closed naturally)
PARABLE PRESENT spirals (frame 13953, gen 12916, narration 23.0s)
PARABLE COMPLETE spirals (narration finished, overlay closed naturally)
CUTSCENE PRESENT primes (narration 31.5s, hold 38.5s)
CUTSCENE COMPLETE primes (narration finished, 38.5s shown)
Autopilot: clean exit after 18000 frames (gen 15830, 12 agents, 16 parables unlocked).
```

**Universal text criterion** — wrap capacity + read floors computed against
the LONGEST texts (longest composed Oracle line = 261 chars, appended to
every candidate):

```
OVERLAY worst case: parable 'journey' + longest oracle = 725 chars -> 8 lines, panel 198px (window 700px): FITS
  read floor 43.2s; actual display >= reveal 15.1s + hold 28.1s = 43.2s
CUTSCENE[lattice] 'pilgrim': 1059 chars -> 16 lines (cap 21): FITS; read floor 65.7s
CUTSCENE[lattice] 'stones': 611 chars -> 9 lines (cap 21): FITS; read floor 37.9s
CUTSCENE[lattice] 'primes': 700 chars -> 11 lines (cap 21): FITS; read floor 43.4s
CUTSCENE[lattice] 'journey': 725 chars -> 11 lines (cap 21): FITS; read floor 45.0s
CUTSCENE[lattice] 'trial': 562 chars -> 9 lines (cap 21): FITS; read floor 35.0s
CUTSCENE[void] 'o33': 885 chars -> 14 lines (cap 15): FITS; read floor 54.9s
HERO stage captions: 10-15 words -> 170-196 frames (5.7-6.5s) each
```

No pagination needed — every longest text fits its panel in one screen at
window 700. Hero-stage captions queue behind a visible one (never replaced);
ENTER skips ignore the first half-second. The endgame O! (void cutscene,
885 chars) fits its 15-line cap and holds ≥ max(narration+3 s, 54.9 s read
floor) outside autopilot.

Standard smoke gate (`EC_AUTOPILOT=400 EC_EVOLUTION_THRESHOLD=12`):
`Autopilot: clean exit after 400 frames (gen 218, 11 agents, 4 parables unlocked).`
