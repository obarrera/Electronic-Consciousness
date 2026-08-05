# Phase 10 — Audio UX + the unseen hand (indie-practice pass)

**Status: awaiting verification**

**Files:** lattice.py (variants, bowls, UI tones, master volume, help bar),
EC-2D-Land.py (volume keys, UI blips, denied thud, drag-to-draw, hand halo,
breathing warmth rings), README.md (controls)

## Why
Standard indie-game audio/UX practice, applied where the game had gaps:

- **Repetition fatigue** — the ear tunes out an identical sample by the third
  hearing; frequent sounds need variants (industry default: 3+, slight
  pitch/level spread).
- **Reward moments deserve reward timbres** — parable/prime were plain sine
  chords; a singing-bowl strike (inharmonic partials ~1/2.71/4.72) reads as
  ceremony, matching the game's meditative Gateway identity.
- **Silent UI is broken UI** — pause/help/speed/volume/mute had no acoustic
  acknowledgment; a refused action (attention empty) failed silently.
- **No volume control** — mute was the only lever; players who find a game
  loud just quit it.
- **The player fantasy** ("the unseen hand") had no embodiment: a single
  click per cell, no cursor presence, no stroke.

All sound remains procedurally generated (numpy) — no external assets were
pulled in: nothing royalty-free matched the game's self-generated ethos
better than extending the existing synth, and it keeps the repo
license-clean by construction.

## Tasks
1. **Variants**: birth/death/footsteps ×3 each (detune ×0.988–1.013, level
   0.88–1.0), picked by a dedicated `random.Random(7)` so humanization never
   touches the sim's deterministic streams.
2. **`_bowl()`**: singing-bowl synth (partials 1/2.71/4.72, higher partials
   decay faster, same declick + Hemi-Sync shimmer) for parable and prime.
3. **UI tones**: `ui` (soft 880 Hz blip, 60 ms cooldown) on SPACE/H/M/[/]/+/-;
   `denied` (low 140 Hz thud) when a touch is refused with attention empty —
   suppressed (`quiet=True`) during drag strokes so a dry drag doesn't drum.
4. **Master volume**: `[` / `]` step 0.1, `EC_VOLUME` env for the start
   value; multiplies every output (bed, binaural, surf, one-shots, emitters,
   narration); help bar shows `VOL n%`; README controls updated.
5. **Drag-to-draw** (the unseen hand): holding the mouse paints each new cell
   crossed with the same touch (warm; chill via right/SHIFT), attention
   permitting — the player draws strokes into the world instead of poking it.
6. **Hand halo**: a soft breathing glow rides the cursor (gold; chill blue
   while SHIFT held or a chill stroke is live), dimming to a shadow when
   attention is spent — reach and cost readable at a glance. Warmed cells'
   rings breathe (~0.3 Hz, ±2 px, phase-shifted per cell). Both
   REDUCED_FLASH safe (slow, small, no luminance jumps).

## Acceptance
- [ ] All tones (incl. variants, bowls, ui, denied) start/end at exactly 0,
      no clipping.
- [ ] Master volume scales one-shots, beds, emitters, narration; clamps 0..1.
- [ ] Drag across N empty cells warms up to attention-budget cells; agents
      under the cursor are never painted over; no denied-spam during drags.
- [ ] EC_TEST_HAND scripted touches still land (quiet param default False).
- [ ] Standard smoke gate passes.

## Verification evidence (implementer's run, 2026-08-05, .venv312)

`ast ok` × 2. Tone sweep: 19 buffers (all variants) — every start=0, end=0,
peak < 6000/32767 (`ALL CLEAN`). Master gain: spawn volume 0.5 at
master=0.5; nudge clamps at 1.0 and 0.0. Variant lists confirmed for
birth/death/step_player/step_ai.

Gates: `Autopilot: clean exit after 400 frames`; with
`EC_TEST_HAND="warm@120:5,5;chill@240:8,8"` — both touches landed, clean
exit after 600 frames.
