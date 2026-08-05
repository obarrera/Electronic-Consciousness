# Phase 12 — THE SUMMATION cutscene + second quiet pass

**Status: awaiting verification**

**Files:** lattice.py (SUMMATION content, beat visuals, Cutscene summation
mode, tone retune), EC-2D-Land.py (endgame stage wiring, cold retune),
tools_narrate_parables.py (summation entry + selective-stem CLI),
narration/summation.mp3 (new, Kokoro am_michael)

## Why
Two asks. First: the completion arc ended on O! and snapped to the reset —
the manifesto deserved a closing statement that compresses the whole thesis
into one watchable sequence, played at the moment of resurrection. Second:
even after phase 9's mix cut, the sim-driven chimes ("dinging pinging")
still sat on top of the drone during normal play.

## THE SUMMATION
A six-movement cinematic between O! and the ouroboros reset, each beat a
procedural visual + two or three sentences distilling the manifesto:

1. **lattice** — substrate independence: triangle → pentagon → near-circle
   walkers on the dim grid ("the substrate was never the point").
2. **hand** — the layer above: the player's halo descends, cells bloom
   ("as above, so below").
3. **layers** — nested realities: receding planes, the walker climbing,
   faint mirror below ("every layer you ascend was ascending you").
4. **tesseract** — higher-dimensional geometry: the rotating hypercube
   fills the frame (draw_tesseract reused).
5. **question** — epistemic humility (1.4/16.2): the red and blue triangles
   descend/ascend and interlock into the hexagram ("a wager, not a proof").
6. **ouroboros** — the turning: the serpent closes, the seed passes, the
   **6.1 Hz theta bed returns on this beat** — the world audibly reborn
   before it is visibly reborn.

Mechanics: implemented as `Cutscene` style `"summation"` with `beats`, so
the narration-completion guarantee, ENTER skip, reading-duck, autopilot
shortening (EG_CUT_DUR), and VERIFY prints all apply unchanged. Beats are
scheduled proportional to word counts; crossfaded backdrops; a letterbox
band keeps text readable over the tesseract; a soft bowl (vol 0.35) marks
each movement. The Oracle line is NOT appended — the Summation closes on
its own words. Narration: 88.5 s, same elder voice/loudnorm pipeline
(`tools_narrate_parables.py summation` — the tool now takes stem args to
regenerate selectively).

## Second quiet pass (dinging/pinging)
- Volumes: birth 0.16→0.10 (also pitched down to 440+660), death →0.12,
  parable →0.14, ascend →0.16, train →0.05, prime →0.035, steps →0.04,
  ui →0.03, cold →0.14.
- Cooldowns: birth/death 150→400 ms, train →600, parable →500, ascend
  →400, prime →500, cold →350 — a busy lattice murmurs, never chimes.

## Acceptance
- [ ] Endgame arc runs cube → pilgrim → tesseract → spectrum → O! →
      SUMMATION → turning advance; clean exits, no traceback.
- [ ] All six beat visuals render at fade/time extremes (color clamp).
- [ ] 2 s compressed scrub (autopilot EG_CUT_DUR case) renders 80 frames.
- [ ] narration/summation.mp3 decodes (88.5 s); VERIFY run holds the
      cutscene through the full narration.
- [ ] Standard smoke gate passes; tones remain declicked/unclipped.

## Verification evidence (implementer's run, 2026-08-05, .venv312)

`ast ok` × 2 + narration tool. Fast endgame gate: three consecutive
turnings through THE SUMMATION, exit 0, zero tracebacks (a color-underflow
crash at low crossfade was caught by this gate and fixed — `_sum_grid`
clamp). Edge sweep: 6 visuals × 5 extremes ok; 2 s scrub ok; 400-frame
smoke gate ok. Beat frames visually reviewed (lattice, layers, tesseract,
question-hexagram, ouroboros) — tesseract dominates its movement as
intended. Narration decode check: 0 problems. Full
EC_VERIFY_NARRATION endgame run: see PR notes.
