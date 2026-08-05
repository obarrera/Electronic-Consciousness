# Phase 11 — Post-Spaceland render fix (frozen 2D, missing endgame scenes)

**Status: awaiting verification**

**Files:** spaceland.py (`leave()` GL-state restore), EC-2D-Land.py (`_present`
hardening)

## Why
Player report: after the first Spaceland visit the game "doesn't render
right" — Flatland unwatchable, and in the completion arc the hypercube
(tesseract), spectrum, and cutscene graphics never appeared.

Root cause: `spaceland.init_gl()` / `init_opengl()` enable `GL_DEPTH_TEST`
for the 3D world, but `spaceland.leave()` only restored fog and the raster
position. Every 2D frame presents via `glDrawPixels`, whose fragments carry
the raster position's z (0). With depth testing left enabled, the first 2D
frame after a return writes depth 0 across the whole screen; every frame
after it fails `GL_LESS` and is discarded — the swap chain then alternates
two stale buffers (frozen/flickering screen). The same discard hid the
tesseract, spectrum, and both endgame cutscenes, which all present through
`_present()` → `glDrawPixels` after `spaceland.leave()` runs at the end of
the cube stage.

Driver-level probe (64×64 window, depth buffer pre-filled at 0.5,
`GL_DEPTH_TEST` on): draw red → reads red; draw green → reads **black**;
draw blue → reads **stale red**. With `glDisable(GL_DEPTH_TEST)`: every
frame reads back exactly what was drawn.

## Tasks
1. `spaceland.leave()`: disable `GL_DEPTH_TEST` (the critical one), plus
   `GL_TEXTURE_2D` and `GL_BLEND` for parity with the pre-ascension state;
   fog and `glWindowPos2i(0, 0)` as before.
2. `_present()`: wrap the pixel push in `glPushAttrib(GL_ENABLE_BIT)` /
   `glPopAttrib` with depth test and fog forced off for the draw — a parable
   cutscene can fire while Spaceland is live, and it must neither be eaten
   by the 3D depth/fog state nor corrupt that state for the next 3D frame.

## Acceptance
- [ ] GL probe: with the leave()-fix state, three consecutive
      `glDrawPixels` frames each read back their own color.
- [ ] Full cycle windowed: ascend → fall/return → 2D keeps animating
      (manual visual check).
- [ ] Completion arc windowed: cube orbit → parable → tesseract → spectrum
      → O! all visibly render (manual visual check).
- [ ] Autopilot gates still exit cleanly.

## Verification evidence (implementer's run, 2026-08-05, .venv312)

`ast ok` × 2. GL probe: bug reproduced (green→black, blue→stale-red), fix
verified (all frames correct). Endgame gate
`EC_AUTOPILOT=6000 EC_EVOLUTION_THRESHOLD=12 EC_LAYERS_TO_COMPLETE=1
EC_VERIFY_NARRATION=1`: completion arc traversed (cube → parable →
tesseract → spectrum → O! → TURNING advanced), no traceback, clean exit.
Same gate without narration: three full turnings, clean exit. Manual
windowed visual confirmation still recommended for the Verifier (the
recorded-frame pipeline captures the CPU surface, which never showed this
GPU-side bug).
