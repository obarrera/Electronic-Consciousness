# Phase 13 — Entropy (no two runs alike) + the recursion, lived

**Status: awaiting verification**

**Files:** ouroboros.py (entropy in the seed river, salted oracle),
EC-2D-Land.py (session-unique ROOT_SEED, salted narration sequencing,
world-below tick + live window), README.md

## Why
Two asks. First: no two runs should ever produce the same messages or
results — the oracle lines, names, worlds, and outcomes must be genuinely
fresh every cycle, including the narration. Second: the game claims to be
a simulation of a recursive simulation; that recursion should be *lived*,
not just stated.

## Entropy — no two runs alike
- `ouroboros._entropy()`: `time_ns ^ pid`, folded into (a) the first seed
  of a fresh world, (b) every `advance()` (each turning's seed), and
  (c) a per-process `session` salt.
- `oracle_fragment_indices(iteration, key, salt)`: the salt
  (`Ouroboros.oracle_salt()` = seed ^ session) picks this run's (opener,
  turn, seal) — shared by the text composer AND the narration sequencer,
  so what the elder *says* is what the screen *shows*, and both recompose
  freshly each cycle. The fragments are recorded once; the sequencing is
  what varies — dynamic narration with no runtime TTS.
- `ROOT_SEED = OURO.seed ^ OURO.session` when EC_SEED is unset: even two
  sessions of the SAME saved turning roll different sim streams — results
  never repeat.
- **EC_SEED pins everything exactly as before** (entropy and salt collapse
  to 0; the classic coprime-stride oracle walk holds; the verification
  gates and phase-6 determinism harness are untouched). README documents
  the knob.

## The recursion, lived
- While the walker is in Spaceland, the world below keeps simulating: the
  environment's Conway layer ticks at one-third speed (agents hold their
  breath; `current_generation` was already advancing).
- `_draw_world_below()`: a live miniature of the 2D lattice (cells, warm
  blooms, goal, agents) drawn into the Spaceland frame's top-right corner
  via glWindowPos/glDrawPixels inside a glPushAttrib(GL_ENABLE_BIT) guard,
  captioned "the world below, still turning". The walker literally watches
  the simulation they came from continue inside the one they walk.

## Acceptance
- [ ] Two fresh worlds (no EC_SEED): different seeds, different oracle
      lines; consecutive turnings all differ.
- [ ] EC_SEED pinned: seeds equal, salt 0, legacy stride walk preserved
      byte-for-byte.
- [ ] Windowed ascend/fall gate passes with the below-tick live.
- [ ] Windowed endgame gate passes (turnings advance with entropy seeds).
- [ ] Manual: corner window visibly updates during Spaceland.

## Verification evidence (implementer's run, 2026-08-05, .venv312)

`ast ok` × 3. Unit checks: fresh seeds differ ✓, oracle lines differ ✓,
three consecutive turnings all distinct ✓, pinned seeds equal with salt 0 ✓,
legacy stride preserved under pin ✓. Windowed gates: ascend/fall 1500-frame
clean exit with world-below tick; endgame 6000-frame run — two more full
turnings through THE SUMMATION (entropy-mixed seeds), zero tracebacks.
Headless 3D remains flaky (segfault) — reproduced identically on stashed
baseline; pre-existing environment limit, tracked in phase 11's notes.
Manual visual check of the corner window recommended for the Verifier.
