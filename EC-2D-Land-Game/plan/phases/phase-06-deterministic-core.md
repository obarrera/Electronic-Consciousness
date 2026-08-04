# Phase 6 — Deterministic headless core

**Status: planned — queued after phases 1–5**

**Files:** EC-2D-Land.py (main loop, sim/render split), lattice.py,
spaceland.py, ouroboros.py (root seed), possibly a new `simcore.py` (allowed;
no src/ package restructure). Highest-ROI item from the external architecture
review. Story-mode behavior (parables, audio, cutscenes, endgame) stays
experientially identical.

## Tasks

1. **Named RNG streams.** An `RngPool` seeded by one root seed (default: the
   ouroboros seed from `.ouroboros.json`; `EC_SEED` overrides). Streams
   derived by stable hash (hashlib, NOT Python's `hash()`): `"world"`,
   `"resources"`, `"agents"`, `"spaceland"`, plus per-agent streams keyed by
   deterministic agent identity. Replace ALL module-global `random` /
   `np.random` uses in simulation logic (agent decisions, births, food
   spawns, mutations, spaceland walker, hazards). Rendering/particle/visual
   randomness uses a separate `"fx"` stream so visuals can never perturb
   behavior. No unseeded numpy generators anywhere in sim code.

2. **Integer-tick sim decoupled from rendering.** Simulation advances via a
   tick function driven by a fixed-timestep accumulator in the main loop;
   rendered FPS must not change sim ticks per unit of sim time; pause must
   not advance ticks; cutscene/parable display pauses must not advance the
   sim (verify this stays true). Autopilot semantics preserved (EC_AUTOPILOT
   counts frames or ticks — pick one, document, keep clean exit).

3. **Headless mode.** `EC_HEADLESS=1`: no window (SDL_VIDEODRIVER=dummy), no
   GL, no audio, no narration waits — pure sim at max speed for N ticks
   (`EC_TICKS` or `EC_AUTOPILOT`). Identical simulation code path; the
   renderer consumes state, never mutates it.

4. **Run manifest + final-state hash.** When `EC_RUN_DIR` is set (and always
   in headless), write `manifest.json`: schema_version, root seed, git commit
   (tolerate failure), dirty-tree flag, python version, tick count, start/end
   timestamps, and `final_state_hash` — canonical SHA-256 over sorted,
   quantized sim state (tick, grid life cells, each agent's
   id/position/energy/consciousness/sides, food positions, spaceland
   layer/position if active). One hash function used by both modes. Run
   artifacts gitignored.

## Acceptance

- [ ] 10 consecutive headless runs, same seed, 2000 ticks → 10 identical
      `final_state_hash` values.
- [ ] Headless vs windowed autopilot, same seed and tick count → identical
      hash.
- [ ] Two windowed runs at different frame pacing (30 fps vs uncapped) →
      identical hash at the same tick.
- [ ] EC_FULL_FLASH=1 vs default → identical hash (visuals provably cannot
      affect behavior).
- [ ] Existing gates still green: ast on all .py, standard autopilot clean
      exit, phases 1–5 features intact (learning CSV still improves,
      chronicle still writes, parable queue works).

Notes: phase 1's reward buffer and phase 3's pacing are sim-state — they hash
too. Player clicks are user input, out of hash scope for these tests (no
clicks in headless/autopilot).
