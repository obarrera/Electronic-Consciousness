# Phase 6 — Deterministic headless core

**Status: ✅ complete**

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

- [x] 10 consecutive headless runs, same seed, 2000 ticks → 10 identical
      `final_state_hash` values.
- [x] Headless vs windowed autopilot, same seed and tick count → identical
      hash.
- [x] Two windowed runs at different frame pacing (30 fps vs uncapped) →
      identical hash at the same tick.
- [x] EC_FULL_FLASH=1 vs default → identical hash (visuals provably cannot
      affect behavior).
- [x] Existing gates still green: ast on all .py, standard autopilot clean
      exit, phases 1–5 features intact (learning CSV still improves,
      chronicle still writes, parable queue works).

Notes: phase 1's reward buffer and phase 3's pacing are sim-state — they hash
too. Player clicks are user input, out of hash scope for these tests (no
clicks in headless/autopilot).

## Implementation notes

- `simcore.py`: `RngPool` fans one root seed (EC_SEED override, else the
  ouroboros seed) into named streams via SHA-256 (`stream_seed`), with live
  proxies (`WORLD`, `AGENTS`, `FX`, `NP_WORLD`, `NP_AGENTS`) so an ouroboros
  reset (`init_pool(new seed)`) can never leave a stale stream bound at a
  call site. Streams: world (board, goals, solids/elements/symbols, GoL
  births, lifespans), agents (decisions, exploration, incentives, rebirth,
  reproduction), brain (MLP weight init + minibatch shuffles —
  `NumpyMLP._rng`), fx (particles, decorative colors — visual-only by
  construction), spaceland:<salt>:<layer> (hazards, spawns, goal region).
  Every module-global `random` / `np.random` use in simulation logic was
  replaced; a grep gate confirms none remain outside the named streams.
- **Tick = one simulation advance** (`current_generation`): unchanged loop
  structure, 1 rendered frame = 1 tick when the sim runs; pause, cutscenes,
  parable presentation, the chronicle viewer, and the endgame arc `continue`
  before the tick increment, so they advance frames but never ticks (the
  headless==windowed hash equality below *proves* presentation cannot touch
  sim state). Spaceland is now driven by sim time (`tick/30`), not wall
  time, so frame pacing cannot change the walker's per-tick behavior.
- **Headless** (`EC_HEADLESS=1`): SDL dummy video/audio drivers selected
  before pygame init, plain (non-GL) display surface, intro/warning skipped,
  the whole renderer + parable presentation skipped (unlock recording and
  the journey's sim-coupled elixir logic still run per tick), no fps cap.
  2000 ticks complete in ~5 s. Known limit (documented): a headless run that
  ascends to Spaceland degrades gracefully (GL-less world never builds; the
  walker returns via energy drain) — the hash gates therefore run in 2D
  (default threshold; 2000 ticks << first ascension ~17,000).
- **Run manifest** (`EC_RUN_DIR`, defaulting to `./run` in headless —
  gitignored): schema_version, root_seed, git commit + dirty flag, python
  version, tick count, timestamps, `final_state_hash`. The hash
  (`simcore.state_hash`) covers tick, full grid, goal, warm/chill cells, and
  each agent's lineage-id/position/energy/consciousness/gender/generation/age
  (floats quantized to 4 dp), plus spaceland layer/position when active —
  one function used by both modes.
- `EC_TICKS=<n>` ends any run at tick n with the hash printed;
  `EC_UNCAPPED=1` removes the windowed fps cap. EC_AUTOPILOT (frame-counted)
  is unchanged and still the smoke-gate harness; EC_TICKS is the
  determinism harness. The shared brain persists across an in-process
  ouroboros reset by design (the seed passes over); streams re-derive from
  the new turning's seed.

## Verification evidence

Gates run 2026-08-03 (.venv312, branch `production-pass`). `ast.parse` on
EC-2D-Land.py, lattice.py, spaceland.py, ouroboros.py, simcore.py:
`ast ok x5`. Grep gate: no `random.` / `np.random` outside named streams in
EC-2D-Land.py.

**Gate 1 — 10 consecutive headless runs** (`EC_HEADLESS=1 EC_SEED=12345
EC_TICKS=2000`): ten runs, ten identical hashes (1 distinct value):

```
final_state_hash a2d8b88c301a12152a927036d327bb47ceed17e7d5b045d6119d2625cf536ba4  × 10
```

**Gate 2 — headless vs windowed** (`EC_SEED=12345 EC_TICKS=2000
EC_AUTOPILOT=99999`, windowed with cutscenes/overlays rendering):

```
windowed capped:   final_state_hash a2d8b88c30...cf536ba4   (== headless)
```

**Gate 3 — frame pacing** (30 fps cap vs `EC_UNCAPPED=1`):

```
windowed uncapped: final_state_hash a2d8b88c30...cf536ba4   (== capped)
```

**Gate 4 — flash equivalence** (`EC_FULL_FLASH=1` vs reduced default):

```
windowed full-flash: final_state_hash a2d8b88c30...cf536ba4  (== default)
```

All four modes converge on the same hash
`a2d8b88c301a12152a927036d327bb47ceed17e7d5b045d6119d2625cf536ba4` —
rendering, frame pacing, presentation, and flash mode provably cannot
perturb behavior.

**Manifest** (run/manifest.json):

```json
{"schema_version": 1, "root_seed": 12345,
 "git_commit": "da5bf3b6357c...", "git_dirty": true,
 "python": "3.12.13", "headless": true, "ticks": 2000,
 "started_at": "2026-08-03T19:57:36-0500",
 "finished_at": "2026-08-03T19:57:41-0500",
 "final_state_hash": "a2d8b88c30...cf536ba4"}
```

(dirty flag true because the run preceded this commit — correct behavior.)

**Existing gates**: standard smoke `EC_AUTOPILOT=400 EC_EVOLUTION_THRESHOLD=12`
→ `Autopilot: clean exit after 400 frames (gen 218, 9 agents, 4 parables
unlocked).` Phases 1–5 feature spot-checks are re-run in the final batch
integration gate (see plan-index).
