# Architecture Reference — EC-2D-Land (Electronic Consciousness game)

> Read this before implementing any phase. Phase files live in `phases/`; the rules for
> sessions are in [`working-agreement.md`](working-agreement.md); found defects in
> [`defects.md`](defects.md); the phase table in [`plan-index.md`](plan-index.md).

## Project overview

**Purpose:** A single-file, self-running (no player input) pygame simulation. AI agents
("Flatland" 2D, then a recursive 3D world) wander a grid, gain/lose energy, reproduce, die and
are reborn, and occasionally transcend into a PyOpenGL 3D "recursive" mode with Platonic-solid
and zodiac-symbol objects. Each agent carries its own small Keras neural net for movement
decisions. There is no player-facing win condition — it's a generative/ambient art piece.

## Tech stack

| Layer | Choice |
|-------|--------|
| Language | Python 3 |
| Rendering (2D) | pygame |
| Rendering (3D) | PyOpenGL (`OpenGL.GL`/`OpenGL.GLU`), driven by pygame's OpenGL display mode |
| Per-agent ML | TensorFlow/Keras (`Sequential`, `Dense`, `Adam`) — one small model per `AI_Agent` |
| Audio | pygame.mixer, loops `binaural_6.1Hz.wav` from the same directory |
| Tests | none currently (no `tests/` dir, no CI) |

Single entry point: `EC-2D-Land.py`. Run with `python EC-2D-Land.py` from this directory (the
working directory matters — the wav file and any relative asset paths resolve from cwd unless
fixed to resolve from `__file__`).

## In-memory data model (no database/persistence — everything resets on process restart)

- **`ai_agents_2d: list[AI_Agent]`** — the live 2D population. Membership in this list, not any
  external ID, is what "alive" means. `MAX_AGENTS` caps its size.
- **`GameOfLifeEnvironment.grid`** — a 2D numpy/array grid holding `0` (empty), `1` (male cell),
  `2` (female cell), `3` (obstacle, in agent sensing). Runs its own semi-independent
  Game-of-Life-style cellular automaton (`update()`), separate from where `AI_Agent` objects
  actually stand.
- **`RecursiveEnvironment3D.objects`** — accumulated `SolidShape3D` / `ZodiacSymbol3D` instances
  for the current 3D session, added via `create_objects(layer)` as the agent's `layer` advances
  or the agent transcends.
- **`AIAgent3D.layer`** — the 3D agent's own progression counter (1 → 2 → 3 → back to 1),
  distinct from `RecursiveEnvironment3D.layer` (set once at construction; do not confuse them).

## Config

No env vars. Everything is a module-level constant near the top of `EC-2D-Land.py` (`MAX_AGENTS`,
`FPS`, grid size, colors, etc.) or a literal inline in the function that uses it (e.g. the wav
filename in `play_audio_on_loop`'s caller). Changing behavior means editing the constant/literal
directly — there is no config file or CLI flag.

## Invariants (what the verifier attacks every phase — grep-checkable)

- **No agent object appears twice in `ai_agents_2d`.** `die_and_rebirth`/`reproduce` must never
  leave a duplicate reference to the same `AI_Agent`, and the list's length must never exceed
  `MAX_AGENTS`. Check: after any simulation run, `len(ai_agents_2d) == len(set(id(a) for a in
  ai_agents_2d))` and `len(ai_agents_2d) <= MAX_AGENTS`.
- **No unbounded placement search.** Every `while True:` loop that hunts for a free grid cell
  (agent respawn, object/solid/element generation) must have an attempt cap (the existing
  convention is 100 tries), so a full-enough grid degrades to "give up" rather than hanging the
  process. Check: `grep -n "while True" EC-2D-Land.py` — every hit inside a grid-search context
  has a bounded `for _ in range(N):` alternative or an explicit break/give-up path.
  Non-grid-search `while True:` loops (e.g. the top-level pygame event loop) are exempt.
  Comment which category a hit falls into if it's ambiguous.
- **Never mutate `ai_agents_2d` while iterating it directly.** Any loop whose body can add/remove
  agents (via `die_and_rebirth`, `reproduce`, culling) must iterate `ai_agents_2d[:]` (a copy),
  never `ai_agents_2d` itself. Check: `grep -n "for .* in ai_agents_2d:" EC-2D-Land.py` — every
  hit whose body calls a population-mutating method must instead read
  `for .* in ai_agents_2d[:]:`.
- **Energy stays in `[0, 100]`.** Every path that increases `self.energy` clamps with
  `min(100, ...)`, matching `apply_element_effect`/`rest`. Check: `grep -n "energy +=" EC-2D-Land.py`
  — every hit is wrapped in a `min(100, ...)` (or otherwise proven bounded) at the assignment.
- **External resource loads never crash startup.** Audio (`pygame.mixer.init`/`music.load`) and
  any other file/device load that isn't guaranteed present must be wrapped so a missing asset or
  unavailable device degrades to "no audio" rather than raising before the game loop starts.
  Check: the audio-loading call site is inside a `try/except`.
- **3D world object growth is bounded per layer, not cumulative forever.** `RecursiveEnvironment3D
  .create_objects()` must not keep appending duplicate Platonic-solid/zodiac objects across
  repeated layer-discovery or transcendence events for the same session. Check: object count
  after N repeated `create_objects()` calls at the same layer does not grow with N.

## Deviations / open questions

None recorded yet. When implementation reveals the spec is wrong or ambiguous, the **planner**
records the resolution here (dated); the **verifier** files defects in [`defects.md`](defects.md).
Implementers never edit this file.
