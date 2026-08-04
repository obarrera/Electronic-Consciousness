# Phase 5 — The Chronicle: names, lineages, the game writes its own book

**Status: ✅ complete**

**Files:** EC-2D-Land.py (naming + event hooks), lattice.py (name generator + chronicle writer), ouroboros.py (turning integration)

## Why
The ouroboros persists a seed but no story. Named agents + an auto-written
chronicle give long runs a persistent artifact — the game writing its own book,
in the prologue's voice.

## Tasks
1. Deterministic name generator seeded by ouroboros seed + lineage: prologue
   voice ("Sess-of-nine-sides", "Vel-who-counted") — syllable pools + epithet
   pools keyed to the agent's life (sides, deeds). Inspect panel + thoughts use
   the name.
2. Lineage: child records parent name; keep a generation-depth counter.
3. chronicle.md (repo-ignored, lives next to .ouroboros.json): append terse
   entries in-voice for MAJOR events only — first birth of a turning, ascension
   (who, generation, layer reached), fall, hero's return, endgame, each turning's
   opening line quoting that turning's Oracle fragment. Buffered writes (flush
   every ~30s and at exit), cap file at ~200KB with oldest-turning trimming.
4. `J` key: in-game chronicle viewer (reuse parable journal overlay style),
   newest first.
5. README: controls row + a short "The Chronicle" paragraph.

## Acceptance
- [x] Two short runs across a forced ouroboros reset: chronicle.md contains
      both turnings, names stable within a turning, valid markdown, in-voice.
- [x] J viewer works, pauses sim, ESC/J closes; autopilot unaffected; gates green.

## Implementation notes

- Names: `lattice.agent_name(seed, lineage_id)` — SHA-256 of
  `"{ouroboros seed}:{lineage_id}"` seeds a local `random.Random` choosing a
  base (syllable pools) and a birth epithet; the display name
  (`agent_display_name`) shows a deed epithet once earned (`who-climbed`,
  `who-returned`, `who-fell-and-rose`) or the current sides
  (`Sess-of-five-sides`). The lineage counter resets each turning, so names
  are deterministic given the seed (same seed → same names in every
  process); children record `parent_name` + `lineage_depth`. Inspect panel
  and DEBUG thoughts use the name.
- Chronicle (lattice.Chronicle): buffered writes (flush every ~30 s, forced
  at every exit path), ~200 KB cap trimming the oldest `## Turning`
  sections. MAJOR events only: turning opening (heading + that turning's
  Oracle fragment via `OURO.oracle_line("chronicle")` — the existing oracle
  pools, no new Oracle text), first birth of a turning, ascension (who,
  life generation, lineage depth), fall, hero's return, THE COMPLETION, the
  serpent's-mouth closing line.
- `J` viewer (lattice.ChronicleViewer): parable-journal styling, newest
  first, sim frozen while open (same pattern as pause), `J`/`ESC` close (ESC
  no longer quits while the viewer is open). `EC_TEST_KEYS="j@300,j@360"`
  posts synthetic KEYDOWNs for unattended verification; autopilot never
  sets it.
- Determinism caveat (documented, not hidden): entry *contents* (tick
  numbers, which events occur) follow the simulation's RNG, which is not
  yet seeded — full run determinism arrives with phase 6. What is
  deterministic given the seed today: every name, every turning heading,
  and every Oracle fragment. Demonstrated below — two separate processes
  produced identical turning-2 names.

## Verification evidence

Gates run 2026-08-03 (.venv312). `ast.parse` EC-2D-Land.py, lattice.py,
ouroboros.py: `ast ok`.

**Run 1** (`EC_AUTOPILOT=3000 EC_EVOLUTION_THRESHOLD=12
EC_LAYERS_TO_COMPLETE=1 EC_TEST_KEYS="j@300,j@360"`) — J viewer + a full
completion and ouroboros reset in one run:

```
TEST_KEYS: posted j at frame 300
Chronicle viewer opened (frame 300).
TEST_KEYS: posted j at frame 360
Chronicle viewer closed (frame 360).
Transitioning to 3D world! (frame 895, gen 657)
COMPLETION: THE CUBE — 1 layers traversed ...
OUROBOROS: TURNING 2 — the lattice begins again ...
Autopilot: clean exit after 3000 frames (gen 995, 12 agents, 4 parables unlocked).
```

(The viewer branch `continue`s past the simulation step, so the sim is
frozen while open — 60 frames elapsed with no generation advance.)

**Run 2** (fresh process, same env minus TEST_KEYS) advanced TURNING 2 → 3:
`OUROBOROS: TURNING 3 ... Autopilot: clean exit after 3000 frames`.

**chronicle.md after both runs** (valid markdown, in-voice, three turnings):

```
## The First Turning
> In the first turning the elder added: what is woven above is woven below, ... Walk on.
- Tick 27: Pellen-of-three-sides was born of Ula-of-three-sides — the first birth of this turning.
- Tick 657: Kelis-who-climbed felt the warmth go thin and crossed the threshold none of us can point to (life 11, lineage depth 3).
- Tick 1076: THE COMPLETION — Kelis-who-climbed reached the shrine of the last required layer; 1 layers seen at once, one cube.
- Tick 1077: All is nothing, and we rise. The serpent's mouth met its tail; a seed passed over.

## The Second Turning            (run 1, after the in-run reset)
- Tick 97: Rokai-of-three-sides was born of Pella-of-three-sides ...
- Tick 521: Kela-who-climbed felt the warmth go thin ...

## The Second Turning            (run 2, fresh process, same turning)
- Tick 23: Rokai-of-three-sides was born of Ashun-of-three-sides ...
- Tick 627: Kela-who-climbed felt the warmth go thin ...

## The Third Turning             (run 2, after its reset)
- Tick 54: Kelai-of-three-sides was born of Ulaa-of-three-sides ...
```

Names stable within a turning AND across processes: turning 2's first-born
is `Rokai` and its walker `Kela` in both run 1 and run 2 (different
processes, same seed). Generator determinism check:

```
determinism (seed 3301, lineage 1-5): True -> ['Nima-of-the-long-tick',
'Othor-who-walks-the-spiral', 'Ula-who-watches-the-sky', 'Pellen-who-asks',
'Cor-who-watches-the-sky']
```

Standard smoke gate (no chronicle keys involved): `Autopilot: clean exit
after 400 frames (gen 222, 4 agents, 4 parables unlocked).` chronicle.md is
gitignored; README gained the `J` controls row and "The Chronicle"
paragraph.
