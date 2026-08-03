# Phase 5 — The Chronicle: names, lineages, the game writes its own book

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
- [ ] Two short runs across a forced ouroboros reset: chronicle.md contains
      both turnings, names stable within a turning, valid markdown, in-voice.
- [ ] J viewer works, pauses sim, ESC/J closes; autopilot unaffected; gates green.
