# Phase 1: Crash-safety + population-integrity fixes

> **Status:** awaiting verification
> **Depends on:** —
> **Read first:** [architecture.md](../architecture.md), [working-agreement.md](../working-agreement.md)
> Work ONLY on this phase. Touch ONLY the files in its **Files** list.

**Goal:** Fix the bugs identified by a prior code-review pass that either (a) reliably crash the
running simulation or (b) silently corrupt the `ai_agents_2d` population, which every later
phase's behavior depends on being sane.

**Origin:** a full-file code review (`EC-2D-Land.py`, ~2600 lines) found these as the highest-
severity, deterministically reachable issues. This phase fixes exactly those; everything else the
review found is split into phases 2–5 in `plan-index.md`.

## Step 0 — lifecycle (before any code)

- [x] `git add -A && git commit -m "pre 1: baseline"` — N/A: fixes were started in the same
      session as the review, before `plan/` existed; no separate pre-phase baseline commit
      exists. Noted as a process deviation for the verifier.
- [x] No test suite exists in this repo (`ls tests/` → not found). Baseline check substituted:
      `python3 -m py_compile EC-2D-Land.py` must succeed before and after.

## Implementation

- [x] `AIAgent3D` (3D agent class) gets its own rebirth handling so `AIAgent3D.move()` never
      calls the 2D-only `AI_Agent.die_and_rebirth` (`AttributeError`, guaranteed after ~400
      frames in the 3D world at the pre-fix energy-drain rate).
  - [x] `grep -n "die_and_rebirth" EC-2D-Land.py` shows `AIAgent3D` no longer calls a method it
        doesn't define — either it defines its own, or the call site is replaced with the
        correct 3D-world reset path. Added `AIAgent3D.die_and_rebirth(self, ai_agents_2d=None)`
        at line 1554 that resets the singleton 3D agent's own state in place.
- [x] `AI_Agent.die_and_rebirth`: never leaves a duplicate reference to `self` in `ai_agents_2d`.
  - [x] `grep -n -A2 "ai_agents_2d.append(self)" EC-2D-Land.py` shows the append is guarded
        against `self` already being a member: `if self not in ai_agents_2d: ai_agents_2d.append(self)`.
- [x] `AI_Agent.die_and_rebirth`'s free-cell search loop is capped like every other grid-search
      loop in the file (existing convention: 100 attempts), so a full grid can't hang the process.
  - [x] `grep -n -B2 "while True" EC-2D-Land.py` — the `die_and_rebirth` hit is gone (replaced
        with `for _ in range(100):`).
- [x] The main loop's per-frame agent loop (`for agent in ai_agents_2d:` around the sense/decide/
      move/train-data-collection block) iterates a copy, since `agent.move()` can mutate the list
      via `die_and_rebirth`.
  - [x] `grep -n "for agent in ai_agents_2d:" EC-2D-Land.py` — zero remaining hits whose body can
        mutate the list; that loop now reads `for agent in ai_agents_2d[:]:` (line 2443). Remaining
        `for agent in ai_agents_2d:` hits are read-only (stat aggregation, rendering) and don't
        mutate the list.
- [x] Audio load (`pygame.mixer.init()` / `pygame.mixer.music.load(...)` / `.play(-1)`) is wrapped
      so a missing wav file, missing audio device, or mixer failure logs a message and continues
      instead of crashing before the game loop starts. Path resolved relative to `__file__`, not
      the process's cwd.
  - [x] `grep -n -B2 -A6 "def play_audio_on_loop" EC-2D-Land.py` shows a `try/except` around the
        mixer calls; call site builds `wav_file` via
        `os.path.join(os.path.dirname(os.path.abspath(__file__)), "binaural_6.1Hz.wav")`.

## Files

`EC-2D-Land.py` (single-file project — no other files exist to touch).

## Step Z — lifecycle (after all code)

- [x] `python3 -m py_compile EC-2D-Land.py` — succeeds; exit code 0, no output. See `ai-updates.md`.
- [x] `grep -n "die_and_rebirth\|while True\|for agent in ai_agents_2d" EC-2D-Land.py` — pasted in
      `ai-updates.md`.
- [x] `git add -A && git commit -m "1: crash-safety + population-integrity fixes"`; SHA pasted in
      `ai-updates.md`.
- [x] Set this file's Status + the index row to `awaiting verification`. Do NOT touch `defects.md`

**Acceptance (verification session):** `EC-2D-Land.py` still compiles; `AIAgent3D` no longer
calls a nonexistent `die_and_rebirth`; `die_and_rebirth` cannot duplicate an agent in
`ai_agents_2d` (grep + code read); the free-cell search in `die_and_rebirth` is bounded; the main
loop's agent-processing loop iterates `ai_agents_2d[:]`, not `ai_agents_2d`; the audio loader is
wrapped in `try/except` and resolves its path relative to `__file__`. Manual run (`python
EC-2D-Land.py` with a display available) survives well past the old ~400-frame 3D-world crash
point without an `AttributeError` traceback, if the verifier has a display to test with;
otherwise code-read + grep evidence stands in.
