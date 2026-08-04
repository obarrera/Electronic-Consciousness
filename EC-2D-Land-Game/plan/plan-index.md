# Plan Index — EC-2D-Land improvements (2026-08 ROI batch)

Three-role workflow: Planner cuts phases → Implementer builds exactly one phase,
sets `awaiting verification` → Verifier (fresh context) reproduces the pasted
gates and alone sets `✅ complete`. Verification = the established autopilot
gates (EC_AUTOPILOT + EC_EVOLUTION_THRESHOLD + frame/console checks) plus
phase-specific checks below.

| Phase | Task | Status | Depends |
|---|---|---|---|
| 1 | [Real learning — make the Gradient a real teacher](phases/phase-01-real-learning.md) | ✅ complete | — |
| 2 | [The player warms the cells](phases/phase-02-player-hand.md) | ✅ complete | — |
| 3 | [First-session pacing](phases/phase-03-pacing.md) | ✅ complete | 1 |
| 4 | [Positional audio in Spaceland](phases/phase-04-positional-audio.md) | ✅ complete | — |
| 5 | [The Chronicle — names, lineages, auto-written history](phases/phase-05-chronicle.md) | ✅ complete | — |
| 6 | [Deterministic headless core](phases/phase-06-deterministic-core.md) | planned — queued after 1–5 | 1–5 |
| 7 | [Narration coverage + quality pass](phases/phase-07-narration-quality.md) | planned — queued after 6 | 3, 6 |
| 8 | [2D legibility — legend, toasts, labeled HUD, hover](phases/phase-08-legibility.md) | planned — queued after 7 (batch closes at 8) | 2, 5, 7 |

Invariants (verifier attacks every phase): read-only guardrails N/A here, but —
no TensorFlow returns; autopilot must stay deterministic-enough to exit cleanly;
REDUCED_FLASH default honored by any new effect; parable/oracle text untouched;
`.ouroboros.json` compatibility preserved; narration completion guarantee (a
started parable narration always plays to the end unless the player skips).

**Universal text-completion criterion** (applies to ALL narrative text —
parables, cutscenes, hero's-journey announcements, the CALL caption,
ascension/fall lines, Oracle lines, endgame captions):
1. *Fully displayed* — no clipping/truncation/overflow; wrap verified against
   the longest text in each pool.
2. *Fully readable* — minimum display = max(narration + 1.5 s,
   word_count / 3.3 wps + 2 s); nothing auto-advances sooner.
3. *Fully heard* — narration plays to completion before auto-dismiss; the
   read-time and narration floors combine, longest wins.
4. *Never lost* — events that would replace visible text queue instead;
   ENTER is the only early exit and must be a deliberate press.
5. Cutscene animation stretches to the text/narration floor, never the
   reverse. Autopilot skip behavior unchanged.
(Implemented in phase 3; extended to new narration in phase 7.)
