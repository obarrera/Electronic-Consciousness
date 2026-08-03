# Plan Index — EC-2D-Land improvements (2026-08 ROI batch)

Three-role workflow: Planner cuts phases → Implementer builds exactly one phase,
sets `awaiting verification` → Verifier (fresh context) reproduces the pasted
gates and alone sets `✅ complete`. Verification = the established autopilot
gates (EC_AUTOPILOT + EC_EVOLUTION_THRESHOLD + frame/console checks) plus
phase-specific checks below.

| Phase | Task | Status | Depends |
|---|---|---|---|
| 1 | [Real learning — make the Gradient a real teacher](phases/phase-01-real-learning.md) | ✅ complete | — |
| 2 | [The player warms the cells](phases/phase-02-player-hand.md) | planned — ready to implement | — |
| 3 | [First-session pacing](phases/phase-03-pacing.md) | planned — ready to implement | 1 |
| 4 | [Positional audio in Spaceland](phases/phase-04-positional-audio.md) | planned — ready to implement | — |
| 5 | [The Chronicle — names, lineages, auto-written history](phases/phase-05-chronicle.md) | planned — ready to implement | — |

Invariants (verifier attacks every phase): read-only guardrails N/A here, but —
no TensorFlow returns; autopilot must stay deterministic-enough to exit cleanly;
REDUCED_FLASH default honored by any new effect; parable/oracle text untouched;
`.ouroboros.json` compatibility preserved.
