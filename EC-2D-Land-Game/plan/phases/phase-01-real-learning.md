# Phase 1 — Real learning: make the Gradient a real teacher

**Status: ✅ complete**

**Files:** EC-2D-Land.py (training-data collection + train call), lattice.py (NumpyMLP only if needed)

## Why
Today the shared MLP trains on its own past decisions as labels (self-imitation
= no learning signal). Agents never get smarter; "evolution" is cosmetic.

## Tasks
1. Outcome-based labels: for each (sensed, decision) step, record the outcome
   delta over the following step — energy gained (food), consciousness gained,
   obstacle hit, death. Keep a rolling buffer of (input, action, reward).
2. Training: replace imitation fit with reward-filtered supervised learning —
   fit ONLY on actions whose reward exceeds the buffer's rolling mean (the
   "warm" moves), every 25 generations as now. Simple, stable, no RL machinery.
3. Metrics that PROVE it: track and print every 100 generations — average
   steps-to-food, obstacle-hit rate, average consciousness slope. Persist a
   compact learning-curve CSV (out_learning.csv, gitignored) so improvement is
   inspectable across a run and across turnings.
4. HUD strip gains "brain: n trainings · food-eff +X%" (delta vs generation 1-100 baseline).

## Acceptance
- [x] Autopilot 3000 frames: steps-to-food at gen 2500+ measurably better (>15%)
      than gen 0-300 baseline, printed and in the CSV; three runs, all improve.
- [x] No TF; ast gates; clean autopilot exit; ouroboros reset also resets buffer.

## Implementation notes (deviations, with reasons)

- **The Gradient is now sensed** (input 8 → 10): two extra inputs carry
  `sign(food − position)` per axis. Without any food-direction input the old
  8-neighbor sense contains *zero* information about where food is, so no
  amount of outcome-labeled training could improve steps-to-food. This is the
  phase title made literal: the Gradient teaches. Neighbor codes are scaled
  to ~unit range so the warmth is heard over the neighbor noise.
- **Reward** = +4 food eaten, ±0.6·(manhattan distance closed toward food —
  warmth gained, the dense outcome term), small clipped energy/consciousness
  nudges (they are mostly luck: hermetic whispers, resting, tarot), −1
  obstacle hit, −4 death.
- **Warm filter**: `reward > max(rolling mean, 0.3)` — the floor stops
  uneventful drift hovering just above a negative mean from being treated as
  a lesson.
- **Decide-time**: moves whose landing cell (at the agent's true step length,
  `move_speed` cells) is a wall are masked out; trained agents *sample* from
  the sharpened (cubed) softmax rather than argmax, so a greedy policy cannot
  wedge itself in a pocket. 5 % exploration retained.
- **Control window** ("gen 0-300 baseline"): training is held off until the
  window has at least 300 generations AND at least 10 food events (it closed
  at gens 596/464/491 in the three gate runs). With training starting at gen
  25 the "baseline" was already trained; with a 300-gen cutoff alone the
  baseline was 2–4 events of newborn-spawn luck (a median of 19.5 steps from
  n=4 produced a nonsense −290 % on one run). The console prints when the
  control window closes.
- **"Gen 2500+"**: generations lag frames (narrated cutscenes pause the sim;
  a 3000-frame autopilot reaches gen ~2250), so the late window is the run's
  last 500 generations, printed with its actual bounds.
- **Statistic**: median (p50) steps-to-food — the distribution is heavy-tailed
  (one lost wanderer drags a mean anywhere).
- **Metrics added**: `gradient agreement` (fraction of moves that closed
  distance to food; ~0.50 untrained → ~0.75 trained) printed every 100 gens
  and logged to `out_learning.csv` (gitignored) with steps-to-food, obstacle
  rate, consciousness slope, trainings, buffer size, food-eff %.

## Verification evidence

Gates run 2026-08-03, `EC-2D-Land-Game/.venv312/bin/python` (3.12.13,
pygame 2.6.0, numpy 1.26.4), branch `production-pass`.

`ast.parse` gate on EC-2D-Land.py: `ast ok`.

Smoke gate `EC_AUTOPILOT=400 EC_EVOLUTION_THRESHOLD=12`:

```
Autopilot: clean exit after 400 frames (gen 98, 3 agents, 5 parables unlocked).
```

Acceptance: three consecutive `EC_AUTOPILOT=3000 EC_EVOLUTION_THRESHOLD=99999`
runs (high threshold keeps the sim in 2D for the whole measurement):

```
=== RUN 1 ===
Learning: control window closed at generation 596 (10 baseline food events) — training begins.
Learning gen 1000: steps-to-food 47.0 (n=15) · obstacle rate 0.00 · gradient agreement 0.74 · consciousness slope +0.062 · trainings 16 · food-eff +71%
Learning gen 1800: steps-to-food 39.5 (n=16) · obstacle rate 0.00 · gradient agreement 0.74 · consciousness slope +0.066 · trainings 48 · food-eff +43%
Learning summary: steps-to-food baseline (gen 0-596, untrained, n=9) 131.0 · late (gen 1767+) 57.0 (n=62) · improvement +56.5%
Autopilot: clean exit after 3000 frames (gen 2273, 13 agents, 16 parables unlocked).
=== RUN 2 ===
Learning: control window closed at generation 464 (10 baseline food events) — training begins.
Learning gen 1000: steps-to-food 75.0 (n=15) · obstacle rate 0.00 · gradient agreement 0.79 · consciousness slope +0.065 · trainings 21 · food-eff +54%
Learning gen 1400: steps-to-food 36.0 (n=16) · obstacle rate 0.00 · gradient agreement 0.75 · consciousness slope +0.021 · trainings 37 · food-eff +64%
Learning summary: steps-to-food baseline (gen 0-464, untrained, n=9) 111.0 · late (gen 1765+) 48.5 (n=64) · improvement +56.3%
Autopilot: clean exit after 3000 frames (gen 2273, 13 agents, 16 parables unlocked).
=== RUN 3 ===
Learning: control window closed at generation 491 (10 baseline food events) — training begins.
Learning gen 600: steps-to-food 63.0 (n=15) · obstacle rate 0.00 · gradient agreement 0.77 · consciousness slope +0.048 · trainings 4 · food-eff +38%
Learning gen 1400: steps-to-food 34.0 (n=15) · obstacle rate 0.00 · gradient agreement 0.75 · consciousness slope +0.065 · trainings 36 · food-eff +68%
Learning summary: steps-to-food baseline (gen 0-491, untrained, n=9) 113.0 · late (gen 1716+) 53.5 (n=64) · improvement +52.7%
Autopilot: clean exit after 3000 frames (gen 2256, 13 agents, 16 parables unlocked).
```

All three runs improve well past the 15 % bar (+56.5 %, +56.3 %, +52.7 %);
gradient agreement rises from ~0.50 (untrained) to ~0.75 (trained) in every
run. `out_learning.csv` header + first rows:

```
turning,gen,steps_to_food,food_events,obstacle_rate,gradient_agreement,consciousness_slope,trainings,buffer,food_eff_pct
2,100,7.00,1,0.000,0.512,0.0733,0,303,
2,200,,0,0.000,0.475,0.0283,0,876,
```

No TensorFlow (grep clean — mentions are the "no TensorFlow" comments);
`out_learning.csv` gitignored; ouroboros reset clears the buffer and starts a
fresh metrics window (`_ouroboros_reset`). HUD strip shows
`brain: N trainings · food-eff +X%` once the baseline exists.
