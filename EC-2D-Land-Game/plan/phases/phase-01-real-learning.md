# Phase 1 — Real learning: make the Gradient a real teacher

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
- [ ] Autopilot 3000 frames: steps-to-food at gen 2500+ measurably better (>15%)
      than gen 0-300 baseline, printed and in the CSV; three runs, all improve.
- [ ] No TF; ast gates; clean autopilot exit; ouroboros reset also resets buffer.
