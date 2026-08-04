#!/usr/bin/env python3
"""tools_experiment.py — multi-seed mirror experiment runner.

The reproducibility rule this enforces: results are measured over many
seeds, never selected from the most interesting run. It executes headless,
genome-frozen runs across N seeds under two conditions — mirror enabled
(agents build world/other/self models) and mirror ablated (EC_MIRROR=0,
the control) — then reports per-condition aggregates from the run
manifests, plus a determinism check (same seed twice must produce the same
final state hash).

Usage:
    python3 tools_experiment.py [--seeds N] [--ticks T] [--out DIR]

What this measures today (Tier 1+2): model fidelities and mirror-moment
counts under identical worlds. What it deliberately does not claim: that
any of it establishes subjective experience. It is the harness the book's
sharper questions (self-model calibration vs. task performance; artifact
detection) can be bolted onto.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

GAME_DIR = os.path.dirname(os.path.abspath(__file__))


def run_one(seed, ticks, mirror_on, out_dir, artifact=0):
    run_dir = os.path.join(
        out_dir, f"seed{seed}-mirror{'1' if mirror_on else '0'}-art{artifact}")
    env = dict(os.environ,
               EC_HEADLESS="1", EC_TICKS=str(ticks), EC_SEED=str(seed),
               EC_MIRROR="1" if mirror_on else "0",
               EC_ARTIFACT=str(artifact),
               EC_GENOME="0",           # pristine constants: worlds match
               EC_RUN_DIR=run_dir)
    proc = subprocess.run([sys.executable, os.path.join(GAME_DIR, "EC-2D-Land.py")],
                          env=env, capture_output=True, text=True, timeout=1800)
    manifest_path = os.path.join(run_dir, "manifest.json")
    if proc.returncode != 0 or not os.path.exists(manifest_path):
        print(f"  seed {seed} mirror={mirror_on}: FAILED "
              f"(exit {proc.returncode})\n{proc.stderr[-400:]}")
        return None
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}"


def report_condition(label, manifests):
    stats = [m.get("mirror") or {} for m in manifests]
    print(f"\n{label}  (n={len(manifests)})")
    print(f"  world-model fidelity : {fmt([s.get('world') for s in stats])}")
    print(f"  other-model fidelity : {fmt([s.get('others') for s in stats])}")
    print(f"  self-model fidelity  : {fmt([s.get('self') for s in stats])}")
    print(f"  mirrored agents      : "
          f"{fmt([float(s['mirrored']) if s else None for s in stats])}")
    seam_counts = {}
    for s in stats:
        for p, n_agents in (s.get("seams") or {}).items():
            seam_counts[int(p)] = seam_counts.get(int(p), 0) + n_agents
    print(f"  seam periods found   : "
          f"{seam_counts if seam_counts else 'none'} "
          f"(period: agent-detections across runs)")
    return seam_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--artifact", type=int, default=0, metavar="PERIOD",
                    help="run the artifact experiment: control worlds vs "
                         "worlds with a hidden goal-teleport every PERIOD "
                         "ticks (try 31)")
    ap.add_argument("--out", default=os.path.join(GAME_DIR, "experiments"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.artifact:
        # Artifact experiment: mirror always on; control vs treatment worlds.
        conditions = [("control  (no seam)", dict(mirror_on=True, artifact=0)),
                      (f"treatment (seam @{args.artifact})",
                       dict(mirror_on=True, artifact=args.artifact))]
    else:
        conditions = [("mirror ON ", dict(mirror_on=True, artifact=0)),
                      ("mirror OFF", dict(mirror_on=False, artifact=0))]

    results = {label: [] for label, _ in conditions}
    for seed in range(args.seeds):
        for label, cond in conditions:
            print(f"run: seed {seed}, {label.strip()}, {args.ticks} ticks...")
            m = run_one(seed, args.ticks, cond["mirror_on"], args.out,
                        cond["artifact"])
            if m:
                results[label].append(m)

    # Determinism check: first condition, seed 0, run again — hashes match.
    print("run: determinism check (seed 0 repeated)...")
    label0, cond0 = conditions[0]
    repeat = run_one(0, args.ticks, cond0["mirror_on"],
                     os.path.join(args.out, "repeat"), cond0["artifact"])
    base = next((m for m in results[label0] if m["root_seed"] == 0), None)
    deterministic = (repeat is not None and base is not None
                     and repeat["final_state_hash"] == base["final_state_hash"])

    title = ("artifact experiment" if args.artifact else "mirror experiment")
    print(f"\n== {title} ==")
    print(f"seeds: {args.seeds} · ticks: {args.ticks} · genome frozen")
    print(f"determinism (seed 0 twice): "
          f"{'IDENTICAL HASH' if deterministic else '*** DIVERGED ***'}")
    seam_by_condition = {}
    for label, _ in conditions:
        seam_by_condition[label] = report_condition(label, results[label])

    verdict = None
    if args.artifact:
        control_label, treatment_label = conditions[0][0], conditions[1][0]
        hit = seam_by_condition[treatment_label].get(args.artifact, 0)
        false_hit = seam_by_condition[control_label].get(args.artifact, 0)
        verdict = bool(hit) and not false_hit
        print(f"\nverdict: seam period {args.artifact} detected in "
              f"{hit} treatment agent-run(s), {false_hit} control "
              f"false-detection(s) -> "
              f"{'ARTIFACT DISTINGUISHED FROM LAW' if verdict else 'NOT DISTINGUISHED (a fair negative result)'}")

    summary = {
        "seeds": args.seeds, "ticks": args.ticks,
        "artifact_period": args.artifact or None,
        "deterministic": deterministic,
        "verdict_artifact_detected": verdict,
        "runs": {label: ms for label, ms in results.items()},
    }
    out_path = os.path.join(args.out, "summary.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nsummary: {out_path}")
    return 0 if deterministic else 1


if __name__ == "__main__":
    sys.exit(main())
