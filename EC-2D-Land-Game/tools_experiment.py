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


def run_one(seed, ticks, mirror_on, out_dir):
    run_dir = os.path.join(out_dir, f"seed{seed}-mirror{'1' if mirror_on else '0'}")
    env = dict(os.environ,
               EC_HEADLESS="1", EC_TICKS=str(ticks), EC_SEED=str(seed),
               EC_MIRROR="1" if mirror_on else "0",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(GAME_DIR, "experiments"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = {True: [], False: []}
    for seed in range(args.seeds):
        for mirror_on in (True, False):
            print(f"run: seed {seed}, mirror {'on' if mirror_on else 'off'}, "
                  f"{args.ticks} ticks...")
            m = run_one(seed, args.ticks, mirror_on, args.out)
            if m:
                results[mirror_on].append(m)

    # Determinism check: seed 0, mirror on, run again — hashes must match.
    print("run: determinism check (seed 0 repeated)...")
    repeat = run_one(0, args.ticks, True, os.path.join(args.out, "repeat"))
    base = next((m for m in results[True] if m["root_seed"] == 0), None)
    deterministic = (repeat is not None and base is not None
                     and repeat["final_state_hash"] == base["final_state_hash"])

    print("\n== mirror experiment ==")
    print(f"seeds: {args.seeds} · ticks: {args.ticks} · genome frozen")
    print(f"determinism (seed 0 twice): "
          f"{'IDENTICAL HASH' if deterministic else '*** DIVERGED ***'}")
    for mirror_on, label in ((True, "mirror ON "), (False, "mirror OFF")):
        ms = results[mirror_on]
        stats = [m.get("mirror") or {} for m in ms]
        print(f"\n{label}  (n={len(ms)})")
        print(f"  world-model fidelity : {fmt([s.get('world') for s in stats])}")
        print(f"  other-model fidelity : {fmt([s.get('others') for s in stats])}")
        print(f"  self-model fidelity  : {fmt([s.get('self') for s in stats])}")
        print(f"  mirrored agents      : {fmt([float(s['mirrored']) if s else None for s in stats])}")

    summary = {
        "seeds": args.seeds, "ticks": args.ticks,
        "deterministic": deterministic,
        "runs": {"mirror_on": results[True], "mirror_off": results[False]},
    }
    out_path = os.path.join(args.out, "summary.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nsummary: {out_path}")
    return 0 if deterministic else 1


if __name__ == "__main__":
    sys.exit(main())
