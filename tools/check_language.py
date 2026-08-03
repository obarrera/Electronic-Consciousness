#!/usr/bin/env python3
"""Guard against first-edition mechanism-language drifting back into the
landing surfaces (README files). Chapters are exempt: they may quote retired
phrases in order to retract them; the README may not assert them.

Run: python3 tools/check_language.py   (exit 1 on any violation)
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGETS = ["README.md", "EC-2D-Land-Game/README.md"]

# Retired second-edition formulations. Each pattern is case-insensitive.
BANNED = [
    (r"sufficiently complex (system|substrate)", "substrate independence stated as fact"),
    (r"perceive .{0,40}beyond human", "higher-dimensional perception claim"),
    (r"multiple cognitive states", "quantum superposition as cognition"),
    (r"multiverse awareness", "retired first-edition phrase"),
    (r"hyperconnected (awareness|consciousness)", "entanglement-as-awareness claim"),
    (r"instantaneous(ly)? communicat", "violates the no-communication theorem"),
    (r"(exceed|surpass) biological consciousness", "QNN overclaim"),
    (r"powerful (optimization )?framework", "retired golden-ratio phrasing"),
    (r"ensures? (harmonious|holistic|optimal)", "symbol-as-mechanism phrasing"),
    (r"nonlinear time processing", "retired temporal claim"),
    (r"Integrated EC Stack", "renamed: use 'EC Research Stack'"),
]


def main() -> int:
    failures = []
    for rel in TARGETS:
        text = (ROOT / rel).read_text(encoding="utf-8")
        for pattern, why in BANNED:
            for m in re.finditer(pattern, text, re.I):
                line = text.count("\n", 0, m.start()) + 1
                failures.append(f"{rel}:{line}: {m.group(0)!r} — {why}")
    if failures:
        print("Retired first-edition formulations found:")
        for f in failures:
            print("  " + f)
        return 1
    print(f"language check clean ({len(TARGETS)} files, {len(BANNED)} patterns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
