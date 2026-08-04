"""genome.py — the game's mutable constitution, and the hand that rewrites it.

Tier 1+2 of the self-modification design (see the book, Chapter 10 —
Consciousness and Self-Modification). A small set of world constants — the
GENES below — no longer live in the source as literals. They live in
`dna.py`: a real Python module that THE GAME ITSELF WRITES. At every
ouroboros turning the game proposes one mutation to one gene; every few
hundred generations mid-run it proposes a smaller one. Each adopted mutation
bumps the genome version, appends to the ledger, and rewrites dna.py — so
the game's own source drifts, run over run, within stated bounds.

Safeguards (the cheap three that make self-modification survivable):
  * every gene is clamped to a stated [lo, hi] — mutation cannot leave it,
  * the rewritten dna.py must pass compile() before it replaces the old one
    (atomic temp-file + os.replace; a half-written file can never load),
  * if dna.py is missing, corrupt, or out of bounds, the embedded defaults
    below win and the game runs pristine.

Mutations are DETERMINISTIC: each derives from (ouroboros seed, turning) or
(seed, generation) through its own random.Random — the named simulation
streams in simcore are never consumed, so the world's replay behavior is
unchanged by the act of mutating. Headless runs (EC_HEADLESS=1) load dna.py
but never mutate or write, so phase-6 equivalence tests stay meaningful.
EC_GENOME=0 disables the whole system (pristine defaults, no reads, no
writes).

dna.py is gitignored, like .ouroboros.json: your copy of the game diverges
from everyone else's. That is the point.
"""
import hashlib
import os
import random
import tempfile

DNA_NAME = "dna.py"

# name -> (default, lo, hi, step fraction of range, kind, what it governs)
GENES = {
    "food_bloom":        (1.00, 0.60, 1.60, 0.10, float, "food-bloom proportion seeded into Flatland"),
    "metabolism":        (0.25, 0.10, 0.60, 0.10, float, "energy an agent burns per step"),
    "rest_recovery":     (5.0,  2.0, 12.0, 0.10, float, "energy regained per resting frame"),
    "reproduction_cost": (20.0, 10.0, 40.0, 0.10, float, "energy parents transfer to a child"),
    "max_reproductions": (2,    1,    4,    0.34, int,   "births allowed per generation"),
    "lifespan_scale":    (1.00, 0.60, 1.60, 0.08, float, "multiplier on cell lifespans"),
    "goal_period":       (10,   4,    20,   0.15, int,   "generations between goal moves"),
    "rift_drain":        (1.50, 0.80, 2.50, 0.10, float, "Spaceland: mind lost per second in a rift"),
    "ambient_drain":     (0.035, 0.015, 0.080, 0.10, float, "Spaceland: ambient mind drain per frame"),
    "mutation_scale":    (1.00, 0.50, 2.00, 0.12, float, "how bold the next mutation may be (meta)"),
}

_HEADER = '''"""dna.py — WRITTEN BY THE GAME ITSELF. Edits by hand will be overwritten.

This module is EC-2D-Land\'s mutable constitution: the world constants the
game is permitted to rewrite as it progresses (see genome.py for bounds and
safeguards, and the book\'s Chapter 10 for why). The LEDGER is this file\'s
own history — every mutation the game has adopted, in order. Delete this
file to return the world to pristine defaults.
"""
'''


def _defaults():
    return {k: spec[0] for k, spec in GENES.items()}


def _clamp(name, value):
    _, lo, hi, _, kind, _ = GENES[name]
    v = max(lo, min(hi, value))
    return int(round(v)) if kind is int else round(float(v), 4)


def _render_dna(genome, version, ledger):
    lines = [_HEADER, f"GENOME_VERSION = {version}", "", "GENOME = {"]
    for name, spec in GENES.items():
        lines.append(f"    {name!r}: {genome[name]!r},  # {spec[5]}")
    lines.append("}")
    lines.append("")
    lines.append("LEDGER = [  # (version, when, gene, old, new)")
    for entry in ledger:
        lines.append(f"    {entry!r},")
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


class Genome:
    """Loads dna.py if the game has written one; mutates and rewrites it."""

    def __init__(self, game_dir, frozen=False):
        self.path = os.path.join(game_dir, DNA_NAME)
        self.frozen = frozen or os.environ.get("EC_GENOME", "1") in ("0", "off")
        self.g = _defaults()
        self.version = 1
        self.ledger = []
        self.last_change = None       # human-readable, for the HUD/console
        if os.environ.get("EC_GENOME", "1") in ("0", "off"):
            return                    # pristine: don't even read dna.py
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as fh:
                source = fh.read()
        except OSError:
            return
        try:
            ns = {}
            exec(compile(source, self.path, "exec"), ns)  # our own artifact
            raw = ns.get("GENOME", {})
            self.version = int(ns.get("GENOME_VERSION", 1))
            self.ledger = list(ns.get("LEDGER", []))[-60:]
            for name in GENES:
                if name in raw:
                    self.g[name] = _clamp(name, raw[name])
        except Exception as exc:                          # corrupt → pristine
            print(f"genome: dna.py unreadable, running pristine ({exc})")
            self.g = _defaults()
            self.version = 1
            self.ledger = []

    def _write(self):
        source = _render_dna(self.g, self.version, self.ledger)
        try:
            compile(source, self.path, "exec")            # syntax gate
        except SyntaxError as exc:                        # never adopt a bad file
            print(f"genome: refused to write invalid dna.py ({exc})")
            return False
        try:
            d = os.path.dirname(self.path) or "."
            fd, tmp = tempfile.mkstemp(dir=d, prefix=".dna-", suffix=".py")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(source)
            os.replace(tmp, self.path)                    # atomic adoption
            return True
        except OSError as exc:
            print(f"genome: could not write dna.py ({exc})")
            return False

    # -- mutation ------------------------------------------------------------
    def _mutate(self, rng, when, scale=1.0):
        name = rng.choice(list(GENES))
        default, lo, hi, step, kind, _ = GENES[name]
        old = self.g[name]
        span = (hi - lo) * step * scale * self.g["mutation_scale"]
        new = _clamp(name, old + rng.uniform(-span, span) * (1 if kind is float else 2))
        if kind is int and new == old:                    # ints must actually move
            new = _clamp(name, old + rng.choice((-1, 1)))
        if new == old:
            return None
        self.g[name] = new
        self.version += 1
        self.ledger.append((self.version, when, name, old, new))
        self.ledger = self.ledger[-60:]
        self.last_change = f"{name} {old} → {new}"
        return name, old, new

    def turning(self, iteration, seed):
        """A full mutation at each ouroboros turning — the world reborn changed."""
        if self.frozen:
            return None
        rng = random.Random(_derive(seed, "turning", iteration))
        changed = self._mutate(rng, f"turning {iteration}", scale=1.0)
        if changed and self._write():
            print(f"genome: turning {iteration} rewrote dna.py — "
                  f"{self.last_change} (v{self.version})")
        return changed

    def milestone(self, generation, seed):
        """A smaller mid-run mutation every few hundred generations."""
        if self.frozen or generation <= 0 or generation % 500:
            return None
        rng = random.Random(_derive(seed, "milestone", generation))
        changed = self._mutate(rng, f"gen {generation}", scale=0.35)
        if changed and self._write():
            print(f"genome: gen {generation} rewrote dna.py — "
                  f"{self.last_change} (v{self.version})")
        return changed

    def hud_tag(self):
        return f"genome v{self.version}"


def _derive(seed, kind, n):
    h = hashlib.sha256(f"{seed}:{kind}:{n}".encode()).digest()
    return int.from_bytes(h[:8], "big")


# Module-level access for satellite modules (spaceland) without circular
# imports: EC-2D-Land.py installs the live instance here at startup.
_CURRENT = None


def install(instance):
    global _CURRENT
    _CURRENT = instance
    return instance


def gene(name):
    """Current value of a gene (pristine default when no instance installed)."""
    if _CURRENT is not None:
        return _CURRENT.g[name]
    return GENES[name][0]
