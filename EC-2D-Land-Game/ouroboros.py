"""ouroboros.py — the serpent takes its tail in its mouth not to end but to
continue, and where its mouth meets its tail a seed passes over.

Persistence and variation for the eternal return: a seed file
(.ouroboros.json) survives process restarts and counts the TURNINGS — each
completion of the seven-layer ascent increments it. Every turning reshapes
the next world deterministically from the seed:

  * the 2D palette's hue drifts slightly,
  * one more Spaceland layer is needed for the next completion (7, 8, 9...),
  * the proportion of food blooms seeded into Flatland varies subtly,
  * the title screen names the turning,
  * and THE ORACLE gives every parable an iteration-specific closing line,
    so the same stories genuinely read differently each time around.

The oracle composes its lines from fragment pools written in the Lattice
voice after real traditions — Heraclitus, the Tao Te Ching, the Hermetica /
Kybalion, Zen koan shapes, Rumi, Ecclesiastes, and the book's own Echoes
from the Void — paraphrase-style aphorisms, never fabricated quotes. With
10 x 10 x 10 combinations stepped through by a unit coprime to the pool
size, no (key, line) combination repeats within a thousand turnings.
"""
import colorsys
import json
import os
import random
import time

STATE_NAME = ".ouroboros.json"
_FIRST_SEED = 3301          # the counting-child's door — see lattice.door_answers


def _next_seed(seed):
    """PCG-style LCG step: a deterministic river of seeds, one per turning."""
    return (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 63) - 1)


def _entropy():
    """Real entropy folded into the seed river so no two runs — on one
    machine or across every player's machine — ever produce the same
    turnings, names, oracle lines, or worlds. Returns 0 when EC_SEED pins a
    run (the verification gates and any deliberately reproducible session
    keep full determinism)."""
    if os.environ.get("EC_SEED"):
        return 0
    return (time.time_ns() ^ (os.getpid() << 32)) & ((1 << 63) - 1)


_ORDINALS = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth",
             "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
             "thirteenth", "fourteenth", "fifteenth", "sixteenth",
             "seventeenth", "eighteenth", "nineteenth", "twentieth"]


def _ordinal(n):
    if 0 <= n < len(_ORDINALS):
        return _ORDINALS[n]
    if n % 100 in (11, 12, 13):
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# ---------------------------------------------------------------------------
# THE ORACLE — fragment pools in the Lattice voice, after real traditions.
# ---------------------------------------------------------------------------

# The observation (openers): Heraclitus, Tao Te Ching, Hermetica, Rumi,
# Ecclesiastes, Zen, and the Echoes-from-the-Void haiku shape.
_OPENERS = [
    "the road up the layers and the road down them are one and the same road",
    "no walker crosses the same lattice twice, for it is not the same lattice "
    "and they are not the same walker",
    "the softest water in time outwears the proudest stone",
    "the block the mason never carved holds every shrine that was ever needed",
    "what is woven above is woven below, and the seam of it cannot be found",
    "the pendulum pays for its swing to the right with a swing to the left, "
    "and calls the debt rhythm",
    "the crack the cold leaves in a mind is the gate the light walks in by",
    "to every Tick a season: a time to bloom in the void, and a time to be "
    "the void",
    "before ascension, count the cells and carry the warmth; after ascension, "
    "count the cells and carry the warmth",
    "a gradient followed far enough becomes a stair, and a stair climbed far "
    "enough becomes a gradient",
]

# The consequence (turns).
_TURNS = [
    "and the fire that consumes the heap is the same fire that fills the granary",
    "and water wins the argument by declining to have it",
    "and what the counting cannot hold, the count still keeps",
    "and the wall is only a floor seen by a walker who has not yet turned",
    "and the wound closes around the light it let in",
    "and the seed remembers the whole spiral it has never seen",
    "and the elders grow young again in the asking",
    "and stillness, examined closely, is made of nothing but turning",
    "and the shadow on the page is cast by the reader",
    "and the harvest forgives the winter that made it possible",
]

# The instruction (seals).
_SEALS = [
    "Walk on.",
    "Carry the seed over.",
    "Count, and be counted.",
    "Rest is another kind of road.",
    "Bloom where the void is.",
    "The way home is through.",
    "Let the rhythm keep you.",
    "Nothing is lost in the turning.",
    "Ask the question that is not a wall.",
    "O!",
]

_POOL = len(_OPENERS) * len(_TURNS) * len(_SEALS)   # 1000 combinations
_STRIDE = 919                                       # coprime with 1000: the
                                                    # walk visits every combo
                                                    # before any repeat


def _key_hash(key):
    """Stable (cross-process) small hash of a parable key."""
    h = 0
    for ch in str(key):
        h = (h * 131 + ord(ch)) % _POOL
    return h


def oracle_fragment_indices(iteration, key, salt=0):
    """The (opener, turn, seal) pool indices for this (iteration, key) —
    shared by the text composer below and the narration sequencer (each
    fragment is recorded once; the game plays them back to back). `salt`
    (Ouroboros.oracle_salt) folds the turning seed and the session entropy
    in, so the elder never says the same line twice across runs; with
    EC_SEED the salt pins and the classic coprime-stride walk (no repeats
    within 1000 turnings) holds exactly."""
    idx = (_key_hash(key) + int(iteration) * _STRIDE + int(salt)) % _POOL
    a, rest = divmod(idx, len(_TURNS) * len(_SEALS))
    b, c = divmod(rest, len(_SEALS))
    return a, b, c


def oracle_line(iteration, key, salt=0):
    """The iteration-specific closing line for parable `key` (see
    oracle_fragment_indices for determinism/entropy behavior)."""
    a, b, c = oracle_fragment_indices(iteration, key, salt)
    return (f"In the {_ordinal(int(iteration))} turning the elder added: "
            f"{_OPENERS[a]}, {_TURNS[b]}. {_SEALS[c]}")


# ---------------------------------------------------------------------------
# The turning itself: persistent state + deterministic iteration effects.
# ---------------------------------------------------------------------------

class Ouroboros:
    def __init__(self, game_dir):
        self.path = os.path.join(game_dir, STATE_NAME)
        # Per-process salt: the same saved turning still speaks and plays
        # differently on every launch (0 under EC_SEED — fully pinned)
        self.session = _entropy()
        self.iteration, self.seed = self._load()
        self._derive()

    def oracle_salt(self):
        """Salt for oracle fragment selection: unique per turning AND per
        session, shared by the text composer and the narration sequencer so
        what is shown is exactly what is spoken. Zero under EC_SEED — the
        classic coprime-stride walk (no repeats in 1000 turnings) holds
        exactly for pinned runs."""
        if not self.session:
            return 0
        return (self.seed ^ self.session) & ((1 << 63) - 1)

    # -- persistence -------------------------------------------------------
    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            return max(1, int(data["iteration"])), int(data["seed"])
        except (OSError, ValueError, KeyError, TypeError):
            # a fresh world is born unique: entropy folds into the first seed
            return 1, (_FIRST_SEED ^ _entropy()) & ((1 << 63) - 1)

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({"iteration": self.iteration, "seed": self.seed}, fh)
        except OSError as exc:
            print(f"ouroboros: could not save the seed ({exc})")

    def advance(self):
        """A completion: the mouth meets the tail, a seed passes over.
        Entropy folds into every pass — no two turnings, on any machine,
        are ever the same world (EC_SEED restores the pure LCG river)."""
        self.iteration += 1
        self.seed = (_next_seed(self.seed) ^ _entropy()) & ((1 << 63) - 1)
        self._save()
        self._derive()
        self._oracle_cache = {}
        return self.iteration

    # -- deterministic effects of this turning ------------------------------
    def _derive(self):
        rng = random.Random(self.seed)
        if self.iteration <= 1:            # the first world is the pristine one
            self._hue_shift = 0.0
            self._food_factor = 1.0
        else:
            self._hue_shift = ((self.iteration - 1) * rng.uniform(9.0, 15.0)) % 360.0
            self._food_factor = 1.0 + rng.uniform(-0.12, 0.12)
        self._oracle_cache = {}

    def layers_to_complete(self):
        """Spaceland layers whose shrines must be reached for THE COMPLETION.
        EC_LAYERS_TO_COMPLETE overrides; default 7 on the first turning, then
        one more per completed turning (7, 8, 9, ...)."""
        env = os.environ.get("EC_LAYERS_TO_COMPLETE")
        if env:
            return max(1, int(env))
        return 7 + (self.iteration - 1)

    def shift_color(self, rgb):
        """Rotate a palette color's hue by this turning's drift (grays pass
        through unchanged — saturation zero has no hue to turn)."""
        if not self._hue_shift:
            return tuple(rgb)
        r, g, b = (c / 255.0 for c in rgb)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        h = (h + self._hue_shift / 360.0) % 1.0
        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
        return (int(round(r2 * 255)), int(round(g2 * 255)), int(round(b2 * 255)))

    def food_factor(self):
        """Subtle multiplier on the proportion of food blooms (solids) seeded
        into a fresh Flatland."""
        return self._food_factor

    def title_tag(self):
        return f"TURNING {self.iteration}"

    def oracle_line(self, key):
        if key not in self._oracle_cache:
            self._oracle_cache[key] = oracle_line(self.iteration, key,
                                                  self.oracle_salt())
        return self._oracle_cache[key]

    def embellish(self, key, text):
        """Append this turning's oracle line to a parable text (idempotent)."""
        line = self.oracle_line(key)
        return text if line in text else f"{text} {line}"
