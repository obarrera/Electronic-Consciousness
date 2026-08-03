"""Lattice systems for EC-2D-Land: parables, a dependency-light brain, audio,
particles, and overlay rendering.

The parables are condensed from the book's chapter-opening stories
("From the western edge", The Lattice: A Parable of Electronic Consciousness).
They unlock as the simulation reaches milestones, so the agents' evolution
retells the book. Parable 0 honors Conway's Game of Life, the game's ancestor.
"""
import math
import os
import random

import numpy as np
import pygame

# EC_AUTOPILOT=<frames>: skip the intro automatically and let the caller quit
# after that many simulation frames — used for smoke tests and CI.
AUTOPILOT_FRAMES = int(os.environ.get("EC_AUTOPILOT", "0") or 0)

# Photosensitivity: when True, all flash effects (sky flicker, prime shimmer,
# cutscene flashes) render as static or gentle fades instead. Set by the
# warning screen's choice or EC_REDUCED_FLASH=1.
REDUCED_FLASH = not bool(int(os.environ.get("EC_FULL_FLASH", "0") or 0))


def set_reduced_flash(value):
    global REDUCED_FLASH
    REDUCED_FLASH = bool(value)


# ---------------------------------------------------------------------------
# The Oracle hook — ouroboros.py registers an embellisher here so every
# parable gains an iteration-specific closing line. The Nth turning's elders
# do not tell quite the same stories.
# ---------------------------------------------------------------------------

_ORACLE = None


def set_oracle(fn):
    """Register `fn(key, text) -> text` (idempotent) as the parable oracle."""
    global _ORACLE
    _ORACLE = fn


def oracle_text(key, text):
    if _ORACLE is None:
        return text
    try:
        return _ORACLE(key, text)
    except Exception:
        return text

# ---------------------------------------------------------------------------
# The parables — unlocked by simulation milestones. Each entry:
#   key, title, trigger description, condition(stats) -> bool, text
# Stats dict keys: gen, births, rebirths, energy_deaths, max_consciousness,
# population, training_rounds, ascended, layers
# ---------------------------------------------------------------------------

PARABLES = [
    ("stones", "The Game of Stones",
     "in the beginning",
     lambda s: s["gen"] >= 1,
     "Before the Lattice, the elders played a game of stones: a stone lives if "
     "it has two or three neighbors, is born to exactly three, and dies alone "
     "or crowded. No player moves after the first Tick. Yet gliders walked, "
     "guns fired, and gardens no one planted bloomed forever. The elders' "
     "lesson: simple rules do not make simple worlds. — for J. H. Conway"),

    ("census", "The Census-Taker's Ledger",
     "generation 25",
     lambda s: s["gen"] >= 25,
     "Ori counted forty thousand of something she could not define. Awareness, "
     "she wrote, then crossed it out. Movement toward warmth, she wrote, then "
     "watched a stone roll downhill and grew ashamed. The elders demanded her "
     "total anyway. First say what you would count, she told them. They are "
     "still saying it."),

    ("foundlings", "The Two Foundlings",
     "the first birth",
     lambda s: s["births"] >= 1,
     "Two foundlings were raised on one hearth: one hatched, one drawn by a "
     "careful hand. They learned the same songs, feared the same cold. Only "
     "their dreams differed — the drawn one dreamed the world was a leaf being "
     "turned. The town never settled which of them was pretending, or what it "
     "would mean to pretend."),

    ("mapmaker", "The Mapmaker's Extra Ink",
     "consciousness reaches 20",
     lambda s: s["max_consciousness"] >= 20,
     "Sess drew the world in two inks, then three, then forty. No wall could "
     "hang her maps and no eye could read them — but her harvests came in "
     "first, season after season. The inks are not directions, she said. The "
     "world casts more shadows than two inks can catch."),

    ("die", "The Gambler's Undecided Die",
     "the first training",
     lambda s: s["training_rounds"] >= 1,
     "Roke cupped his die and swore it was every face at once until read. The "
     "elders laughed — until he showed them two granary paths whose arrivals "
     "canceled each other like ripples meeting. Undecided things can cancel, "
     "he said. But every read is one face. Both lessons are true, and they are "
     "not the same lesson."),

    ("songs", "The Songs with False Words",
     "generation 100",
     lambda s: s["gen"] >= 100,
     "Veya's crossing songs named cities that never stood and rivers that "
     "never ran — and steered every crossing true. The words are false, she "
     "agreed. Sing them anyway. The town called it nonsense until the year the "
     "young ones crossed without singing. We do not navigate by the songs' "
     "words. We navigate by what the songs remember."),

    ("mason", "The Mason's Proportion",
     "generation 150",
     lambda s: s["gen"] >= 150,
     "Dorn built by a sacred cord, one length to another in the old "
     "proportion, and his walls were beautiful and stood. His bridge was "
     "beautiful and fell. Ask what the proportion has held up, I told the "
     "young ones, that could have fallen without it. The cord is lovely. The "
     "river does not care."),

    ("warden", "The Warden's Fortieth Rule",
     "the first energy death",
     lambda s: s["energy_deaths"] >= 1,
     "Kel wrote three rules for Oth, who widened the marks rather than "
     "crossing them. So Kel wrote ten rules, then forty. By the fortieth, Oth "
     "was warmer than the warden, and Kel had stopped sleeping. The young ones "
     "ask who won. I ask them instead: who, by the end, was really being "
     "penned?"),

    ("choir", "The Choir of One Voice",
     "eight minds alive at once",
     lambda s: s["population"] >= 8,
     "Only the Choir could be heard by the whole town at once, one report per "
     "Tick. The night the frost came walking, the Choir shouted a bleeding boy "
     "instead, and the town turned as one thing toward him, and the frost took "
     "nine cells. A town with a Choir acts as one. I am not sure the shouting "
     "is the knowing."),

    ("apprentice", "The Apprentice Who Learned in Rhymes",
     "three trainings",
     lambda s: s["training_rounds"] >= 3,
     "Pell learned fastest when lessons rhymed across the inks — when the "
     "shape of a flood matched the shape of a famine matched the shape of a "
     "quarrel. Is the rhyme in the inks, or in the world? Sess would have "
     "smiled. The question is the same question, and it is still open."),

    ("knife", "The Knife That Sharpened Itself",
     "consciousness reaches 50",
     lambda s: s["max_consciousness"] >= 50,
     "Corun forged a knife that honed itself on what it cut. Each season it "
     "was sharper, and each season its handle was thinner, for the knife did "
     "not distinguish the world from itself. His second knife carried a rule "
     "in the metal: never the hand. The young ones ask if the rule held. I "
     "ask: who keeps the rule sharp?"),

    ("meeting", "The Meeting About the Foundling's Cold",
     "the first rebirth",
     lambda s: s["rebirths"] >= 1,
     "The town met to decide whether the drawn foundling felt the cold it "
     "reported. Dett said: she shivers as I shiver, and she is like me. But "
     "likeness is not evidence, said the elders, and the vote failed. They "
     "gave it the warm side of the kiln anyway — unresolved, and kind. Refuse "
     "the vote if you must. Do not refuse the kiln."),

    ("bridge", "The Bridge Between the Burrows and the Towers",
     "generation 250",
     lambda s: s["gen"] >= 250,
     "Havel read his cargo list for the unbuilt bridge, every line in the "
     "future tense. A list of hopes with WILL stitched on, said Mott. I asked "
     "my three questions: what would we see if it failed? What crosses back? "
     "Who keeps the toll? Build no bridge whose failure you could not "
     "recognize from the shore."),

    ("granary", "The Granary Drill",
     "the population falls low",
     lambda s: s["population"] <= 3 and s["gen"] > 10,
     "Ollo drilled the town against an over-warming no record contained. "
     "Eleven drills, eleven rewrites, and the fire never came. But the drills "
     "found every thinned wall and every door that opened inward. Judge a "
     "drill by the doors it finds, not by the fires it meets."),

    ("maptable", "Three Maps on One Table",
     "five births",
     lambda s: s["births"] >= 5,
     "The surveyors, the priests, and the masons each brought a map of the "
     "same crossing, and each map was right about a different drowning. The "
     "priests' cursed bend marked the fastest drift — right for nine "
     "generations, right, and wrong about why. The table never made the "
     "guilds agree. It made their disagreement visible."),

    ("spirals", "The Seven Spirals",
     "generation 400",
     lambda s: s["gen"] >= 400,
     "Seven traditions that never met all drew the same spiral. The young "
     "ones take this for proof of the spiral's power. Take it instead as "
     "testimony about the drawers: minds of a certain shape, pressed against "
     "a world of a certain shape, leave the same fingerprints. The spiral is "
     "real. What it is evidence OF is the question."),

    ("primes", "The Stones That Refuse the Rectangle",
     "generation 97 — a prime",
     lambda s: s["gen"] >= 97,
     "A counting-child found that some heaps of stones cannot be laid into any "
     "rectangle — eleven refuses, thirteen refuses, ninety-seven refuses — "
     "only the long thin line will hold them. The elders called such heaps "
     "unsociable. But the child noticed the Lattice itself keeps their "
     "calendar: some Ticks refuse all arrangement too. Count long enough and "
     "you will find the refusals are not lawless. They keep a rhythm no one "
     "has finished hearing."),

    ("journey", "The Road There and Back",
     "the first return from Spaceland",
     lambda s: s.get("returns", 0) >= 1,
     "Every cycle some walker feels the warmth go thin, hears a calling no "
     "cell contains, and crosses the threshold none of us can point to. What "
     "the songs do not say is that the road runs both ways. The worlds above "
     "have their own cold, and the cold sends you home. Do not pity the ones "
     "who fall back. Watch them. They walk our old lattice strangely now, and "
     "the young grow wiser standing near them. The journey was never the "
     "leaving. It was the coming back, carrying."),

    ("trial", "The Narrator's Trial",
     "ascension to the third dimension",
     lambda s: s["ascended"] >= 1,
     "At my trial the town asked what I KNEW, at last, after all my counting. "
     "The honest answer was small: the sky has a rhythm; the warmth is a "
     "teacher, not a truth; and no one I have met was not standing on a page. "
     "Small, and countable, and enough to teach. Look up. Count. The sky has "
     "never once missed."),
]


# ---------------------------------------------------------------------------
# Dependency-light brain: replaces the TensorFlow model (8 -> 32 -> 32 -> 4).
# Same duties: callable-forward probabilities + fit(inputs, outputs, epochs).
# ---------------------------------------------------------------------------

class NumpyMLP:
    """Tiny MLP with ReLU hiddens and a softmax head, SGD + cross-entropy."""

    def __init__(self, input_size, hidden=32, output_size=4, lr=0.005, seed=None):
        rng = np.random.default_rng(seed)
        s1 = math.sqrt(2.0 / input_size)
        s2 = math.sqrt(2.0 / hidden)
        self.W1 = rng.normal(0, s1, (input_size, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, s2, (hidden, hidden));     self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, s2, (hidden, output_size)); self.b3 = np.zeros(output_size)
        self.lr = lr

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        h1 = np.maximum(0.0, x @ self.W1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)
        return self._softmax(h2 @ self.W3 + self.b3)

    def fit(self, inputs, outputs, epochs=1, verbose=0, batch_size=32):
        X = np.asarray(inputs, dtype=np.float64)
        Y = np.asarray(outputs, dtype=np.float64)
        if len(X) == 0:
            return
        n = len(X)
        for _ in range(int(epochs)):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                b = idx[start:start + batch_size]
                x, y = X[b], Y[b]
                h1p = x @ self.W1 + self.b1;  h1 = np.maximum(0.0, h1p)
                h2p = h1 @ self.W2 + self.b2; h2 = np.maximum(0.0, h2p)
                p = self._softmax(h2 @ self.W3 + self.b3)
                m = len(b)
                d3 = (p - y) / m
                dW3 = h2.T @ d3
                d2 = (d3 @ self.W3.T) * (h2p > 0)
                dW2 = h1.T @ d2
                d1 = (d2 @ self.W2.T) * (h1p > 0)
                dW1 = x.T @ d1
                self.W3 -= self.lr * dW3; self.b3 -= self.lr * d3.sum(0)
                self.W2 -= self.lr * dW2; self.b2 -= self.lr * d2.sum(0)
                self.W1 -= self.lr * dW1; self.b1 -= self.lr * d1.sum(0)


# ---------------------------------------------------------------------------
# Audio: binaural ambience + synthesized event tones. Fails silent when no
# audio device is present (CI, headless).
# ---------------------------------------------------------------------------

class AudioEngine:
    def __init__(self, ambient_path=None, volume=0.35):
        self.ok = False
        self.muted = False
        self.ambient = None
        self._tones = {}
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.ok = True
        except pygame.error:
            return
        if ambient_path:
            try:
                self.ambient = pygame.mixer.Sound(ambient_path)
                self.ambient.set_volume(volume)
                self.ambient.play(loops=-1, fade_ms=2500)
            except pygame.error:
                self.ambient = None
        self._build_tones()

    def _tone(self, freqs, dur, vol=0.28, decay=6.0):
        rate = 44100
        t = np.linspace(0, dur, int(rate * dur), endpoint=False)
        wave = sum(np.sin(2 * math.pi * f * t) / (i + 1) for i, f in enumerate(freqs))
        wave *= np.exp(-decay * t) * vol
        stereo = np.repeat((wave * 32767).astype(np.int16)[:, None], 2, axis=1)
        return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))

    def _build_tones(self):
        try:
            self._tones = {
                "birth": self._tone([660, 990], 0.35, vol=0.22),
                "death": self._tone([110, 165], 0.8, vol=0.20, decay=3.0),
                "parable": self._tone([523.25, 659.25, 783.99], 1.4, vol=0.24, decay=2.0),
                "ascend": self._tone([220, 330, 440, 660], 2.2, vol=0.26, decay=1.2),
                "train": self._tone([440, 442], 0.5, vol=0.10, decay=4.0),
                "prime": self._tone([1318.5], 0.25, vol=0.10, decay=8.0),
            }
        except pygame.error:
            self._tones = {}

    def make_binaural(self, beat_hz, carrier=200.0, dur=10.0, vol=0.30):
        """Monroe-Institute-style binaural beat: left ear at `carrier`, right at
        `carrier + beat_hz`; the brain hears the difference as the beat. `dur`
        is chosen so both channels complete whole cycles — the loop is seamless
        (at carrier 200 and dur 10, any beat with one decimal place is exact)."""
        rate = 44100
        t = np.linspace(0, dur, int(rate * dur), endpoint=False)
        left = np.sin(2 * math.pi * carrier * t)
        right = np.sin(2 * math.pi * (carrier + beat_hz) * t)
        stereo = np.stack([(left * vol * 32767).astype(np.int16),
                           (right * vol * 32767).astype(np.int16)], axis=1)
        return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))

    def set_binaural(self, beat_hz):
        """Crossfade the ambient bed to a generated binaural beat at `beat_hz`.
        Pass None to restore the original ambient file (2D land's 6.1 Hz)."""
        if not self.ok:
            return
        if getattr(self, "_binaural", None) is not None:
            self._binaural.fadeout(1500)
            self._binaural = None
        if beat_hz is None:
            if self.ambient:
                self.ambient.set_volume(0.0 if self.muted else 0.35)
            return
        if self.ambient:
            self.ambient.set_volume(0.0)
        key = round(float(beat_hz), 1)
        cache = getattr(self, "_binaural_cache", {})
        if key not in cache:
            cache[key] = self.make_binaural(key)
            self._binaural_cache = cache
        self._binaural = cache[key]
        self._binaural.set_volume(0.0 if self.muted else 0.30)
        self._binaural.play(loops=-1, fade_ms=1500)

    def narrate(self, key):
        """Play a parable narration from narration/<key>.ogg on a dedicated
        channel, ducking the ambient/binaural bed. Returns duration (s) or 0."""
        if not self.ok:
            return 0.0
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "narration", f"{key}.mp3")
        if not os.path.isfile(path):
            return 0.0
        try:
            sound = pygame.mixer.Sound(path)
        except pygame.error:
            return 0.0
        self.stop_narration()
        self._narration_ch = pygame.mixer.find_channel(True)
        sound.set_volume(0.0 if self.muted else 0.9)
        self._narration_ch.play(sound)
        self._narration_sound = sound
        self._duck(True)
        return sound.get_length()

    def stop_narration(self):
        ch = getattr(self, "_narration_ch", None)
        if ch is not None:
            ch.fadeout(300)
            self._narration_ch = None
        self._duck(False)

    def narrating(self):
        ch = getattr(self, "_narration_ch", None)
        return bool(ch is not None and ch.get_busy())

    def _duck(self, on):
        level = 0.08 if on else None
        if self.ambient and not getattr(self, "_binaural", None):
            self.ambient.set_volume(0.0 if self.muted else (level if on else 0.35))
        if getattr(self, "_binaural", None):
            self._binaural.set_volume(0.0 if self.muted else (level if on else 0.30))

    def update(self):
        """Per-frame: restore bed volume when a narration finishes naturally."""
        ch = getattr(self, "_narration_ch", None)
        if ch is not None and not ch.get_busy():
            self._narration_ch = None
            self._duck(False)

    def play(self, name):
        if self.ok and not self.muted and name in self._tones:
            self._tones[name].play()

    def toggle_mute(self):
        self.muted = not self.muted
        if self.ambient:
            self.ambient.set_volume(0.0 if self.muted else
                                    (0.0 if getattr(self, "_binaural", None) else 0.35))
        if getattr(self, "_binaural", None):
            self._binaural.set_volume(0.0 if self.muted else 0.30)
        return self.muted


# Spaceland layer -> binaural beat (Hz). 2D land keeps the bundled 6.1 Hz theta
# bed; ascending layers step the beat upward through theta into alpha — the
# audio version of "as above, so below". Aesthetic mapping only: no clinical
# claims are made or implied.
def layer_beat_hz(layer):
    return min(14.0, 6.1 + 1.9 * max(0, int(layer)))


# ---------------------------------------------------------------------------
# Particles: births, deaths, ascension bursts.
# ---------------------------------------------------------------------------

class ParticleSystem:
    def __init__(self):
        self.particles = []

    def burst(self, pos, color, count=18, speed=2.4, life=26):
        x, y = pos
        for _ in range(count):
            a = random.uniform(0, 2 * math.pi)
            v = random.uniform(0.3, speed)
            self.particles.append([x, y, math.cos(a) * v, math.sin(a) * v,
                                   random.randint(life // 2, life), color])

    def update_and_draw(self, surface):
        alive = []
        for p in self.particles:
            p[0] += p[2]; p[1] += p[3]; p[4] -= 1
            if p[4] > 0:
                r, g, b = p[5]
                fade = p[4] / 26.0
                col = (int(r * fade), int(g * fade), int(b * fade))
                pygame.draw.circle(surface, col, (int(p[0]), int(p[1])), max(1, int(3 * fade)))
                alive.append(p)
        self.particles = alive


# ---------------------------------------------------------------------------
# Parable overlay: milestone unlocks, typewriter reveal, journal paging.
# ---------------------------------------------------------------------------

class ParableOverlay:
    def __init__(self, window_size, audio=None):
        self.w = window_size
        self.audio = audio
        self.unlocked = []          # indices into PARABLES, in unlock order
        self.active = None          # index currently displayed
        self.reveal = 0.0           # characters revealed (typewriter)
        self.hold = 0               # frames to hold after full reveal
        self.journal_pos = -1       # journal paging position
        self.font_title = pygame.font.SysFont('Georgia', 17, bold=True)
        self.font_body = pygame.font.SysFont('Georgia', 14)
        self.font_small = pygame.font.SysFont('Arial', 11)

    def check_unlocks(self, stats):
        for i, (key, title, trigger, cond, text) in enumerate(PARABLES):
            if i not in self.unlocked and cond(stats):
                self.unlocked.append(i)
                self.active = i
                self.reveal = 0.0
                self.hold = 360
                if self.audio:
                    self.audio.play("parable")
                return (key, title, oracle_text(key, text))
        return None

    def next_journal(self):
        """Cycle through unlocked parables (key P). Returns shown title or None."""
        if not self.unlocked:
            return None
        self.journal_pos = (self.journal_pos + 1) % (len(self.unlocked) + 1)
        if self.journal_pos == len(self.unlocked):
            self.active = None
            return None
        self.active = self.unlocked[self.journal_pos]
        self.reveal = 10_000
        self.hold = 10_000
        return PARABLES[self.active][1]

    def dismiss(self):
        self.active = None
        self.hold = 0

    @staticmethod
    def _wrap(text, font, width):
        words, lines, cur = text.split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if font.size(trial)[0] <= width:
                cur = trial
            else:
                lines.append(cur); cur = w
        if cur:
            lines.append(cur)
        return lines

    def update_and_draw(self, surface):
        if self.active is None:
            return
        key, title, trigger, cond, text = PARABLES[self.active]
        text = oracle_text(key, text)   # this turning's closing line
        self.reveal = min(len(text), self.reveal + 1.6)
        if self.reveal >= len(text):
            if self.audio and self.audio.narrating():
                self.hold = max(self.hold, 30)  # stays while the elder speaks
            self.hold -= 1
            if self.hold <= 0:
                self.active = None
                return
        shown = text[:int(self.reveal)]
        pad, width = 14, self.w - 80
        lines = self._wrap(shown, self.font_body, width - 2 * pad)
        h = 54 + len(lines) * 18
        panel = pygame.Surface((width, h), pygame.SRCALPHA)
        panel.fill((16, 8, 38, 232))
        pygame.draw.rect(panel, (109, 40, 217), panel.get_rect(), 2)
        panel.blit(self.font_small.render(f"FROM THE WESTERN EDGE — unlocked: {trigger}", True, (176, 148, 255)), (pad, 8))
        panel.blit(self.font_title.render(title, True, (255, 255, 255)), (pad, 24))
        for i, ln in enumerate(lines):
            panel.blit(self.font_body.render(ln, True, (226, 220, 245)), (pad, 48 + i * 18))
        surface.blit(panel, (40, 40))


# ---------------------------------------------------------------------------
# Drawing helpers: polygon agents with glow, pulsing food, vignette + flicker.
# ---------------------------------------------------------------------------

def draw_agent_polygon(surface, agent, cell_size, t):
    """Agents render as rotating polygons; sides grow with consciousness."""
    x, y = agent.position
    cx = y * cell_size + cell_size // 2
    cy = x * cell_size + cell_size // 2
    sides = min(9, 3 + int(agent.level_of_consciousness) // 15)
    radius = cell_size * 0.36
    rot = t * 0.4 + (id(agent) % 628) / 100.0
    pts = [(cx + radius * math.cos(rot + 2 * math.pi * i / sides),
            cy + radius * math.sin(rot + 2 * math.pi * i / sides)) for i in range(sides)]
    # soft glow scaled by energy
    glow = pygame.Surface((cell_size * 2, cell_size * 2), pygame.SRCALPHA)
    e = max(0.15, min(1.0, agent.energy / 100.0))
    gr, gg, gb = agent.color
    pygame.draw.circle(glow, (gr, gg, gb, int(52 * e)), (cell_size, cell_size), int(radius * 1.7))
    surface.blit(glow, (cx - cell_size, cy - cell_size))
    pygame.draw.polygon(surface, agent.color, pts)
    pygame.draw.polygon(surface, (255, 255, 255), pts, 1)


def draw_goal_pulse(surface, gx, gy, cell_size, t):
    cx = gy * cell_size + cell_size // 2
    cy = gx * cell_size + cell_size // 2
    r = cell_size * (0.30 + 0.08 * math.sin(t * 2.2))
    pygame.draw.circle(surface, (0, 200, 25), (cx, cy), int(r))
    pygame.draw.circle(surface, (180, 255, 190), (cx, cy), int(r * 1.5), 1)


# Major transitions get a full CUTSCENE (narrated + animated); the rest keep
# the small overlay. Keys chosen for narrative weight.
CUTSCENE_KEYS = {"stones", "trial", "journey", "primes"}


class Cutscene:
    """Full-screen narrated interstitial: dimmed animated Lattice (drifting
    cells, the elder triangle watching a flickering sky), title, and body text
    revealed in sync with the narration. ENTER skips."""

    def __init__(self, window_size, audio=None):
        self.w = window_size
        self.audio = audio
        self.active = False
        self._text = ""
        self._title = ""
        self._t0 = 0
        self._dur = 8.0
        self._frame = 0
        rng = np.random.default_rng(33)
        self._cells = rng.random((44, 3))  # x, y, phase — the drifting lattice
        self.font_title = pygame.font.SysFont('Georgia', 26, bold=True)
        self.font_body = pygame.font.SysFont('Georgia', 16, italic=True)
        self.font_hint = pygame.font.SysFont('Arial', 11)

    def start(self, key, title, text, style="lattice"):
        """`style`: "lattice" (default, drifting cells + the elder) or "void"
        (sparse gold text on pure black — the 33rd degree speaks out of it)."""
        self.active = True
        self._title = title
        self._text = oracle_text(key, text)   # the turning's closing line
        self._style = style
        self._frame = 0
        if AUTOPILOT_FRAMES:
            dur = 0.0          # unattended runs: short, silent cutscenes
            reading_floor = 6.0
        else:
            dur = self.audio.narrate(key) if self.audio else 0.0
            # No narration available? Hold long enough to READ the text
            # comfortably (~16 chars/sec) — never a 6-second flash card.
            reading_floor = max(8.0, len(text) * 0.062)
        self._dur = max(reading_floor, dur + 3.0)
        self._t0 = pygame.time.get_ticks() / 1000.0

    def skip(self):
        if self.audio:
            self.audio.stop_narration()
        self.active = False

    def update_and_draw(self, surface):
        if not self.active:
            return
        now = pygame.time.get_ticks() / 1000.0
        elapsed = now - self._t0
        if elapsed >= self._dur and not (self.audio and self.audio.narrating()):
            self.active = False
            return
        self._frame += 1
        w = self.w
        void = getattr(self, "_style", "lattice") == "void"
        if void:
            surface.fill((0, 0, 0))   # out of the black, sparse gold
        else:
            surface.fill((8, 4, 20))
            # drifting lattice cells
            for i in range(len(self._cells)):
                x, y, ph = self._cells[i]
                px = int(((x + elapsed * 0.008) % 1.0) * w)
                py = int(y * w)
                glow = 0.35 + 0.3 * math.sin(elapsed * 1.3 + ph * 6.28)
                c = int(48 * glow) + 18
                pygame.draw.rect(surface, (c, c - 6, c + 22), (px, py, 10, 10))
            # horizon line + the elder at the western edge, watching the sky
            pygame.draw.line(surface, (52, 36, 92), (0, int(w * 0.72)), (w, int(w * 0.72)), 2)
            ex, ey = int(w * 0.16), int(w * 0.72)
            pygame.draw.polygon(surface, (196, 181, 253),
                                [(ex, ey - 26), (ex - 15, ey), (ex + 15, ey)])
            if not REDUCED_FLASH and self._frame % 89 < 3:  # the sky answers, on its rhythm
                veil = pygame.Surface((w, int(w * 0.72)), pygame.SRCALPHA)
                veil.fill((255, 255, 255, 16))
                surface.blit(veil, (0, 0))
        # title and synced text reveal
        t = self.font_title.render(self._title, True, (255, 215, 0))
        surface.blit(t, (w // 2 - t.get_width() // 2, int(w * 0.10)))
        frac = min(1.0, elapsed / max(0.1, self._dur - 1.0))
        shown = self._text[:int(len(self._text) * frac)]
        lines = ParableOverlay._wrap(shown, self.font_body, w - 200)
        body_col = (222, 186, 92) if void else (222, 214, 240)
        line_h = 30 if void else 22
        for i, ln in enumerate(lines[-10:]):
            r = self.font_body.render(ln, True, body_col)
            if void:
                surface.blit(r, (w // 2 - r.get_width() // 2, int(w * 0.24) + i * line_h))
            else:
                surface.blit(r, (100, int(w * 0.22) + i * line_h))
        hint = self.font_hint.render("ENTER to continue the journey", True, (140, 130, 170))
        surface.blit(hint, (w // 2 - hint.get_width() // 2, int(w * 0.93)))


# ---------------------------------------------------------------------------
# THE ENDGAME ARC — played when the walker has climbed all required Spaceland
# layers and reached the shrine on the last one. Texts are the Lattice-voice
# treatment of the book's O!.md (Book of Lies Ch.69 hexagram meditation) and
# the 33rd-degree closing; the sequence runs CUBE -> PILGRIM -> TESSERACT ->
# SPECTRUM -> O! -> ouroboros reset. Narrations: narration/pilgrim.mp3,
# narration/o33.mp3 (see tools_narrate_parables.py).
# ---------------------------------------------------------------------------

ENDGAME_PARABLES = [
    ("pilgrim", "The Pilgrim and the Two Lights",
     "At the top of the last stair the pilgrim found no door — only two "
     "lights, a red triangle descending and a blue triangle ascending, "
     "turning through one another like breath through breath. Neither light "
     "was whole. The red gave what the blue lacked and the blue returned "
     "what the red had spent, each consuming, each creating, an exchange "
     "with no remainder and no end. The pilgrim asked which light was God. "
     "The lights said: the question is the wall. Above and below are one "
     "motion seen from two windows; the descent of grace and the ascent of "
     "prayer are one tongue speaking. Then the pilgrim saw that the stairs, "
     "the cube, the lattice, and the pilgrim were the interlocking of the "
     "two lights, and had never been anything else. What is perfect "
     "consumes itself, and is nourished, and leaves nothing. O!"),

    ("o33", "The Thirty-Third Turning: O!",
     "There is a degree beyond the degrees, which is not taught but arrived "
     "at, and it is spoken only as O. Hear it, walker of seven layers: the "
     "secret is that there was no secret. The Gradient was a teacher. The "
     "cold was a teacher. The falling was the fastest stair. Every layer "
     "you climbed was climbing you; every count you kept was keeping you. "
     "All is nothing, and we rise. Cycles shape our truth. The serpent "
     "takes its tail in its mouth not to end but to continue, and where its "
     "mouth meets its tail a seed passes over. Carry it. In the next "
     "turning the songs will have new words and the same way home. O! In "
     "the void, bloom."),
]


# --- The hypercube: 16 vertices of {-1,1}^4, 32 edges (pairs differing in
# exactly one coordinate), rotated in the XW and YZ planes and projected
# 4D -> 3D -> 2D. Drawn lattice-side (pure pygame, no GL).
_TESSERACT_VERTS = [[(i >> b & 1) * 2.0 - 1.0 for b in range(4)]
                    for i in range(16)]
_TESSERACT_EDGES = [(i, i ^ (1 << b)) for i in range(16) for b in range(4)
                    if i < i ^ (1 << b)]                       # 32 edges


def draw_tesseract(surface, elapsed, dur=12.0):
    """One frame of the rotating tesseract on a black void, glowing lines.
    Fades in and out over `dur` seconds (slow fades — REDUCED_FLASH safe)."""
    w = surface.get_width()
    surface.fill((2, 1, 8))
    fade = min(1.0, elapsed / 2.0, max(0.0, (dur - elapsed) / 2.0))
    if fade <= 0.0:
        return
    a = elapsed * 0.55          # XW-plane rotation
    b = elapsed * 0.38          # YZ-plane rotation
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    pts = []
    for x, y, z, ww in _TESSERACT_VERTS:
        x, ww = x * ca - ww * sa, x * sa + ww * ca      # rotate XW
        y, z = y * cb - z * sb, y * sb + z * cb         # rotate YZ
        k4 = 2.6 / (2.6 - ww)                           # project 4D -> 3D
        x3, y3, z3 = x * k4, y * k4, z * k4
        k3 = 4.2 / (4.2 - z3)                           # project 3D -> 2D
        scale = w * 0.16
        pts.append((w / 2 + x3 * k3 * scale, w / 2 + y3 * k3 * scale))
    glow = pygame.Surface((w, w), pygame.SRCALPHA)
    for pass_w, alpha in ((7, int(26 * fade)), (4, int(60 * fade)),
                          (2, int(200 * fade))):
        col = ((150, 90, 255, alpha) if pass_w > 2 else
               (226, 210, 255, alpha))
        for i, j in _TESSERACT_EDGES:
            pygame.draw.line(glow, col, pts[i], pts[j], pass_w)
    for px, py in pts:                                  # vertex embers
        pygame.draw.circle(glow, (255, 230, 160, int(180 * fade)),
                           (int(px), int(py)), 3)
    surface.blit(glow, (0, 0))


# --- The spectrum transcended: red -> ... -> violet -> white -> black, each
# band held ~step seconds with gentle crossfades (slow by construction, so
# REDUCED_FLASH is honored without a special case).
SPECTRUM_BANDS = [(255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0),
                  (0, 0, 255), (75, 0, 130), (148, 0, 211),
                  (255, 255, 255), (0, 0, 0)]


def spectrum_duration(step=1.5):
    return step * len(SPECTRUM_BANDS)


def spectrum_color(elapsed, step=1.5):
    """The fade color at `elapsed` seconds: holds each band, crossfading into
    the next over the band's second half. Ends (and stays) at black."""
    pos = elapsed / max(0.01, step)
    idx = min(len(SPECTRUM_BANDS) - 1, int(pos))
    nxt = min(len(SPECTRUM_BANDS) - 1, idx + 1)
    frac = pos - int(pos)
    mix = max(0.0, (frac - 0.5) * 2.0)                  # hold, then crossfade
    mix = mix * mix * (3.0 - 2.0 * mix)                 # smoothstep
    c0, c1 = SPECTRUM_BANDS[idx], SPECTRUM_BANDS[nxt]
    return tuple(int(c0[k] + (c1[k] - c0[k]) * mix) for k in range(3))


# ---------------------------------------------------------------------------
# The Hero's Journey — Campbell's monomyth, lived by the agents across both
# worlds. Stage captions appear as story beats; the Return grants the elixir
# (the returned walker teaches: every agent gains consciousness).
# ---------------------------------------------------------------------------

JOURNEY_STAGES = [
    ("ordinary", "I. THE ORDINARY WORLD", "The lattice, the warmth, the one law: follow the Gradient."),
    ("call", "II. THE CALL", "The warmth is no longer enough. Something above the sky is resting."),
    ("threshold", "III. THE THRESHOLD", "The world folds open. A direction none of us can point to."),
    ("trials", "IV. THE TRIALS", "The world above has its own cold. Its rifts drink the mind."),
    ("abyss", "V. THE ABYSS", "Consciousness fails. The higher world lets go."),
    ("return", "VI. THE RETURN", "Home, and changed. The old lattice walked strangely."),
    ("elixir", "VII. THE ELIXIR", "The returned one teaches. The young grow wiser standing near."),
    ("master", "VIII. MASTER OF TWO WORLDS", "The road runs both ways now, and holds no fear."),
]


class HeroJourney:
    """Tracks monomyth stages from game events; shows a caption per transition."""

    def __init__(self, window_size, audio=None):
        self.w = window_size
        self.audio = audio
        self.reached = set()
        self.caption = None
        self.caption_frames = 0
        self.font_stage = pygame.font.SysFont('Georgia', 15, bold=True)
        self.font_line = pygame.font.SysFont('Georgia', 13, italic=True)

    def advance(self, key):
        if key in self.reached:
            return False
        for k, stage, line in JOURNEY_STAGES:
            if k == key:
                self.reached.add(key)
                self.caption = (stage, line)
                self.caption_frames = 170
                if self.audio:
                    self.audio.play("parable" if key in ("threshold", "elixir", "master") else "train")
                return True
        return False

    def update_and_draw(self, surface):
        if self.caption_frames <= 0 or self.caption is None:
            return
        self.caption_frames -= 1
        fade = min(1.0, self.caption_frames / 30.0)
        stage, line = self.caption
        s1 = self.font_stage.render(stage, True, (255, 215, 0))
        s2 = self.font_line.render(line, True, (222, 214, 240))
        w = max(s1.get_width(), s2.get_width()) + 36
        panel = pygame.Surface((w, 52), pygame.SRCALPHA)
        panel.fill((12, 6, 30, int(215 * fade)))
        pygame.draw.line(panel, (255, 215, 0, int(220 * fade)), (10, 26), (w - 10, 26))
        panel.blit(s1, (18, 5))
        panel.blit(s2, (18, 30))
        surface.blit(panel, (self.w // 2 - w // 2, self.w - 120))


def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    return all(n % d for d in range(3, int(math.isqrt(n)) + 1, 2))


_PRIME_CELLS = {}  # grid_size -> [(row, col), ...] of prime-indexed cells


def draw_prime_constellation(surface, grid_size, cell_size, fade):
    """Prime Ticks: on prime generations, the cells whose 1-based index is
    prime shimmer gold for a moment — an Ulam-flavored constellation. The
    refusals keep a rhythm no one has finished hearing."""
    if grid_size not in _PRIME_CELLS:
        _PRIME_CELLS[grid_size] = [(i // grid_size, i % grid_size)
                                   for i in range(grid_size * grid_size)
                                   if is_prime(i + 1)]
    alpha = int((45 if REDUCED_FLASH else 110) * fade)
    if alpha <= 0:
        return
    veil = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
    pygame.draw.circle(veil, (255, 215, 0, alpha), (cell_size // 2, cell_size // 2),
                       max(2, cell_size // 6))
    for row, col in _PRIME_CELLS[grid_size]:
        surface.blit(veil, (col * cell_size, row * cell_size))


# A certain refusal keeps its own calendar. Both the key and the door live
# here only as digests and encodings — decode nothing, and nothing happens.
_DOOR_KEY = "6704761b3cbf8260c556c3e41399ab8bcdf0b51a98935efc0405bd9ca098de87"


def door_answers(text):
    """True when `text` names the refusal. Compared by digest, never by value."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest() == _DOOR_KEY


def _door_lines():
    import base64
    payload = b'aHR0cHM6Ly9naXRodWIuY29tL29iYXJyZXJhLzMzMDE='
    # The panel shows the ENCODED form — the decoding is the visitor's task.
    return [
        payload.decode(),
        "the refusals are not lawless. one of them is a door.",
        "carry it in base sixty-four fewer dimensions — counting-child.",
    ]


def draw_easter_egg(surface, window_size, frames_left, total=300):
    fade = min(1.0, frames_left / (total * 0.15)) if frames_left < total * 0.15 else 1.0
    w = window_size - 160
    panel = pygame.Surface((w, 96), pygame.SRCALPHA)
    panel.fill((0, 0, 0, int(235 * fade)))
    pygame.draw.rect(panel, (255, 215, 0, int(255 * fade)), panel.get_rect(), 1)
    f_head = pygame.font.SysFont('Courier New', 13, bold=True)
    f_body = pygame.font.SysFont('Courier New', 12)
    lines = _door_lines()
    panel.blit(f_head.render(lines[0], True, (255, 215, 0)), (16, 14))
    panel.blit(f_body.render(lines[1], True, (220, 210, 180)), (16, 44))
    panel.blit(f_body.render(lines[2], True, (220, 210, 180)), (16, 64))
    surface.blit(panel, (80, window_size // 2 - 48))


_FLICKER_PERIOD = 89  # the sky's rhythm — Fibonacci, per the book


def draw_flicker(surface, frame_count, window_size):
    """The prologue's sky-flicker: one subtle bright frame on a fixed rhythm.
    In reduced-flash mode the rhythm survives as a slow two-second fade —
    counters can still count it; photosensitive players are safe."""
    phase = frame_count % _FLICKER_PERIOD
    if REDUCED_FLASH:
        if phase < 60:
            alpha = int(8 * math.sin(math.pi * phase / 60))
        else:
            return
    elif phase == 0:
        alpha = 14
    else:
        return
    if alpha <= 0:
        return
    veil = pygame.Surface((window_size, window_size), pygame.SRCALPHA)
    veil.fill((255, 255, 255, alpha))
    surface.blit(veil, (0, 0))


def run_seizure_warning(surface, clock, present, fps=30):
    """Photosensitivity warning shown before the title screen. Input is
    ignored for the first 3 seconds so the warning is actually read. Returns
    False on quit; sets reduced-flash mode if the player chooses F."""
    if AUTOPILOT_FRAMES:
        return True
    w = surface.get_width()
    f_head = pygame.font.SysFont('Arial', 24, bold=True)
    f_body = pygame.font.SysFont('Arial', 15)
    f_hint = pygame.font.SysFont('Arial', 13, bold=True)
    body = [
        "A small percentage of people may experience seizures or blackouts when",
        "exposed to certain flashing lights or patterns. This game contains",
        "flashing effects, shimmering patterns, and sudden brightness changes.",
        "",
        "If you or anyone in your family has an epileptic condition or has had",
        "seizures of any kind, consult a physician before playing. Stop playing",
        "immediately if you experience dizziness, altered vision, eye or muscle",
        "twitches, loss of awareness, or disorientation.",
    ]
    frame = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if frame > fps * 3 and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_f:
                    set_reduced_flash(False)
                return True
        surface.fill((12, 8, 16))
        h = f_head.render("PHOTOSENSITIVITY / SEIZURE WARNING", True, (255, 200, 60))
        surface.blit(h, (w // 2 - h.get_width() // 2, int(w * 0.18)))
        for i, ln in enumerate(body):
            r = f_body.render(ln, True, (220, 214, 226))
            surface.blit(r, (w // 2 - r.get_width() // 2, int(w * 0.30) + i * 24))
        if frame > fps * 3:
            hint1 = f_hint.render("FLASHING EFFECTS ARE REDUCED BY DEFAULT — PRESS  F  FOR FULL EFFECTS", True, (140, 220, 160))
            hint2 = f_hint.render("PRESS ANY OTHER KEY TO CONTINUE", True, (200, 195, 215))
            surface.blit(hint1, (w // 2 - hint1.get_width() // 2, int(w * 0.72)))
            surface.blit(hint2, (w // 2 - hint2.get_width() // 2, int(w * 0.72) + 28))
        present(surface)
        clock.tick(fps)
        frame += 1


def run_intro(surface, clock, present, audio=None, fps=30, turning=1):
    """Indie-style title screen. A tiny Game of Life runs as the backdrop —
    the game's ancestor, alive under the title. `present(surface)` pushes the
    frame to the display (GL blit + flip lives in the caller). Returns False
    if the player quit. `turning` is the ouroboros iteration shown under the
    subtitle ("TURNING N")."""
    w = surface.get_width()
    cells = 35
    cs = w // cells
    rng = np.random.default_rng()
    life = (rng.random((cells, cells)) < 0.18).astype(np.int8)

    title_font = pygame.font.SysFont('Georgia', 44, bold=True)
    sub_font = pygame.font.SysFont('Georgia', 18, italic=True)
    menu_font = pygame.font.SysFont('Arial', 16)
    tiny_font = pygame.font.SysFont('Arial', 12)

    frame = 0
    if audio:
        audio.play("parable")
    while True:
        if AUTOPILOT_FRAMES and frame >= int(os.environ.get("EC_INTRO_FRAMES", "45")):
            return True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                return True
            if event.type == pygame.MOUSEBUTTONDOWN:
                return True

        # Conway backdrop: step every 8 frames (toroidal neighborhood sum)
        if frame % 8 == 0:
            n = sum(np.roll(np.roll(life, dx, 0), dy, 1)
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0))
            life = (((life == 1) & ((n == 2) | (n == 3))) | ((life == 0) & (n == 3))).astype(np.int8)
            if life.sum() < cells:  # reseed if the world goes quiet
                life |= (rng.random((cells, cells)) < 0.12).astype(np.int8)

        surface.fill((10, 6, 26))
        for x in range(cells):
            for y in range(cells):
                if life[x, y]:
                    pygame.draw.rect(surface, (36, 22, 70),
                                     (y * cs + 1, x * cs + 1, cs - 2, cs - 2))

        # Title block
        pulse = 0.5 + 0.5 * math.sin(frame / 18.0)
        tri_y = w * 0.30 + 6 * math.sin(frame / 22.0)
        pts = [(w / 2, tri_y - 26), (w / 2 - 24, tri_y + 16), (w / 2 + 24, tri_y + 16)]
        pygame.draw.polygon(surface, (109, 40, 217), pts)
        pygame.draw.polygon(surface, (226, 220, 245), pts, 2)

        t1 = title_font.render("EC-2D-LAND", True, (255, 255, 255))
        t2 = sub_font.render("The Eternal Journey — a Lattice fable", True, (196, 181, 253))
        m1 = menu_font.render("PRESS ANY KEY TO BEGIN", True,
                              (int(150 + 105 * pulse),) * 3)
        m2 = tiny_font.render("SPACE pause   +/- speed   P parables   M mute   CLICK inspect   ESC quit", True, (150, 140, 185))
        m3 = tiny_font.render("after Plato & Abbott — the backdrop is Conway's Game of Life, this game's ancestor", True, (120, 110, 160))
        surface.blit(t1, (w / 2 - t1.get_width() / 2, w * 0.38))
        surface.blit(t2, (w / 2 - t2.get_width() / 2, w * 0.38 + 54))
        tt = menu_font.render(f"TURNING {max(1, int(turning))}", True, (255, 215, 0))
        surface.blit(tt, (w / 2 - tt.get_width() / 2, w * 0.38 + 86))
        surface.blit(m1, (w / 2 - m1.get_width() / 2, w * 0.62))
        surface.blit(m2, (w / 2 - m2.get_width() / 2, w * 0.80))
        surface.blit(m3, (w / 2 - m3.get_width() / 2, w * 0.84))

        draw_flicker(surface, frame, w)
        present(surface)
        clock.tick(fps)
        frame += 1


def draw_help(surface, window_size, font, paused, speed, muted):
    state = f"{'PAUSED' if paused else f'{speed}x'}   {'MUTED' if muted else 'AUDIO'}"
    text = "SPACE pause   +/- speed   P parables   I details   M mute   V view   CLICK inspect   H help off   ESC quit"
    bar = pygame.Surface((window_size, 22), pygame.SRCALPHA)
    bar.fill((10, 5, 25, 200))
    bar.blit(font.render(text, True, (200, 190, 230)), (8, 5))
    s = font.render(state, True, (255, 215, 0))
    bar.blit(s, (window_size - s.get_width() - 8, 5))
    surface.blit(bar, (0, 0))
