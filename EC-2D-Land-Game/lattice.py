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

    def play(self, name):
        if self.ok and not self.muted and name in self._tones:
            self._tones[name].play()

    def toggle_mute(self):
        self.muted = not self.muted
        if self.ambient:
            self.ambient.set_volume(0.0 if self.muted else 0.35)
        return self.muted


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
                self.hold = 240
                if self.audio:
                    self.audio.play("parable")
                return title
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
        self.reveal = min(len(text), self.reveal + 2.2)
        if self.reveal >= len(text):
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
    alpha = int(110 * fade)
    if alpha <= 0:
        return
    veil = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
    pygame.draw.circle(veil, (255, 215, 0, alpha), (cell_size // 2, cell_size // 2),
                       max(2, cell_size // 6))
    for row, col in _PRIME_CELLS[grid_size]:
        surface.blit(veil, (col * cell_size, row * cell_size))


def draw_easter_egg(surface, window_size, frames_left, total=300):
    """Generation 3301 keeps its own calendar. (Also answers those who type it.)"""
    fade = min(1.0, frames_left / (total * 0.15)) if frames_left < total * 0.15 else 1.0
    w = window_size - 160
    panel = pygame.Surface((w, 96), pygame.SRCALPHA)
    panel.fill((0, 0, 0, int(235 * fade)))
    pygame.draw.rect(panel, (255, 215, 0, int(255 * fade)), panel.get_rect(), 1)
    f_head = pygame.font.SysFont('Courier New', 15, bold=True)
    f_body = pygame.font.SysFont('Courier New', 13)
    lines = [
        "3 3 0 1",
        "The refusals are not lawless. One of them has a door.",
        "github.com/obarrera/3301        — good luck, counting-child.",
    ]
    panel.blit(f_head.render(lines[0], True, (255, 215, 0)), (w // 2 - 34, 12))
    panel.blit(f_body.render(lines[1], True, (220, 210, 180)), (16, 42))
    panel.blit(f_body.render(lines[2], True, (220, 210, 180)), (16, 64))
    surface.blit(panel, (80, window_size // 2 - 48))


_FLICKER_PERIOD = 89  # the sky's rhythm — Fibonacci, per the book


def draw_flicker(surface, frame_count, window_size):
    """The prologue's sky-flicker: one subtle bright frame on a fixed rhythm."""
    if frame_count % _FLICKER_PERIOD == 0:
        veil = pygame.Surface((window_size, window_size), pygame.SRCALPHA)
        veil.fill((255, 255, 255, 14))
        surface.blit(veil, (0, 0))


def run_intro(surface, clock, present, audio=None, fps=30):
    """Indie-style title screen. A tiny Game of Life runs as the backdrop —
    the game's ancestor, alive under the title. `present(surface)` pushes the
    frame to the display (GL blit + flip lives in the caller). Returns False
    if the player quit."""
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
        surface.blit(m1, (w / 2 - m1.get_width() / 2, w * 0.62))
        surface.blit(m2, (w / 2 - m2.get_width() / 2, w * 0.80))
        surface.blit(m3, (w / 2 - m3.get_width() / 2, w * 0.84))

        draw_flicker(surface, frame, w)
        present(surface)
        clock.tick(fps)
        frame += 1


def draw_help(surface, window_size, font, paused, speed, muted):
    state = f"{'PAUSED' if paused else f'{speed}x'}   {'MUTED' if muted else 'AUDIO'}"
    text = "SPACE pause   +/- speed   P parables   I details   M mute   CLICK inspect   H help off   ESC quit"
    bar = pygame.Surface((window_size, 22), pygame.SRCALPHA)
    bar.fill((10, 5, 25, 200))
    bar.blit(font.render(text, True, (200, 190, 230)), (8, 5))
    s = font.render(state, True, (255, 215, 0))
    bar.blit(s, (window_size - s.get_width() - 8, 5))
    surface.blit(bar, (0, 0))
