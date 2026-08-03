"""spaceland.py — Doom-style first-person Spaceland for EC-2D-Land.

The book's dimensional-ascent thesis, embodied: when an agent ascends, the
camera drops INSIDE the very 2D lattice it lived on. Obstacle cells (grid 3)
extrude into purple-stone walls, life-cells (1/2) glow as rune tiles, the 2D
goal becomes a pulsing golden shrine, and above the extruded grid there is
no ceiling — only stars, and beyond them the faint inverted lattice of the
layer above ("as above, so below"; the layer beneath glows up through the
floor). Retro-FPS aesthetic: no GL lighting (brightness baked into vertex
colors, distance handled by fog), 64x64 procedural textures with
nearest-neighbor filtering, heavy exp2 purple fog that warms per layer.

The world above has its own cold: dark pulsing COLD RIFTS drain the walker's
consciousness (with a blue vignette chill), one DESCENT WELL per map drops
the walker a layer down, and losing consciousness in stages (66 / 33 / 0)
descends the stack too. Layer 0 is Flatland: the walker falls back to 2D.

Public API (all degrade gracefully to no-ops if GL/pygame are missing):
    init_gl(window_size)
    enter(grid, goal, layer)
    toggle_overview()
    leave()
    update_and_render(t, keys_pressed) -> None | "goal" | "hazard" | "fell"
"""
import math
import os
import random

import numpy as np
import pygame

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    _GL_OK = True
except Exception:  # PyOpenGL missing: every entry point becomes a no-op
    _GL_OK = False

FOG_BASE = (0.078, 0.039, 0.180)    # deep purple #140a2e (layer 1)
EYE_HEIGHT = 0.5                    # eye height: half a cell
WALL_H = 1.0
WALK_SPEED = 1.7                    # AI walker, cells/sec
PLAYER_SPEED = 2.3
TURN_SPEED = 2.6                    # rad/sec
PLAYER_TIMEOUT = 3.0                # seconds of no input before AI resumes
FOG_FPS = 0.11                      # exp2 density tuned for a 20-cell world
FOG_OVERVIEW = 0.028
RIFT_DRAIN = 1.5                    # consciousness per frame while in a rift
AMBIENT_DRAIN = float(os.environ.get("EC_SPACELAND_DRAIN", "0.035"))
STAGES = (66.0, 33.0)               # consciousness stage thresholds

# Module state (single world at a time, mirroring the game's single 3D agent)
_S = {"ready": False, "overview": False, "win": 700, "font": None,
      "textures": {}, "last_t": None, "mind": 100.0, "stage": 0, "chill": 0.0}

# --------------------------------------------------------------------------
# Platonic solids — own vertex/face tables (unit-ish scale, centered)
# --------------------------------------------------------------------------
_PHI = (1.0 + math.sqrt(5.0)) / 2.0

_TETRA = ([(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)],
          [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)])
_CUBE = ([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
          (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
         [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
          (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)])
_OCTA = ([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
         [(0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
          (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5)])
_ICOSA = ([(-1, _PHI, 0), (1, _PHI, 0), (-1, -_PHI, 0), (1, -_PHI, 0),
           (0, -1, _PHI), (0, 1, _PHI), (0, -1, -_PHI), (0, 1, -_PHI),
           (_PHI, 0, -1), (_PHI, 0, 1), (-_PHI, 0, -1), (-_PHI, 0, 1)],
          [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
           (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
           (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
           (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)])
_SOLID_TABLE = [_TETRA, _CUBE, _OCTA, _ICOSA]
_SOLID_COLORS = [(0.45, 0.90, 1.00), (1.00, 0.55, 0.90),
                 (0.55, 1.00, 0.60), (1.00, 0.82, 0.42)]


def _fog_color(layer):
    """Fog hue shifts slightly warmer as the layer number rises."""
    k = min(layer - 1, 8)
    return (min(0.24, FOG_BASE[0] + 0.020 * k),
            min(0.12, FOG_BASE[1] + 0.007 * k),
            max(0.10, FOG_BASE[2] - 0.010 * k))


# --------------------------------------------------------------------------
# Procedural 64x64 textures (numpy -> glTexImage2D, generated once)
# --------------------------------------------------------------------------
def _upload_texture(pixels):
    tid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tid)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 64, 64, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, np.ascontiguousarray(pixels, np.uint8).tobytes())
    return tid


def _noise(rng, base, amp):
    return np.clip(np.array(base, float) +
                   rng.normal(0, amp, (64, 64, 3)), 0, 255)


def _tex_wall(rng):
    """Purple stone with darker mortar lines (offset brick courses)."""
    img = _noise(rng, (98, 76, 148), 14)
    ys, xs = np.mgrid[0:64, 0:64]
    course = ys // 16
    mortar = (ys % 16 < 1) | (((xs + course * 32) % 64) < 1)
    img[mortar] *= 0.42
    img[(ys % 16 == 1)] *= 0.8              # soft shadow under each course
    return img


def _tex_floor(rng):
    img = _noise(rng, (52, 44, 82), 10)
    ys, xs = np.mgrid[0:64, 0:64]
    lines = (ys % 32 < 1) | (xs % 32 < 1)
    img[lines] *= 0.5
    return img


def _tex_rune(rng):
    """Floor tile with a glowing rune: a ring crossed by diagonals."""
    img = _tex_floor(rng) * 0.7
    ys, xs = np.mgrid[0:64, 0:64]
    r = np.hypot(xs - 31.5, ys - 31.5)
    glyph = (np.abs(r - 20) < 1.6) | \
            ((np.abs(xs - ys) < 1.5) & (r < 22)) | \
            ((np.abs(xs + ys - 63) < 1.5) & (r < 22))
    img[glyph] = (120, 255, 190)
    return img


def _tex_gold(rng):
    img = _noise(rng, (222, 176, 66), 18)
    ys, xs = np.mgrid[0:64, 0:64]
    img[(ys % 8 < 1) | (xs % 8 < 1)] *= 0.75
    return img


# --------------------------------------------------------------------------
# GL setup
# --------------------------------------------------------------------------
def init_gl(window_size):
    """Perspective + depth + purple exp2 fog + blend. Doom look: NO lighting."""
    if not _GL_OK:
        return
    try:
        _S["win"] = int(window_size)
        glViewport(0, 0, _S["win"], _S["win"])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(70.0, 1.0, 0.05, 120.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_EXP2)
        glFogfv(GL_FOG_COLOR, (*FOG_BASE, 1.0))
        glFogf(GL_FOG_DENSITY, FOG_FPS)
        glClearColor(*FOG_BASE, 1.0)
        if _S["font"] is None:
            try:
                _S["font"] = pygame.font.SysFont("Courier New", 14, bold=True)
            except Exception:
                _S["font"] = None
    except Exception as exc:
        print(f"spaceland: init_gl degraded ({exc})")


def leave():
    """Restore GL state the 2D glDrawPixels path relies on."""
    if not (_GL_OK and _S["ready"]):
        return
    try:
        glDisable(GL_FOG)
        glWindowPos2i(0, 0)
    except Exception:
        pass
    _S["ready"] = False


def toggle_overview():
    _S["overview"] = not _S["overview"]


def current_layer():
    """The layer the walker is on right now (descents change it internally)."""
    return _S.get("layer", 1)


# --------------------------------------------------------------------------
# World building
# --------------------------------------------------------------------------
def _bfs_field(start, walk, n):
    """BFS distance field over walkable cells from `start`."""
    field = {start: 0}
    frontier = [start]
    while frontier:
        nxt = []
        for (r, c) in frontier:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                cell = (r + dr, c + dc)
                if cell in walk and cell not in field:
                    field[cell] = field[(r, c)] + 1
                    nxt.append(cell)
        frontier = nxt
    return field


def _path_to_goal(cell):
    """Walk the precomputed distance-from-goal field downhill to the shrine."""
    field = _S["field"]
    if cell not in field:
        return []
    path, cur = [], cell
    while field[cur] > 0:
        cur = min(((cur[0] + dr, cur[1] + dc)
                   for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))),
                  key=lambda c: field.get(c, 1 << 30))
        path.append(cur)
    return path


def _emit_floor(cells, shade):
    glColor3f(*shade)
    glBegin(GL_QUADS)
    for (r, c) in cells:
        glTexCoord2f(0, 0); glVertex3f(c, 0, r)
        glTexCoord2f(1, 0); glVertex3f(c + 1, 0, r)
        glTexCoord2f(1, 1); glVertex3f(c + 1, 0, r + 1)
        glTexCoord2f(0, 1); glVertex3f(c, 0, r + 1)
    glEnd()


def _build_world_list():
    """Static geometry (floors + walls) into one display list."""
    n, walls, walk = _S["n"], _S["walls"], _S["walk"]
    grid, goal = _S["grid"], _S["goal"]
    lst = glGenLists(1)
    glNewList(lst, GL_COMPILE)
    glEnable(GL_TEXTURE_2D)

    plain = [c for c in walk if grid[c] not in (1, 2) and c != goal]
    runes = [c for c in walk if grid[c] in (1, 2) and c != goal]
    glBindTexture(GL_TEXTURE_2D, _S["textures"]["floor"])
    _emit_floor(plain, (0.82, 0.82, 0.88))
    glBindTexture(GL_TEXTURE_2D, _S["textures"]["rune"])
    _emit_floor(runes, (0.95, 1.0, 0.95))
    glBindTexture(GL_TEXTURE_2D, _S["textures"]["gold"])
    _emit_floor([goal], (1.0, 1.0, 1.0))

    # Walls: interior wall cubes (grid==3) plus a perimeter ring just outside
    all_walls = set(walls)
    for i in range(-1, n + 1):
        all_walls |= {(-1, i), (n, i), (i, -1), (i, n)}
    glBindTexture(GL_TEXTURE_2D, _S["textures"]["wall"])
    glBegin(GL_QUADS)
    for (r, c) in all_walls:
        # One quad per face that borders a walkable interior cell
        if (r, c + 1) in walk:      # east face, +x
            glColor3f(0.9, 0.9, 0.9)
            glTexCoord2f(0, 0); glVertex3f(c + 1, 0, r)
            glTexCoord2f(1, 0); glVertex3f(c + 1, 0, r + 1)
            glTexCoord2f(1, 1); glVertex3f(c + 1, WALL_H, r + 1)
            glTexCoord2f(0, 1); glVertex3f(c + 1, WALL_H, r)
        if (r, c - 1) in walk:      # west face, -x
            glColor3f(0.9, 0.9, 0.9)
            glTexCoord2f(0, 0); glVertex3f(c, 0, r)
            glTexCoord2f(1, 0); glVertex3f(c, 0, r + 1)
            glTexCoord2f(1, 1); glVertex3f(c, WALL_H, r + 1)
            glTexCoord2f(0, 1); glVertex3f(c, WALL_H, r)
        if (r + 1, c) in walk:      # south face, +z (darker: Doom facing shade)
            glColor3f(0.68, 0.68, 0.74)
            glTexCoord2f(0, 0); glVertex3f(c, 0, r + 1)
            glTexCoord2f(1, 0); glVertex3f(c + 1, 0, r + 1)
            glTexCoord2f(1, 1); glVertex3f(c + 1, WALL_H, r + 1)
            glTexCoord2f(0, 1); glVertex3f(c, WALL_H, r + 1)
        if (r - 1, c) in walk:      # north face, -z
            glColor3f(0.68, 0.68, 0.74)
            glTexCoord2f(0, 0); glVertex3f(c, 0, r)
            glTexCoord2f(1, 0); glVertex3f(c + 1, 0, r)
            glTexCoord2f(1, 1); glVertex3f(c + 1, WALL_H, r)
            glTexCoord2f(0, 1); glVertex3f(c, WALL_H, r)
        if 0 <= r < n and 0 <= c < n:   # top cap (seen from the overview)
            glColor3f(0.5, 0.48, 0.58)
            glTexCoord2f(0, 0); glVertex3f(c, WALL_H, r)
            glTexCoord2f(1, 0); glVertex3f(c + 1, WALL_H, r)
            glTexCoord2f(1, 1); glVertex3f(c + 1, WALL_H, r + 1)
            glTexCoord2f(0, 1); glVertex3f(c, WALL_H, r + 1)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    glEndList()
    return lst


def _build_star_list(layer):
    """Open starfield sky: ~300 unlit points on a far dome (fog disabled)."""
    rng = random.Random(1042 + layer)
    lst = glGenLists(1)
    glNewList(lst, GL_COMPILE)
    glDisable(GL_FOG)
    glPointSize(2.0)
    glBegin(GL_POINTS)
    for _ in range(300):
        az = rng.uniform(0, 2 * math.pi)
        el = math.asin(rng.uniform(0.03, 1.0))
        b = rng.uniform(0.35, 1.0)
        glColor3f(b, b, min(1.0, b * 1.15))
        glVertex3f(70 * math.cos(el) * math.cos(az),
                   70 * math.sin(el),
                   70 * math.cos(el) * math.sin(az))
    glEnd()
    glEnable(GL_FOG)
    glEndList()
    return lst


def enter(grid, goal, layer=1):
    """Build the extruded lattice world for `layer` and (re)spawn the walker.

    Consciousness persists across ascents/descents within one 3D visit; it
    resets to 100 only on a fresh entry from Flatland (after leave()).
    """
    if not _GL_OK:
        return
    try:
        fresh = not _S.get("ready", False)
        layer = max(1, int(layer))
        g = np.asarray([list(row) for row in grid], dtype=int)
        n = len(g)
        walls = {(r, c) for r in range(n) for c in range(n) if g[r, c] == 3}
        walk = {(r, c) for r in range(n) for c in range(n) if g[r, c] != 3}
        goal = (int(goal[0]), int(goal[1]))
        if goal not in walk:
            walk.add(goal)   # never wall the shrine in

        field = _bfs_field(goal, walk, n)
        if len(field) < max(4, len(walk) // 10):
            # Goal is boxed into a tiny pocket: move the shrine to the
            # largest connected open region so the walk is a real journey.
            seen, best = set(), set()
            for cell in walk:
                if cell not in seen:
                    comp = set(_bfs_field(cell, walk, n))
                    seen |= comp
                    if len(comp) > len(best):
                        best = comp
            goal = random.Random(layer).choice(sorted(best))
            field = _bfs_field(goal, walk, n)

        far = max(field.values())
        spawn = random.Random(layer * 31).choice(
            [c for c, d in field.items() if d >= max(3, far - 2)] or [goal])

        # Hazards: cold rifts (+1 per layer) and one descent well, placed on
        # reachable open cells so they genuinely threaten the walk.
        rng = random.Random(104729 * layer + 7)
        pool = [c for c in sorted(field) if g[c] == 0
                and c not in (goal, spawn)]
        rng.shuffle(pool)
        n_rifts = min(3 + layer, 10)
        rifts = set(pool[:n_rifts])
        well = pool[n_rifts] if len(pool) > n_rifts else None

        # Free any stale GL objects from a previous layer
        for key in ("world_list", "star_list"):
            if _S.get(key):
                try:
                    glDeleteLists(_S[key], 1)
                except Exception:
                    pass
        if not _S["textures"]:
            trng = np.random.default_rng(7)
            _S["textures"] = {"wall": _upload_texture(_tex_wall(trng)),
                              "floor": _upload_texture(_tex_floor(trng)),
                              "rune": _upload_texture(_tex_rune(trng)),
                              "gold": _upload_texture(_tex_gold(trng))}

        _S.update(grid=g, n=n, walls=walls, walk=walk, goal=goal,
                  field=field, layer=layer, done=False, last_t=None,
                  last_input=-1e9, bob=0.0, rifts=rifts, well=well,
                  haz_t=-1e9, desc_t=-1e9)
        if fresh:
            _S["mind"] = 100.0
            _S["stage"] = 0
            _S["chill"] = 0.0
        _S["walker"] = {"pos": [spawn[1] + 0.5, spawn[0] + 0.5],
                        "yaw": 0.0, "path": _path_to_goal(spawn)}
        _S["solids"] = _place_solids(layer, rng)
        # Ghost footprint of the layer beneath (glows up through the floor)
        brng = random.Random(104729 * (layer - 1) + 7)
        opens = sorted(walk)
        _S["below_cells"] = [opens[brng.randrange(len(opens))]
                             for _ in range(min(40, len(opens)))]
        # Per-layer fog hue (warmer as N rises)
        fog = _fog_color(layer)
        glFogfv(GL_FOG_COLOR, (*fog, 1.0))
        glClearColor(*fog, 1.0)
        _S["world_list"] = _build_world_list()
        _S["star_list"] = _build_star_list(layer)
        # EC_SPACELAND_OVERVIEW=1 forces the orbit camera (video/CI hook)
        if os.environ.get("EC_SPACELAND_OVERVIEW"):
            _S["overview"] = True
        _S["ready"] = True
    except Exception as exc:
        print(f"spaceland: enter degraded ({exc})")
        _S["ready"] = False


def _place_solids(layer, rng):
    """Rotating Platonic solids floating in open cells (+1 per layer)."""
    open_cells = [c for c in sorted(_S["walk"])
                  if _S["grid"][c] == 0 and c != _S["goal"]
                  and c not in _S["rifts"] and c != _S["well"]]
    rng.shuffle(open_cells)
    return [{"cell": cell, "kind": i % len(_SOLID_TABLE),
             "phase": rng.uniform(0, 6.28)}
            for i, cell in enumerate(open_cells[:min(3 + layer, 9)])]


# --------------------------------------------------------------------------
# Movement: AI walker (BFS path) with WASD/arrow player override
# --------------------------------------------------------------------------
def _is_wall_at(x, z):
    c, r = int(math.floor(x)), int(math.floor(z))
    return (r, c) not in _S["walk"]


def _clear_at(x, z, rad=0.22):
    return not any(_is_wall_at(x + dx, z + dz)
                   for dx in (-rad, rad) for dz in (-rad, rad))


def _ease_angle(cur, target, k):
    d = (target - cur + math.pi) % (2 * math.pi) - math.pi
    return cur + d * min(1.0, k)


def _descend():
    """Drop one layer down the stack. Layer 0 is Flatland: 'fell'."""
    if _S["layer"] - 1 < 1:
        return "fell"
    enter(_S["grid"], _S["goal"], _S["layer"] - 1)   # preserves mind/stage
    return "hazard"


def _advance(t, dt, keys):
    w = _S["walker"]
    moved = 0.0
    event = None
    player_keys = False
    if keys:
        try:
            fwd = keys[pygame.K_w] or keys[pygame.K_UP]
            back = keys[pygame.K_s] or keys[pygame.K_DOWN]
            left = keys[pygame.K_a] or keys[pygame.K_LEFT]
            right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
            player_keys = bool(fwd or back or left or right)
        except Exception:
            fwd = back = left = right = False
    if player_keys:
        _S["last_input"] = t
        w["path"] = []                       # player took the wheel

    if t - _S["last_input"] < PLAYER_TIMEOUT:
        # PLAYER OVERRIDE — turn with A/D or arrows, move with wall collision
        if player_keys:
            w["yaw"] += (right - left) * TURN_SPEED * dt
            step = (fwd - back) * PLAYER_SPEED * dt
            nx = w["pos"][0] + math.cos(w["yaw"]) * step
            nz = w["pos"][1] + math.sin(w["yaw"]) * step
            if _clear_at(nx, w["pos"][1]):
                w["pos"][0] = nx
            if _clear_at(w["pos"][0], nz):
                w["pos"][1] = nz
            moved = abs(step)
    else:
        # AI resumes: BFS path to the shrine, smooth interpolated steps
        if not w["path"]:
            cell = (int(w["pos"][1]), int(w["pos"][0]))
            w["path"] = _path_to_goal(cell)
        if w["path"]:
            tr, tc = w["path"][0]
            dx, dz = tc + 0.5 - w["pos"][0], tr + 0.5 - w["pos"][1]
            dist = math.hypot(dx, dz)
            if dist > 1e-6:
                w["yaw"] = _ease_angle(w["yaw"], math.atan2(dz, dx), dt * 4.0)
                step = min(dist, WALK_SPEED * dt)
                w["pos"][0] += dx / dist * step
                w["pos"][1] += dz / dist * step
                moved = step
            if dist < 0.10:
                w["path"].pop(0)

    _S["bob"] += moved * 7.0
    _S["chill"] = max(0.0, _S["chill"] - dt)

    # --- Consciousness: cold rifts + ambient drain, staged descents -------
    here = (int(w["pos"][1]), int(w["pos"][0]))
    if here in _S["rifts"]:
        _S["mind"] -= RIFT_DRAIN * 30.0 * dt
        _S["chill"] = 0.6
        if t - _S["haz_t"] > 1.2:
            _S["haz_t"] = t
            event = "hazard"
    _S["mind"] -= AMBIENT_DRAIN * 30.0 * dt

    if _S["mind"] <= 0.0:
        return "fell"
    stage_now = sum(1 for th in STAGES if _S["mind"] < th)
    if stage_now > _S["stage"]:
        _S["stage"] = stage_now
        _S["chill"] = 0.8
        ev = _descend()
        if ev == "fell":
            return "fell"
        return "hazard"

    # Descent well: a dark spiral that drops the walker one layer down
    if _S["well"] and t - _S["desc_t"] > 2.0:
        wr, wc = _S["well"]
        if math.hypot(w["pos"][0] - (wc + 0.5), w["pos"][1] - (wr + 0.5)) < 0.38:
            _S["desc_t"] = t
            _S["chill"] = 0.8
            ev = _descend()
            if ev == "fell":
                return "fell"
            return "hazard"

    gr, gc = _S["goal"]
    if (not _S["done"] and
            math.hypot(w["pos"][0] - (gc + 0.5), w["pos"][1] - (gr + 0.5)) < 0.45):
        _S["done"] = True
        _S["mind"] = min(100.0, _S["mind"] + 15.0)   # the shrine restores
        return "goal"
    return event


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def _draw_platonic(kind, color, dim=0.22):
    verts, faces = _SOLID_TABLE[kind]
    m = max(math.hypot(v[0], math.hypot(v[1], v[2])) for v in verts)
    verts = [(v[0] / m, v[1] / m, v[2] / m) for v in verts]
    glColor4f(color[0] * dim, color[1] * dim, color[2] * dim, 0.9)
    for face in faces:                        # dim solid body
        glBegin(GL_POLYGON)
        for i in face:
            glVertex3f(*verts[i])
        glEnd()
    glColor3f(*color)                         # emissive wireframe over it
    glLineWidth(2.0)
    for face in faces:
        glBegin(GL_LINE_LOOP)
        for i in face:
            glVertex3f(*verts[i])
        glEnd()
    glLineWidth(1.0)


def _draw_solids(t):
    for s in _S["solids"]:
        r, c = s["cell"]
        glPushMatrix()
        glTranslatef(c + 0.5, 0.55 + 0.10 * math.sin(t * 1.3 + s["phase"]),
                     r + 0.5)
        glRotatef(t * 47.0 + s["phase"] * 57.3, 0.3, 1.0, 0.2)
        glScalef(0.26, 0.26, 0.26)
        _draw_platonic(s["kind"], _SOLID_COLORS[s["kind"]])
        glPopMatrix()


def _draw_shrine(t):
    gr, gc = _S["goal"]
    pulse = 0.5 + 0.5 * math.sin(t * 3.0)
    glPushMatrix()
    glTranslatef(gc + 0.5, 0.55 + 0.06 * pulse, gr + 0.5)
    glRotatef(t * 35.0, 0, 1, 0)
    s = 0.30 + 0.06 * pulse
    glScalef(s, s, s)
    _draw_platonic(3, (1.0, 0.84 + 0.12 * pulse, 0.30), dim=0.45)
    glPopMatrix()
    # Translucent glow column rising from the shrine
    glDepthMask(GL_FALSE)
    glColor4f(1.0, 0.85, 0.35, 0.10 + 0.10 * pulse)
    glBegin(GL_QUAD_STRIP)
    for i in range(9):
        a = i * math.pi / 4.0
        x = gc + 0.5 + 0.33 * math.cos(a)
        z = gr + 0.5 + 0.33 * math.sin(a)
        glVertex3f(x, 0.0, z)
        glVertex3f(x, 3.2, z)
    glEnd()
    glDepthMask(GL_TRUE)


def _draw_hazards(t):
    """Cold rifts (dark pulsing floor cells) and the descent well spiral."""
    glDepthMask(GL_FALSE)
    for i, (r, c) in enumerate(sorted(_S["rifts"])):
        pulse = 0.5 + 0.5 * math.sin(t * 2.6 + i * 1.7)
        glColor4f(0.01, 0.02, 0.10, 0.55 + 0.30 * pulse)
        glBegin(GL_QUADS)
        glVertex3f(c + 0.05, 0.02, r + 0.05)
        glVertex3f(c + 0.95, 0.02, r + 0.05)
        glVertex3f(c + 0.95, 0.02, r + 0.95)
        glVertex3f(c + 0.05, 0.02, r + 0.95)
        glEnd()
        glColor4f(0.30, 0.60, 1.00, 0.20 + 0.45 * pulse)   # icy rim
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(c + 0.10, 0.03, r + 0.10)
        glVertex3f(c + 0.90, 0.03, r + 0.10)
        glVertex3f(c + 0.90, 0.03, r + 0.90)
        glVertex3f(c + 0.10, 0.03, r + 0.90)
        glEnd()
        glLineWidth(1.0)
    if _S["well"]:
        wr, wc = _S["well"]
        cx, cz = wc + 0.5, wr + 0.5
        glColor4f(0.0, 0.0, 0.02, 0.9)                     # the dark mouth
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(cx, 0.02, cz)
        for i in range(17):
            a = i * math.pi / 8.0
            glVertex3f(cx + 0.44 * math.cos(a), 0.02, cz + 0.44 * math.sin(a))
        glEnd()
        glColor4f(0.55, 0.35, 1.00, 0.8)                   # rotating spiral
        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)
        for i in range(49):
            th = i * (4.0 * math.pi / 48.0)
            rad = 0.04 + 0.40 * th / (4.0 * math.pi)
            glVertex3f(cx + rad * math.cos(th - t * 1.8), 0.035,
                       cz + rad * math.sin(th - t * 1.8))
        glEnd()
        glLineWidth(1.0)
    glDepthMask(GL_TRUE)


def _draw_stack(t):
    """The layer stack made visible: the layer above's floor hangs inverted
    over the starfield; the layer beneath glows up through the floor."""
    n = _S["n"]
    shim = 0.5 + 0.5 * math.sin(t * 0.8)
    glDepthMask(GL_FALSE)

    # ABOVE: faint translucent inverted lattice at the sky's edge
    ya = 5.6
    glColor4f(0.55, 0.45, 0.95, 0.05 + 0.05 * shim)
    glBegin(GL_LINES)
    for i in range(n + 1):
        glVertex3f(i, ya, 0); glVertex3f(i, ya, n)
        glVertex3f(0, ya, i); glVertex3f(n, ya, i)
    glEnd()
    glColor4f(0.50, 0.40, 0.92, 0.09 + 0.05 * shim)
    glBegin(GL_QUADS)
    for (r, c) in _S["walls"]:      # as above, so below: the same footprint
        glVertex3f(c + 0.08, ya, r + 0.08)
        glVertex3f(c + 0.92, ya, r + 0.08)
        glVertex3f(c + 0.92, ya, r + 0.92)
        glVertex3f(c + 0.08, ya, r + 0.92)
    glEnd()

    # BELOW: a dimmer lattice glowing up through the walkable floor
    yb = 0.012
    glColor4f(0.85, 0.55, 0.95, 0.05 + 0.03 * shim)
    glBegin(GL_LINES)
    for i in range(n + 1):
        glVertex3f(i, yb, 0); glVertex3f(i, yb, n)
        glVertex3f(0, yb, i); glVertex3f(n, yb, i)
    glEnd()
    glColor4f(0.95, 0.65, 0.85, 0.045 + 0.035 * shim)
    glBegin(GL_QUADS)
    for (r, c) in _S["below_cells"]:
        glVertex3f(c + 0.15, yb, r + 0.15)
        glVertex3f(c + 0.85, yb, r + 0.15)
        glVertex3f(c + 0.85, yb, r + 0.85)
        glVertex3f(c + 0.15, yb, r + 0.85)
    glEnd()
    glDepthMask(GL_TRUE)


def _draw_vignette():
    """Blue screen-edge chill when a rift is draining the mind."""
    a = min(1.0, _S["chill"] / 0.6) * 0.55
    if a <= 0.02:
        return
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, 1, 0, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_FOG)
    e = 0.20
    quads = [[(0, 0, a), (e, 0, 0), (e, 1, 0), (0, 1, a)],          # left
             [(1, 0, a), (1 - e, 0, 0), (1 - e, 1, 0), (1, 1, a)],  # right
             [(0, 0, a), (1, 0, a), (1, e, 0), (0, e, 0)],          # bottom
             [(0, 1, a), (1, 1, a), (1, 1 - e, 0), (0, 1 - e, 0)]]  # top
    glBegin(GL_QUADS)
    for quad in quads:
        for x, y, alpha in quad:
            glColor4f(0.25, 0.45, 1.0, alpha)
            glVertex2f(x, y)
    glEnd()
    glEnable(GL_FOG)
    glEnable(GL_DEPTH_TEST)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def _draw_hud():
    if _S["font"] is None:
        return
    try:
        win = _S["win"]
        text = (f"SPACELAND — LAYER {_S['layer']} · as above, so below · "
                f"WASD walk · V overview")
        strip = pygame.Surface((win, 24), pygame.SRCALPHA)
        strip.fill((16, 8, 38, 200))
        strip.blit(_S["font"].render(text, True, (255, 215, 0)), (8, 4))
        # Consciousness meter: gold when whole, icy blue as the cold drinks it
        frac = max(0.0, min(1.0, _S["mind"] / 100.0))
        bw, x0 = 130, win - 140
        col = (int(90 + 165 * frac), int(160 + 55 * frac), int(255 - 255 * frac))
        pygame.draw.rect(strip, (200, 200, 220), (x0, 5, bw, 14), 1)
        pygame.draw.rect(strip, col, (x0 + 2, 7, max(1, int((bw - 4) * frac)), 10))
        data = pygame.image.tostring(strip, "RGBA", True)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_FOG)
        glWindowPos2i(0, win - 24)
        glDrawPixels(win, 24, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glWindowPos2i(0, 0)     # restore for the 2D path's glDrawPixels
        glEnable(GL_FOG)
        glEnable(GL_DEPTH_TEST)
    except Exception:
        _S["font"] = None       # HUD unavailable: keep rendering the world


def _camera(t):
    n, w = _S["n"], _S["walker"]
    if _S["overview"]:
        # High slow-rotating orbit over the whole extruded lattice
        cx = cz = n / 2.0
        a = t * 0.15
        eye = (cx + n * 0.95 * math.cos(a), n * 0.80,
               cz + n * 0.95 * math.sin(a))
        glFogf(GL_FOG_DENSITY, FOG_OVERVIEW)
        gluLookAt(*eye, cx, 0.0, cz, 0, 1, 0)
    else:
        bob = 0.035 * math.sin(_S["bob"])
        ex, ez = w["pos"]
        ey = EYE_HEIGHT + bob
        glFogf(GL_FOG_DENSITY, FOG_FPS)
        gluLookAt(ex, ey, ez,
                  ex + math.cos(w["yaw"]), ey, ez + math.sin(w["yaw"]),
                  0, 1, 0)


def update_and_render(t, keys_pressed=None):
    """Advance the walker, draw one frame.

    Returns None, "goal" (shrine reached — caller re-enters with layer+1),
    "hazard" (cold rift bite or a one-layer descent — caller may play a cold
    tone), or "fell" (consciousness gone / below layer 1 — back to Flatland).
    """
    if not (_GL_OK and _S["ready"]):
        return None
    try:
        dt = 1.0 / 30.0 if _S["last_t"] is None else \
            min(0.1, max(0.0, t - _S["last_t"]))
        _S["last_t"] = t
        event = _advance(t, dt, keys_pressed)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        _camera(t)

        # Starfield sky, glued to the camera so it reads as infinitely far
        glDepthMask(GL_FALSE)
        glPushMatrix()
        if _S["overview"]:
            glTranslatef(_S["n"] / 2.0, 0.0, _S["n"] / 2.0)
        else:
            glTranslatef(_S["walker"]["pos"][0], 0.0, _S["walker"]["pos"][1])
        glCallList(_S["star_list"])
        glPopMatrix()
        glDepthMask(GL_TRUE)

        glCallList(_S["world_list"])
        _draw_stack(t)
        _draw_hazards(t)
        _draw_solids(t)
        _draw_shrine(t)
        _draw_vignette()
        _draw_hud()
        return event
    except Exception as exc:
        print(f"spaceland: frame degraded ({exc})")
        return None
