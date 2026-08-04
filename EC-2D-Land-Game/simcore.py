"""simcore.py — deterministic core for EC-2D-Land: named RNG streams, the
canonical state hash, and the run manifest.

One root seed (default: the ouroboros seed; EC_SEED overrides) fans out into
named streams via a stable hash (hashlib, never Python's salted hash()):

  * "world"   — lattice/board randomness: initial board, goal placement and
                movement, solids/elements/symbols, Game-of-Life births
  * "agents"  — agent life: decisions, exploration, lifespans, incentives,
                rebirth placement, reproduction
  * "brain"   — shared-MLP weight init and minibatch shuffling
  * "fx"      — particles and other visual-only randomness; consuming it can
                never perturb behavior (separate stream by construction)
  * "spaceland:<layer>" — per-layer hazard/spawn/solid layout

Streams survive an ouroboros reset only by re-initializing the pool with the
new seed (reset_pool). Proxies (WORLD, AGENTS, FX, NP_*) always resolve the
CURRENT pool, so call sites never hold a stale stream.
"""
import hashlib
import json
import os
import platform
import random
import subprocess
import time

import numpy as np

SCHEMA_VERSION = 1

_POOL = None


def stream_seed(root, name):
    """Stable 64-bit seed for stream `name` under `root` (cross-process)."""
    digest = hashlib.sha256(f"{root}:{name}".encode()).hexdigest()
    return int(digest[:16], 16)


class RngPool:
    def __init__(self, root_seed):
        self.root = int(root_seed)
        self._streams = {}
        self._np_streams = {}

    def stream(self, name):
        if name not in self._streams:
            self._streams[name] = random.Random(stream_seed(self.root, name))
        return self._streams[name]

    def np_stream(self, name):
        if name not in self._np_streams:
            self._np_streams[name] = np.random.default_rng(
                stream_seed(self.root, name))
        return self._np_streams[name]


def init_pool(root_seed):
    """(Re)initialize the global pool — startup and each ouroboros reset."""
    global _POOL
    _POOL = RngPool(root_seed)
    return _POOL


def pool():
    if _POOL is None:
        init_pool(0)
    return _POOL


def seed_for(name):
    return stream_seed(pool().root, name)


class _Proxy:
    """Live proxy to the current pool's stream: reset-safe at every call."""

    def __init__(self, name, numpy=False):
        self._name = name
        self._numpy = numpy

    def __getattr__(self, attr):
        src = (pool().np_stream(self._name) if self._numpy
               else pool().stream(self._name))
        return getattr(src, attr)


WORLD = _Proxy("world")
AGENTS = _Proxy("agents")
FX = _Proxy("fx")
NP_WORLD = _Proxy("world", numpy=True)
NP_AGENTS = _Proxy("agents", numpy=True)


# ---------------------------------------------------------------------------
# Canonical state hash + run manifest
# ---------------------------------------------------------------------------

def state_hash(tick, environment, agents, spaceland_layer=None,
               spaceland_pos=None):
    """SHA-256 over the sorted, quantized simulation state. The same
    function serves headless and windowed runs — that equality is the test.
    """
    h = hashlib.sha256()
    h.update(f"tick:{tick};".encode())
    h.update(("grid:" + ";".join(
        ",".join(str(int(v)) for v in row)
        for row in environment.grid)).encode())
    h.update(f";goal:{tuple(environment.goal)};".encode())
    h.update(("warm:" + repr(sorted(environment.warm_cells.items()))).encode())
    h.update(("chill:" + repr(sorted(environment.chill_cells.items()))).encode())
    for a in sorted(agents, key=lambda a: getattr(a, "lineage_id", 0)):
        h.update((f"agent:{getattr(a, 'lineage_id', 0)},"
                  f"{tuple(a.position)},{round(float(a.energy), 4)},"
                  f"{round(float(a.level_of_consciousness), 4)},"
                  f"{a.gender},{a.generation},{a.age};").encode())
    if spaceland_layer is not None:
        pos = (None if spaceland_pos is None else
               tuple(round(float(p), 4) for p in spaceland_pos))
        h.update(f"space:{spaceland_layer},{pos};".encode())
    return h.hexdigest()


def _git_info(cwd):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd,
                                capture_output=True, text=True,
                                timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=cwd,
                                    capture_output=True, text=True,
                                    timeout=5).stdout.strip())
        return commit, dirty
    except Exception:
        return None, None


def write_manifest(run_dir, root_seed, tick, final_hash, started_at,
                   headless):
    """manifest.json for a completed run; failures are non-fatal."""
    try:
        os.makedirs(run_dir, exist_ok=True)
        commit, dirty = _git_info(os.path.dirname(os.path.abspath(__file__)))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "root_seed": int(root_seed),
            "git_commit": commit,
            "git_dirty": dirty,
            "python": platform.python_version(),
            "headless": bool(headless),
            "ticks": int(tick),
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "final_state_hash": final_hash,
        }
        path = os.path.join(run_dir, "manifest.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        return path
    except OSError as exc:
        print(f"simcore: could not write manifest ({exc})")
        return None
