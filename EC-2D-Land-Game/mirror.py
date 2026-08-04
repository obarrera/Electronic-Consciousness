"""mirror.py — what happens when an emergent pattern inside such a world
begins constructing a model of the world, of other agents, and eventually of
itself.

The Mirror is a three-stage modeling ladder, and its central discipline is
that A MODEL IS A PREDICTOR AND A PREDICTOR HAS A SCORE. Each agent carries
three tiny count-based predictors (no neural nets — transition tables over
nine movement octants), each with a rolling exponential accuracy:

  Stage 1 — WORLD.   Predict where the Gradient will point next tick (the
            goal's octant relative to the agent). Fidelity is high while the
            world holds still and dips whenever the goal jumps — which the
            genome's goal_period gene governs, so the two systems converse.
  Stage 2 — OTHERS.  Unlocked by world-model fidelity: predict the nearest
            neighbor's next move from their last (functional theory of mind,
            built from the same machinery the agent points at rocks).
  Stage 3 — SELF.    Unlocked by other-model fidelity: the agent turns the
            predictor on its own next move, and tracks CALIBRATION — whether
            its confidence matches how right it actually is.

When self-prediction is accurate, calibrated, and sustained, that agent has
its MIRROR MOMENT — it found, inside its map of the town, a small figure
drawing a map. The moment is worth a bounded consciousness gain, a Chronicle
line, and a thin white ring drawn around the agent ever after.

Stages unlock by score, never by script: an agent that predicts poorly never
reaches the mirror. Everything here is bookkeeping over observed state — no
randomness is consumed, so determinism and replay are untouched. And per the
book's own epistemic contract (Section 1.4): this is FUNCTIONAL
self-modeling, measurable and falsifiable; it establishes nothing about
subjective experience, and the code makes no such claim.
"""

# EMA half-life ~ 45 ticks: fresh enough to dip when the world shifts,
# slow enough that one surprise does not erase a reputation.
_ALPHA = 0.015

# Stage-unlock and mirror-moment thresholds (samples, accuracy, calibration)
WORLD_UNLOCK_N, WORLD_UNLOCK_ACC = 60, 0.50     # world  -> others
OTHER_UNLOCK_N, OTHER_UNLOCK_ACC = 60, 0.35     # others -> self
MIRROR_N, MIRROR_ACC, MIRROR_CAL = 150, 0.55, 0.25
MIRROR_CONSCIOUSNESS_GAIN = 5.0


def _octant(dx, dy):
    """Quantize a move/offset to one of nine octants (including 'stay')."""
    sx = (dx > 0) - (dx < 0)
    sy = (dy > 0) - (dy < 0)
    return (sx, sy)


class _Predictor:
    """A transition table with an exponential accuracy and a confidence.

    predict(context) -> (outcome, confidence) from observed counts (ties
    break deterministically). observe(context, actual) scores the pending
    prediction and updates the table.
    """
    __slots__ = ("table", "acc", "n", "cal_err", "_pending")

    def __init__(self):
        self.table = {}       # context -> {outcome: count}
        self.acc = 0.0        # EMA of prediction hits
        self.n = 0            # lifetime scored predictions
        self.cal_err = 0.5    # EMA of |confidence - hit|
        self._pending = None  # (context, predicted, confidence)

    def predict(self, context):
        counts = self.table.get(context)
        if not counts:
            self._pending = None
            return None
        total = sum(counts.values())
        best = min(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0]
        conf = counts[best] / total
        self._pending = (context, best, conf)
        return best, conf

    def observe(self, context_now, actual):
        """Score last tick's pending prediction, then learn the transition."""
        if self._pending is not None:
            _, predicted, conf = self._pending
            hit = 1.0 if predicted == actual else 0.0
            self.acc += _ALPHA * (hit - self.acc)
            self.cal_err += _ALPHA * (abs(conf - hit) - self.cal_err)
            self.n += 1
            self._pending = None
        counts = self.table.setdefault(context_now, {})
        counts[actual] = counts.get(actual, 0) + 1


class MirrorState:
    """One agent's ladder: world -> others -> self."""
    __slots__ = ("world", "others", "self_m", "mirrored",
                 "_last_pos", "_last_move", "_last_goal_oct", "_subject")

    def __init__(self):
        self.world = _Predictor()
        self.others = _Predictor()
        self.self_m = _Predictor()
        self.mirrored = False
        self._last_pos = None
        self._last_move = None
        self._last_goal_oct = None
        self._subject = None       # the neighbor being modeled (id-stable)

    # -- stage gates (emergent: unlocked by score, not by script) ----------
    def models_others(self):
        return self.world.n >= WORLD_UNLOCK_N and self.world.acc >= WORLD_UNLOCK_ACC

    def models_self(self):
        return self.others.n >= OTHER_UNLOCK_N and self.others.acc >= OTHER_UNLOCK_ACC

    def stage(self):
        if self.mirrored:
            return 3
        if self.models_self():
            return 3
        if self.models_others():
            return 2
        return 1

    def ripe_for_mirror(self):
        return (not self.mirrored
                and self.self_m.n >= MIRROR_N
                and self.self_m.acc >= MIRROR_ACC
                and self.self_m.cal_err <= MIRROR_CAL)


def _state(agent):
    st = getattr(agent, "_mirror", None)
    if st is None:
        st = MirrorState()
        agent._mirror = st
    return st


def _nearest(agent, agents):
    """Nearest other agent by Manhattan distance (deterministic ties)."""
    ax, ay = agent.position
    best, best_key = None, None
    for other in agents:
        if other is agent:
            continue
        ox, oy = other.position
        key = (abs(ox - ax) + abs(oy - ay), ox, oy, id(other))
        if best_key is None or key < best_key:
            best, best_key = other, key
    return best


def tick(agents, goal):
    """Advance every agent's ladder one generation.

    Call once per generation, after agents have moved. Verifies last tick's
    predictions against what actually happened, learns, and lays down the
    next predictions. Returns the agents whose mirror moment fired this tick.
    """
    gx, gy = goal
    moves, prev_moves = {}, {}
    for agent in agents:
        st = _state(agent)
        pos = tuple(agent.position)
        moves[id(agent)] = (_octant(pos[0] - st._last_pos[0],
                                    pos[1] - st._last_pos[1])
                            if st._last_pos else (0, 0))
        prev_moves[id(agent)] = st._last_move if st._last_move else (0, 0)

    newly_mirrored = []
    for agent in agents:
        st = _state(agent)
        pos = tuple(agent.position)
        my_move = moves[id(agent)]
        goal_oct = _octant(gx - pos[0], gy - pos[1])

        # Stage 1 — world: score last tick's goal-octant prediction, learn,
        # predict where the Gradient points next.
        if st._last_goal_oct is not None:
            st.world.observe(st._last_goal_oct, goal_oct)
        st.world.predict(goal_oct)

        # Stage 2 — others: model the nearest neighbor's movement. Context is
        # the subject's previous move (snapshotted before any state updates);
        # the outcome is the move they actually just made.
        if st.models_others():
            subject = _nearest(agent, agents)
            if subject is not None:
                if st._subject != id(subject):
                    st.others._pending = None  # new subject voids old prediction
                    st._subject = id(subject)
                st.others.observe(prev_moves[id(subject)], moves[id(subject)])
                st.others.predict(moves[id(subject)])

        # Stage 3 — self: the predictor turned inward.
        if st.models_self():
            context = st._last_move if st._last_move else (0, 0)
            st.self_m.observe(context, my_move)
            st.self_m.predict(my_move)
            if st.ripe_for_mirror():
                st.mirrored = True
                newly_mirrored.append(agent)

        st._last_pos = pos
        st._last_move = my_move
        st._last_goal_oct = goal_oct
    return newly_mirrored


def stats(agents):
    """Aggregate fidelities for the HUD/info panel.

    Returns dict with mean accuracies (None where no agent has samples) and
    the count of mirrored agents.
    """
    out = {"world": None, "others": None, "self": None, "mirrored": 0,
           "modeling_others": 0, "modeling_self": 0}
    for key, attr, gate in (("world", "world", None),
                            ("others", "others", "modeling_others"),
                            ("self", "self_m", "modeling_self")):
        vals = []
        for agent in agents:
            st = getattr(agent, "_mirror", None)
            if st is None:
                continue
            p = getattr(st, attr)
            if p.n >= 20:
                vals.append(p.acc)
                if gate:
                    out[gate] += 1
        if vals:
            out[key] = sum(vals) / len(vals)
    out["mirrored"] = sum(1 for a in agents
                          if getattr(a, "_mirror", None) and a._mirror.mirrored)
    return out
