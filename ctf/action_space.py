"""Action spaces: the network's categorical heads and how they become engine actions.

* `bins` (default): throttle bins x turn bins x jump, sent to the engine as is.
* `goal`: heading bins (relative to yaw) x speed bins x jump; the fixed controller
  of ctf/controller.py turns the goal into (throttle, turn) every tick.

Each space owns its feature layout (the previous-action block differs), its
action tables, its head sizes, the PlayerMemory flavour, the decoding of
sampled indices into engine actions, and the projection of a scripted
teacher's action onto its heads (kickstarting labels).
"""

import math

from ctf import features as F
from ctf.controller import (CTL_ACCEL, CTL_DT, CTL_FRICTION, CTL_MAX_SPEED, CTL_TURN_BASE,
                            CTL_TURN_PENALTY, goal_step)

PROJECT_TICKS = 3.0  # physics projection: extrapolate the teacher's velocity change this far


def _nearest(table, value):
    best, best_d = 0, abs(table[0] - value)
    for i in range(1, len(table)):
        d = abs(table[i] - value)
        if d < best_d:
            best, best_d = i, d
    return best


def _nearest_index(table, value):
    """Nearest bin of an engine value (clamped to [-1, 1]; unparsable values count as 0)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    value = -1.0 if value < -1.0 else 1.0 if value > 1.0 else value
    best, best_d = 0, abs(table[0] - value)
    for i in range(1, len(table)):
        d = abs(table[i] - value)
        if d < best_d:
            best, best_d = i, d
    return best


def _clean(value, low=-1.0, high=1.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return low if value < low else high if value > high else value


def _norm(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def heading_index(relative, headings=F.HEADING_VALUES):
    """Nearest heading bin (circular) to an angle in radians relative to yaw."""
    best, best_d = 0, 10.0
    for i, h in enumerate(headings):
        d = abs(_norm(relative - math.radians(h)))
        if d < best_d - 1e-12:
            best, best_d = i, d
    return best


def goal_indices(relative, speed, headings=F.HEADING_VALUES, speeds=F.SPEED_VALUES):
    return heading_index(relative, headings), _nearest(speeds, speed)


def project_action(me, throttle, turn, ticks=PROJECT_TICKS):
    """(heading relative to yaw, speed) that explains an engine action physically.

    One engine step of (throttle, turn) from the current state gives the velocity
    change dv; the intended velocity is taken as v + ticks * dv (capped at the top
    speed), i.e. where the teacher is pushing the velocity. Without intended motion
    the heading is where the teacher is turning.
    """
    yaw = _clean(me.get("yaw"), -1e9, 1e9)
    vx, vy = _clean(me.get("vx"), -100.0, 100.0), _clean(me.get("vy"), -100.0, 100.0)
    throttle, turn = _clean(throttle), _clean(turn)
    sp = math.hypot(vx, vy)
    rate = CTL_TURN_BASE * (1.0 - CTL_TURN_PENALTY * min(1.0, sp / CTL_MAX_SPEED)) * CTL_DT
    new_yaw = yaw + turn * rate
    nvx = vx + CTL_ACCEL * throttle * math.sin(new_yaw)
    nvy = vy + CTL_ACCEL * throttle * math.cos(new_yaw)
    n = math.hypot(nvx, nvy)
    if n > CTL_MAX_SPEED:
        nvx, nvy = nvx * CTL_MAX_SPEED / n, nvy * CTL_MAX_SPEED / n
    nvx, nvy = nvx * CTL_FRICTION, nvy * CTL_FRICTION
    tx, ty = vx + ticks * (nvx - vx), vy + ticks * (nvy - vy)
    speed = math.hypot(tx, ty)
    if speed < 0.5:
        return ticks * turn * rate, 0.0
    return _norm(math.atan2(tx, ty) - yaw), min(speed, CTL_MAX_SPEED * CTL_FRICTION)


def teacher_goal(decide, action, observation):
    """(heading relative to yaw, speed) a scripted teacher intends on this tick.

    Bots built from opponents/_core.py record the (absolute heading, target speed)
    their velocity-vector control was given in `M.goal`; that commanded goal is
    used when present, else the physics projection of the action.
    """
    me = observation.get("self") or {}
    memory = getattr(decide, "__globals__", {}).get("M")
    intent = getattr(memory, "goal", None)
    if intent is not None:
        a, vt = intent
        return _norm(a - _clean(me.get("yaw"), -1e9, 1e9)), _clean(vt, 0.0, 18.0)
    return project_action(me, action[0], action[1])


def sync_teacher(decide, throttle):
    """Tell a shadow _core teacher which throttle was actually executed this tick.

    Its bump detector predicts the velocity from its own last throttle (`M.prev`);
    when the learner drove differently it would see phantom bumps (and plan around
    phantom obstacles), so the record is replaced by the executed throttle.
    """
    memory = getattr(decide, "__globals__", {}).get("M")
    prev = getattr(memory, "prev", None)
    if isinstance(prev, tuple) and len(prev) == 7:
        memory.prev = prev[:5] + (throttle,) + prev[6:]


class BinsSpace:
    name = "bins"
    feature_names = F.FEATURE_NAMES
    feature_dim = F.FEATURE_DIM
    tables = F.ACTION_TABLES
    head_names = ("throttle", "turn", "jump")
    head_sizes = (F.N_THROTTLE, F.N_TURN, 2)

    @staticmethod
    def memory():
        return F.PlayerMemory()

    @staticmethod
    def apply(memory, action, observation=None):
        """Engine action for sampled indices; records the action in `memory`."""
        t, r, j = action
        memory.set_action(t, r, j)
        return F.THROTTLE_VALUES[t], F.TURN_VALUES[r], bool(j)

    @staticmethod
    def teacher_label(decide, observation):
        """A scripted teacher's action for this documented observation, as head indices."""
        try:
            action = decide(observation)
            return (_nearest_index(F.THROTTLE_VALUES, action[0]),
                    _nearest_index(F.TURN_VALUES, action[1]), 1 if action[2] else 0)
        except Exception:
            return None


class GoalSpace:
    name = "goal"
    feature_names = F.GOAL_FEATURE_NAMES
    feature_dim = F.GOAL_FEATURE_DIM
    tables = F.GOAL_ACTION_TABLES
    head_names = ("heading", "speed", "jump")
    head_sizes = (F.N_HEADING, F.N_SPEED, 2)

    @staticmethod
    def memory():
        return F.PlayerMemory(goal=True)

    @staticmethod
    def apply(memory, action, observation):
        h, s, j = action
        return goal_step(memory, observation, h, s, j, F.HEADING_VALUES, F.SPEED_VALUES)

    @staticmethod
    def teacher_label(decide, observation):
        try:
            action = decide(observation)
            relative, speed = teacher_goal(decide, action, observation)
            h, s = goal_indices(relative, speed)
            return h, s, 1 if action[2] else 0
        except Exception:
            return None


BINS, GOAL = BinsSpace(), GoalSpace()
SPACES = {"bins": BINS, "goal": GOAL}


def get(name):
    if isinstance(name, (BinsSpace, GoalSpace)):
        return name
    try:
        return SPACES[name or "bins"]
    except KeyError:
        raise ValueError(f"unknown action space {name!r}") from None


def of_tables(tables):
    """The action space whose tables have these heads (files store their tables)."""
    return GOAL if "heading" in tables else BINS
