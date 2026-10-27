"""Distilling a goal-space policy into the bins action space (a pure end-to-end student).

The teacher is a goal-space policy (heading/speed/jump heads driven by the fixed
controller of ctf/controller.py). The student outputs throttle/turn/jump bins
directly. Every student player gets a shadow teacher (`PolicyTeacher`): its own
goal-space PlayerMemory and controller state, fed the same documented observation.
Each tick the teacher's whole goal distribution is pushed through the controller
(vectorized here, `control_batch`), mapped onto the student's bins (`bins_weights`:
nearest, speed-capped nearest, or stochastic rounding between the two bracketing
bins), and summed into per-head soft targets:

    q_throttle[b] = sum over goals g of p(g) * weight of bin b for throttle(g),  same for turn,
    q_jump = the teacher's jump head.

These per-head marginals are exactly the best a factorized (per-head) student can
match. Stochastic rounding makes the student's expected throttle and turn equal to
the controller's, so a sampling student dithers between bins instead of rounding.

The shadow teacher then commits a goal to its memory (its previous-goal features
and the controller's forward/reverse bit). With commit='sample' it is a draw from its
own distribution, as the deployed teacher would; but that draw is invisible to the
student, so labels then condition on a hidden random history (the noise that sank
the scripted-teacher kickstarts). The default commit='posterior' draws the goal
conditioned on having produced the bins the student actually executed, so the
teacher's history follows the student's own (which the student sees through its
previous-action features). The throttle its bump detector believes it sent is the
one the student executed (as `action_space.sync_teacher`).

Numpy only; `snap_indices` also serves the rounding-floor evaluation
(`ctf/evaluate.py` option `@snap`).
"""

import math

import numpy as np

from ctf import features as F
from ctf.controller import (CTL_ACCEL, CTL_ALIGN_DV, CTL_ALIGN_FRAC, CTL_ALIGN_MIN, CTL_DT,
                            CTL_FRICTION, CTL_HYSTERESIS, CTL_MAX_SPEED, CTL_MISALIGN,
                            CTL_TURN_BASE, CTL_TURN_PENALTY)
from ctf.npnets import sample_logits

_PI = math.pi
_TAU = 2.0 * math.pi


def _edges(values):
    values = np.asarray(values, dtype=np.float64)
    return (values[1:] + values[:-1]) / 2.0


_THROTTLE_EDGES = _edges(F.THROTTLE_VALUES)
_TURN_EDGES = _edges(F.TURN_VALUES)


def snap_indices(throttle, turn):
    """Nearest student bins (ties to the lower value, like action_space._nearest_index)."""
    t = np.clip(np.asarray(throttle, dtype=np.float64), -1.0, 1.0)
    r = np.clip(np.asarray(turn, dtype=np.float64), -1.0, 1.0)
    return (np.searchsorted(_THROTTLE_EDGES, t, side="left"),
            np.searchsorted(_TURN_EDGES, r, side="left"))


def snap_action(throttle, turn):
    """(throttle, turn) rounded to the student's bin values (floats)."""
    ti, ri = snap_indices(throttle, turn)
    return F.THROTTLE_VALUES[int(ti)], F.TURN_VALUES[int(ri)]


_THROTTLE = np.asarray(F.THROTTLE_VALUES, dtype=np.float64)
_TURN = np.asarray(F.TURN_VALUES, dtype=np.float64)


def snap_capped(yaw, vx, vy, throttle, turn, target):
    """Bins rounding that keeps the controller's speed cap (arrays broadcast).

    Nearest bins, except that a throttle bin whose push would take the velocity
    above the controller's limit (max(target, 0.5) / friction, with the rounded
    turn) is replaced by the bin on the other side of the controller's throttle
    when that one stays under the limit: rounding 0.6 up to 1.0 near the lethal
    speed would otherwise turn a safe goal into a lethal one.
    """
    yaw, vx, vy = (np.asarray(v, dtype=np.float64) for v in (yaw, vx, vy))
    throttle = np.clip(np.asarray(throttle, dtype=np.float64), -1.0, 1.0)
    ti, ri = snap_indices(throttle, turn)
    value = _THROTTLE[ti]
    toward_zero = np.where(throttle > 0, ti - 1, ti + 1)
    alt = np.where(value > throttle, ti - 1, np.where(value < throttle, ti + 1, toward_zero))
    alt = np.clip(alt, 0, len(_THROTTLE) - 1)
    sp = np.hypot(vx, vy)
    rate = CTL_TURN_BASE * (1.0 - CTL_TURN_PENALTY * np.minimum(sp / CTL_MAX_SPEED, 1.0)) * CTL_DT
    new_yaw = yaw + _TURN[ri] * rate
    fx, fy = np.sin(new_yaw), np.cos(new_yaw)
    lim = np.maximum(np.clip(np.asarray(target, dtype=np.float64), 0.0,
                             CTL_MAX_SPEED * CTL_FRICTION), 0.5) / CTL_FRICTION
    lim2 = lim * lim + 1e-9

    def over(u):
        nvx, nvy = vx + CTL_ACCEL * u * fx, vy + CTL_ACCEL * u * fy
        return nvx * nvx + nvy * nvy > lim2

    ti = np.where(over(value) & ~over(_THROTTLE[alt]), alt, ti)
    return ti, ri


def _bracket(values, x):
    """Bins (lo, hi) around x and the weight of hi that interpolates x linearly."""
    lo = np.clip(np.searchsorted(values, x, side="right") - 1, 0, len(values) - 2)
    hi = lo + 1
    w = np.clip((x - values[lo]) / (values[hi] - values[lo]), 0.0, 1.0)
    return lo, hi, w


def bins_weights(mode, yaw, vx, vy, throttle, turn, target):
    """Bins for controller actions as two candidates per head with weights (arrays broadcast).

    Returns (t_lo, t_hi, w_t, r_lo, r_hi, w_r): head index `hi` gets weight w, `lo` 1 - w.
      near    nearest bins (w = 0)
      cap     snap_capped (w = 0)
      dither  stochastic rounding: the bracketing bins with linear-interpolation weights,
              so the expected throttle and turn (hence, by the linear dynamics, the
              expected velocity and yaw change) equal the controller's
      dcap    dither, except that the throttle bin that would exceed the controller's
              speed limit (checked along the unrounded turn) gives its weight to the
              other one when that one stays under the limit
      dlethal dither, except that when the controller's push keeps the speed under
              LETHAL_GUARD, a bin that would reach it gives its weight to the other
              one (if that one stays under): rounding never makes a safe speed lethal
    """
    throttle = np.clip(np.asarray(throttle, dtype=np.float64), -1.0, 1.0)
    turn = np.clip(np.asarray(turn, dtype=np.float64), -1.0, 1.0)
    if mode in ("near", "cap"):
        if mode == "cap":
            ti, ri = snap_capped(yaw, vx, vy, throttle, turn, target)
        else:
            ti, ri = snap_indices(throttle, turn)
        zero = np.zeros(np.broadcast(ti, ri).shape)
        return ti, ti, zero, ri, ri, zero
    if mode not in ("dither", "dcap", "dlethal"):
        raise ValueError(f"unknown snap mode {mode!r}")
    t_lo, t_hi, w_t = _bracket(_THROTTLE, throttle)
    r_lo, r_hi, w_r = _bracket(_TURN, turn)
    if mode in ("dcap", "dlethal"):
        yaw, vx, vy = (np.asarray(v, dtype=np.float64) for v in (yaw, vx, vy))
        sp = np.hypot(vx, vy)
        rate = (CTL_TURN_BASE * (1.0 - CTL_TURN_PENALTY * np.minimum(sp / CTL_MAX_SPEED, 1.0))
                * CTL_DT)
        new_yaw = yaw + turn * rate
        fx, fy = np.sin(new_yaw), np.cos(new_yaw)

        def speed2(u):  # squared pre-friction speed after a push u along the new facing
            nvx, nvy = vx + CTL_ACCEL * u * fx, vy + CTL_ACCEL * u * fy
            return nvx * nvx + nvy * nvy

        if mode == "dcap":
            lim = np.maximum(np.clip(np.asarray(target, dtype=np.float64), 0.0,
                                     CTL_MAX_SPEED * CTL_FRICTION), 0.5) / CTL_FRICTION
            lim2 = lim * lim + 1e-9
            over_lo, over_hi = speed2(_THROTTLE[t_lo]) > lim2, speed2(_THROTTLE[t_hi]) > lim2
        else:  # dlethal: only rounding that crosses the lethal speed is avoided
            lethal2 = (LETHAL_GUARD / CTL_FRICTION) ** 2
            safe = speed2(throttle) < lethal2
            over_lo = safe & (speed2(_THROTTLE[t_lo]) >= lethal2)
            over_hi = safe & (speed2(_THROTTLE[t_hi]) >= lethal2)
        w_t = np.where(over_hi & ~over_lo, 0.0, np.where(over_lo & ~over_hi, 1.0, w_t))
    return t_lo, t_hi, w_t, r_lo, r_hi, w_r


def snap_engine(mode, me, throttle, turn, target, rng=None):
    """One engine action rounded to bins values by `mode` (SNAP_MODES; the dithering modes
    draw from `rng`); `me` is the documented self observation, `target` the goal speed the
    controller drove toward."""
    t_lo, t_hi, w_t, r_lo, r_hi, w_r = bins_weights(
        mode, _num(me.get("yaw")), _num(me.get("vx")), _num(me.get("vy")), throttle, turn, target)
    if mode in ("dither", "dcap", "dlethal"):
        ti = int(t_hi) if rng.random() < float(w_t) else int(t_lo)
        ri = int(r_hi) if rng.random() < float(w_r) else int(r_lo)
    else:
        ti, ri = int(t_lo), int(r_lo)
    return F.THROTTLE_VALUES[ti], F.TURN_VALUES[ri]


def _norm(angle):
    return np.mod(angle + _PI, _TAU) - _PI


def control_batch(yaw, vx, vy, heading, speed, reverse):
    """Vectorized ctf.controller.control: arrays broadcast against each other.

    yaw, vx, vy, reverse: per player (e.g. shape [N, 1]); heading (radians relative
    to yaw), speed: per goal (e.g. [1, G]). Returns (throttle, turn, new reverse).
    """
    yaw, vx, vy = (np.asarray(v, dtype=np.float64) for v in (yaw, vx, vy))
    reverse = np.asarray(reverse, dtype=bool)
    sp = np.hypot(vx, vy)
    target = np.clip(np.asarray(speed, dtype=np.float64), 0.0, CTL_MAX_SPEED * CTL_FRICTION)
    a = yaw + heading
    k = target / CTL_FRICTION
    dvx = k * np.sin(a) - vx
    dvy = k * np.cos(a) - vy
    dm = np.hypot(dvx, dvy)
    align = np.clip(CTL_ALIGN_FRAC * k, CTL_ALIGN_MIN, CTL_ALIGN_DV)
    reached = dm <= align
    da = np.where(reached, a, np.arctan2(dvx, dvy))
    ef = _norm(da - yaw)
    er = _norm(da - yaw - _PI)
    rev = np.where(reverse, ~(np.abs(ef) + CTL_HYSTERESIS < np.abs(er)),
                   np.abs(er) + CTL_HYSTERESIS < np.abs(ef))
    err = np.where(rev, er, ef)
    err = np.where(reached & rev, _norm(a - yaw - _PI), err)
    rate = CTL_TURN_BASE * (1.0 - CTL_TURN_PENALTY * np.minimum(sp / CTL_MAX_SPEED, 1.0)) * CTL_DT
    turn = np.clip(err / rate, -1.0, 1.0)
    new_yaw = yaw + turn * rate
    fx, fy = np.sin(new_yaw), np.cos(new_yaw)
    u = np.clip((dvx * fx + dvy * fy) / CTL_ACCEL, -1.0, 1.0)
    along = vx * fx + vy * fy
    off = np.abs(_norm(da - new_yaw - np.where(rev, _PI, 0.0)))
    u = np.where((off > CTL_MISALIGN) & (u * along >= 0.0), 0.0, u)
    lim = np.maximum(target, 0.5) / CTL_FRICTION
    nvx, nvy = vx + CTL_ACCEL * u * fx, vy + CTL_ACCEL * u * fy
    over = nvx * nvx + nvy * nvy > lim * lim
    disc = along * along - (vx * vx + vy * vy - lim * lim)
    root = np.sqrt(np.maximum(disc, 0.0))
    low, high = (-along - root) / CTL_ACCEL, (-along + root) / CTL_ACCEL
    inside = np.clip(np.where(u < low, low, np.where(u > high, high, u)), -1.0, 1.0)
    outside = np.clip(-along / CTL_ACCEL, -1.0, 1.0)
    u = np.where(over, np.where(disc >= 0.0, inside, outside), u)
    return u, turn, rev


_HEADINGS = np.radians(np.asarray(F.HEADING_VALUES, dtype=np.float64))
_SPEEDS = np.asarray(F.SPEED_VALUES, dtype=np.float64)
GOAL_H = np.repeat(_HEADINGS, F.N_SPEED)[None, :]   # goal g = heading * N_SPEED + speed
GOAL_S = np.tile(_SPEEDS, F.N_HEADING)[None, :]
N_SOFT = F.N_THROTTLE + F.N_TURN + 2                 # soft target layout: student logits
SNAP_MODES = ("near", "cap", "dither", "dcap", "dlethal")  # see bins_weights
LETHAL_GUARD = 10.95                                 # post-friction speed (lethal at 11)
COMMIT_MODES = ("posterior", "sample")               # see PolicyTeacher.executed


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _num(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


class PolicyTeacher:
    """Shadow goal-space policy that labels student states with soft bins targets."""

    def __init__(self, actor, snap="dither", commit="posterior"):
        if getattr(actor, "space", "bins") != "goal" or actor.recurrent:
            raise ValueError("the policy teacher must be a feed-forward goal-space policy")
        if snap not in SNAP_MODES:
            raise ValueError(f"unknown snap mode {snap!r} (choose from {SNAP_MODES})")
        if commit not in COMMIT_MODES:
            raise ValueError(f"unknown commit mode {commit!r} (choose from {COMMIT_MODES})")
        self.actor = actor
        self.snap = snap
        self.commit = commit

    @staticmethod
    def memory():
        memory = F.PlayerMemory(goal=True)
        memory.distill_pending = None
        return memory

    def label(self, memories, docs, rng):
        """Soft targets [N, N_SOFT] (throttle, turn, jump distributions) and hard labels [N, 3]
        (per-head argmax); every memory commits a sampled teacher goal."""
        rows = [F.encode(doc, memory) for doc, memory in zip(docs, memories)]
        logits = self.actor.logits(np.asarray(rows, dtype=np.float32)).astype(np.float64)
        nh, ns = F.N_HEADING, F.N_SPEED
        ph = _softmax(logits[:, :nh])
        ps = _softmax(logits[:, nh:nh + ns])
        pj = _softmax(logits[:, nh + ns:])
        n = len(docs)
        yaw = np.empty((n, 1))
        vx = np.empty((n, 1))
        vy = np.empty((n, 1))
        reverse = np.empty((n, 1), dtype=bool)
        ticks = []
        for i, (doc, memory) in enumerate(zip(docs, memories)):
            me = doc.get("self") or {}
            yaw[i, 0], vx[i, 0], vy[i, 0] = _num(me.get("yaw")), _num(me.get("vx")), _num(me.get("vy"))
            tick = int(_num(doc.get("tick")))
            last_tick, before, after = memory.ctrl
            reverse[i, 0] = before if last_tick == tick else after
            ticks.append(tick)
        u, turn, rev = control_batch(yaw, vx, vy, GOAL_H, GOAL_S, reverse)
        t_lo, t_hi, w_t, r_lo, r_hi, w_r = bins_weights(self.snap, yaw, vx, vy, u, turn, GOAL_S)
        joint = (ph[:, :, None] * ps[:, None, :]).reshape(n, nh * ns)
        base = np.arange(n)[:, None]
        nt, nr = F.N_THROTTLE, F.N_TURN

        def marginal(lo, hi, w, size):
            index = np.concatenate([(lo + size * base).ravel(), (hi + size * base).ravel()])
            mass = np.concatenate([(joint * (1.0 - w)).ravel(), (joint * w).ravel()])
            return np.bincount(index, weights=mass, minlength=n * size)

        qt = marginal(t_lo, t_hi, w_t, nt)
        qr = marginal(r_lo, r_hi, w_r, nr)
        soft = np.concatenate([qt.reshape(n, nt), qr.reshape(n, nr), pj], axis=1)
        hard = np.stack([soft[:, :nt].argmax(1), soft[:, nt:nt + nr].argmax(1),
                         soft[:, nt + nr:].argmax(1)], axis=1)
        if self.commit == "posterior":
            for i, memory in enumerate(memories):
                memory.distill_pending = (ticks[i], bool(reverse[i, 0]), joint[i], t_lo[i],
                                          t_hi[i], w_t[i], r_lo[i], r_hi[i], w_r[i], rev[i])
        else:
            goals, _ = sample_logits(logits.astype(np.float32), rng, False, self.actor.head_sizes)
            for i, memory in enumerate(memories):
                h, s, j = (int(v) for v in goals[i])
                g = h * ns + s
                memory.ctrl = (ticks[i], bool(reverse[i, 0]), bool(rev[i, g]))
                memory.set_goal(h, s, j, float(u[i, g]))
        return soft.astype(np.float32), hard

    def executed(self, memory, action, throttle, rng):
        """Record what the student actually did: `action` its (throttle, turn, jump) bins,
        `throttle` the engine value (the teacher's bump detector uses it).

        With commit='posterior' the shadow teacher commits a goal drawn from its
        distribution conditioned on having produced the student's bins (p(g) times the
        probability that g rounds to them), so its previous-goal features and
        controller state follow the student's actual history; if no goal explains the
        action, from the unconditioned distribution. The jump is the executed one.
        """
        pending = getattr(memory, "distill_pending", None)
        if pending is not None:
            memory.distill_pending = None
            tick, reverse, joint, t_lo, t_hi, w_t, r_lo, r_hi, w_r, rev = pending
            ti, ri, jump = (int(v) for v in action)
            pt = np.where(t_lo == ti, 1.0 - w_t, 0.0) + np.where(t_hi == ti, w_t, 0.0)
            pr = np.where(r_lo == ri, 1.0 - w_r, 0.0) + np.where(r_hi == ri, w_r, 0.0)
            post = joint * pt * pr
            total = post.sum()
            if not total > 1e-12:
                post, total = joint, joint.sum()
            g = int(np.searchsorted(np.cumsum(post), rng.random() * total, side="right"))
            g = min(g, len(post) - 1)
            h, s = divmod(g, F.N_SPEED)
            memory.ctrl = (tick, reverse, bool(rev[g]))
            memory.set_goal(h, s, jump, throttle)
        memory.prev_push = throttle
