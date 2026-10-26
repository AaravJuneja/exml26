"""Fixed goal controller of the `goal` action space: (heading, speed) -> (throttle, turn).

Pure Python (only `math`), so the same source runs in training and is inlined
into the exported `model.py` after ctf/features.py. It is deterministic
physics preprocessing: the network chooses where and how fast to go; this
inverts the engine's sphere dynamics (engine/sphere.py) for one tick:

    yaw' = yaw + turn * 3.2 * (1 - 0.85 * min(1, speed / 18)) * dt
    v'   = 0.98 * cap18(v + 6 * throttle * (sin yaw', cos yaw'))

Velocity does not rotate with yaw, so the controller thrusts (forward or in
reverse) along the velocity error: it turns the facing toward the error
direction and pushes by the projection of the error on the new facing. This is
the velocity-vector control of opponents/_core.py `_drive` (no avoidance,
planning or speed governor): forward/reverse choice with hysteresis, no thrust
when misaligned (it would bend the velocity into an orbit), and thrust clamped
so that the speed never exceeds the target. One change: the velocity error
below which the facing simply follows the goal heading scales with the target
speed (relay uses a fixed 1.5 u/s, which leaves slow goals up to ~20 degrees
off for a second or more).

The only state is the controller's own previous output (whether it was
thrusting in reverse), kept in the player's memory by `goal_step`.
"""

import math

CTL_ACCEL = 6.0          # engine ACCEL_RATE (velocity per tick at throttle 1)
CTL_FRICTION = 0.98      # engine FRICTION, applied after the action
CTL_MAX_SPEED = 18.0     # engine MAX_SPEED, applied before friction
CTL_TURN_BASE = 3.2      # engine TURN_RATE_BASE (rad/s)
CTL_TURN_PENALTY = 0.85  # engine TURN_SPEED_PENALTY
CTL_DT = 0.05            # engine TICK_DT
CTL_HYSTERESIS = 0.4     # rad: switch forward/reverse only when the other is this much closer
CTL_ALIGN_DV = 1.5       # velocity error treated as reached (then face the goal heading): at
CTL_ALIGN_FRAC = 0.1     # most CTL_ALIGN_DV, at most this fraction of the target speed (so slow
CTL_ALIGN_MIN = 0.3      # goals still converge in direction), at least CTL_ALIGN_MIN
CTL_MISALIGN = 0.7       # rad: no thrust when the facing is this far off the error direction

_CTL_PI = math.pi
_CTL_TAU = 2.0 * math.pi


def _ctl_norm(angle):
    return (angle + _CTL_PI) % _CTL_TAU - _CTL_PI


def _ctl_clamp(value):
    return -1.0 if value < -1.0 else 1.0 if value > 1.0 else value


def _ctl_num(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def control(me, heading, speed, reverse=False):
    """One tick of control toward velocity `speed` along `heading` (radians, relative to yaw).

    `me` is the documented `self` observation (yaw, vx, vy are read). Returns
    (throttle, turn, reverse): `reverse` is the new forward/reverse state, to be
    passed back on the next tick.
    """
    yaw = _ctl_num(me.get("yaw"))
    vx, vy = _ctl_num(me.get("vx")), _ctl_num(me.get("vy"))
    sp = math.hypot(vx, vy)
    target = speed if speed > 0.0 else 0.0
    if target > CTL_MAX_SPEED * CTL_FRICTION:
        target = CTL_MAX_SPEED * CTL_FRICTION
    a = yaw + heading
    k = target / CTL_FRICTION  # pre-friction velocity that friction brings to `target`
    dvx = k * math.sin(a) - vx
    dvy = k * math.cos(a) - vy
    dm = math.hypot(dvx, dvy)
    align = CTL_ALIGN_FRAC * k
    align = CTL_ALIGN_MIN if align < CTL_ALIGN_MIN else CTL_ALIGN_DV if align > CTL_ALIGN_DV else align
    da = math.atan2(dvx, dvy) if dm > align else a
    ef = _ctl_norm(da - yaw)
    er = _ctl_norm(da - yaw - _CTL_PI)
    if reverse:
        if abs(ef) + CTL_HYSTERESIS < abs(er):
            reverse = False
    elif abs(er) + CTL_HYSTERESIS < abs(ef):
        reverse = True
    err = er if reverse else ef
    if dm <= align and reverse:
        err = _ctl_norm(a - yaw - _CTL_PI)
    ratio = sp / CTL_MAX_SPEED
    rate = CTL_TURN_BASE * (1.0 - CTL_TURN_PENALTY * (ratio if ratio < 1.0 else 1.0)) * CTL_DT
    turn = _ctl_clamp(err / rate)
    new_yaw = yaw + turn * rate
    fx, fy = math.sin(new_yaw), math.cos(new_yaw)
    u = _ctl_clamp((dvx * fx + dvy * fy) / CTL_ACCEL)
    along = vx * fx + vy * fy
    if (abs(_ctl_norm(da - new_yaw - (_CTL_PI if reverse else 0.0))) > CTL_MISALIGN
            and u * along >= 0.0):
        u = 0.0  # misaligned thrust would only bend the velocity into an orbit
    # Never let the resulting speed exceed the target (perpendicular thrust adds speed).
    lim = (target if target > 0.5 else 0.5) / CTL_FRICTION
    nvx, nvy = vx + CTL_ACCEL * u * fx, vy + CTL_ACCEL * u * fy
    if nvx * nvx + nvy * nvy > lim * lim:
        c0 = vx * vx + vy * vy - lim * lim
        disc = along * along - c0
        if disc >= 0.0:
            root = math.sqrt(disc)
            low, high = (-along - root) / CTL_ACCEL, (-along + root) / CTL_ACCEL
            u = _ctl_clamp(low if u < low else high if u > high else u)
        else:
            u = _ctl_clamp(-along / CTL_ACCEL)
    return u, turn, reverse


def goal_step(memory, observation, heading_index, speed_index, jump, headings, speeds):
    """Engine action (throttle, turn, jump) for goal indices; records them in `memory`.

    `headings` (degrees relative to yaw) and `speeds` are the goal tables of
    ctf/features.py. `memory.ctrl` = (tick, reverse state before, after) of the
    last call, so deciding the same tick twice gives the same action.
    """
    tick = int(_ctl_num(observation.get("tick")))
    last_tick, before, after = memory.ctrl
    reverse = before if last_tick == tick else after
    me = observation.get("self") or {}
    throttle, turn, new_reverse = control(me, math.radians(headings[heading_index]),
                                          speeds[speed_index], reverse)
    memory.set_goal(heading_index, speed_index, jump, throttle)
    memory.ctrl = (tick, reverse, new_reverse)
    return throttle, turn, bool(jump)
