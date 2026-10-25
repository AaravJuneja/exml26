"""Observation encoder and per-player memory.

Pure Python (only `math`), so the exact same source is used in training and is
inlined into the exported single-file `model.py`. Everything here is fixed
preprocessing of documented observation fields; no function chooses an action.

Frames
------
* Canonical frame: the world rotated by 180 degrees when needed so that the own
  home lies on the negative x side. Rotation (not mirroring) keeps turn
  direction meaningful.
* Body frame: components along the sphere's facing (`fwd`) and to its right
  (`right`). Rotation invariant, so it is computed from world coordinates.
"""

import math

THROTTLE_VALUES = (-1.0, -0.3, -0.1, -0.03, 0.0, 0.015, 0.025, 0.034, 0.05, 0.1, 0.3, 1.0)
TURN_VALUES = (-1.0, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0)
N_THROTTLE = len(THROTTLE_VALUES)
N_TURN = len(TURN_VALUES)
IDLE_THROTTLE = THROTTLE_VALUES.index(0.0)
IDLE_TURN = TURN_VALUES.index(0.0)

ARENA_LIMIT = 99.0
LETHAL_SPEED = 11.0
ENEMY_SLOTS = 3
ENEMY_MEMORY_TICKS = 40
BUMP_SLOTS = 12
BUMP_RANGE = 30.0
BUMP_CAPACITY = 96
DANGER_SLOTS = 4
DANGER_TICKS = 600
HOME_CLEAR = 16.0

SELF_DIM = 21
GAME_DIM = 4
FLAG_DIM = 22
ENEMY_DIM = 11
BUMP_DIM = 3
DANGER_DIM = 5
ACTION_DIM = N_THROTTLE + N_TURN + 1
FEATURE_DIM = (SELF_DIM + GAME_DIM + FLAG_DIM + ENEMY_SLOTS * ENEMY_DIM
               + BUMP_SLOTS * BUMP_DIM + DANGER_SLOTS * DANGER_DIM + ACTION_DIM)


def _clip(value, low, high):
    return low if value < low else high if value > high else value


def _num(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class PlayerMemory:
    """Everything one player remembers between callbacks within a match."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.last_tick = None
        self.own_home = None
        self.enemy_home = None
        self.side = None
        self.prev_self = None
        self.prev_tick = None
        self.prev_own_flag = None
        self.flag_velocity = (0.0, 0.0)
        self.bumps = []
        self.deaths = {}
        self.enemies = {}
        self.prev_throttle = IDLE_THROTTLE
        self.prev_turn = IDLE_TURN
        self.prev_jump = False

    def set_action(self, throttle_index, turn_index, jump):
        self.prev_throttle = throttle_index
        self.prev_turn = turn_index
        self.prev_jump = bool(jump)

    # ----------------------------------------------------------------- updates
    def _add_bump(self, x, y):
        # The map generator keeps obstacles out of an 18-unit radius around homes,
        # so stops there are teammate bounces, not terrain.
        for home in (self.own_home, self.enemy_home):
            if home is not None and (home[0] - x) ** 2 + (home[1] - y) ** 2 < HOME_CLEAR ** 2:
                return
        for bx, by in self.bumps:
            if (bx - x) ** 2 + (by - y) ** 2 < 2.25:
                return
        self.bumps.append((x, y))
        if len(self.bumps) > BUMP_CAPACITY:
            del self.bumps[0]

    def update(self, observation):
        tick = int(_num(observation.get("tick"), 0))
        if self.last_tick is not None and tick < self.last_tick:
            self.reset()
        me = observation.get("self") or {}
        x, y = _num(me.get("x")), _num(me.get("y"))

        own_flag = observation.get("own_flag") or {}
        if own_flag.get("at_home"):
            self.own_home = (_num(own_flag.get("x")), _num(own_flag.get("y")))
        for flag in observation.get("other_flags") or ():
            if flag.get("at_home"):
                self.enemy_home = (_num(flag.get("x")), _num(flag.get("y")))
                break
        if self.own_home is not None:
            self.side = -1.0 if self.own_home[0] > 0 else 1.0
        elif self.enemy_home is not None:
            self.side = 1.0 if self.enemy_home[0] > 0 else -1.0
        elif self.side is None:
            self.side = -1.0 if x > 0 else 1.0

        gap = tick - self.prev_tick if self.prev_tick is not None else 0
        prev = self.prev_self
        speed = _num(me.get("speed"))
        prev_speed = speed
        if prev is not None and gap == 1:
            prev_speed = _num(prev.get("speed"))
            if (prev_speed > 0.5 and speed < 1e-9
                    and abs(x) < ARENA_LIMIT - 0.5 and abs(y) < ARENA_LIMIT - 0.5):
                pvx, pvy = _num(prev.get("vx")), _num(prev.get("vy"))
                norm = math.hypot(pvx, pvy) or 1.0
                self._add_bump(x + pvx / norm, y + pvy / norm)

        if own_flag.get("carried_by", -1) != -1 and self.prev_own_flag is not None and gap >= 1:
            px, py, pcarried = self.prev_own_flag
            if pcarried:
                fx, fy = _num(own_flag.get("x")), _num(own_flag.get("y"))
                self.flag_velocity = ((fx - px) / (0.05 * gap), (fy - py) / (0.05 * gap))
        else:
            self.flag_velocity = (0.0, 0.0)
        self.prev_own_flag = (_num(own_flag.get("x")), _num(own_flag.get("y")),
                              own_flag.get("carried_by", -1) != -1)

        for message in observation.get("death_messages") or ():
            key = (message.get("player"), message.get("tick"))
            if key in self.deaths:
                continue
            dx, dy = _num(message.get("x")), _num(message.get("y"))
            wall = message.get("reason") == "wall"
            self.deaths[key] = (dx, dy, int(_num(message.get("tick"))), wall)
            if wall:
                self._add_bump(dx, dy)
        if len(self.deaths) > 64:
            for key in sorted(self.deaths, key=lambda k: self.deaths[k][2])[:-64]:
                del self.deaths[key]

        for enemy in observation.get("visible_enemies") or ():
            self.enemies[enemy.get("id")] = (
                tick, _num(enemy.get("x")), _num(enemy.get("y")), _num(enemy.get("z")),
                _num(enemy.get("vx")), _num(enemy.get("vy")), _num(enemy.get("speed")),
                1.0 if enemy.get("carrying") else 0.0,
            )
        for key in [k for k, v in self.enemies.items() if tick - v[0] > ENEMY_MEMORY_TICKS]:
            del self.enemies[key]

        self.prev_self = me
        self.prev_tick = tick
        self.last_tick = tick
        return gap, prev_speed


def encode(observation, memory):
    """Update `memory` with `observation` and return the feature list."""
    gap, prev_speed = memory.update(observation)
    me = observation.get("self") or {}
    tick = int(_num(observation.get("tick"), 0))
    x, y, z = _num(me.get("x")), _num(me.get("y")), _num(me.get("z"), 1.0)
    vx, vy = _num(me.get("vx")), _num(me.get("vy"))
    speed = _num(me.get("speed"))
    yaw = _num(me.get("yaw"))
    side = memory.side
    sin_yaw, cos_yaw = math.sin(yaw), math.cos(yaw)

    def body(dx, dy):
        return dx * sin_yaw + dy * cos_yaw, dx * cos_yaw - dy * sin_yaw

    cx, cy = side * x, side * y
    out = [
        cx / 100.0, cy / 100.0, (z - 1.0) / 2.0,
    ]
    fwd_v, right_v = body(vx, vy)
    out += [fwd_v / 18.0, right_v / 18.0, _num(me.get("vz")) / 10.0,
            speed / 18.0, _clip((speed - LETHAL_SPEED) / 5.0, -2.0, 1.5)]
    canonical_yaw = yaw if side > 0 else yaw + math.pi
    out += [math.sin(canonical_yaw), math.cos(canonical_yaw),
            1.0 if me.get("grounded") else 0.0, 1.0 if me.get("carrying") else 0.0]
    out += [_clip((ARENA_LIMIT - cx) / 20.0, 0.0, 1.0), _clip((ARENA_LIMIT + cx) / 20.0, 0.0, 1.0),
            _clip((ARENA_LIMIT - cy) / 20.0, 0.0, 1.0), _clip((ARENA_LIMIT + cy) / 20.0, 0.0, 1.0)]
    player = int(_num(me.get("player"), 1))
    out += [1.0 if player == 1 else 0.0, 1.0 if player == 2 else 0.0, 1.0 if player == 3 else 0.0]
    out += [_clip(gap / 60.0, 0.0, 2.0), _clip((speed - prev_speed) / 18.0, -1.0, 1.0)]

    score = observation.get("score") or {}
    team = me.get("team")
    mine = _num(score.get(team))
    theirs = sum(_num(v) for k, v in score.items() if k != team)
    out += [tick / 6000.0, _clip((mine - theirs) / 5.0, -2.0, 2.0),
            _clip(mine / 10.0, 0.0, 3.0), _clip(theirs / 10.0, 0.0, 3.0)]

    my_id = me.get("id")
    own_flag = observation.get("own_flag") or {}
    fdx, fdy = _num(own_flag.get("x")) - x, _num(own_flag.get("y")) - y
    ff, fr = body(fdx, fdy)
    fvf, fvr = body(*memory.flag_velocity)
    out += [ff / 100.0, fr / 100.0, _clip(math.hypot(fdx, fdy) / 100.0, 0.0, 2.0),
            1.0 if own_flag.get("at_home") else 0.0,
            1.0 if own_flag.get("carried_by", -1) != -1 else 0.0,
            fvf / 18.0, fvr / 18.0]
    enemy_flag = None
    for flag in observation.get("other_flags") or ():
        enemy_flag = flag
        break
    if enemy_flag is not None:
        edx, edy = _num(enemy_flag.get("x")) - x, _num(enemy_flag.get("y")) - y
        ef, er = body(edx, edy)
        carrier = enemy_flag.get("carried_by", -1)
        out += [1.0, ef / 100.0, er / 100.0, _clip(math.hypot(edx, edy) / 100.0, 0.0, 2.0),
                1.0 if enemy_flag.get("at_home") else 0.0,
                1.0 if carrier == my_id else 0.0,
                1.0 if carrier not in (-1, my_id) else 0.0]
    else:
        out += [0.0] * 7
    for home in (memory.own_home, memory.enemy_home):
        if home is None:
            out += [0.0, 0.0, 0.0, 0.0]
        else:
            hdx, hdy = home[0] - x, home[1] - y
            hf, hr = body(hdx, hdy)
            out += [1.0, hf / 100.0, hr / 100.0, _clip(math.hypot(hdx, hdy) / 100.0, 0.0, 2.0)]

    enemies = []
    for record in memory.enemies.values():
        etick, ex, ey, ez, evx, evy, espeed, ecarry = record
        dx, dy = ex - x, ey - y
        enemies.append((tick - etick, dx * dx + dy * dy, dx, dy, ez, evx, evy, espeed, ecarry))
    enemies.sort(key=lambda e: (e[0] > 0, e[1]))
    for i in range(ENEMY_SLOTS):
        if i < len(enemies):
            age, dist2, dx, dy, ez, evx, evy, espeed, ecarry = enemies[i]
            ef, er = body(dx, dy)
            rvf, rvr = body(evx - vx, evy - vy)
            out += [1.0, age / ENEMY_MEMORY_TICKS, ef / 40.0, er / 40.0,
                    _clip(math.sqrt(dist2) / 40.0, 0.0, 2.0), rvf / 18.0, rvr / 18.0,
                    espeed / 18.0, (speed - espeed) / 18.0, ecarry, ez - z]
        else:
            out += [0.0] * ENEMY_DIM

    near = []
    limit2 = BUMP_RANGE * BUMP_RANGE
    for bx, by in memory.bumps:
        dx, dy = bx - x, by - y
        d2 = dx * dx + dy * dy
        if d2 <= limit2:
            near.append((d2, dx, dy))
    near.sort()
    for i in range(BUMP_SLOTS):
        if i < len(near):
            _, dx, dy = near[i]
            bf, br = body(dx, dy)
            out += [1.0, bf / BUMP_RANGE, br / BUMP_RANGE]
        else:
            out += [0.0, 0.0, 0.0]

    deaths = sorted((d for d in memory.deaths.values() if tick - d[2] <= DANGER_TICKS),
                    key=lambda d: -d[2])
    for i in range(DANGER_SLOTS):
        if i < len(deaths):
            dx, dy = deaths[i][0] - x, deaths[i][1] - y
            df, dr = body(dx, dy)
            out += [1.0, _clip(df / 60.0, -2.0, 2.0), _clip(dr / 60.0, -2.0, 2.0),
                    (tick - deaths[i][2]) / DANGER_TICKS, 0.0 if deaths[i][3] else 1.0]
        else:
            out += [0.0] * DANGER_DIM

    action = [0.0] * ACTION_DIM
    action[memory.prev_throttle] = 1.0
    action[N_THROTTLE + memory.prev_turn] = 1.0
    action[-1] = 1.0 if memory.prev_jump else 0.0
    out += action
    return out
