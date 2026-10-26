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

Every column has a unique name (`FEATURE_NAMES`). Checkpoints and policy files
store the names and the action tables, so a warm start can match inputs by
name whatever the layout change (see ctf/feature_layouts.py).
"""

import math

FEATURE_VERSION = 5
THROTTLE_VALUES = (-1.0, -0.3, -0.1, -0.05, -0.034, 0.0, 0.015, 0.025, 0.034, 0.037, 0.05,
                   0.06, 0.1, 0.3, 1.0)
TURN_VALUES = (-1.0, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0)
ACTION_TABLES = {"throttle": THROTTLE_VALUES, "turn": TURN_VALUES}
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
HOME_CLEAR = 12.0
RESPAWN_TICKS = 60
CELL = 2.2   # cells are centred on multiples of CELL: boundaries avoid 0, +-90 (homes) and
GRID = 93    # +-99 (arena clamp), and a 180-degree rotation maps cells exactly onto cells
GRID_HALF = GRID // 2
RAY_RANGE = 24.0
RAY_ANGLES = (0.0, 0.25 * math.pi, -0.25 * math.pi, 0.5 * math.pi, -0.5 * math.pi,
              0.75 * math.pi, -0.75 * math.pi, math.pi)
UNKNOWN, FREE, BLOCKED = 0, 1, 2
TRAIL_SHORT = 20     # decisions
TRAIL_LONG = 60
STUCK_THROTTLE = 0.1  # |throttle| that counts as pressing
STUCK_TICKS = 40
BUMP_AGE_TICKS = 600
COLLIDE_HORIZON = 3.0  # seconds

SELF_DIM = 22
GAME_DIM = 5
FLAG_DIM = 22
ENEMY_DIM = 14
BUMP_DIM = 3
DANGER_DIM = 5
FREE_DIM = len(RAY_ANGLES) + 3
TEAM_DIM = 8
MEMORY_DIM = 18
ACTION_DIM = N_THROTTLE + N_TURN + 1
FEATURE_DIM = (SELF_DIM + GAME_DIM + FLAG_DIM + ENEMY_SLOTS * ENEMY_DIM
               + BUMP_SLOTS * BUMP_DIM + DANGER_SLOTS * DANGER_DIM + FREE_DIM + TEAM_DIM
               + MEMORY_DIM + ACTION_DIM)


def _feature_names():
    names = ["self.x", "self.y", "self.z", "self.vfwd", "self.vright", "self.vz", "self.speed",
             "self.lethal", "self.yaw_sin", "self.yaw_cos", "self.grounded", "self.carrying",
             "self.edge_px", "self.edge_nx", "self.edge_py", "self.edge_ny",
             "self.player1", "self.player2", "self.player3", "self.gap", "self.dspeed",
             "self.side_known"]
    names += ["game.tick", "game.lead", "game.mine", "game.theirs", "game.dead_mates"]
    names += ["own_flag.fwd", "own_flag.right", "own_flag.dist", "own_flag.at_home",
              "own_flag.carried", "own_flag.vfwd", "own_flag.vright"]
    names += ["enemy_flag.seen", "enemy_flag.fwd", "enemy_flag.right", "enemy_flag.dist",
              "enemy_flag.at_home", "enemy_flag.mine", "enemy_flag.mate"]
    for home in ("own_home", "enemy_home"):
        names += [home + ".known", home + ".fwd", home + ".right", home + ".dist"]
    for k in range(ENEMY_SLOTS):
        names += ["enemy%d.%s" % (k, f) for f in (
            "present", "age", "fwd", "right", "dist", "rvfwd", "rvright", "speed", "speed_adv",
            "carrying", "dz", "yaw_sin", "yaw_cos", "grounded")]
    for k in range(BUMP_SLOTS):
        names += ["bump%d.%s" % (k, f) for f in ("present", "fwd", "right")]
    for k in range(DANGER_SLOTS):
        names += ["danger%d.%s" % (k, f) for f in ("present", "fwd", "right", "age", "not_wall")]
    names += ["ray.fwd", "ray.r45", "ray.l45", "ray.r90", "ray.l90", "ray.r135", "ray.l135",
              "ray.back", "ray.motion", "ray.motion_blocked", "ray.known_free"]
    names += ["team.mate_a_carrying", "team.mate_b_carrying", "team.mate_a_respawn",
              "team.mate_b_respawn", "team.home_enemy_dist", "team.home_enemy_count",
              "team.flag_closing", "team.enemy_flag_waited"]
    names += ["trail20.fwd", "trail20.right", "trail60.fwd", "trail60.right", "trail60.path",
              "trail.fill", "stuck.duration",
              "nbump.normal_fwd", "nbump.normal_right", "nbump.age", "nbump.dist",
              "collide.present", "collide.time", "collide.dist", "collide.speed_adv",
              "carrier.escape_dist", "carrier.escape_speed", "carrier.home_known"]
    names += ["prev.throttle=%r" % v for v in THROTTLE_VALUES]
    names += ["prev.turn=%r" % v for v in TURN_VALUES]
    names.append("prev.jump")
    return tuple(names)


FEATURE_NAMES = _feature_names()
if len(FEATURE_NAMES) != FEATURE_DIM or len(set(FEATURE_NAMES)) != FEATURE_DIM:
    raise ValueError("FEATURE_NAMES must name every feature column exactly once")


def _clip(value, low, high):
    return low if value < low else high if value > high else value


def _num(value, default=0.0):
    if type(value) is float:
        return value if value - value == 0.0 else default  # finite check without a call
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


_INV_CELL = 1.0 / CELL
_INF = float("inf")


def _rays(grid, x, y, directions):
    """PlayerMemory.ray from (x, y) along each unit direction: [(distance, blocked), ...]."""
    gx, gy = x / CELL + 0.5, y / CELL + 0.5  # cell (i, j) spans [i, i+1) here
    ix, iy = int(math.floor(gx)), int(math.floor(gy))
    n = GRID
    cx0, cy0 = ix + GRID_HALF, iy + GRID_HALF
    if cx0 < 0 or cx0 >= n or cy0 < 0 or cy0 >= n:
        return [(0.0, 1.0)] * len(directions)
    start = cy0 * n + cx0
    state = grid[start]
    if state != FREE:
        return [(0.0, 1.0 if state == BLOCKED else 0.0)] * len(directions)
    # Offsets from the start point to the next cell boundary on each side, in world units.
    x_pos, x_neg = (ix + 1 - gx) * CELL, (gx - ix) * CELL
    y_pos, y_neg = (iy + 1 - gy) * CELL, (gy - iy) * CELL
    hits = []
    for dx, dy in directions:
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1
        adx, ady = abs(dx), abs(dy)
        if adx > 1e-12:
            t_dx = _INV_CELL / adx * CELL * CELL
            t_x = (x_pos if dx > 0 else x_neg) / adx
        else:
            t_dx = t_x = _INF
        if ady > 1e-12:
            t_dy = _INV_CELL / ady * CELL * CELL
            t_y = (y_pos if dy > 0 else y_neg) / ady
        else:
            t_dy = t_y = _INF
        row_step = step_y * n
        cx, cy, index = cx0, cy0, start
        # Step to the next cell; check the range, then the bounds of the coordinate that moved.
        while True:
            if t_x < t_y:
                t = t_x
                if t >= RAY_RANGE:
                    hit = (RAY_RANGE, 0.0)
                    break
                t_x += t_dx
                cx += step_x
                if cx < 0 or cx >= n:
                    hit = (t, 1.0)
                    break
                index += step_x
            else:
                t = t_y
                if t >= RAY_RANGE:
                    hit = (RAY_RANGE, 0.0)
                    break
                t_y += t_dy
                cy += step_y
                if cy < 0 or cy >= n:
                    hit = (t, 1.0)
                    break
                index += row_step
            state = grid[index]
            if state != FREE:
                hit = (t, 1.0 if state == BLOCKED else 0.0)
                break
        hits.append(hit)
    return hits


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
        self.flag_streak = 0
        self.enemy_home_since = None
        self.side_known = False
        self.cached = None
        self.flag_velocity = (0.0, 0.0)
        self.bumps = []  # (x, y, tick, normal x, normal y); normal (0, 0) when unknown
        self.trail = []  # (x, y, odometer) of the last TRAIL_LONG + 1 decisions of this life
        self.stuck = 0
        self.grid = bytearray(GRID * GRID)
        self.homes_marked = set()
        self._homes_seen = set()
        self.deaths = {}
        self.enemies = {}
        self.prev_throttle = IDLE_THROTTLE
        self.prev_turn = IDLE_TURN
        self.prev_jump = False

    def set_action(self, throttle_index, turn_index, jump):
        self.prev_throttle = throttle_index
        self.prev_turn = turn_index
        self.prev_jump = bool(jump)

    # ------------------------------------------------------------ clear map
    def _cell(self, x, y):
        ix = int(math.floor(x / CELL + 0.5)) + GRID_HALF
        iy = int(math.floor(y / CELL + 0.5)) + GRID_HALF
        if 0 <= ix < GRID and 0 <= iy < GRID:
            return iy * GRID + ix
        return -1

    def mark_free(self, x, y):
        index = self._cell(x, y)
        if index >= 0 and self.grid[index] == UNKNOWN:
            self.grid[index] = FREE

    def mark_blocked(self, x, y):
        index = self._cell(x, y)
        if index >= 0:
            self.grid[index] = BLOCKED

    def _mark_home(self, home):
        if home in self._homes_seen:  # exact coordinates already handled (skips round())
            return
        self._homes_seen.add(home)
        key = (round(home[0], 3), round(home[1], 3))
        if key in self.homes_marked:
            return
        self.homes_marked.add(key)
        steps = int(HOME_CLEAR / CELL)
        for i in range(-steps, steps + 1):
            for j in range(-steps, steps + 1):
                if (i * CELL) ** 2 + (j * CELL) ** 2 <= (HOME_CLEAR - CELL) ** 2:
                    self.mark_free(home[0] + i * CELL, home[1] + j * CELL)

    def ray(self, x, y, dx, dy):
        """Known-clear distance along unit direction (dx, dy) and whether it ended at a block.

        Exact grid traversal (Amanatides & Woo): every cell the ray crosses is checked,
        so a diagonal ray cannot slip past a blocked cell through its corner.
        """
        return _rays(self.grid, x, y, ((dx, dy),))[0]

    # ----------------------------------------------------------------- updates
    def _add_bump(self, x, y, inferred=False, tick=0, nx=0.0, ny=0.0):
        """Record a contact point; (nx, ny) is the unit direction from it to the player."""
        # Obstacles are never generated within 18 units of a home; ramps can come closer.
        # Inferred stops right next to a home are almost always teammate bounces.
        if inferred:
            for home in (self.own_home, self.enemy_home):
                if home is not None and (home[0] - x) ** 2 + (home[1] - y) ** 2 < HOME_CLEAR ** 2:
                    return
        self.mark_blocked(x, y)
        for i, (bx, by, btick, bnx, bny) in enumerate(self.bumps):
            if (bx - x) ** 2 + (by - y) ** 2 < 2.25:
                # Same point again: refresh its age, and its normal when this one is known.
                if nx or ny:
                    bnx, bny = nx, ny
                self.bumps[i] = (bx, by, tick if tick > btick else btick, bnx, bny)
                return
        self.bumps.append((x, y, tick, nx, ny))
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
        enemy_at_home = False
        for flag in observation.get("other_flags") or ():
            if flag.get("at_home"):
                self.enemy_home = (_num(flag.get("x")), _num(flag.get("y")))
                enemy_at_home = True
            break
        if not enemy_at_home:
            self.enemy_home_since = None
        elif self.enemy_home_since is None:
            self.enemy_home_since = tick
        if self.own_home is not None:
            self.side = -1.0 if self.own_home[0] > 0 else 1.0
            self.side_known = True
        elif self.enemy_home is not None:
            self.side = 1.0 if self.enemy_home[0] > 0 else -1.0
            self.side_known = True
        elif self.side is None:
            self.side = -1.0 if x > 0 else 1.0
        for home in (self.own_home, self.enemy_home):
            if home is not None and home not in self._homes_seen:
                self._mark_home(home)

        gap = tick - self.prev_tick if self.prev_tick is not None else 0
        prev = self.prev_self
        speed = _num(me.get("speed"))
        prev_speed = speed
        if prev is not None and gap == 1:
            prev_speed = _num(prev.get("speed"))
            pushed = abs(THROTTLE_VALUES[self.prev_throttle]) * 6.0 > 0.05
            if ((prev_speed > 0.5 or pushed) and speed < 1e-9
                    and abs(x) < ARENA_LIMIT - 1.5 and abs(y) < ARENA_LIMIT - 1.5):
                if prev_speed > 0.5:
                    dx, dy = _num(prev.get("vx")), _num(prev.get("vy"))
                else:
                    sign = 1.0 if THROTTLE_VALUES[self.prev_throttle] > 0 else -1.0
                    yaw = _num(me.get("yaw"))
                    dx, dy = sign * math.sin(yaw), sign * math.cos(yaw)
                norm = math.hypot(dx, dy) or 1.0
                self._add_bump(x + dx / norm, y + dy / norm, inferred=True, tick=tick,
                               nx=-dx / norm, ny=-dy / norm)

        # Trail of this life (reset across a respawn or a long gap) and stuck duration.
        trail = self.trail
        if prev is not None and 1 <= gap <= 3 and trail:
            tx, ty, odometer = trail[-1]
            trail.append((x, y, odometer + math.hypot(x - tx, y - ty)))
            if len(trail) > TRAIL_LONG + 1:
                del trail[0]
        else:
            self.trail = [(x, y, 0.0)]
        if (gap == 1 and speed < 1.0
                and abs(THROTTLE_VALUES[self.prev_throttle]) >= STUCK_THROTTLE):
            self.stuck += 1
        else:
            self.stuck = 0

        carrier = own_flag.get("carried_by", -1)
        fx, fy = _num(own_flag.get("x")), _num(own_flag.get("y"))
        same = (self.prev_own_flag is not None and self.prev_own_flag[2] == carrier
                and 1 <= gap <= 3)
        self.flag_streak = self.flag_streak + 1 if same else 1
        # Public flag coordinates lag a new carrier by one tick, so the first
        # displacement after a carrier change is a catch-up jump, not motion.
        if carrier != -1 and self.flag_streak >= 3:
            px, py = self.prev_own_flag[0], self.prev_own_flag[1]
            self.flag_velocity = ((fx - px) / (0.05 * gap), (fy - py) / (0.05 * gap))
        else:
            self.flag_velocity = (0.0, 0.0)
        self.prev_own_flag = (fx, fy, carrier)

        # Ground known to be clear: where this player, seen enemies and flag carriers have been.
        # Only grounded, ground-level positions count (not airborne, not on ramps).
        if me.get("grounded") and _num(me.get("z"), 1.0) < 1.05:
            self.mark_free(x, y)
        if carrier != -1 and self.flag_streak >= 2:
            self.mark_free(fx, fy)
        for flag in observation.get("other_flags") or ():
            if flag.get("carried_by", -1) not in (-1, me.get("id")):
                self.mark_free(_num(flag.get("x")), _num(flag.get("y")))
        for enemy in observation.get("visible_enemies") or ():
            if enemy.get("grounded") and _num(enemy.get("z"), 1.0) < 1.05:
                self.mark_free(_num(enemy.get("x")), _num(enemy.get("y")))

        for message in observation.get("death_messages") or ():
            key = (message.get("player"), message.get("tick"))
            if key in self.deaths:
                continue
            dx, dy = _num(message.get("x")), _num(message.get("y"))
            wall = message.get("reason") == "wall"
            dtick = int(_num(message.get("tick")))
            self.deaths[key] = (dx, dy, dtick, wall)
            if wall:
                self._add_bump(dx, dy, tick=dtick)
        if len(self.deaths) > 64:
            for key in sorted(self.deaths, key=lambda k: self.deaths[k][2])[:-64]:
                del self.deaths[key]

        enemies = self.enemies
        for enemy in observation.get("visible_enemies") or ():
            eid = enemy.get("id")
            espeed = _num(enemy.get("speed"))
            old = enemies.get(eid)
            # Speed change per tick, when the same enemy was also seen on the previous tick.
            accel = espeed - old[6] if old is not None and old[0] == tick - 1 else 0.0
            enemies[eid] = (
                tick, _num(enemy.get("x")), _num(enemy.get("y")), _num(enemy.get("z")),
                _num(enemy.get("vx")), _num(enemy.get("vy")), espeed,
                1.0 if enemy.get("carrying") else 0.0, _num(enemy.get("yaw")),
                1.0 if enemy.get("grounded") else 0.0, accel,
            )
        for key in [k for k, v in self.enemies.items() if tick - v[0] > ENEMY_MEMORY_TICKS]:
            del self.enemies[key]

        self.prev_self = me
        self.prev_tick = tick
        self.last_tick = tick
        return gap, prev_speed


def encode(observation, memory):
    """Update `memory` with `observation` and return the feature list.

    Encoding the same tick twice returns the cached features unchanged.
    """
    tick = int(_num(observation.get("tick"), 0))
    if memory.cached is not None and memory.last_tick == tick:
        return list(memory.cached)
    gap, prev_speed = memory.update(observation)
    me = observation.get("self") or {}
    x, y, z = _num(me.get("x")), _num(me.get("y")), _num(me.get("z"), 1.0)
    vx, vy = _num(me.get("vx")), _num(me.get("vy"))
    speed = _num(me.get("speed"))
    yaw = _num(me.get("yaw"))
    side = memory.side
    # Body frame of a world vector (dx, dy): (dx * s + dy * c, dx * c - dy * s).
    s, c = math.sin(yaw), math.cos(yaw)

    cx, cy = side * x, side * y
    lethal = (speed - LETHAL_SPEED) / 5.0
    canonical_yaw = yaw if side > 0 else yaw + math.pi
    e0 = (ARENA_LIMIT - cx) / 20.0
    e1 = (ARENA_LIMIT + cx) / 20.0
    e2 = (ARENA_LIMIT - cy) / 20.0
    e3 = (ARENA_LIMIT + cy) / 20.0
    player = int(_num(me.get("player"), 1))
    g = gap / 60.0
    dv = (speed - prev_speed) / 18.0
    out = [
        cx / 100.0, cy / 100.0, (z - 1.0) / 2.0,
        (vx * s + vy * c) / 18.0, (vx * c - vy * s) / 18.0, _num(me.get("vz")) / 10.0,
        speed / 18.0, -2.0 if lethal < -2.0 else 1.5 if lethal > 1.5 else lethal,
        math.sin(canonical_yaw), math.cos(canonical_yaw),
        1.0 if me.get("grounded") else 0.0, 1.0 if me.get("carrying") else 0.0,
        0.0 if e0 < 0.0 else 1.0 if e0 > 1.0 else e0,
        0.0 if e1 < 0.0 else 1.0 if e1 > 1.0 else e1,
        0.0 if e2 < 0.0 else 1.0 if e2 > 1.0 else e2,
        0.0 if e3 < 0.0 else 1.0 if e3 > 1.0 else e3,
        1.0 if player == 1 else 0.0, 1.0 if player == 2 else 0.0, 1.0 if player == 3 else 0.0,
        0.0 if g < 0.0 else 2.0 if g > 2.0 else g,
        -1.0 if dv < -1.0 else 1.0 if dv > 1.0 else dv,
        1.0 if memory.side_known else 0.0,
    ]

    score = observation.get("score") or {}
    team = me.get("team")
    mine = _num(score.get(team))
    theirs = sum([_num(v) for k, v in score.items() if k != team])
    dead_mates = 0
    for d_key, d in memory.deaths.items():
        if d_key[0] != player and 0 <= tick - d[2] <= RESPAWN_TICKS:
            dead_mates += 1
    lead = (mine - theirs) / 5.0
    mine_n, theirs_n = mine / 10.0, theirs / 10.0
    out += (tick / 6000.0, -2.0 if lead < -2.0 else 2.0 if lead > 2.0 else lead,
            0.0 if mine_n < 0.0 else 3.0 if mine_n > 3.0 else mine_n,
            0.0 if theirs_n < 0.0 else 3.0 if theirs_n > 3.0 else theirs_n, dead_mates / 2.0)

    my_id = me.get("id")
    own_flag = observation.get("own_flag") or {}
    fdx, fdy = _num(own_flag.get("x")) - x, _num(own_flag.get("y")) - y
    fvx, fvy = memory.flag_velocity
    dist = math.hypot(fdx, fdy) / 100.0
    out += ((fdx * s + fdy * c) / 100.0, (fdx * c - fdy * s) / 100.0,
            0.0 if dist < 0.0 else 2.0 if dist > 2.0 else dist,
            1.0 if own_flag.get("at_home") else 0.0,
            1.0 if own_flag.get("carried_by", -1) != -1 else 0.0,
            (fvx * s + fvy * c) / 18.0, (fvx * c - fvy * s) / 18.0)
    enemy_flag = None
    for flag in observation.get("other_flags") or ():
        enemy_flag = flag
        break
    if enemy_flag is not None:
        edx, edy = _num(enemy_flag.get("x")) - x, _num(enemy_flag.get("y")) - y
        carrier = enemy_flag.get("carried_by", -1)
        dist = math.hypot(edx, edy) / 100.0
        out += (1.0, (edx * s + edy * c) / 100.0, (edx * c - edy * s) / 100.0,
                0.0 if dist < 0.0 else 2.0 if dist > 2.0 else dist,
                1.0 if enemy_flag.get("at_home") else 0.0,
                1.0 if carrier == my_id else 0.0,
                1.0 if carrier not in (-1, my_id) else 0.0)
    else:
        out += _ZEROS[:7]
    for home in (memory.own_home, memory.enemy_home):
        if home is None:
            out += _ZEROS[:4]
        else:
            hdx, hdy = home[0] - x, home[1] - y
            dist = math.hypot(hdx, hdy) / 100.0
            out += (1.0, (hdx * s + hdy * c) / 100.0, (hdx * c - hdy * s) / 100.0,
                    0.0 if dist < 0.0 else 2.0 if dist > 2.0 else dist)

    enemies = []
    for record in memory.enemies.values():
        etick, ex, ey, ez, evx, evy, espeed, ecarry, eyaw, egrounded, eaccel = record
        dx, dy = ex - x, ey - y
        enemies.append((tick - etick, dx * dx + dy * dy, dx, dy, ez, evx, evy, espeed, ecarry,
                        eyaw, egrounded, eaccel))
    if len(enemies) > 1:
        enemies.sort(key=_enemy_order)
    for age, dist2, dx, dy, ez, evx, evy, espeed, ecarry, eyaw, egrounded, _ in enemies[:ENEMY_SLOTS]:
        rx, ry = evx - vx, evy - vy
        dist = math.sqrt(dist2) / 40.0
        dz = ez - z
        out += (1.0, age / ENEMY_MEMORY_TICKS, (dx * s + dy * c) / 40.0, (dx * c - dy * s) / 40.0,
                0.0 if dist < 0.0 else 2.0 if dist > 2.0 else dist,
                (rx * s + ry * c) / 18.0, (rx * c - ry * s) / 18.0,
                espeed / 18.0, (speed - espeed) / 18.0, ecarry,
                -3.0 if dz < -3.0 else 3.0 if dz > 3.0 else dz,
                math.sin(eyaw - yaw), math.cos(eyaw - yaw), egrounded)
    if len(enemies) < ENEMY_SLOTS:
        out += _ZEROS[:ENEMY_DIM * (ENEMY_SLOTS - len(enemies))]

    near = []
    limit2 = BUMP_RANGE * BUMP_RANGE
    for bx, by, btick, bnx, bny in memory.bumps:
        dx, dy = bx - x, by - y
        d2 = dx * dx + dy * dy
        if d2 <= limit2:
            near.append((d2, dx, dy, btick, bnx, bny))
    if near:
        near.sort()
        for _, dx, dy, _, _, _ in near[:BUMP_SLOTS]:
            out += (1.0, (dx * s + dy * c) / BUMP_RANGE, (dx * c - dy * s) / BUMP_RANGE)
    if len(near) < BUMP_SLOTS:
        out += _ZEROS[:BUMP_DIM * (BUMP_SLOTS - len(near))]

    deaths = ()
    if memory.deaths:
        deaths = [d for d in memory.deaths.values() if tick - d[2] <= DANGER_TICKS]
        deaths.sort(key=_newest_first)
        for ddx, ddy, dtick, wall in deaths[:DANGER_SLOTS]:
            dx, dy = ddx - x, ddy - y
            df, dr = (dx * s + dy * c) / 60.0, (dx * c - dy * s) / 60.0
            out += (1.0, -2.0 if df < -2.0 else 2.0 if df > 2.0 else df,
                    -2.0 if dr < -2.0 else 2.0 if dr > 2.0 else dr,
                    (tick - dtick) / DANGER_TICKS, 0.0 if wall else 1.0)
    if len(deaths) < DANGER_SLOTS:
        out += _ZEROS[:DANGER_DIM * (DANGER_SLOTS - len(deaths))]

    directions = [(math.sin(yaw + angle), math.cos(yaw + angle)) for angle in RAY_ANGLES]
    if speed > 0.5:
        directions.append((vx / speed, vy / speed))
    grid = memory.grid
    hits = _rays(grid, x, y, directions)
    for i in range(len(RAY_ANGLES)):
        out.append(hits[i][0] / RAY_RANGE)
    # Without speed the last ray is the facing ray: angle 0, (sin(yaw), cos(yaw)).
    distance, blocked = hits[-1] if speed > 0.5 else hits[0]
    # Known-clear cells of the 5x5 block around the player (_cell is separable in x, y).
    cols = []
    rows = []
    for i in (-2, -1, 0, 1, 2):
        col = int(math.floor((x + i * CELL) / CELL + 0.5)) + GRID_HALF
        if 0 <= col < GRID:
            cols.append(col)
        row = int(math.floor((y + i * CELL) / CELL + 0.5)) + GRID_HALF
        if 0 <= row < GRID:
            rows.append(row * GRID)
    known = 0
    if len(cols) == 5 and cols[4] - cols[0] == 4 and cols[1] - cols[0] == 1 \
            and cols[2] - cols[1] == 1 and cols[3] - cols[2] == 1:
        first, stop = cols[0], cols[4] + 1
        for row in rows:
            known += grid.count(FREE, row + first, row + stop)
    else:
        for row in rows:
            for col in cols:
                if grid[row + col] == FREE:
                    known += 1
    out += (distance / RAY_RANGE, blocked, known / 25.0)

    # Team block: which teammate carries, teammates' respawn countdowns, enemies near our
    # home, closing speed on our stolen flag, and how long the enemy flag has sat at home.
    # Global ids are allocated per team in player order, so id - (player - 1) is the
    # team's first id (documented `id`/`player` fields of our own state).
    first_id = (my_id if isinstance(my_id, int) else 0) - (player - 1)
    mates = [p for p in (1, 2, 3) if p != player]
    carrier_player = -1
    if enemy_flag is not None:
        carrier = enemy_flag.get("carried_by", -1)
        if isinstance(carrier, int) and carrier not in (-1, my_id) and 0 <= carrier - first_id <= 2:
            carrier_player = carrier - first_id + 1
    team = [1.0 if carrier_player == p else 0.0 for p in mates]
    for p in mates:
        remaining = 0.0
        for d_key, d in memory.deaths.items():
            if d_key[0] == p and 0 <= tick - d[2] <= RESPAWN_TICKS:
                remaining = max(remaining, (RESPAWN_TICKS - (tick - d[2])) / RESPAWN_TICKS)
        team.append(remaining)
    nearest, count = 2.0, 0.0
    if memory.own_home is not None:
        hx, hy = memory.own_home
        for record in memory.enemies.values():
            d = math.hypot(record[1] - hx, record[2] - hy)
            nearest = min(nearest, d / 60.0)
            if d < 20.0:
                count += 1.0
    team += (nearest, count / 3.0)
    closing = 0.0
    if own_flag.get("carried_by", -1) != -1:
        norm = math.hypot(fdx, fdy)
        if norm > 1e-6:
            closing = (fdx * (fvx - vx) + fdy * (fvy - vy)) / norm / 18.0
            closing = -2.0 if closing < -2.0 else 2.0 if closing > 2.0 else closing
    waited = 0.0
    if memory.enemy_home_since is not None:
        waited = (tick - memory.enemy_home_since) / 100.0
        waited = 2.0 if waited > 2.0 else waited
    team += (closing, waited)
    out += team

    # Memory block. Trail of this life: displacement over the last TRAIL_SHORT and
    # TRAIL_LONG decisions (body frame) and path length, so progress, orbiting and being
    # stuck look different; then how long throttle has been pressed without moving.
    trail = memory.trail
    n = len(trail)
    tx, ty, _ = trail[n - 1 - TRAIL_SHORT] if n > TRAIL_SHORT else trail[0]
    lx, ly, odometer = trail[n - 1 - TRAIL_LONG] if n > TRAIL_LONG else trail[0]
    dx, dy, ex, ey = x - tx, y - ty, x - lx, y - ly
    path = (trail[-1][2] - odometer) / (0.9 * TRAIL_LONG)
    stuck = memory.stuck / STUCK_TICKS
    out += ((dx * s + dy * c) / (0.9 * TRAIL_SHORT), (dx * c - dy * s) / (0.9 * TRAIL_SHORT),
            (ex * s + ey * c) / (0.9 * TRAIL_LONG), (ex * c - ey * s) / (0.9 * TRAIL_LONG),
            2.0 if path > 2.0 else path, (n - 1) / TRAIL_LONG, 1.0 if stuck > 1.0 else stuck)
    # Nearest remembered contact: surface normal (towards where we stood), age, distance.
    if near:
        d2, _, _, btick, bnx, bny = near[0]
        age = (tick - btick) / BUMP_AGE_TICKS
        out += (bnx * s + bny * c, bnx * c - bny * s,
                0.0 if age < 0.0 else 1.0 if age > 1.0 else age, math.sqrt(d2) / BUMP_RANGE)
    else:
        out += (0.0, 0.0, 1.0, 1.0)
    # Closest approach to the nearest visible enemy at constant velocities, and the speed
    # advantage there (speeds extrapolated with their last per-tick change): the faster
    # sphere wins a collision.
    if enemies and enemies[0][0] == 0:
        _, _, dx, dy, _, evx, evy, espeed, _, _, _, eaccel = enemies[0]
        rx, ry = evx - vx, evy - vy
        rv2 = rx * rx + ry * ry
        t = -(dx * rx + dy * ry) / rv2 if rv2 > 1e-9 else 0.0
        t = 0.0 if t < 0.0 else COLLIDE_HORIZON if t > COLLIDE_HORIZON else t
        dx, dy = dx + rx * t, dy + ry * t
        dist = math.sqrt(dx * dx + dy * dy) / 20.0
        mine = speed + (speed - prev_speed) * 20.0 * t
        theirs = espeed + eaccel * 20.0 * t
        mine = 0.0 if mine < 0.0 else 18.0 if mine > 18.0 else mine
        theirs = 0.0 if theirs < 0.0 else 18.0 if theirs > 18.0 else theirs
        out += (1.0, t / COLLIDE_HORIZON, 2.0 if dist > 2.0 else dist, (mine - theirs) / 18.0)
    else:
        out += (0.0, 1.0, 2.0, 0.0)
    # Our flag's carrier: distance to its own home (the observed enemy home, else our home
    # rotated by 180 degrees) and its speed towards it.
    home = None
    if own_flag.get("carried_by", -1) != -1:
        home, known = memory.enemy_home, 1.0
        if home is None and memory.own_home is not None:
            home, known = (-memory.own_home[0], -memory.own_home[1]), 0.5
    if home is not None:
        hx, hy = home[0] - _num(own_flag.get("x")), home[1] - _num(own_flag.get("y"))
        norm = math.hypot(hx, hy)
        escape = (fvx * hx + fvy * hy) / norm / 18.0 if norm > 1e-6 else 0.0
        out += (2.0 if norm > 200.0 else norm / 100.0,
                -2.0 if escape < -2.0 else 2.0 if escape > 2.0 else escape, known)
    else:
        out += (0.0, 0.0, 0.0)

    action = [0.0] * ACTION_DIM
    action[memory.prev_throttle] = 1.0
    action[N_THROTTLE + memory.prev_turn] = 1.0
    action[-1] = 1.0 if memory.prev_jump else 0.0
    out += action
    memory.cached = out
    return list(out)


_ZEROS = [0.0] * 64


def _enemy_order(e):
    return (e[0] > 0, e[1])


def _newest_first(d):
    return -d[2]
