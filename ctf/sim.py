"""Headless match simulator built from the unmodified engine functions.

`Match` reproduces `engine.game.Game` exactly (construction order, step order,
death log, respawns and sonar), without worker processes, telemetry or
real-time pacing. Each tick is split into two phases so that a trainer can batch
decisions:

    observations = match.observe()      # {sphere_id: observation}, alive only
    match.act({sphere_id: action})      # applies actions, advances one tick

Collision and line-of-sight checks receive only nearby obstacles/ramps
(bounding-circle broad phase); the engine functions still decide every
outcome, and `ctf.tests.test_sim` checks bit-for-bit parity with `Game`.
"""

import math

from engine import physics
from engine.constants import (
    DEATH_HISTORY_TICKS,
    RESPAWN_DELAY_TICKS,
    SONAR_RANGE,
    SPHERE_RADIUS,
    TICK_DT,
)
from engine.flags import Flag, update_flags
from engine.game import _normalize_action, _team_home
from engine.geometry import distance_point_to_segment, segment_intersects_polygon
from engine.map_gen import generate_map, generate_ramps
from engine.sphere import Sphere
from engine.vision import VISION_HALF_ANGLE, VISION_RANGE, _normalize_angle

DOCUMENTED_KEYS = (
    "tick", "self", "visible_enemies", "own_flag", "other_flags",
    "score", "sonar_receive", "death_messages",
)
_MARGIN = SPHERE_RADIUS + 3.0


def _ramp_radius(ramp):
    return math.hypot(ramp.length / 2.0, ramp.width / 2.0)


class Match:
    def __init__(self, seed, team_names=("red", "blue"), spheres_per_team=3,
                 max_ticks=6000, broadphase=True, teammates_field=False):
        self.seed = seed
        self.max_ticks = max_ticks
        self.tick = 0
        self.team_names = tuple(team_names)
        self.score = {name: 0 for name in self.team_names}
        self.broadphase = broadphase
        self.teammates_field = teammates_field

        homes = [_team_home(i, len(self.team_names)) for i in range(len(self.team_names))]
        self.homes = dict(zip(self.team_names, homes))
        self.spheres = []
        sphere_id = 0
        for name, (hx, hy) in zip(self.team_names, homes):
            for i in range(spheres_per_team):
                offset = (i - spheres_per_team / 2.0) * 3.0
                self.spheres.append(Sphere(sphere_id, name, hx + offset, hy, player=i + 1))
                sphere_id += 1
        self.flags = [Flag(name, hx, hy) for name, (hx, hy) in zip(self.team_names, homes)]
        ramp_seed = None if seed is None else (seed + 0x9E37) & 0xFFFFFFFF
        self.obstacles = generate_map(homes, seed=seed)
        self.ramps = generate_ramps(self.obstacles, homes, seed=ramp_seed)
        self._ramp_radii = [_ramp_radius(r) for r in self.ramps]
        self._respawn_at = {}
        self._last_sonar = {}
        self._death_log = []
        self._observed_ids = []

    # ----------------------------------------------------------------- queries
    def is_over(self):
        return self.tick >= self.max_ticks

    def recent_deaths(self):
        cutoff = self.tick - DEATH_HISTORY_TICKS
        return [m for m in self._death_log if m["tick"] >= cutoff]

    def death_log(self):
        return self._death_log

    # --------------------------------------------------------------- broadphase
    def _near_obstacles(self, sphere):
        if not self.broadphase:
            return self.obstacles
        x, y = sphere.x, sphere.y
        return [
            o for o in self.obstacles
            if math.hypot(x - o.centroid[0], y - o.centroid[1]) <= o.bounding_radius + _MARGIN
        ]

    def _near_ramps(self, sphere):
        if not self.broadphase:
            return self.ramps
        x, y = sphere.x, sphere.y
        return [
            r for r, radius in zip(self.ramps, self._ramp_radii)
            if math.hypot(x - r.cx, y - r.cy) <= radius + _MARGIN
        ]

    def _blocked(self, a, b):
        ax, ay, bx, by = a.x, a.y, b.x, b.y
        low = min(a.z, b.z)
        for obstacle in self.obstacles:
            if obstacle.height <= low:
                continue
            if self.broadphase:
                cx, cy = obstacle.centroid
                dist, _, _ = distance_point_to_segment(cx, cy, ax, ay, bx, by)
                if dist > obstacle.bounding_radius + 1e-6:
                    continue
            if segment_intersects_polygon((ax, ay), (bx, by), obstacle.points):
                return True
        return False

    def _visible_enemies(self, sphere, alive):
        visible = []
        for other in alive:
            if other.team == sphere.team or not other.alive:
                continue
            dx, dy = other.x - sphere.x, other.y - sphere.y
            dist = math.hypot(dx, dy)
            if dist == 0 or dist > VISION_RANGE:
                continue
            relative = _normalize_angle(math.atan2(dx, dy) - sphere.yaw)
            if abs(relative) > VISION_HALF_ANGLE:
                continue
            if self._blocked(sphere, other):
                continue
            visible.append(other.to_state_dict())
        return visible

    def _audible_sonars(self, receiver, by_id, sonar_snapshot):
        heard = []
        for sender_id, value in sonar_snapshot.items():
            if value is None or sender_id == receiver.id:
                continue
            sender = by_id.get(sender_id)
            if sender is None or not sender.alive or sender.team != receiver.team:
                continue
            if math.hypot(sender.x - receiver.x, sender.y - receiver.y) > SONAR_RANGE:
                continue
            if self._blocked(sender, receiver):
                continue
            heard.append({"from_id": sender_id, "from_player": sender.player, "value": value})
        return heard

    # ------------------------------------------------------------------- phases
    def observe(self):
        """Observations for every alive sphere, exactly as Game.step builds them."""
        alive = [s for s in self.spheres if s.alive]
        flag_states = {f.team: f.to_state_dict() for f in self.flags}
        recent = self.recent_deaths()
        sonar_snapshot = dict(self._last_sonar)
        by_id = {s.id: s for s in self.spheres}
        observations = {}
        for sphere in alive:
            observation = {
                "tick": self.tick,
                "self": sphere.to_state_dict(),
                "visible_enemies": self._visible_enemies(sphere, alive),
                "own_flag": dict(flag_states[sphere.team]),
                "other_flags": [dict(state) for team, state in flag_states.items()
                                if team != sphere.team],
                "score": dict(self.score),
                "sonar_receive": self._audible_sonars(sphere, by_id, sonar_snapshot),
                "death_messages": [dict(m) for m in recent if m["team"] == sphere.team],
            }
            if self.teammates_field:
                # Undocumented backend extras; only for the provided scripted opponents.
                observation["teammates"] = [
                    s.to_state_dict() for s in alive if s.team == sphere.team and s.id != sphere.id
                ]
                observation["sonar_recieve"] = observation["sonar_receive"]
            observations[sphere.id] = observation
        self._observed_ids = [s.id for s in alive]
        return observations

    def act(self, actions):
        """Apply actions for the spheres returned by the last observe() and advance."""
        by_id = {s.id: s for s in self.spheres}
        for sid in self._observed_ids:
            throttle, turn, jump, sonar_val = _normalize_action(actions.get(sid))
            sphere = by_id[sid]
            sphere.apply_action(throttle, turn, jump, sonar_val, TICK_DT)
            if sonar_val is not None:
                self._last_sonar[sphere.id] = sonar_val
            elif sphere.id in self._last_sonar:
                del self._last_sonar[sphere.id]

        spheres = self.spheres
        for sphere in spheres:
            sphere.integrate(TICK_DT)
        physics.clamp_to_arena(spheres)
        if self.broadphase:
            for sphere in spheres:
                ramps = self._near_ramps(sphere)
                if ramps:
                    physics.resolve_ramp_collisions([sphere], ramps)
            for sphere in spheres:
                obstacles = self._near_obstacles(sphere)
                if obstacles:
                    physics.resolve_obstacle_collisions([sphere], obstacles)
            for sphere in spheres:
                physics.resolve_ground([sphere], self._near_ramps(sphere), self._near_obstacles(sphere))
        else:
            physics.resolve_ramp_collisions(spheres, self.ramps)
            physics.resolve_obstacle_collisions(spheres, self.obstacles)
            physics.resolve_ground(spheres, self.ramps, self.obstacles)
        physics.resolve_sphere_collisions(spheres)
        update_flags(self.flags, spheres, self.score)
        self._collect_deaths()
        self._process_respawns()
        alive_ids = {s.id for s in spheres if s.alive}
        for sid in list(self._last_sonar):
            if sid not in alive_ids:
                del self._last_sonar[sid]
        self.tick += 1

    def _collect_deaths(self):
        for sphere in self.spheres:
            if sphere.alive or sphere.death_reported:
                continue
            sphere.death_reported = True
            self._death_log.append({
                "player": sphere.player,
                "team": sphere.team,
                "tick": self.tick,
                "x": sphere.death_x,
                "y": sphere.death_y,
                "reason": sphere.death_reason or "unknown",
            })
        del self._death_log[:-200]

    def _process_respawns(self):
        for sphere in self.spheres:
            if sphere.alive:
                continue
            if sphere.id not in self._respawn_at:
                self._respawn_at[sphere.id] = self.tick + RESPAWN_DELAY_TICKS
            elif self._respawn_at[sphere.id] <= self.tick:
                sphere.respawn()
                del self._respawn_at[sphere.id]
