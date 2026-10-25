"""Exact broad-phase filter for the unmodified engine collision functions.

Each sphere is resolved alone against only the obstacles/ramps whose bounding
circle lies within a generous margin, preserving the engine's iteration order.
The engine functions are called unchanged; only their input lists shrink.
"""

import math

import engine.game as game_module
from engine import physics
from engine.constants import SPHERE_RADIUS

_MARGIN = SPHERE_RADIUS + 3.0
_ORIGINAL = {
    "resolve_obstacle_collisions": physics.resolve_obstacle_collisions,
    "resolve_ramp_collisions": physics.resolve_ramp_collisions,
    "resolve_ground": physics.resolve_ground,
}


def _ramp_radius(ramp):
    radius = getattr(ramp, "_bench_radius", None)
    if radius is None:
        radius = math.hypot(ramp.length / 2.0, ramp.width / 2.0)
        ramp._bench_radius = radius
    return radius


def _near_obstacles(sphere, obstacles):
    return [
        o for o in obstacles or ()
        if math.hypot(sphere.x - o.centroid[0], sphere.y - o.centroid[1])
        <= o.bounding_radius + _MARGIN
    ]


def _near_ramps(sphere, ramps):
    return [
        r for r in ramps or ()
        if math.hypot(sphere.x - r.cx, sphere.y - r.cy) <= _ramp_radius(r) + _MARGIN
    ]


def _obstacles(spheres, obstacles):
    for sphere in spheres:
        near = _near_obstacles(sphere, obstacles)
        if near:
            _ORIGINAL["resolve_obstacle_collisions"]([sphere], near)


def _ramps(spheres, ramps):
    for sphere in spheres:
        near = _near_ramps(sphere, ramps)
        if near:
            _ORIGINAL["resolve_ramp_collisions"]([sphere], near)


def _ground(spheres, ramps, obstacles):
    for sphere in spheres:
        _ORIGINAL["resolve_ground"](
            [sphere], _near_ramps(sphere, ramps), _near_obstacles(sphere, obstacles)
        )


def install():
    game_module.resolve_obstacle_collisions = _obstacles
    game_module.resolve_ramp_collisions = _ramps
    game_module.resolve_ground = _ground


def uninstall():
    for name, function in _ORIGINAL.items():
        setattr(game_module, name, function)
