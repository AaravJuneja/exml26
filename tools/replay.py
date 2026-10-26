"""Record one match as a compact JSON replay for docs/viewer.html.

    python tools/replay.py SPEC_A SPEC_B --seed 1000 --side red --ticks 6000 \
        --out replay.json [--every 1] [--actions npz|all|none] [--html replay.html]

The match is played exactly as `ctf.evaluate.run_match` plays it (same specs,
team controllers, RNG seeding and simulator), so the final score matches the
evaluation record for the same (A, B, seed, side, ticks).

Format (version 1)
------------------
    meta     specs, labels, seed, sides, ticks, every, final score, result
    map      arena size, homes, obstacles [{p: [[x, y], ...], h}],
             ramps [{c: corners (low-left, low-right, high-right, high-left), h, ...}]
    spheres  [{id, team, player, label, policy}] (index == sphere id)
    frames   [{t, s, f, sc[, a]}] sampled every `every` ticks plus the final tick;
             frame t is the state *before* the actions of tick t are applied
               s   per sphere [x, y, z, yaw, speed, alive, carrying]
               f   per flag (team order) [x, y, carrier_id or -1]
               sc  score in team order
               a   per sphere [throttle, turn, jump] chosen at tick t, or null
    events   deaths / pickups / captures; `tick` is the step during which the event
             happened (the engine death-log tick), so it is visible in frame tick + 1.

Coordinates are world units (arena -100..100); yaw 0 points to +y, forward is
(sin yaw, cos yaw). Floats are rounded to 2 decimals (actions to 3).
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.constants import (  # noqa: E402
    ARENA_SIZE,
    FLAG_CAPTURE_RADIUS,
    SONAR_RANGE,
    SPHERE_RADIUS,
    TICK_RATE,
    VISION_HALF_ANGLE_DEG,
    VISION_RANGE,
)
from engine.game import _normalize_action  # noqa: E402
from ctf.evaluate import (  # noqa: E402
    DEFAULT_TICKS,
    PLAYERS_PER_TEAM,
    SIDES,
    _make_team,
    parse_spec,
)
from ctf.sim import Match  # noqa: E402

REPLAY_VERSION = 1
SPHERE_FIELDS = ("x", "y", "z", "yaw", "speed", "alive", "carrying")
ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "docs" / "viewer.html"


def _r(value, digits=2):
    value = round(float(value), digits)
    return 0.0 if value == 0 else value  # no "-0.0"


def _yaw(yaw):
    return math.atan2(math.sin(yaw), math.cos(yaw))


def _map(match):
    return {
        "arena": ARENA_SIZE,
        "homes": {team: [_r(x), _r(y)] for team, (x, y) in match.homes.items()},
        "obstacles": [{"p": [[_r(x), _r(y)] for x, y in o.points], "h": _r(o.height)}
                      for o in match.obstacles],
        "ramps": [{"c": [[_r(x), _r(y)] for x, y in r.corners()], "h": _r(r.height),
                   "cx": _r(r.cx), "cy": _r(r.cy), "dx": _r(r.dx, 4), "dy": _r(r.dy, 4),
                   "length": _r(r.length), "width": _r(r.width)}
                  for r in match.ramps],
    }


def _frame(match, tick, actions=None):
    spheres = [[_r(s.x), _r(s.y), _r(s.z), _r(_yaw(s.yaw)), _r(s.speed),
                int(s.alive), int(s.carrying_flag is not None)] for s in match.spheres]
    flags = [[_r(f.x), _r(f.y), f.carrier.id if f.carrier is not None else -1]
             for f in match.flags]
    frame = {"t": tick, "s": spheres, "f": flags,
             "sc": [match.score[team] for team in match.team_names]}
    if actions is not None:
        frame["a"] = actions
    return frame


def _action_row(match, chosen, recorded_ids):
    row = []
    for sphere in match.spheres:
        if sphere.id not in recorded_ids or sphere.id not in chosen:
            row.append(None)
            continue
        throttle, turn, jump, _ = _normalize_action(chosen[sphere.id])
        row.append([_r(throttle, 3), _r(turn, 3), int(bool(jump))])
    return row


def record_match(spec_a, spec_b, seed, a_side="red", max_ticks=DEFAULT_TICKS, every=1,
                 actions="npz"):
    """Play one match like `run_match`; return (replay dict, finished Match)."""
    if a_side not in SIDES:
        raise ValueError(f"a_side must be 'red' or 'blue', not {a_side!r}")
    if every < 1:
        raise ValueError("every must be >= 1")
    if actions not in ("npz", "all", "none"):
        raise ValueError(f"actions must be npz, all or none, not {actions!r}")
    spec_a, spec_b = parse_spec(spec_a), parse_spec(spec_b)
    b_side = SIDES[1 - SIDES.index(a_side)]
    need_full = any(s.kind == "scripted" and not s.documented_only for s in (spec_a, spec_b))
    match = Match(seed, team_names=SIDES, spheres_per_team=PLAYERS_PER_TEAM,
                  max_ticks=max_ticks, teammates_field=need_full)
    team_of = {s.id: s.team for s in match.spheres}
    spec_of = {a_side: spec_a, b_side: spec_b}
    controllers = {}
    for side, spec, role in ((a_side, spec_a, 0), (b_side, spec_b, 1)):
        ids = [s.id for s in match.spheres if s.team == side]
        controllers[side] = _make_team(spec, ids, seed, role)

    recorded_ids = set()
    if actions != "none":
        recorded_ids = {s.id for s in match.spheres
                        if actions == "all" or spec_of[s.team].kind == "npz"}
    flags, score, death_log = match.flags, match.score, match.death_log()
    by_key = {(s.team, s.player): s for s in match.spheres}
    frames, events = [], []

    while not match.is_over():
        tick = match.tick
        observations = match.observe()
        split = {side: {} for side in SIDES}
        for sid, observation in observations.items():
            split[team_of[sid]][sid] = observation
        chosen = {}
        for side in SIDES:
            chosen.update(controllers[side].act(split[side]))
        if tick % every == 0:
            row = _action_row(match, chosen, recorded_ids) if recorded_ids else None
            frames.append(_frame(match, tick, row))
        carriers = [flag.carrier for flag in flags]
        before = dict(score)
        match.act(chosen)

        for flag, carrier in zip(flags, carriers):
            new = flag.carrier
            if new is not None and new is not carrier:
                events.append({"tick": tick, "type": "pickup", "id": new.id, "team": new.team,
                               "player": new.player, "flag": flag.team,
                               "x": _r(new.x), "y": _r(new.y)})
            if (carrier is not None and new is None and carrier.alive
                    and score[carrier.team] > before[carrier.team]):
                events.append({"tick": tick, "type": "capture", "id": carrier.id,
                               "team": carrier.team, "player": carrier.player,
                               "flag": flag.team, "x": _r(carrier.x), "y": _r(carrier.y),
                               "score": [score[t] for t in match.team_names]})
        new_deaths = []
        for message in reversed(death_log):
            if message["tick"] != tick:
                break
            new_deaths.append(message)
        for message in reversed(new_deaths):
            sphere = by_key[(message["team"], message["player"])]
            carried = next((f.team for f, c in zip(flags, carriers) if c is sphere), None)
            events.append({"tick": tick, "type": "death", "id": sphere.id,
                           "team": message["team"], "player": message["player"],
                           "reason": message["reason"], "x": _r(message["x"]),
                           "y": _r(message["y"]), "carrying": carried})

    if not frames or frames[-1]["t"] != match.tick:
        frames.append(_frame(match, match.tick, None))

    score_a, score_b = score[a_side], score[b_side]
    result = "win" if score_a > score_b else "loss" if score_a < score_b else "draw"
    replay = {
        "version": REPLAY_VERSION,
        "meta": {
            "spec_a": str(spec_a), "spec_b": str(spec_b),
            "label_a": spec_a.label, "label_b": spec_b.label,
            "seed": seed, "a_side": a_side, "b_side": b_side,
            "teams": list(match.team_names),
            "labels": {a_side: spec_a.label, b_side: spec_b.label},
            "max_ticks": max_ticks, "ticks": match.tick, "every": every,
            "tick_rate": TICK_RATE, "actions": actions,
            "score": dict(score), "score_a": score_a, "score_b": score_b, "result": result,
            "errors": {side: controllers[side].errors for side in SIDES},
        },
        "constants": {
            "sphere_radius": SPHERE_RADIUS, "flag_capture_radius": FLAG_CAPTURE_RADIUS,
            "vision_range": VISION_RANGE, "vision_half_angle_deg": VISION_HALF_ANGLE_DEG,
            "sonar_range": SONAR_RANGE,
        },
        "map": _map(match),
        "spheres": [{"id": s.id, "team": s.team, "player": s.player,
                     "label": spec_of[s.team].label, "kind": spec_of[s.team].kind,
                     "actions": s.id in recorded_ids} for s in match.spheres],
        "sphere_fields": list(SPHERE_FIELDS),
        "frames": frames,
        "events": events,
    }
    return replay, match


def dumps(replay):
    """Compact JSON: one frame per line keeps the file diffable and greppable."""
    head = {k: v for k, v in replay.items() if k not in ("frames", "events")}
    parts = [json.dumps(head, separators=(",", ":"))[:-1], ',"events":[']
    parts.append(",\n".join(json.dumps(e, separators=(",", ":")) for e in replay["events"]))
    parts.append('],"frames":[\n')
    parts.append(",\n".join(json.dumps(f, separators=(",", ":")) for f in replay["frames"]))
    parts.append("\n]}\n")
    return "".join(parts)


def embed_html(replay_text, out, viewer=VIEWER):
    """Copy of the viewer with the replay inlined (opens with a double click)."""
    html = Path(viewer).read_text(encoding="utf-8")
    marker = "<!--REPLAY_DATA-->"
    if marker not in html:
        raise ValueError(f"{viewer} has no {marker} marker")
    payload = replay_text.replace("</", "<\\/")
    script = f'<script id="replay-data" type="application/json">{payload}</script>'
    Path(out).write_text(html.replace(marker, script, 1), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec_a")
    parser.add_argument("spec_b")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--side", choices=SIDES, default="red", help="side played by A")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument("--every", type=int, default=1, help="keep every Nth tick")
    parser.add_argument("--actions", choices=("npz", "all", "none"), default="npz",
                        help="record chosen actions for .npz players, everyone, or nobody")
    parser.add_argument("--out", default="replay.json")
    parser.add_argument("--html", help="also write a self-contained viewer with this replay")
    args = parser.parse_args(argv)

    replay, _ = record_match(args.spec_a, args.spec_b, args.seed, args.side, args.ticks,
                             args.every, args.actions)
    text = dumps(replay)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    if args.html:
        embed_html(text, args.html)
    meta = replay["meta"]
    kinds = {}
    for event in replay["events"]:
        key = event["type"] + (f"/{event['reason']}" if event["type"] == "death" else "")
        kinds[key] = kinds.get(key, 0) + 1
    print(f"{meta['label_a']} ({meta['a_side']}) {meta['score_a']}-{meta['score_b']} "
          f"{meta['label_b']} ({meta['b_side']})  seed {meta['seed']}  {meta['result']}  "
          f"{len(replay['frames'])} frames  events {kinds}  -> {out} "
          f"({out.stat().st_size / 1e6:.2f} MB)")
    return replay


if __name__ == "__main__":
    main()
