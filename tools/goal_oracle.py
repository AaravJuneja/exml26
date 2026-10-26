"""Goal-space oracle: relay's goals projected onto the goal bins, executed by ctf/controller.py.

Measures what the `goal` action space loses against the original bot: each
oracle player runs a private relay instance on its documented observation,
projects relay's choice onto (heading bin, speed bin, jump) exactly as the
kickstarting labels do, and drives with the fixed controller. Variants:

    plain     relay itself (control)
    intent    relay's commanded goal (M.goal), quantized to the goal bins
    cont      relay's commanded goal, not quantized (controller loss only)
    physics   physics projection of relay's action (no access to M.goal)

    python -m tools.goal_oracle intent,physics relay --seeds 2000-2015 --procs 3
"""

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor

from ctf import action_space as A
from ctf import features as F
from ctf.controller import control
from ctf.policies import SCRIPTED, _silent, load_decide
from ctf.sim import DOCUMENTED_KEYS, Match

TEACHER = "relay"


class _Player:
    def __init__(self, variant, headings, speeds, sync):
        self.decide = load_decide(SCRIPTED[TEACHER])
        self.decide.__globals__["print"] = _silent
        self.variant, self.headings, self.speeds, self.sync = variant, headings, speeds, sync
        self.reverse = False
        self.labels = self.intents = 0

    def act(self, observation):
        out = self.decide(observation)
        if self.variant == "plain":
            return out
        me = observation["self"]
        M = self.decide.__globals__["M"]
        self.labels += 1
        if self.variant == "physics":
            relative, speed = A.project_action(me, out[0], out[1])
        else:
            if M.goal is not None:
                self.intents += 1
            relative, speed = A.teacher_goal(self.decide, out, observation)
        if self.variant != "cont":
            h, s = A.goal_indices(relative, speed, self.headings, self.speeds)
            relative, speed = math.radians(self.headings[h]), self.speeds[s]
        throttle, turn, self.reverse = control(me, relative, speed, self.reverse)
        if self.sync:  # relay's memory of its own throttle, as in kickstarting rollouts
            A.sync_teacher(self.decide, throttle)
        return (throttle, turn, out[2]) + tuple(out[3:4])


def play(job):
    seed, side, variant, opponent, headings, speeds, sync, ticks = job
    match = Match(seed, max_ticks=ticks, teammates_field=True)
    players = {}
    for s in match.spheres:
        if s.team == side:
            players[s.id] = _Player(variant, headings, speeds, sync)
        else:
            decide = load_decide(SCRIPTED.get(opponent, opponent))
            decide.__globals__["print"] = _silent
            players[s.id] = decide
    while not match.is_over():
        actions = {}
        for sid, observation in match.observe().items():
            observation = {k: observation[k] for k in DOCUMENTED_KEYS}
            player = players[sid]
            try:
                actions[sid] = player.act(observation) if isinstance(player, _Player) \
                    else player(observation)
            except Exception:
                actions[sid] = (0.0, 0.0, False, None)
        match.act(actions)
    other = next(t for t in match.score if t != side)
    mine = [p for p in players.values() if isinstance(p, _Player)]
    walls = sum(1 for m in match.death_log() if m["team"] == side and m["reason"] == "wall")
    return (variant, opponent, match.score[side], match.score[other], walls,
            sum(p.intents for p in mine), sum(p.labels for p in mine))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("variants")
    p.add_argument("opponents")
    p.add_argument("--seeds", default="2000-2015")
    p.add_argument("--procs", type=int, default=3)
    p.add_argument("--headings", default="",
                   help="N uniform heading bins, or comma-separated non-negative angles "
                        "(mirrored); default: the features' table")
    p.add_argument("--speeds", default="", help="comma-separated speed bins (default: features)")
    p.add_argument("--no-sync", action="store_true", help="leave relay's own throttle memory alone")
    p.add_argument("--ticks", type=int, default=6000)
    args = p.parse_args(argv)
    low, _, high = args.seeds.partition("-")
    seeds = range(int(low), int(high or low) + 1)
    if "," in args.headings:  # non-negative angles, mirrored
        half = sorted({float(v) for v in args.headings.split(",")})
        headings = tuple(sorted({-v for v in half if 0.0 < v < 180.0} | set(half)))
    elif int(args.headings or 0):
        n = int(args.headings)
        headings = tuple(360.0 / n * k - 180.0 + 360.0 / n for k in range(n))
    else:
        headings = F.HEADING_VALUES
    speeds = tuple(float(v) for v in args.speeds.split(",")) if args.speeds else F.SPEED_VALUES
    jobs = [(seed, side, v, o, headings, speeds, not args.no_sync, args.ticks)
            for v in args.variants.split(",") for o in args.opponents.split(",")
            for seed in seeds for side in ("red", "blue")]
    started = time.time()
    results = {}
    with ProcessPoolExecutor(args.procs) as ex:
        for v, o, gf, ga, walls, intents, labels in ex.map(play, jobs):
            r = results.setdefault((v, o), [0, 0, 0, 0, 0, 0, 0, 0])
            r[0 if gf > ga else 1 if gf == ga else 2] += 1
            r[3] += gf
            r[4] += ga
            r[5] += walls
            r[6] += intents
            r[7] += labels
    print(f"headings {headings} speeds {speeds} sync {not args.no_sync}")
    for (v, o), r in sorted(results.items()):
        n = r[0] + r[1] + r[2]
        intent = f" intent {r[6] / r[7]:.3f}" if r[7] else ""
        print(f"{v:8s} vs {o:9s} W-D-L {r[0]}-{r[1]}-{r[2]}  gf {r[3] / n:.2f} ga {r[4] / n:.2f} "
              f"gd {(r[3] - r[4]) / n:+.2f} wall/match {r[5] / n:.2f}{intent}", flush=True)
    print("secs", round(time.time() - started))


if __name__ == "__main__":
    main()
