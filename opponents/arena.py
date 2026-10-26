"""Headless matches and round-robins for the training opponents.

    python -m opponents.arena match balanced baseline --seed 2000
    python -m opponents.arena rr --seeds 2000-2015 --procs 10 --out docs/eval/opponents_round_robin.json

Provided bots (baseline/example/example2) receive the backend's undocumented
`teammates`/`sonar_recieve` extras, as on the real backend; every bot under
`opponents/` receives only the documented observation keys.
"""

import argparse
import contextlib
import io
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ctf.policies import ROOT, SCRIPTED, load_decide
from ctf.sim import DOCUMENTED_KEYS, Match

OPPONENTS = ("rusher", "guardian", "camper", "relay", "balanced", "chaos", "steady")
PROVIDED = ("baseline", "example", "example2")


def _idle(observation):
    return 0.0, 0.0, False


def _load(name):
    if name == "idle":
        return _idle
    return load_decide(bot_path(name))


def bot_path(name):
    if name.endswith(".py"):
        return Path(name)
    if name in SCRIPTED:
        return ROOT / SCRIPTED[name]
    return ROOT / "opponents" / name / "model.py"


def play(seed, red, blue, ticks=6000, timing=False, trace=None, trace_every=100):
    """One match. Returns per-team goals, wall/player deaths and (optionally) decide time."""
    with contextlib.redirect_stdout(io.StringIO()):
        return _play(seed, red, blue, ticks, timing, trace, trace_every)


def _play(seed, red, blue, ticks, timing, trace, trace_every):
    import sys
    if trace is True:
        trace = sys.stderr
    match = Match(seed, max_ticks=ticks, teammates_field=True)
    names = {"red": red, "blue": blue}
    strip = {team: name not in PROVIDED for team, name in names.items()}
    decide = {s.id: _load(names[s.team]) for s in match.spheres}
    for fn in decide.values():
        if "SEED" in getattr(fn, "__globals__", {}):
            fn.__globals__["SEED"] = seed  # reproducible chaos draws
    team_of = {s.id: s.team for s in match.spheres}
    stats = {t: {"goals": 0, "wall": 0, "player": 0, "errors": 0, "calls": 0, "secs": 0.0}
             for t in names}
    clock = time.perf_counter
    while not match.is_over():
        observations = match.observe()
        actions = {}
        for sid, obs in observations.items():
            team = team_of[sid]
            if strip[team]:
                obs = {k: obs[k] for k in DOCUMENTED_KEYS}
            st = stats[team]
            t0 = clock() if timing else 0.0
            try:
                actions[sid] = decide[sid](obs)
            except Exception:
                st["errors"] += 1
                actions[sid] = (0.0, 0.0, False, None)
            if timing:
                st["secs"] += clock() - t0
                st["calls"] += 1
        alive = {s.id for s in match.spheres if s.alive}
        carriers = {f.team: (f.carrier.id if f.carrier else -1) for f in match.flags}
        score = dict(match.score)
        match.act(actions)
        for s in match.spheres:
            if s.id in alive and not s.alive:
                reason = s.death_reason or "unknown"
                if reason in ("wall", "player"):
                    stats[s.team][reason] += 1
                if trace:
                    print(f"t={match.tick} DEATH {s.team}{s.player} {reason} "
                          f"at ({s.x:.1f},{s.y:.1f}) v={s.speed:.1f}", file=trace)
        if trace:
            for f in match.flags:
                cid = f.carrier.id if f.carrier else -1
                if cid != carriers[f.team] and cid != -1:
                    c = f.carrier
                    print(f"t={match.tick} PICKUP {c.team}{c.player} took {f.team}", file=trace)
            for t in names:
                if match.score[t] != score[t]:
                    print(f"t={match.tick} SCORE {t} {match.score}", file=trace)
            if match.tick % trace_every == 0:
                row = " ".join(f"{s.team[0]}{s.player}:({s.x:.0f},{s.y:.0f},{s.speed:.0f})"
                               if s.alive else f"{s.team[0]}{s.player}:dead" for s in match.spheres)
                print(f"t={match.tick} {row}", file=trace)
    for t in names:
        stats[t]["goals"] = match.score[t]
    return stats


def _job(args):
    seed, a, b, ticks = args
    st = play(seed, a, b, ticks)
    return seed, a, b, st["red"], st["blue"]


def round_robin(bots, seeds, procs=10, ticks=6000):
    jobs = []
    for a, b in itertools.combinations(bots, 2):
        for seed in seeds:
            jobs.append((seed, a, b, ticks))
            jobs.append((seed, b, a, ticks))
    table = {a: {b: {"W": 0, "D": 0, "L": 0, "gf": 0, "ga": 0, "wall": 0, "killed": 0}
                 for b in bots if b != a} for a in bots}
    seats = {"red_wins": 0, "blue_wins": 0, "draws": 0}
    with ProcessPoolExecutor(procs) as pool:
        for seed, a, b, ra, rb in pool.map(_job, jobs, chunksize=1):
            key = ("red_wins" if ra["goals"] > rb["goals"] else
                   "blue_wins" if rb["goals"] > ra["goals"] else "draws")
            seats[key] += 1
            for me, op, rm, ro in ((a, b, ra, rb), (b, a, rb, ra)):
                cell = table[me][op]
                cell["gf"] += rm["goals"]
                cell["ga"] += ro["goals"]
                cell["wall"] += rm["wall"]
                cell["killed"] += rm["player"]
                cell["W" if rm["goals"] > ro["goals"] else "L" if rm["goals"] < ro["goals"] else "D"] += 1
    return table, seats


def panel(candidates, opponents, seeds, procs=10, ticks=6000):
    """Each candidate against each panel opponent, both seats. Returns {cand: {opp: cell}}."""
    jobs = []
    for a in candidates:
        for b in opponents:
            for seed in seeds:
                jobs.append((seed, a, b, ticks))
                jobs.append((seed, b, a, ticks))
    table = {a: {b: {"W": 0, "D": 0, "L": 0, "gf": 0, "ga": 0, "wall": 0, "killed": 0}
                 for b in opponents} for a in candidates}
    with ProcessPoolExecutor(procs) as pool:
        for seed, a, b, ra, rb in pool.map(_job, jobs, chunksize=1):
            if a in table and b in table[a]:
                me, op, rm, ro = a, b, ra, rb
            else:
                me, op, rm, ro = b, a, rb, ra
            cell = table[me][op]
            cell["gf"] += rm["goals"]
            cell["ga"] += ro["goals"]
            cell["wall"] += rm["wall"]
            cell["killed"] += rm["player"]
            cell["W" if rm["goals"] > ro["goals"] else "L" if rm["goals"] < ro["goals"] else "D"] += 1
    return table


def summary(table):
    rows = {}
    for a, row in table.items():
        tot = {k: sum(c[k] for c in row.values()) for k in ("W", "D", "L", "gf", "ga", "wall", "killed")}
        rows[a] = tot
    return rows


def _seeds(text):
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("match")
    m.add_argument("red")
    m.add_argument("blue")
    m.add_argument("--seed", type=int, default=2000)
    m.add_argument("--ticks", type=int, default=6000)
    m.add_argument("--trace", type=int, default=0)
    v = sub.add_parser("vs")
    v.add_argument("red")
    v.add_argument("blue")
    v.add_argument("--seeds", default="2000-2007")
    v.add_argument("--procs", type=int, default=6)
    v.add_argument("--ticks", type=int, default=6000)
    q = sub.add_parser("panel")
    q.add_argument("cands")
    q.add_argument("opps")
    q.add_argument("--seeds", default="2100-2107")
    q.add_argument("--procs", type=int, default=10)
    q.add_argument("--ticks", type=int, default=6000)
    r = sub.add_parser("rr")
    r.add_argument("--bots", default=",".join(OPPONENTS + PROVIDED))
    r.add_argument("--seeds", default="2000-2015")
    r.add_argument("--procs", type=int, default=10)
    r.add_argument("--ticks", type=int, default=6000)
    r.add_argument("--out", default="")
    args = p.parse_args()
    if args.cmd == "match":
        t0 = time.time()
        st = play(args.seed, args.red, args.blue, args.ticks, timing=True,
                  trace=bool(args.trace), trace_every=args.trace or 100)
        for t, s in st.items():
            ms = 1000 * s["secs"] / max(1, s["calls"])
            print(t, {k: v for k, v in s.items() if k not in ("secs", "calls")}, f"{ms:.3f} ms/call")
        print(f"{time.time() - t0:.1f}s")
    elif args.cmd == "panel":
        table = panel(args.cands.split(","), args.opps.split(","), _seeds(args.seeds),
                      args.procs, args.ticks)
        for a, row in table.items():
            tot = {k: sum(c[k] for c in row.values()) for k in ("W", "D", "L", "gf", "ga", "wall")}
            cells = " ".join(f"{b.rsplit('/', 1)[-1][:8]}:{c['W']}-{c['D']}-{c['L']}({c['gf']}:{c['ga']})"
                             for b, c in row.items())
            print(f"{a.rsplit('/', 1)[-1]:12s} {tot} | {cells}")
    elif args.cmd == "vs":
        table, _ = round_robin([args.red, args.blue], _seeds(args.seeds), args.procs, args.ticks)
        print(json.dumps(summary(table)))
    else:
        bots = args.bots.split(",")
        t0 = time.time()
        table, seats = round_robin(bots, _seeds(args.seeds), args.procs, args.ticks)
        out = {"seeds": args.seeds, "ticks": args.ticks, "both_sides": True,
               "notes": "W/D/L by goals; gf/ga goals for/against; wall = own wall deaths; "
                        "killed = own deaths by enemy ram. Provided bots get backend extras; "
                        "opponents/* get documented keys only.",
               "summary": summary(table), "seats": seats, "pairs": table,
               "elapsed_s": round(time.time() - t0, 1)}
        text = json.dumps(out, indent=1)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text + "\n")
        for a, s in out["summary"].items():
            print(f"{a:10s} {s}")


if __name__ == "__main__":
    main()
