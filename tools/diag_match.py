"""Behavioural diagnosis of a policy against an opponent over many matches.

    python tools/diag_match.py A_SPEC B_SPEC --seeds 4000-4031 --workers 3 --out diag.json

Plays matches exactly as ctf.evaluate.run_match does and, from the full simulator
state (analysis only, never a model input), measures per team:
  * carries: pickups, outcome (capture / killed by player / wall), carry duration,
    and for kills the killer's distance to its own home (defender vs chaser);
  * flag availability: ticks a reset flag sits at home before it is re-taken,
    split by reset cause (capture vs carrier death);
  * roles per tick: carrier, near the enemy home (<25), near own home (<25), midfield,
    dead; conditioned on whether the enemy is carrying our flag;
  * defence: while the enemy carries our flag, the number of our players within
    20 of the carrier, and whether an enemy "waiter" sits at our flag home;
  * deaths by cause and zone; speeds by role.
A is the team being diagnosed ("a"), B the opponent ("b").
"""

import argparse
import json
import math
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctf.evaluate import PLAYERS_PER_TEAM, SIDES, _make_team, parse_seeds, parse_spec  # noqa: E402
from ctf.sim import Match  # noqa: E402

ZONE = 25.0


def _new():
    return {
        "goals": 0, "pickups": 0, "carry_capture": 0, "carry_killed": 0, "carry_wall": 0,
        "carry_ticks_capture": [], "carry_ticks_fail": [], "killer_home_dist": [],
        "kill_by_carrierless_defender": 0, "kill_dist_from_our_home": [],
        "avail_after_capture": [], "avail_after_return": [],
        "role_ticks": {"carrier": 0, "enemy_zone": 0, "own_zone": 0, "mid": 0, "dead": 0},
        "role_ticks_defending": {"carrier": 0, "enemy_zone": 0, "own_zone": 0, "mid": 0, "dead": 0},
        "defending_ticks": 0, "chasers_near_carrier": 0, "waiter_present_ticks": 0,
        "deaths": {"wall": 0, "player": 0}, "deaths_zone": {"own_zone": 0, "enemy_zone": 0, "mid": 0},
        "player_kills": 0, "kills_of_carrier": 0, "kills_of_waiter": 0,
        "speed_sum": {"carrier": 0.0, "enemy_zone": 0.0, "own_zone": 0.0, "mid": 0.0},
        "speed_n": {"carrier": 0, "enemy_zone": 0, "own_zone": 0, "mid": 0},
        "carry_path_progress": [],  # per carry: ticks per 100 units of home-ward progress
        "first_goal_tick": None,
    }


def diagnose(job):
    spec_a, spec_b, seed, a_side, ticks = job
    spec_a, spec_b = parse_spec(spec_a), parse_spec(spec_b)
    b_side = SIDES[1 - SIDES.index(a_side)]
    need_full = any(s.kind == "scripted" and not s.documented_only for s in (spec_a, spec_b))
    match = Match(seed, team_names=SIDES, spheres_per_team=PLAYERS_PER_TEAM, max_ticks=ticks,
                  teammates_field=need_full)
    teams = {}
    for side, spec, role in ((a_side, spec_a, 0), (b_side, spec_b, 1)):
        ids = [s.id for s in match.spheres if s.team == side]
        teams[side] = _make_team(spec, ids, seed, role)
    role_of = {a_side: "a", b_side: "b"}
    other = {a_side: b_side, b_side: a_side}
    st = {"a": _new(), "b": _new()}
    flag_of = {f.team: f for f in match.flags}
    homes = match.homes
    team_of = {s.id: s.team for s in match.spheres}
    carry_start = {}       # flag team -> (tick, carrier id, start dist home)
    avail_since = {}       # flag team -> (tick, cause)
    death_log = match.death_log()
    details = []           # one row per player death
    respawned_at = {sp.id: 0 for sp in match.spheres}
    was_alive = {sp.id: True for sp in match.spheres}

    while not match.is_over():
        tick = match.tick
        observations = match.observe()
        split = {side: {} for side in SIDES}
        for sid, obs in observations.items():
            split[team_of[sid]][sid] = obs
        actions = {}
        for side in SIDES:
            actions.update(teams[side].act(split[side]))
        # per-tick roles (before the step)
        for side in SIDES:
            s = st[role_of[side]]
            hx, hy = homes[side]
            ex, ey = homes[other[side]]
            defending = flag_of[side].carrier is not None
            if defending:
                s["defending_ticks"] += 1
                ec = flag_of[side].carrier
                near = sum(1 for sp in match.spheres if sp.team == side and sp.alive
                           and math.hypot(sp.x - ec.x, sp.y - ec.y) < 20.0)
                s["chasers_near_carrier"] += near
                waiter = any(sp.team == other[side] and sp.alive and sp.carrying_flag is None
                             and math.hypot(sp.x - hx, sp.y - hy) < 15.0 for sp in match.spheres)
                s["waiter_present_ticks"] += int(waiter)
            for sp in match.spheres:
                if sp.team != side:
                    continue
                if not sp.alive:
                    role = "dead"
                elif sp.carrying_flag is not None:
                    role = "carrier"
                elif math.hypot(sp.x - ex, sp.y - ey) < ZONE:
                    role = "enemy_zone"
                elif math.hypot(sp.x - hx, sp.y - hy) < ZONE:
                    role = "own_zone"
                else:
                    role = "mid"
                s["role_ticks"][role] += 1
                if defending:
                    s["role_ticks_defending"][role] += 1
                if role != "dead":
                    s["speed_sum"][role] += sp.speed
                    s["speed_n"][role] += 1
        carriers = {f.team: f.carrier for f in match.flags}
        pre = {sp.id: (sp.x, sp.y, sp.speed, sp.alive) for sp in match.spheres}
        for sp in match.spheres:
            if sp.alive and not was_alive[sp.id]:
                respawned_at[sp.id] = tick
            was_alive[sp.id] = sp.alive
        before = dict(match.score)
        match.act(actions)
        # captures
        for side in SIDES:
            gained = match.score[side] - before[side]
            if gained:
                s = st[role_of[side]]
                s["goals"] += gained
                if s["first_goal_tick"] is None:
                    s["first_goal_tick"] = tick
        # flag transitions
        for f in match.flags:
            old, new = carriers[f.team], f.carrier
            if old is None and new is not None:
                taker = role_of[new.team]
                st[taker]["pickups"] += 1
                if f.team in avail_since:
                    t0, cause = avail_since.pop(f.team)
                    key = "avail_after_capture" if cause == "capture" else "avail_after_return"
                    st[taker][key].append(tick - t0)
                carry_start[f.team] = (tick, new.id)
            elif old is not None and new is not old:
                taker = role_of[old.team]
                t0, _ = carry_start.pop(f.team, (tick, None))
                dur = tick - t0
                if old.alive:  # captured
                    st[taker]["carry_capture"] += 1
                    st[taker]["carry_ticks_capture"].append(dur)
                    avail_since[f.team] = (tick + 1, "capture")
                else:
                    reason = old.death_reason
                    st[taker]["carry_wall" if reason == "wall" else "carry_killed"] += 1
                    st[taker]["carry_ticks_fail"].append(dur)
                    avail_since[f.team] = (tick + 1, "return")
                if new is not None:  # immediate re-take on the same tick
                    pass
        # deaths
        new_deaths = []
        for message in reversed(death_log):
            if message["tick"] != tick:
                break
            new_deaths.append(message)
        for m in new_deaths:
            side = m["team"]
            s = st[role_of[side]]
            reason = m["reason"] if m["reason"] in ("wall", "player") else "player"
            s["deaths"][reason] += 1
            hx, hy = homes[side]
            ex, ey = homes[other[side]]
            dx, dy = m["x"], m["y"]
            zone = ("enemy_zone" if math.hypot(dx - ex, dy - ey) < ZONE else
                    "own_zone" if math.hypot(dx - hx, dy - hy) < ZONE else "mid")
            s["deaths_zone"][zone] += 1
            if reason == "player":
                victim = next(sp for sp in match.spheres if sp.team == side and sp.player == m["player"])
                was_carrier = carriers[other[side]] is victim
                killers = [sp for sp in match.spheres if sp.team == other[side] and sp.alive
                           and math.hypot(sp.x - dx, sp.y - dy) < 3.5]
                k = st[role_of[other[side]]]
                kill = min(killers, key=lambda sp: math.hypot(sp.x - dx, sp.y - dy)) if killers else None
                vh = homes[side]
                details.append({
                    "victim": role_of[side], "t": tick, "zone": zone,
                    "v_carrier": was_carrier, "v_speed": round(pre[victim.id][2], 2),
                    "v_since_spawn": tick - respawned_at[victim.id],
                    "v_home_d": round(math.hypot(dx - vh[0], dy - vh[1]), 1),
                    "k_speed": round(pre[kill.id][2], 2) if kill else None,
                    "k_carrier": bool(kill is not None and kill.carrying_flag is not None),
                    "k_carrier_pre": bool(kill is not None and carriers[side] is kill),
                    "k_home_d": round(math.hypot(kill.x - homes[kill.team][0],
                                                 kill.y - homes[kill.team][1]), 1) if kill else None,
                    "k_since_spawn": tick - respawned_at[kill.id] if kill else None,
                    "own_flag_carried": carriers[side] is not None,
                    "enemy_flag_carried_by_mate": carriers[other[side]] is not None,
                })
                k["player_kills"] += 1
                if was_carrier:
                    k["kills_of_carrier"] += 1
                    if killers:
                        killer = min(killers, key=lambda sp: math.hypot(sp.x - dx, sp.y - dy))
                        kh = homes[killer.team]
                        k["killer_home_dist"].append(round(math.hypot(killer.x - kh[0], killer.y - kh[1]), 1))
                # a victim hovering near the killer team's home without carrying = waiter
                kh = homes[other[side]]
                if not was_carrier and math.hypot(dx - kh[0], dy - kh[1]) < 15.0:
                    k["kills_of_waiter"] += 1
    for side in SIDES:
        teams[side].close()
    out = {"seed": seed, "a_side": a_side, "ticks": match.tick, "score_a": match.score[a_side],
           "score_b": match.score[b_side], "a": st["a"], "b": st["b"], "deaths": details}
    return out


def _mean(xs):
    return round(sum(xs) / len(xs), 2) if xs else None


def _median(xs):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[len(xs) // 2]


def aggregate(records):
    n = len(records)
    T = records[0]["ticks"]
    rep = {"n": n, "win_a": sum(r["score_a"] > r["score_b"] for r in records) / n,
           "gd_a": _mean([r["score_a"] - r["score_b"] for r in records])}
    for t in ("a", "b"):
        S = [r[t] for r in records]
        tot = lambda k: sum(s[k] for s in S)  # noqa: E731
        allv = lambda k: [v for s in S for v in s[k]]  # noqa: E731
        role = {k: round(sum(s["role_ticks"][k] for s in S) / (n * 3 * T), 3) for k in S[0]["role_ticks"]}
        dticks = tot("defending_ticks")
        roled = {k: round(sum(s["role_ticks_defending"][k] for s in S) / max(1, 3 * dticks), 3)
                 for k in S[0]["role_ticks_defending"]}
        speed = {k: round(sum(s["speed_sum"][k] for s in S) / max(1, sum(s["speed_n"][k] for s in S)), 2)
                 for k in S[0]["speed_sum"]}
        pk = tot("pickups")
        rep[t] = {
            "goals_pm": round(tot("goals") / n, 2), "pickups_pm": round(pk / n, 2),
            "conv": round(tot("carry_capture") / max(1, pk), 3),
            "carry_killed_pm": round(tot("carry_killed") / n, 2),
            "carry_wall_pm": round(tot("carry_wall") / n, 2),
            "carry_ticks_capture_med": _median(allv("carry_ticks_capture")),
            "carry_ticks_capture_mean": _mean(allv("carry_ticks_capture")),
            "carry_ticks_fail_med": _median(allv("carry_ticks_fail")),
            "avail_after_capture_med": _median(allv("avail_after_capture")),
            "avail_after_capture_mean": _mean(allv("avail_after_capture")),
            "avail_after_return_med": _median(allv("avail_after_return")),
            "avail_after_return_mean": _mean(allv("avail_after_return")),
            "n_avail_capture": len(allv("avail_after_capture")),
            "n_avail_return": len(allv("avail_after_return")),
            "role_frac": role, "role_frac_while_defending": roled,
            "defending_frac": round(dticks / (n * T), 3),
            "near_enemy_carrier_avg": round(tot("chasers_near_carrier") / max(1, dticks), 2),
            "enemy_waiter_present_frac": round(tot("waiter_present_ticks") / max(1, dticks), 3),
            "deaths_pm": {k: round(sum(s["deaths"][k] for s in S) / n, 2) for k in S[0]["deaths"]},
            "deaths_zone_pm": {k: round(sum(s["deaths_zone"][k] for s in S) / n, 2)
                               for k in S[0]["deaths_zone"]},
            "player_kills_pm": round(tot("player_kills") / n, 2),
            "kills_of_carrier_pm": round(tot("kills_of_carrier") / n, 2),
            "kills_of_waiter_pm": round(tot("kills_of_waiter") / n, 2),
            "carrier_killer_home_dist_med": _median(allv("killer_home_dist")),
            "speed": speed,
            "first_goal_tick_med": _median([s["first_goal_tick"] for s in S
                                            if s["first_goal_tick"] is not None]),
        }
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--seeds", default="4000-4031")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--ticks", type=int, default=6000)
    p.add_argument("--out", default="")
    args = p.parse_args()
    a, b = str(parse_spec(args.a)), str(parse_spec(args.b))
    jobs = [(a, b, s, side, args.ticks) for s in parse_seeds(args.seeds) for side in SIDES]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        records = list(pool.map(diagnose, jobs))
    rep = aggregate(records)
    print(json.dumps(rep, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"report": rep, "records": records}))


if __name__ == "__main__":
    main()
