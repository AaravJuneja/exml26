"""Evaluation harness: the single source of truth for "is A better than B".

Every comparison plays full matches in the bit-exact simulator `ctf.sim.Match`,
on fixed seeds, with *both* side assignments per seed. Statistics resample
seeds (keeping each seed's two side-swapped games together), because the two
games on one map are strongly correlated.

Player specs (strings or `PlayerSpec`)
--------------------------------------
    idle                       does nothing
    baseline|example|example2  provided scripted models
    path/to/model.py           any module with `decide(observation)`
    path/to/policy.npz         batched NumpyActor over three PlayerMemory objects

Options follow an `@`, comma separated:
    @greedy      argmax actions (default for .npz)
    @sample      sample actions; @seed=N sets the sampling seed (default 0)
    @doc         pass only the documented observation keys
    @full        also pass the backend's undocumented `teammates` field

The provided scripted models read `teammates`, so they default to `@full`.
Any other `model.py` defaults to `@doc` (it stands in for a submission), and a
policy `.npz` always receives documented keys only.

CLI
---
    python -m ctf.evaluate A B --seeds 1000-1049 --workers 10 --out runs/eval/x.jsonl
    python -m ctf.evaluate A --gauntlet [--opponents idle baseline ...] --seeds 1000-1031
    python -m ctf.evaluate --round-robin baseline example example2 idle --seeds 1000-1031
    python -m ctf.evaluate --summarize runs/eval/x.jsonl
    python -m ctf.evaluate --compare runs/eval/new.jsonl runs/eval/old.jsonl
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ctf.features import FEATURE_DIM, THROTTLE_VALUES, TURN_VALUES, PlayerMemory, encode
from ctf.policies import ROOT, SCRIPTED
from ctf.sim import DOCUMENTED_KEYS, Match

SIDES = ("red", "blue")
PLAYERS_PER_TEAM = 3
DEFAULT_TICKS = 6000
DEFAULT_OPPONENTS = ("idle", "baseline", "example", "example2")
RECORD_VERSION = 1
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS")


# ------------------------------------------------------------------ specs
@dataclass(frozen=True)
class PlayerSpec:
    kind: str                 # "idle", "scripted" or "npz"
    path: str = ""            # absolute path for scripted/npz
    name: str = ""            # builtin name, if any
    greedy: bool = True       # npz only
    sample_seed: int = 0      # npz only, used when greedy is False
    documented_only: bool = True

    @property
    def label(self):
        """Short human-readable name; options shown only when not the default."""
        if self.name:
            base = self.name
        elif self.kind == "npz":
            base = Path(self.path).stem
        else:
            base = f"{Path(self.path).parent.name}/{Path(self.path).stem}"
        if self.kind == "npz" and not self.greedy:
            base += f"@sample,seed={self.sample_seed}"
        elif self.kind == "scripted":
            default_documented = not self.name
            if self.documented_only != default_documented:
                base += "@doc" if self.documented_only else "@full"
        return base

    def __str__(self):
        if self.kind == "idle":
            return "idle"
        base = self.name or self.path
        if self.kind == "npz":
            opts = "greedy" if self.greedy else f"sample,seed={self.sample_seed}"
        else:
            opts = "doc" if self.documented_only else "full"
        return f"{base}@{opts}"


def _resolve(path):
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and not candidate.exists() and (ROOT / candidate).exists():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"player spec file not found: {path}")
    return str(candidate)


def parse_spec(spec):
    """Parse a spec string (or pass a PlayerSpec through) into a PlayerSpec."""
    if isinstance(spec, PlayerSpec):
        return spec
    text = str(spec).strip()
    base, _, opt_text = text.partition("@")
    options = [o.strip() for o in opt_text.split(",") if o.strip()]
    if base == "idle":
        result = PlayerSpec("idle", name="idle")
    elif base in SCRIPTED:
        result = PlayerSpec("scripted", path=_resolve(SCRIPTED[base]), name=base,
                            documented_only=False)
    elif base.endswith(".npz"):
        result = PlayerSpec("npz", path=_resolve(base))
    elif base.endswith(".py"):
        resolved = _resolve(base)
        builtin = {_resolve(p): n for n, p in SCRIPTED.items()}.get(resolved, "")
        result = PlayerSpec("scripted", path=resolved, name=builtin,
                            documented_only=not builtin)
    else:
        raise ValueError(f"unknown player spec {spec!r}")
    for option in options:
        key, _, value = option.partition("=")
        if key == "greedy" and result.kind == "npz":
            result = replace(result, greedy=True)
        elif key == "sample" and result.kind == "npz":
            result = replace(result, greedy=False)
        elif key == "seed" and result.kind == "npz":
            result = replace(result, sample_seed=int(value))
        elif key == "doc" and result.kind != "npz":
            result = replace(result, documented_only=True)
        elif key == "full" and result.kind == "scripted":
            result = replace(result, documented_only=False)
        elif result.kind == "idle" and key in ("doc", "full"):
            continue
        else:
            raise ValueError(f"option {option!r} is not valid for spec {spec!r}")
    return result


def file_digest(spec):
    spec = parse_spec(spec)
    if not spec.path:
        return None
    return hashlib.sha256(Path(spec.path).read_bytes()).hexdigest()[:16]


def parse_seeds(*items):
    """'1000-1049', '1,2,7-9' or ints (any mix) -> ordered list without duplicates."""
    seeds = []
    for item in items:
        if isinstance(item, (list, tuple, range)):
            seeds += parse_seeds(*item)
            continue
        if isinstance(item, (int, np.integer)):
            seeds.append(int(item))
            continue
        for token in str(item).replace(" ", ",").split(","):
            if not token:
                continue
            low, dash, high = token.partition("-")
            if dash:
                low, high = int(low), int(high)
                if high < low:
                    raise ValueError(f"empty seed range {token!r}")
                seeds += range(low, high + 1)
            else:
                seeds.append(int(token))
    return list(dict.fromkeys(seeds))


# --------------------------------------------------------- team controllers
_CACHE = {}


def _documented(observation):
    return {key: observation[key] for key in DOCUMENTED_KEYS}


def _loaded(spec):
    """Per-process cache of the expensive part of a spec (compiled code / actor)."""
    key = str(spec)
    if key not in _CACHE:
        if spec.kind == "scripted":
            source = Path(spec.path).read_text()
            _CACHE[key] = compile(source, spec.path, "exec")
        elif spec.kind == "npz":
            from ctf.policy_io import load_policy

            actor, meta = load_policy(spec.path)
            _CACHE[key] = actor
        else:
            _CACHE[key] = None
    return _CACHE[key]


def _silent(*args, **kwargs):
    pass


class _IdleTeam:
    def __init__(self):
        self.errors = 0
        self.times = []

    def act(self, observations):
        return {}


class _ScriptedTeam:
    """Three fresh module instances (module-level state is per player, per match)."""

    def __init__(self, spec, sphere_ids):
        code = _loaded(spec)
        self.documented_only = spec.documented_only
        self.decide = {}
        for sid in sphere_ids:
            module = types.ModuleType(f"ctf_eval_player_{sid}")
            module.__file__ = spec.path
            module.print = _silent  # the provided models print events every match
            exec(code, module.__dict__)
            self.decide[sid] = module.decide
        self.errors = 0
        self.times = []

    def act(self, observations):
        actions = {}
        clock = time.perf_counter
        for sid, decide in self.decide.items():
            observation = observations.get(sid)
            if observation is None:
                continue
            if self.documented_only:
                observation = _documented(observation)
            start = clock()
            try:
                actions[sid] = decide(observation)
            except Exception:
                self.errors += 1
                actions[sid] = (0.0, 0.0, False, None)
            self.times.append(clock() - start)
        return actions


class _PolicyTeam:
    """Batched NumpyActor: one feature batch and one forward pass per tick."""

    def __init__(self, spec, sphere_ids, rng):
        self.actor = _loaded(spec)
        self.greedy = spec.greedy
        self.rng = rng
        self.ids = sorted(sphere_ids)
        self.memories = {sid: PlayerMemory() for sid in self.ids}
        self.errors = 0
        self.times = []

    def act(self, observations):
        ids = [sid for sid in self.ids if sid in observations]
        if not ids:
            return {}
        start = time.perf_counter()
        batch = np.empty((len(ids), FEATURE_DIM), dtype=np.float32)
        for row, sid in enumerate(ids):
            batch[row] = encode(_documented(observations[sid]), self.memories[sid])
        chosen, _ = self.actor.sample(batch, self.rng, self.greedy)
        actions = {}
        for sid, (throttle, turn, jump) in zip(ids, chosen.tolist()):
            self.memories[sid].set_action(throttle, turn, jump)
            actions[sid] = (THROTTLE_VALUES[throttle], TURN_VALUES[turn], bool(jump))
        self.times.append(time.perf_counter() - start)
        return actions


def _make_team(spec, sphere_ids, seed, role):
    if spec.kind == "idle":
        return _IdleTeam()
    if spec.kind == "scripted":
        return _ScriptedTeam(spec, sphere_ids)
    rng = np.random.default_rng([spec.sample_seed, seed, role])
    return _PolicyTeam(spec, sphere_ids, rng)


def _new_team_stats():
    return {"captures": 0, "pickups": 0, "wall_deaths": 0, "player_deaths": 0,
            "unknown_deaths": 0, "carrier_deaths": 0, "first_capture_tick": None}


def _latency(times):
    if not times:
        return {"calls": 0, "mean_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    arr = np.asarray(times) * 1000.0
    return {"calls": int(arr.size), "mean_ms": round(float(arr.mean()), 4),
            "p99_ms": round(float(np.percentile(arr, 99)), 4),
            "max_ms": round(float(arr.max()), 4)}


# -------------------------------------------------------------- one match
def run_match(spec_a, spec_b, seed, a_side="red", max_ticks=DEFAULT_TICKS):
    """Play one full match; return a JSON-serializable record from A's perspective."""
    if a_side not in SIDES:
        raise ValueError(f"a_side must be 'red' or 'blue', not {a_side!r}")
    spec_a, spec_b = parse_spec(spec_a), parse_spec(spec_b)
    started = time.perf_counter()
    b_side = SIDES[1 - SIDES.index(a_side)]
    need_full = any(s.kind == "scripted" and not s.documented_only for s in (spec_a, spec_b))
    match = Match(seed, team_names=SIDES, spheres_per_team=PLAYERS_PER_TEAM,
                  max_ticks=max_ticks, teammates_field=need_full)
    team_of = {s.id: s.team for s in match.spheres}
    role_of = {a_side: "a", b_side: "b"}
    controllers = {}
    for side, spec, role in ((a_side, spec_a, 0), (b_side, spec_b, 1)):
        ids = [s.id for s in match.spheres if s.team == side]
        controllers[side] = _make_team(spec, ids, seed, role)
    stats = {"a": _new_team_stats(), "b": _new_team_stats()}
    flags = match.flags
    score = match.score
    death_log = match.death_log()

    while not match.is_over():
        tick = match.tick
        observations = match.observe()
        split = {side: {} for side in SIDES}
        for sid, observation in observations.items():
            split[team_of[sid]][sid] = observation
        actions = {}
        for side in SIDES:
            actions.update(controllers[side].act(split[side]))
        carriers = [flag.carrier for flag in flags]
        before = dict(score)
        match.act(actions)

        for side in SIDES:
            gained = score[side] - before[side]
            if gained:
                team = stats[role_of[side]]
                team["captures"] += gained
                if team["first_capture_tick"] is None:
                    team["first_capture_tick"] = tick
        for flag, carrier in zip(flags, carriers):
            if flag.carrier is not None and flag.carrier is not carrier:
                stats[role_of[flag.carrier.team]]["pickups"] += 1
            if carrier is not None and not carrier.alive:
                stats[role_of[carrier.team]]["carrier_deaths"] += 1
        for message in reversed(death_log):
            if message["tick"] != tick:
                break
            reason = message["reason"]
            key = reason + "_deaths" if reason in ("wall", "player") else "unknown_deaths"
            stats[role_of[message["team"]]][key] += 1

    score_a, score_b = score[a_side], score[b_side]
    result = "win" if score_a > score_b else "loss" if score_a < score_b else "draw"
    for role, side in (("a", a_side), ("b", b_side)):
        stats[role]["side"] = side
        stats[role]["score"] = score[side]
        stats[role]["errors"] = controllers[side].errors
    return {
        "version": RECORD_VERSION,
        "spec_a": str(spec_a),
        "spec_b": str(spec_b),
        "seed": seed,
        "a_side": a_side,
        "b_side": b_side,
        "players": PLAYERS_PER_TEAM,
        "max_ticks": max_ticks,
        "ticks": match.tick,
        "score": dict(score),
        "score_a": score_a,
        "score_b": score_b,
        "goal_diff": score_a - score_b,
        "result": result,
        "a": stats["a"],
        "b": stats["b"],
        "wall_time": round(time.perf_counter() - started, 4),
        "timing": {"decide_a": _latency(controllers[a_side].times),
                   "decide_b": _latency(controllers[b_side].times)},
    }


def deterministic_view(record):
    """A record without wall-clock fields (for reproducibility checks)."""
    return {k: v for k, v in record.items() if k not in ("wall_time", "timing")}


# ------------------------------------------------------------ parallel runs
def _run_job(job):
    spec_a, spec_b, seed, a_side, max_ticks = job
    return run_match(spec_a, spec_b, seed, a_side, max_ticks)


def _job_key(record):
    return (record["spec_a"], record["spec_b"], record["seed"], SIDES.index(record["a_side"]))


def run_jobs(jobs, workers=1, progress=None):
    """Run (spec_a, spec_b, seed, a_side, max_ticks) jobs; yield records as they finish.

    workers <= 1 runs in this process. Otherwise a spawn-context process pool is
    used; each worker caches compiled code / actors per spec, so specs load once.
    """
    jobs = [(str(parse_spec(a)), str(parse_spec(b)), int(s), side, int(t))
            for a, b, s, side, t in jobs]
    total, done, started = len(jobs), 0, time.perf_counter()

    def report(record):
        if progress:
            elapsed = time.perf_counter() - started
            progress(f"[{done}/{total}] {elapsed:7.1f}s  seed {record['seed']} "
                     f"{record['a_side']:>4}  {record['score_a']}-{record['score_b']} "
                     f"{record['result']}  ({record['wall_time']:.1f}s)")

    if workers <= 1 or total <= 1:
        for job in jobs:
            record = _run_job(job)
            done += 1
            report(record)
            yield record
        return
    for var in _THREAD_VARS:
        os.environ.setdefault(var, "1")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, total), mp_context=context) as pool:
        futures = [pool.submit(_run_job, job) for job in jobs]
        try:
            for future in as_completed(futures):
                record = future.result()
                done += 1
                report(record)
                yield record
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def _write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    os.replace(tmp, path)


def load_records(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _collect(jobs, workers, out=None, progress=None):
    """Run jobs, stream records to `out`.partial, then write `out` sorted."""
    records = []
    partial = None
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = open(out.with_name(out.name + ".partial"), "w")
    started = time.perf_counter()
    try:
        for record in run_jobs(jobs, workers, progress):
            records.append(record)
            if partial is not None:
                partial.write(json.dumps(record) + "\n")
                partial.flush()
    finally:
        if partial is not None:
            partial.close()
    elapsed = time.perf_counter() - started
    records.sort(key=_job_key)
    if out is not None:
        _write_jsonl(out, records)
        os.remove(out.with_name(out.name + ".partial"))
    return records, elapsed


def _throughput(records, elapsed, workers):
    ticks = sum(r["ticks"] for r in records)
    return {"workers": workers, "elapsed_s": round(elapsed, 2),
            "matches_per_min": round(60.0 * len(records) / elapsed, 2) if elapsed else None,
            "ticks_per_s": round(ticks / elapsed, 1) if elapsed else None,
            "mean_match_wall_s": round(float(np.mean([r["wall_time"] for r in records])), 3)
            if records else None}


def evaluate(spec_a, spec_b, seeds, workers=1, out=None, max_ticks=DEFAULT_TICKS,
             n_boot=10000, boot_seed=0, progress=None):
    """Play every seed on both sides; return (summary, records sorted by seed/side)."""
    spec_a, spec_b = parse_spec(spec_a), parse_spec(spec_b)
    seeds = parse_seeds(seeds)
    jobs = [(spec_a, spec_b, seed, side, max_ticks) for seed in seeds for side in SIDES]
    records, elapsed = _collect(jobs, workers, out, progress)
    summary = summarize(records, n_boot=n_boot, boot_seed=boot_seed)
    summary["meta"] = {"spec_a": str(spec_a), "spec_b": str(spec_b),
                       "label_a": spec_a.label, "label_b": spec_b.label,
                       "digest_a": file_digest(spec_a), "digest_b": file_digest(spec_b),
                       "seeds": _seed_text(seeds), "max_ticks": max_ticks,
                       "out": str(out) if out is not None else None}
    summary["throughput"] = _throughput(records, elapsed, workers)
    if out is not None:
        with open(Path(out).with_suffix(".summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2)
    return summary, records


def _seed_text(seeds):
    if not seeds:
        return ""
    parts, start, prev = [], seeds[0], seeds[0]
    for seed in seeds[1:] + [None]:
        if seed is not None and seed == prev + 1:
            prev = seed
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if seed is not None:
            start = prev = seed
    return ",".join(parts)


# -------------------------------------------------------------- statistics
def _bootstrap_ratio(sums, counts, n_boot, rng, chunk=2000):
    """Bootstrap distribution of sum(sums[i]) / sum(counts[i]) over resampled seeds."""
    n = len(sums)
    out = np.empty(n_boot)
    for begin in range(0, n_boot, chunk):
        size = min(chunk, n_boot - begin)
        idx = rng.integers(0, n, size=(size, n))
        out[begin:begin + size] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return out


def _ci(samples):
    low, high = np.percentile(samples, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def paired_bootstrap(records, value, n_boot=10000, boot_seed=0):
    """Mean of value(record) per match, with a 95% CI that resamples whole seeds.

    Returns (mean, [low, high], P(bootstrap mean > 0)).
    """
    groups = {}
    for record in records:
        total, count = groups.get(record["seed"], (0.0, 0))
        groups[record["seed"]] = (total + float(value(record)), count + 1)
    sums = np.array([g[0] for g in groups.values()], dtype=np.float64)
    counts = np.array([g[1] for g in groups.values()], dtype=np.float64)
    if not len(sums):
        return None, [None, None], None
    mean = float(sums.sum() / counts.sum())
    samples = _bootstrap_ratio(sums, counts, n_boot, np.random.default_rng(boot_seed))
    return round(mean, 4), _ci(samples), round(float((samples > 0).mean()), 4)


def _rate(part, whole):
    return round(part / whole, 4) if whole else None


def summarize(records, n_boot=10000, boot_seed=0):
    """Aggregate per-match records (from A's perspective)."""
    n = len(records)
    wins = sum(r["result"] == "win" for r in records)
    draws = sum(r["result"] == "draw" for r in records)
    losses = sum(r["result"] == "loss" for r in records)

    def boot(fn):
        return paired_bootstrap(records, fn, n_boot, boot_seed)

    gd, gd_ci, gd_pos = boot(lambda r: r["score_a"] - r["score_b"])
    win_rate, win_ci, _ = boot(lambda r: r["result"] == "win")
    adjusted, adjusted_ci, _ = boot(lambda r: {"win": 1.0, "draw": 0.5}.get(r["result"], 0.0))
    summary = {
        "n_matches": n,
        "n_seeds": len({r["seed"] for r in records}),
        "wins": wins, "draws": draws, "losses": losses,
        "win_rate": win_rate, "win_rate_ci": win_ci,
        "draw_rate": _rate(draws, n), "loss_rate": _rate(losses, n),
        "match_points": wins,
        "match_points_per_match": win_rate, "match_points_ci": win_ci,
        "draw_adjusted_score": adjusted, "draw_adjusted_ci": adjusted_ci,
        "goal_diff_mean": gd, "goal_diff_ci": gd_ci, "p_goal_diff_positive": gd_pos,
        "goals_for_mean": _rate(sum(r["score_a"] for r in records), n),
        "goals_against_mean": _rate(sum(r["score_b"] for r in records), n),
        "per_side": {},
    }
    for side in SIDES:
        subset = [r for r in records if r["a_side"] == side]
        m = len(subset)
        summary["per_side"][side] = {
            "n": m,
            "wins": sum(r["result"] == "win" for r in subset),
            "draws": sum(r["result"] == "draw" for r in subset),
            "losses": sum(r["result"] == "loss" for r in subset),
            "win_rate": _rate(sum(r["result"] == "win" for r in subset), m),
            "goal_diff_mean": _rate(sum(r["score_a"] - r["score_b"] for r in subset), m),
        }
    if records and "a" in records[0]:
        players = records[0].get("players", PLAYERS_PER_TEAM)
        for role in ("a", "b"):
            teams = [r[role] for r in records]
            captures = sum(t["captures"] for t in teams)
            pickups = sum(t["pickups"] for t in teams)
            firsts = [t["first_capture_tick"] for t in teams if t["first_capture_tick"] is not None]
            summary[role] = {
                "captures_per_match": _rate(captures, n),
                "pickups_per_match": _rate(pickups, n),
                "conversion": _rate(captures, pickups),
                "wall_deaths_per_player_per_match":
                    _rate(sum(t["wall_deaths"] for t in teams), n * players),
                "player_deaths_per_player_per_match":
                    _rate(sum(t["player_deaths"] for t in teams), n * players),
                "carrier_deaths_per_match": _rate(sum(t["carrier_deaths"] for t in teams), n),
                "matches_with_capture": len(firsts),
                "median_first_capture_tick": float(np.median(firsts)) if firsts else None,
                "errors": sum(t.get("errors", 0) for t in teams),
            }
            timings = [r["timing"][f"decide_{role}"] for r in records if "timing" in r]
            if timings and any(t["calls"] for t in timings):
                summary[role]["decide_p99_ms_max"] = max(t["p99_ms"] for t in timings)
                summary[role]["decide_max_ms"] = max(t["max_ms"] for t in timings)
    return summary


def compare(records_x, records_y, n_boot=10000, boot_seed=0):
    """Paired difference X - Y between two record sets on the same (seed, side) games.

    Typical use: challenger-vs-opponent records against incumbent-vs-opponent
    records. Only games present in both sets are used; seeds are resampled as
    units, so both sides and both candidates stay paired.
    """
    y_by_key = {(r["seed"], r["a_side"]): r for r in records_y}
    pairs = [(r, y_by_key[(r["seed"], r["a_side"])]) for r in records_x
             if (r["seed"], r["a_side"]) in y_by_key]
    points = {"win": 1.0, "draw": 0.0, "loss": 0.0}
    merged = [{"seed": x["seed"],
               "gd": (x["score_a"] - x["score_b"]) - (y["score_a"] - y["score_b"]),
               "pts": points[x["result"]] - points[y["result"]]} for x, y in pairs]
    gd, gd_ci, gd_pos = paired_bootstrap(merged, lambda r: r["gd"], n_boot, boot_seed)
    pts, pts_ci, pts_pos = paired_bootstrap(merged, lambda r: r["pts"], n_boot, boot_seed)
    return {"n_pairs": len(pairs), "n_seeds": len({m["seed"] for m in merged}),
            "goal_diff_delta": gd, "goal_diff_delta_ci": gd_ci, "p_goal_diff_delta_positive": gd_pos,
            "match_points_delta": pts, "match_points_delta_ci": pts_ci,
            "p_match_points_delta_positive": pts_pos}


# ------------------------------------------------------ gauntlet / round robin
def _multi(pairs, seeds, workers, out_dir, max_ticks, n_boot, progress):
    """Evaluate several (A, B) pairings with one shared process pool."""
    seeds = parse_seeds(seeds)
    pairs = [(parse_spec(a), parse_spec(b)) for a, b in pairs]
    jobs = [(a, b, seed, side, max_ticks) for a, b in pairs for seed in seeds for side in SIDES]
    records, elapsed = _collect(jobs, workers, None, progress)
    results = []
    for a, b in pairs:
        subset = [r for r in records if r["spec_a"] == str(a) and r["spec_b"] == str(b)]
        summary = summarize(subset, n_boot=n_boot)
        summary["meta"] = {"spec_a": str(a), "spec_b": str(b), "label_a": a.label,
                           "label_b": b.label, "digest_a": file_digest(a),
                           "digest_b": file_digest(b), "seeds": _seed_text(seeds),
                           "max_ticks": max_ticks}
        if out_dir is not None:
            path = Path(out_dir) / f"{_slug(a.label)}_vs_{_slug(b.label)}.jsonl"
            _write_jsonl(path, subset)
            summary["meta"]["out"] = str(path)
        results.append(summary)
    return results, _throughput(records, elapsed, workers)


def _slug(text):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def gauntlet(spec, opponents=DEFAULT_OPPONENTS, seeds=range(1000, 1032), workers=1,
             out_dir=None, max_ticks=DEFAULT_TICKS, n_boot=10000, progress=None, show=True):
    """Evaluate `spec` against each opponent; print a compact table, return summaries."""
    pairs = [(spec, opponent) for opponent in opponents]
    results, throughput = _multi(pairs, seeds, workers, out_dir, max_ticks, n_boot, progress)
    if show:
        print(format_table(results))
        print(f"{throughput['matches_per_min']} matches/min, {throughput['elapsed_s']}s total")
    return {"spec": str(parse_spec(spec)), "results": results, "throughput": throughput}


def round_robin(specs, seeds=range(1000, 1032), workers=1, out_dir=None,
                max_ticks=DEFAULT_TICKS, n_boot=10000, progress=None, show=True):
    """Every unordered pair of specs, both sides of every seed."""
    specs = [parse_spec(s) for s in specs]
    pairs = [(specs[i], specs[j]) for i in range(len(specs)) for j in range(i + 1, len(specs))]
    results, throughput = _multi(pairs, seeds, workers, out_dir, max_ticks, n_boot, progress)
    if show:
        print(format_table(results))
        print(f"{throughput['matches_per_min']} matches/min, {throughput['elapsed_s']}s total")
    return {"specs": [str(s) for s in specs], "results": results, "throughput": throughput}


def format_table(summaries):
    header = (f"{'A':<14} {'B':<14} {'W-D-L':>9} {'win%':>5} {'win CI':>11} "
              f"{'GD':>6} {'GD CI':>14} {'caps A/B':>11} {'wall/pl A/B':>11}")
    lines = [header, "-" * len(header)]
    for s in summaries:
        meta = s.get("meta", {})
        a, b = s.get("a", {}), s.get("b", {})
        wr = s["win_rate"] or 0.0
        lines.append(
            f"{meta.get('label_a', 'A')[:14]:<14} {meta.get('label_b', 'B')[:14]:<14} "
            f"{s['wins']:>3}-{s['draws']}-{s['losses']:<3} {100 * wr:>5.1f} "
            f"{_fmt_ci(s['win_rate_ci'], 100, 0):>11} {s['goal_diff_mean']:>+6.2f} "
            f"{_fmt_ci(s['goal_diff_ci'], 1, 2):>14} "
            f"{a.get('captures_per_match', 0):>5.2f}/{b.get('captures_per_match', 0):<5.2f} "
            f"{a.get('wall_deaths_per_player_per_match', 0):>5.2f}/"
            f"{b.get('wall_deaths_per_player_per_match', 0):<5.2f}")
    return "\n".join(lines)


def _fmt_ci(ci, scale, digits):
    if not ci or ci[0] is None:
        return "-"
    return f"[{ci[0] * scale:.{digits}f},{ci[1] * scale:.{digits}f}]"


# --------------------------------------------------------------------- CLI
def _label_of(text):
    """Label for a recorded spec string, even if its file lives on another machine."""
    try:
        return parse_spec(text).label
    except (FileNotFoundError, ValueError):
        base = text.partition("@")[0]
        return Path(base).stem if base.endswith(".npz") else Path(base).parent.name or base


def _print_summary(summary):
    print(format_table([summary]))
    for side, part in summary["per_side"].items():
        print(f"  A as {side:<4}: {part['wins']}-{part['draws']}-{part['losses']}  "
              f"GD {part['goal_diff_mean']:+.2f}")
    print(f"  match points {summary['match_points']}/{summary['n_matches']}  "
          f"draw-adjusted {summary['draw_adjusted_score']} {summary['draw_adjusted_ci']}  "
          f"P(GD>0) {summary['p_goal_diff_positive']}")
    for role in ("a", "b"):
        if role in summary:
            part = summary[role]
            print(f"  {role.upper()}: pickups/match {part['pickups_per_match']}  "
                  f"conversion {part['conversion']}  player deaths/pl "
                  f"{part['player_deaths_per_player_per_match']}  errors {part['errors']}"
                  + (f"  decide p99 max {part['decide_p99_ms_max']} ms"
                     if "decide_p99_ms_max" in part else ""))
    if "throughput" in summary:
        t = summary["throughput"]
        print(f"  {t['matches_per_min']} matches/min, {t['ticks_per_s']} ticks/s, "
              f"{t['elapsed_s']}s on {t['workers']} workers")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ctf.evaluate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", nargs="*", help="player specs (A [B])")
    parser.add_argument("--seeds", nargs="+", default=["1000-1031"],
                        help="seed ranges/lists, e.g. 1000-1049 or 1,5,9-12")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    parser.add_argument("--out", help="JSONL path (pairwise) or directory (gauntlet/round robin)")
    parser.add_argument("--json", help="write the full summary JSON here")
    parser.add_argument("--max-ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument("--boot", type=int, default=10000, help="bootstrap replicates")
    parser.add_argument("--gauntlet", action="store_true", help="A against --opponents")
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    parser.add_argument("--round-robin", action="store_true", help="every pair of the specs")
    parser.add_argument("--summarize", nargs="+", metavar="JSONL")
    parser.add_argument("--compare", nargs=2, metavar=("X_JSONL", "Y_JSONL"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    progress = None if args.quiet else (lambda line: print(line, file=sys.stderr, flush=True))
    seeds = parse_seeds(args.seeds)
    result = None
    if args.summarize:
        for path in args.summarize:
            records = load_records(path)
            result = summarize(records, n_boot=args.boot)
            if records:
                result["meta"] = {"label_a": _label_of(records[0]["spec_a"]),
                                  "label_b": _label_of(records[0]["spec_b"]),
                                  "out": path}
            _print_summary(result)
    elif args.compare:
        result = compare(load_records(args.compare[0]), load_records(args.compare[1]),
                         n_boot=args.boot)
        print(json.dumps(result, indent=2))
    elif args.gauntlet:
        if len(args.specs) != 1:
            parser.error("--gauntlet takes exactly one spec")
        result = gauntlet(args.specs[0], args.opponents, seeds, args.workers, args.out,
                          args.max_ticks, args.boot, progress)
    elif args.round_robin:
        if len(args.specs) < 2:
            parser.error("--round-robin needs at least two specs")
        result = round_robin(args.specs, seeds, args.workers, args.out, args.max_ticks,
                             args.boot, progress)
    else:
        if len(args.specs) != 2:
            parser.error("give two specs A B (or use --gauntlet / --round-robin)")
        result, _ = evaluate(args.specs[0], args.specs[1], seeds, args.workers, args.out,
                             args.max_ticks, args.boot, progress=progress)
        _print_summary(result)
    if args.json and result is not None:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=2)
    return result


if __name__ == "__main__":
    main()
