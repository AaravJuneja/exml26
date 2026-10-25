"""Headless simulator throughput benchmark.

Runs the unmodified engine/Game.step loop in-process: model workers, telemetry
and the thread pool are replaced at runtime by synchronous stand-ins, so the
measurement covers physics, observation building and policy callbacks only.

Usage:
    python -m bench.sim_speed --ticks 1000 --procs 1,2,4,6
"""

import argparse
import importlib.util
import itertools
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import engine.game as game_module

ROOT = Path(__file__).resolve().parents[1]
_POLICY_SOURCES = {
    "idle": "def decide(observation):\n    return 0.0, 0.0, False\n",
    "random": (
        "import random\n"
        "def decide(observation):\n"
        "    return random.uniform(-1, 1), random.uniform(-1, 1), random.random() < 0.02\n"
    ),
}
_module_counter = itertools.count()


def _policy_path(name):
    if name in _POLICY_SOURCES:
        directory = ROOT / "bench" / ".policies"
        directory.mkdir(exist_ok=True)
        path = directory / f"{name}-{os.getpid()}.py"
        if not path.exists():
            path.write_text(_POLICY_SOURCES[name], encoding="utf-8")
        return str(path)
    return str(ROOT / name)


class _InProcessWorker:
    """One independent module instance per sphere, like the per-player workers."""

    decide_seconds = 0.0

    def __init__(self, team, model_path, sphere_id=None):
        spec = importlib.util.spec_from_file_location(
            f"bench_model_{next(_module_counter)}", model_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._decide = module.decide

    def wait_ready(self, timeout=None):
        return True

    def get_action(self, observation, timeout):
        observation.pop("teammates", None)
        observation.pop("sonar_recieve", None)
        start = time.perf_counter()
        action = self._decide(observation)
        _InProcessWorker.decide_seconds += time.perf_counter() - start
        return action

    def close(self):
        pass


class _NullTelemetry:
    file_path = None

    def record_map(self, *args, **kwargs):
        pass

    def record_tick(self, *args, **kwargs):
        pass

    def close(self):
        pass


class _Done:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _SyncPool:
    def submit(self, fn, *args):
        return _Done(fn(*args))


def _patch():
    game_module.ModelWorker = _InProcessWorker
    game_module.TelemetryRecorder = _NullTelemetry


def run_game(red, blue, seed, ticks):
    _patch()
    _InProcessWorker.decide_seconds = 0.0
    start = time.perf_counter()
    teams = [
        {"name": "red", "model_path": _policy_path(red)},
        {"name": "blue", "model_path": _policy_path(blue)},
    ]
    game = game_module.Game(teams, spheres_per_team=3, seed=seed, max_ticks=ticks)
    game._pool.shutdown(wait=False)
    game._pool = _SyncPool()
    setup = time.perf_counter() - start
    deaths = 0
    start = time.perf_counter()
    while not game.is_over():
        game.step()
    elapsed = time.perf_counter() - start
    deaths = len(game._death_log)
    return {
        "ticks": game.tick,
        "setup_s": setup,
        "step_s": elapsed,
        "decide_s": _InProcessWorker.decide_seconds,
        "ticks_per_s": game.tick / elapsed,
        "score": dict(game.score),
        "deaths": deaths,
    }


def _worker(args):
    return run_game(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--procs", default="1,2,4,6")
    parser.add_argument("--policies", default="idle,random,teams/example_team/model.py")
    args = parser.parse_args()

    results = {"single": {}, "parallel": {}}
    for policy in args.policies.split(","):
        r = run_game(policy, policy, seed=1234, ticks=args.ticks)
        r["engine_only_ticks_per_s"] = r["ticks"] / (r["step_s"] - r["decide_s"])
        results["single"][policy] = r
        print(json.dumps({"policy": policy, **r}), flush=True)

    policy = "random"
    for procs in (int(p) for p in args.procs.split(",")):
        jobs = [(policy, policy, 5000 + i, args.ticks) for i in range(procs)]
        start = time.perf_counter()
        with mp.get_context("spawn").Pool(procs) as pool:
            outs = pool.map(_worker, jobs)
        wall = time.perf_counter() - start
        total_ticks = sum(o["ticks"] for o in outs)
        step_rate = sum(o["ticks_per_s"] for o in outs)
        results["parallel"][procs] = {
            "aggregate_step_ticks_per_s": step_rate,
            "wall_ticks_per_s_incl_setup": total_ticks / wall,
        }
        print(json.dumps({"procs": procs, **results["parallel"][procs]}), flush=True)

    best = max(v["aggregate_step_ticks_per_s"] for v in results["parallel"].values())
    print(json.dumps({
        "best_aggregate_ticks_per_s": best,
        "agent_transitions_per_hour_6_agents": best * 6 * 3600,
        "full_6000_tick_games_per_hour": best * 3600 / 6000,
    }), flush=True)


if __name__ == "__main__":
    main()
