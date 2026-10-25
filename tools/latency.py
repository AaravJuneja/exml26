"""Deadline behaviour of an exported model under the real engine and RPC workers.

Runs `engine.game.Game` matches (six real `rpc.worker.ModelWorker` processes,
the exported model on both teams) stepping as fast as possible, and times every
`ModelWorker.get_action` call from send to response. Telemetry is disabled by
monkeypatching in this process only; engine and rpc code are untouched.

    python tools/latency.py --hidden 128 128 --ticks 6000 [--stress N]
    python tools/latency.py --policy runs/x/policy.npz --stress 10

Prints one JSON summary. Thread-count environment variables are removed before
the workers start (unless --keep-thread-env), so the model's own `setdefault`
is what limits BLAS threads, as it would be in the tournament.

Keep top-level imports light: spawned children re-import this module.
"""

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")
DEADLINE_S = 0.03


def _burn(stop):
    x = 0
    while not stop.is_set():
        for _ in range(100000):
            x = (x * 1103515245 + 12345) & 0xFFFFFFFF


def _percentiles(values_s):
    import numpy as np

    if not values_s:
        return {}
    ms = np.asarray(values_s) * 1000.0
    return {
        "n": int(ms.size),
        "mean_ms": round(float(ms.mean()), 4),
        "p50_ms": round(float(np.percentile(ms, 50)), 4),
        "p90_ms": round(float(np.percentile(ms, 90)), 4),
        "p99_ms": round(float(np.percentile(ms, 99)), 4),
        "p99.9_ms": round(float(np.percentile(ms, 99.9)), 4),
        "max_ms": round(float(ms.max()), 4),
        "over_10ms": int((ms > 10).sum()),
        "over_20ms": int((ms > 20).sum()),
        "over_30ms": int((ms > 30).sum()),
    }


def _load_module(path, name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    start = time.perf_counter()
    spec.loader.exec_module(module)
    return module, time.perf_counter() - start


def _inprocess(model_path, ticks, seed):
    """Warm decide() times for six in-process players on a headless Match."""
    from ctf.sim import Match

    match = Match(seed, max_ticks=ticks)
    modules, import_s = {}, []
    for sphere in match.spheres:
        modules[sphere.id], took = _load_module(model_path, f"latency_model_{sphere.id}")
        import_s.append(took)
    times = []
    while not match.is_over():
        actions = {}
        for sid, observation in match.observe().items():
            start = time.perf_counter()
            actions[sid] = modules[sid].decide(observation)
            times.append(time.perf_counter() - start)
        match.act(actions)
    return {"import_s": round(max(import_s), 4), "decide": _percentiles(times)}


class _NoTelemetry:
    file_path = None

    def __init__(self, *args, **kwargs):
        pass

    def record_map(self, *args, **kwargs):
        pass

    def record_tick(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _run_games(teams_path, matches, ticks, seed, err_path):
    import threading

    import engine.game as game_mod
    from rpc import worker as worker_mod

    game_mod.TelemetryRecorder = _NoTelemetry
    Worker = worker_mod.ModelWorker
    original_get, original_respawn = Worker.get_action, Worker._respawn
    original_spawn, original_ready = Worker._spawn, Worker.wait_ready
    lock = threading.Lock()
    rec = {"times": [], "respawns": 0, "exceptions": 0, "ready_s": [], "idle": 0}
    spawned_at = {}

    def get_action(self, observation, timeout):
        start = time.perf_counter()
        try:
            result = original_get(self, observation, timeout)
        except Exception:
            with lock:
                rec["exceptions"] += 1
            return worker_mod.IDLE_ACTION
        took = time.perf_counter() - start
        with lock:
            rec["times"].append(took)
            if result is worker_mod.IDLE_ACTION:
                rec["idle"] += 1
        return result

    def respawn(self):
        with lock:
            rec["respawns"] += 1
        return original_respawn(self)

    def spawn(self):
        spawned_at[id(self)] = time.perf_counter()
        return original_spawn(self)

    def wait_ready(self, timeout=worker_mod.READY_TIMEOUT_S):
        was_ready = self._ready
        ready = original_ready(self, timeout)
        if ready and not was_ready:
            rec["ready_s"].append(time.perf_counter() - spawned_at.get(id(self), time.perf_counter()))
        return ready

    Worker.get_action, Worker._respawn = get_action, respawn
    Worker._spawn, Worker.wait_ready = spawn, wait_ready

    with open(teams_path) as handle:
        teams = json.load(handle)

    # Workers inherit fd 2; capture it to count decide() errors reported by the model.
    sys.stderr.flush()
    saved_fd = os.dup(2)
    err_file = open(err_path, "w")
    os.dup2(err_file.fileno(), 2)
    construct_s, step_s, scores = [], [], []
    try:
        for i in range(matches):
            start = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                game = game_mod.Game(teams, spheres_per_team=3, seed=seed + i, max_ticks=ticks)
            construct_s.append(time.perf_counter() - start)
            try:
                while not game.is_over():
                    start = time.perf_counter()
                    game.step()
                    step_s.append(time.perf_counter() - start)
                scores.append(dict(game.score))
            finally:
                for client in game.clients.values():
                    client.close()
                game._pool.shutdown()
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        err_file.close()
        Worker.get_action, Worker._respawn = original_get, original_respawn
        Worker._spawn, Worker.wait_ready = original_spawn, original_ready

    err_text = Path(err_path).read_text(errors="replace")
    decide_errors = err_text.count("model.py: decide error")
    return {
        "callback": _percentiles(rec["times"]),
        "timeouts_or_respawns": rec["respawns"],
        "harness_exceptions": rec["exceptions"],
        "idle_results": rec["idle"],
        "decide_errors_reported": decide_errors,
        "worker_stderr_tail": err_text[-400:] if err_text.strip() else "",
        "ready_s": {"max": round(max(rec["ready_s"]), 3) if rec["ready_s"] else None,
                    "mean": round(sum(rec["ready_s"]) / len(rec["ready_s"]), 3)
                    if rec["ready_s"] else None},
        "game_construct_s": [round(t, 3) for t in construct_s],
        "step": _percentiles(step_s),
        "scores": scores,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", help="policy .npz (default: random policy of --hidden)")
    parser.add_argument("--model", help="already exported model.py (skips export)")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--decode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--matches", type=int, default=1)
    parser.add_argument("--ticks", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stress", type=int, default=0, help="number of busy-loop processes")
    parser.add_argument("--inprocess-ticks", type=int, default=1000,
                        help="ticks of the in-process warm decide() benchmark (0 to skip)")
    parser.add_argument("--keep-thread-env", action="store_true")
    parser.add_argument("--workdir", help="scratch parent dir (default: ROOT/runs)")
    args = parser.parse_args(argv)

    if not args.keep_thread_env:
        for var in THREAD_VARS:
            os.environ.pop(var, None)

    scratch = Path(args.workdir) if args.workdir else ROOT / "runs"
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="latency_", dir=scratch))
    try:
        _measure(args, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _measure(args, tmp):
    import multiprocessing as mp

    if args.model:
        model_path = Path(args.model).resolve()
        source = str(model_path)
    else:
        from ctf.export import export

        policy = args.policy
        if policy is None:
            from ctf.policy_io import random_policy

            policy = tmp / "random.npz"
            random_policy(policy, hidden=tuple(args.hidden), seed=args.seed)
        model_path = export(policy, tmp / "model.py", decode=args.decode).resolve()
        source = str(args.policy or f"random{tuple(args.hidden)}")
    teams_path = tmp / "teams.json"
    teams_path.write_text(json.dumps([
        {"name": "red", "model_path": str(model_path), "color": "#ff4d4d"},
        {"name": "blue", "model_path": str(model_path), "color": "#3d9bff"},
    ]))

    summary = {
        "policy": source,
        "model_kib": round(model_path.stat().st_size / 1024, 1),
        "decode": args.decode,
        "stress_procs": args.stress,
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "thread_env_kept": args.keep_thread_env,
    }
    if args.inprocess_ticks > 0:
        summary["inprocess"] = _inprocess(model_path, args.inprocess_ticks, args.seed)

    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    burners = [ctx.Process(target=_burn, args=(stop,), daemon=True) for _ in range(args.stress)]
    for proc in burners:
        proc.start()
    try:
        if burners:
            time.sleep(1.0)
        summary["engine"] = _run_games(teams_path, args.matches, args.ticks, args.seed,
                                       tmp / "worker_stderr.txt")
    finally:
        stop.set()
        for proc in burners:
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
    callback = summary["engine"]["callback"]
    summary["verdict"] = {
        "callback_p99_le_10ms": callback.get("p99_ms", 1e9) <= 10.0,
        "callback_max_le_20ms": callback.get("max_ms", 1e9) <= 20.0,
        "zero_timeouts": summary["engine"]["timeouts_or_respawns"] == 0,
    }
    if "inprocess" in summary:
        summary["verdict"]["decide_p99_le_3ms"] = summary["inprocess"]["decide"]["p99_ms"] <= 3.0
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
