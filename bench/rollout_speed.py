"""Rollout (`ctf.rollout.collect`) throughput benchmark and profiler.

    python -m bench.rollout_speed --policy PATH.npz --ticks 2000 --repeat 3
    python -m bench.rollout_speed --policy PATH.npz --profile 40      # cProfile, top 40
    python -m bench.rollout_speed --digest                            # output fingerprint
    python -m bench.rollout_speed --policy PATH.npz --breakdown       # time per stage

Without --policy a random tanh MLP (128, 128) is used with a forward-throttle bias,
so players drive around like a trained policy instead of jittering in place.
"""

import argparse
import cProfile
import hashlib
import pstats
import time

import numpy as np

from ctf import features as F
from ctf import rollout
from ctf.npnets import NumpyActor
from ctf.policy_io import load_policy

DEFAULT_JOBS = "relay,baseline,mirror"


def random_actor(seed=0, hidden=(128, 128)):
    rng = np.random.default_rng(seed)
    sizes = (F.FEATURE_DIM, *hidden, F.N_THROTTLE + F.N_TURN + 2)
    weights = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        w = rng.normal(0.0, 1.0 / np.sqrt(n_in), (n_in, n_out)).astype(np.float32)
        weights.append((w, np.zeros(n_out, dtype=np.float32)))
    bias = weights[-1][1]
    bias[F.THROTTLE_VALUES.index(1.0)] = 2.0
    bias[F.THROTTLE_VALUES.index(0.3)] = 1.0
    bias[F.N_THROTTLE + F.N_TURN] = 2.0  # mostly no jump
    return NumpyActor(weights, np.zeros(F.FEATURE_DIM), np.ones(F.FEATURE_DIM))


def make_jobs(names, seed=1000):
    sides = ("red", "blue")
    return [{"seed": seed + i, "side": sides[i % 2], "opponent": name}
            for i, name in enumerate(names.split(","))]


def digest(out):
    h = hashlib.sha256()
    for key in ("obs", "actions", "logp", "rewards", "alive"):
        h.update(np.ascontiguousarray(out[key]).tobytes())
    h.update(repr(out["stats"]).encode())
    return h.hexdigest()


def install_timers():
    """Wrap the hot-path stages with wall-clock accumulators; returns the totals dict."""
    from ctf import npnets, policies, rewards, sim
    totals = {}

    def wrap(owner, name, label):
        inner = getattr(owner, name)

        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return inner(*args, **kwargs)
            finally:
                totals[label] = totals.get(label, 0.0) + time.perf_counter() - start
        setattr(owner, name, timed)

    wrap(sim.Match, "observe", "Match.observe")
    wrap(sim.Match, "act", "Match.act (physics)")
    wrap(F, "encode", "features.encode")
    wrap(policies.ScriptedTeam, "act", "scripted opponents")
    for cls in (npnets.NumpyActor, npnets.NumpyGRUActor):
        wrap(cls, "logits" if cls is npnets.NumpyActor else "step", "network forward")
    wrap(npnets, "sample_logits", "sampling")
    wrap(rewards.RewardTracker, "before", "rewards")
    wrap(rewards.RewardTracker, "after", "rewards")
    return totals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy")
    p.add_argument("--jobs", default=DEFAULT_JOBS)
    p.add_argument("--ticks", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--wipe", type=float, default=0.02)
    p.add_argument("--profile", type=int, default=0)
    p.add_argument("--sort", default="tottime")
    p.add_argument("--digest", action="store_true")
    p.add_argument("--breakdown", action="store_true", help="wall-clock time per stage")
    args = p.parse_args()

    actor = load_policy(args.policy)[0] if args.policy else random_actor()
    jobs = make_jobs(args.jobs)
    weights = {}

    def run():
        return rollout.collect(actor, jobs, weights, args.ticks, args.seed, args.wipe)

    if args.breakdown:
        totals = install_timers()
        start = time.perf_counter()
        run()
        total = time.perf_counter() - start
        totals["other (collect loop, setup)"] = total - sum(totals.values())
        for label, seconds in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"{label:30s} {seconds:7.3f}s {100 * seconds / total:5.1f}%")
        print(f"{'total':30s} {total:7.3f}s")
        return
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        out = run()
        profiler.disable()
        print(f"{out['ticks']} ticks in {out['seconds']:.2f}s (profiled)")
        pstats.Stats(profiler).sort_stats(args.sort).print_stats(args.profile)
        return
    times, cpu = [], []
    for _ in range(args.repeat):
        start, start_cpu = time.perf_counter(), time.process_time()
        out = run()
        times.append(time.perf_counter() - start)
        cpu.append(time.process_time() - start_cpu)
    ticks = out["ticks"]
    print(f"jobs={args.jobs} ticks={args.ticks} best {min(times):.3f}s "
          f"({ticks / min(times):.0f} match-ticks/s) all={[round(t, 3) for t in times]} "
          f"cpu={[round(t, 3) for t in cpu]}")
    if args.digest:
        print("digest", digest(out))


if __name__ == "__main__":
    main()
