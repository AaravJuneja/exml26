"""Histogram of the throttle/turn a goal-space policy's controller sends, and its bins rounding.

    python tools/distill_probe.py runs/b1/policies/u00970.npz --opponents relay guardian \
        --seeds 3100-3103 [--sample]

Plays the policy (seeded sampling unless --greedy) and reports, over all alive decisions,
the distribution of the controller's throttle and turn, how far they are from the nearest
bins value, and the most common rounding errors.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctf import features as F  # noqa: E402
from ctf.distill import snap_action  # noqa: E402
from ctf.evaluate import parse_seeds  # noqa: E402
from ctf.policy_io import load_policy  # noqa: E402
from ctf.rollout import PolicyTeam, make_opponent  # noqa: E402
from ctf.sim import Match  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("policy")
    p.add_argument("--opponents", nargs="+", default=["relay", "guardian", "rusher"])
    p.add_argument("--seeds", default="3100-3101")
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()
    actor = load_policy(args.policy)[0]
    throttle, turn = [], []
    for opponent in args.opponents:
        for seed in parse_seeds(args.seeds):
            match = Match(seed, max_ticks=6000, teammates_field=True)
            red, blue = match.team_names
            rng = np.random.default_rng(seed)
            mine = PolicyTeam(actor, [s.id for s in match.spheres if s.team == red], rng,
                              greedy=args.greedy)
            theirs = make_opponent(opponent, [s.id for s in match.spheres if s.team == blue], rng)
            while not match.is_over():
                obs = match.observe()
                acts = mine.act(obs)
                for t, r, _ in acts.values():
                    throttle.append(t)
                    turn.append(r)
                acts.update(theirs.act(obs))
                match.act(acts)
    throttle, turn = np.array(throttle), np.array(turn)
    snapped = np.array([snap_action(t, r) for t, r in zip(throttle, turn)])
    for name, raw, snap, table in (("throttle", throttle, snapped[:, 0], F.THROTTLE_VALUES),
                                   ("turn", turn, snapped[:, 1], F.TURN_VALUES)):
        err = np.abs(raw - snap)
        print(f"{name}: n={len(raw)} |err| mean {err.mean():.4f} p90 {np.quantile(err, 0.9):.4f} "
              f"p99 {np.quantile(err, 0.99):.4f}; exact-ish (<1e-3) {np.mean(err < 1e-3):.3f}")
        print("  quantiles:", np.round(np.quantile(raw, np.linspace(0, 1, 21)), 4).tolist())
        counts = Counter(np.round(snap, 4).tolist())
        print("  snapped bins:", {v: round(counts.get(v, 0) / len(raw), 4) for v in table})
        edges = np.array([-1.0001, -0.9, -0.6, -0.4, -0.2, -0.12, -0.07, -0.03, -0.01, -1e-6, 1e-6,
                          0.01, 0.02, 0.03, 0.04, 0.055, 0.08, 0.12, 0.2, 0.4, 0.6, 0.9, 0.9999,
                          1.0001])
        hist, _ = np.histogram(raw, edges)
        print("  histogram:", [(f"{edges[i]:.3g}..{edges[i + 1]:.3g}", round(h / len(raw), 4))
                               for i, h in enumerate(hist) if h])


if __name__ == "__main__":
    main()
