"""Copy an MLP policy with its output logits divided by per-head temperatures.

    python tools/temper.py IN.npz OUT.npz --tau 0.7            # all heads
    python tools/temper.py IN.npz OUT.npz --tau 1,0.5,1        # per head (heading, speed, jump)

Sampling the copy is exactly sampling the original at temperature tau (the last
layer's weights and bias are scaled by 1/tau), so the deployed model.py needs no
code change; greedy decoding is unchanged.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctf import action_space as A  # noqa: E402
from ctf.policy_io import read_policy, save_policy  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--tau", required=True)
    args = p.parse_args()
    arch, weights, mean, std, meta = read_policy(args.src)
    if arch != "mlp":
        raise SystemExit("MLP policies only")
    space = A.get(meta.get("action_space", "bins"))
    taus = [float(t) for t in args.tau.split(",")]
    if len(taus) == 1:
        taus = taus * len(space.head_sizes)
    scale = np.concatenate([np.full(s, 1.0 / t, np.float32) for s, t in zip(space.head_sizes, taus)])
    w, b = weights[-1]
    weights = list(weights[:-1]) + [(w * scale[None, :], b * scale)]
    extra = {k: v for k, v in meta.items() if k in ("run", "update", "samples")}
    save_policy(args.dst, weights, mean, std, space=space.name,
                temperature=json.dumps(taus), source=str(args.src), **extra)
    print(f"wrote {args.dst} (tau {taus})")


if __name__ == "__main__":
    main()
