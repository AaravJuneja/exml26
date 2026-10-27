"""Average the weights of several MLP policy snapshots (one run, same layout) into one.

    python tools/average_policies.py OUT.npz runs/b1/policies/u014[0-8]0.npz

Uniform weight averaging of nearby snapshots of one training run (as in stochastic
weight averaging). All inputs must share the architecture, the feature/action layout
and the observation normalisation (which is frozen early in training).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctf.policy_io import read_policy, save_policy  # noqa: E402

LAYOUT_KEYS = ("arch", "action_space", "feature_dim", "feature_version", "feature_names", "action_tables",
               "head_sizes", "activation")


def average(paths):
    first = None
    sums = None
    for path in paths:
        arch, weights, mean, std, meta = read_policy(path)
        if arch != "mlp":
            raise ValueError(f"{path}: only MLP policies can be averaged")
        if first is None:
            first = (weights, mean, std, meta)
            sums = [(w.astype(np.float64), b.astype(np.float64)) for w, b in weights]
            continue
        _, mean0, std0, meta0 = first
        for key in LAYOUT_KEYS:
            if meta.get(key) != meta0.get(key):
                raise ValueError(f"{path}: {key} differs from {paths[0]}")
        if not (np.array_equal(mean, mean0) and np.array_equal(std, std0)):
            raise ValueError(f"{path}: observation normalisation differs from {paths[0]}")
        if [w.shape for w, _ in weights] != [w.shape for w, _ in first[0]]:
            raise ValueError(f"{path}: layer shapes differ from {paths[0]}")
        sums = [(sw + w, sb + b) for (sw, sb), (w, b) in zip(sums, weights)]
    n = len(paths)
    averaged = [((sw / n).astype(np.float32), (sb / n).astype(np.float32)) for sw, sb in sums]
    return averaged, first[1], first[2], first[3]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out")
    p.add_argument("inputs", nargs="+")
    args = p.parse_args(argv)
    if len(args.inputs) < 2:
        p.error("need at least two snapshots")
    weights, mean, std, meta = average(args.inputs)
    space = meta.get("action_space", "bins")
    extra = {"run": meta.get("run"), "averaged_from": [Path(x).name for x in args.inputs]}
    save_policy(args.out, weights, mean, std, space=space, **extra)
    print(json.dumps({"out": args.out, "n": len(args.inputs)}))


if __name__ == "__main__":
    main()
