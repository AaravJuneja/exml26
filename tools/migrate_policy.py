"""Migrate a v3 MLP policy file (.npz) to the current v4 feature/action layout.

    python tools/migrate_policy.py OLD.npz NEW.npz
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctf.policy_io import save_policy  # noqa: E402
from ctf.train import migrate_v3_to_v4  # noqa: E402


def main(src, dst):
    with np.load(src, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        layers, i = [], 0
        while f"W{i}" in data:
            layers.append((data[f"W{i}"], data[f"b{i}"]))
            i += 1
        mean, std = data["obs_mean"], data["obs_std"]
    actor = {}
    for k, (w, b) in enumerate(layers):
        actor[f"net.{2 * k}.weight"] = torch.as_tensor(w.T.copy())
        actor[f"net.{2 * k}.bias"] = torch.as_tensor(b.copy())
    state = {"actor": actor, "critic": {},
             "norm": {"mean": mean.astype(np.float64), "var": std.astype(np.float64) ** 2,
                      "count": 1.0}}
    new = migrate_v3_to_v4(state)
    weights = []
    for k in range(len(layers)):
        weights.append((new["actor"][f"net.{2 * k}.weight"].numpy().T.copy(),
                        new["actor"][f"net.{2 * k}.bias"].numpy().copy()))
    save_policy(dst, weights, new["norm"]["mean"], np.sqrt(new["norm"]["var"]),
                migrated_from=str(src), **{k: v for k, v in meta.items()
                                           if k in ("run", "update", "samples")})
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
