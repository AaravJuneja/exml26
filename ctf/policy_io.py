"""Portable policy files: actor weights + feature normalization, numpy only.

A policy file is an `.npz` with `obs_mean`, `obs_std`, a JSON string `meta`
(arch, feature_dim, head_sizes, activation, feature_version, notes) and the
actor weights, by `meta["arch"]`:

* "mlp" (the default; files without an arch are MLPs): arrays W0, b0, W1, b1,
  ... (W_i shaped [n_in, n_out], applied as `h @ W + b`, tanh on every layer
  except the last).
* "gru": arrays W_in, b_in, W_ih, b_ih, W_hh, b_hh, W_out, b_out (see
  `npnets.NumpyGRUActor`), with `meta["gru_hidden"]` the hidden-state size.
"""

import json

import numpy as np

from ctf.features import FEATURE_DIM, N_THROTTLE, N_TURN
from ctf.npnets import GRU_KEYS, HEAD_SIZES, NumpyActor, NumpyGRUActor

FEATURE_VERSION = 1
ARCHS = ("mlp", "gru")


def save_policy(path, weights, obs_mean, obs_std, **meta):
    """`weights`: list of (W, b) for an MLP, or a dict of GRU_KEYS arrays for a GRU."""
    arrays = {}
    if isinstance(weights, dict):
        arch = "gru"
        for key in GRU_KEYS:
            arrays[key] = np.asarray(weights[key], dtype=np.float32)
        meta["gru_hidden"] = int(arrays["W_hh"].shape[0])
    else:
        arch = "mlp"
        for i, (w, b) in enumerate(weights):
            arrays[f"W{i}"] = np.asarray(w, dtype=np.float32)
            arrays[f"b{i}"] = np.asarray(b, dtype=np.float32)
    meta = {
        "arch": arch,
        "feature_dim": FEATURE_DIM,
        "head_sizes": list(HEAD_SIZES),
        "n_throttle": N_THROTTLE,
        "n_turn": N_TURN,
        "activation": "tanh",
        "feature_version": FEATURE_VERSION,
        **meta,
    }
    arrays["obs_mean"] = np.asarray(obs_mean, dtype=np.float32)
    arrays["obs_std"] = np.asarray(obs_std, dtype=np.float32)
    arrays["meta"] = np.array(json.dumps(meta))
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)


def load_policy(path):
    """Return (NumpyActor or NumpyGRUActor, meta dict)."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta["feature_dim"] != FEATURE_DIM or meta["feature_version"] != FEATURE_VERSION:
            raise ValueError(f"policy {path} was built for a different feature encoder")
        meta.setdefault("arch", "mlp")
        mean, std = data["obs_mean"].copy(), data["obs_std"].copy()
        if meta["arch"] == "gru":
            actor = NumpyGRUActor({key: data[key].copy() for key in GRU_KEYS}, mean, std)
        elif meta["arch"] == "mlp":
            weights, i = [], 0
            while f"W{i}" in data:
                weights.append((data[f"W{i}"].copy(), data[f"b{i}"].copy()))
                i += 1
            actor = NumpyActor(weights, mean, std)
        else:
            raise ValueError(f"policy {path} has unknown arch {meta['arch']!r}")
    return actor, meta


def random_policy(path, hidden=(128, 128), seed=0, arch="mlp"):
    """A randomly initialized policy file (for tooling tests)."""
    import torch

    from ctf.nets import actor_to_numpy, build_actor

    torch.manual_seed(seed)
    actor = build_actor(arch, hidden)
    save_policy(path, actor_to_numpy(actor), np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM),
                notes="random init")
