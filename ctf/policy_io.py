"""Portable policy files: actor weights + feature normalization, numpy only.

A policy file is an `.npz` with arrays W0, b0, W1, b1, ... (W_i shaped
[n_in, n_out], applied as `h @ W + b`, tanh on every layer except the last),
`obs_mean`, `obs_std`, and a JSON string `meta` (feature_dim, head_sizes,
activation, feature_version, notes).
"""

import json

import numpy as np

from ctf.features import FEATURE_DIM, N_THROTTLE, N_TURN
from ctf.nets import HEAD_SIZES, NumpyActor

FEATURE_VERSION = 1


def save_policy(path, weights, obs_mean, obs_std, **meta):
    arrays = {}
    for i, (w, b) in enumerate(weights):
        arrays[f"W{i}"] = np.asarray(w, dtype=np.float32)
        arrays[f"b{i}"] = np.asarray(b, dtype=np.float32)
    meta = {
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
    """Return (NumpyActor, meta dict)."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if meta["feature_dim"] != FEATURE_DIM or meta["feature_version"] != FEATURE_VERSION:
            raise ValueError(f"policy {path} was built for a different feature encoder")
        weights, i = [], 0
        while f"W{i}" in data:
            weights.append((data[f"W{i}"].copy(), data[f"b{i}"].copy()))
            i += 1
        actor = NumpyActor(weights, data["obs_mean"].copy(), data["obs_std"].copy())
    return actor, meta


def random_policy(path, hidden=(128, 128), seed=0):
    """A randomly initialized policy file (for tooling tests)."""
    import torch

    from ctf.nets import Actor, actor_to_numpy

    torch.manual_seed(seed)
    actor = Actor(hidden)
    save_policy(path, actor_to_numpy(actor), np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM),
                notes="random init")
