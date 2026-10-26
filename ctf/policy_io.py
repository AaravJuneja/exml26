"""Portable policy files: actor weights + feature normalization, numpy only.

A policy file is an `.npz` with `obs_mean`, `obs_std`, a JSON string `meta`
(arch, feature_dim, head_sizes, activation, feature_version, feature_names,
action_tables, notes) and the actor weights, by `meta["arch"]`:

* "mlp" (the default; files without an arch are MLPs): arrays W0, b0, W1, b1,
  ... (W_i shaped [n_in, n_out], applied as `h @ W + b`, tanh on every layer
  except the last).
* "gru": arrays W_in, b_in, W_ih, b_ih, W_hh, b_hh, W_out, b_out (see
  `npnets.NumpyGRUActor`), with `meta["gru_hidden"]` the hidden-state size.

Files written before feature version 5 have no feature names; their layout is
reconstructed from ctf/feature_layouts.py. An MLP file built for another layout
can be migrated explicitly (`load_policy(..., migrate=True)` or tools/migrate_policy.py);
by default a different layout is an error, so old snapshots are never silently altered.
"""

import json

import numpy as np

from ctf import feature_layouts as L
from ctf.features import (ACTION_TABLES, FEATURE_DIM, FEATURE_NAMES, FEATURE_VERSION, N_THROTTLE,
                          N_TURN)
from ctf.npnets import GRU_KEYS, HEAD_SIZES, NumpyActor, NumpyGRUActor

ARCHS = ("mlp", "gru")
CURRENT_LAYOUT = (FEATURE_VERSION, FEATURE_NAMES, ACTION_TABLES)


def layout_meta():
    """Meta entries describing the current feature encoder and action tables."""
    return {
        "feature_dim": FEATURE_DIM,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "action_tables": {head: list(values) for head, values in ACTION_TABLES.items()},
        "head_sizes": list(HEAD_SIZES),
        "n_throttle": N_THROTTLE,
        "n_turn": N_TURN,
    }


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
    meta = {"arch": arch, **layout_meta(), "activation": "tanh", **meta}
    arrays["obs_mean"] = np.asarray(obs_mean, dtype=np.float32)
    arrays["obs_std"] = np.asarray(obs_std, dtype=np.float32)
    arrays["meta"] = np.array(json.dumps(meta))
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)


def policy_layout(meta):
    """(feature names, action tables) a policy file was built for."""
    return L.resolve_layout(meta.get("feature_names"), meta.get("action_tables"),
                            meta.get("feature_version"), meta.get("feature_dim"),
                            current=CURRENT_LAYOUT)


def read_policy(path):
    """Raw contents without any migration: (arch, weights, mean, std, meta)."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        meta.setdefault("arch", "mlp")
        mean, std = data["obs_mean"].copy(), data["obs_std"].copy()
        if meta["arch"] == "gru":
            weights = {key: data[key].copy() for key in GRU_KEYS}
        elif meta["arch"] == "mlp":
            weights, i = [], 0
            while f"W{i}" in data:
                weights.append((data[f"W{i}"].copy(), data[f"b{i}"].copy()))
                i += 1
        else:
            raise ValueError(f"policy {path} has unknown arch {meta['arch']!r}")
    return meta["arch"], weights, mean, std, meta


def migrate_policy_arrays(weights, mean, std, meta):
    """Bring MLP `weights`, `mean`, `std` built for `meta`'s layout to the current one.

    Returns (weights, mean, std, meta); unchanged if the layout is already current.
    """
    names, tables = policy_layout(meta)
    if len(mean) != len(names):
        raise ValueError(f"normalization has {len(mean)} features, layout has {len(names)}")
    if L.same_layout(names, tables, FEATURE_NAMES, ACTION_TABLES):
        return weights, mean, std, meta
    if meta.get("arch", "mlp") != "mlp":
        raise ValueError("only MLP policies can be migrated to another feature layout")
    weights, mean, std = L.migrate_mlp(weights, mean, std, names, tables,
                                       FEATURE_NAMES, ACTION_TABLES)
    meta = dict(meta, **layout_meta(), migrated_from={
        "feature_version": meta.get("feature_version"), "feature_dim": len(names)})
    return weights, mean, std, meta


def load_policy(path, migrate=False):
    """Return (NumpyActor or NumpyGRUActor, meta dict).

    With `migrate`, an MLP built for an older feature layout is migrated in memory
    (meta gains `migrated_from`); otherwise a different layout is an error.
    """
    arch, weights, mean, std, meta = read_policy(path)
    names, tables = policy_layout(meta)
    if not L.same_layout(names, tables, FEATURE_NAMES, ACTION_TABLES):
        if not migrate or arch != "mlp":
            raise ValueError(f"policy {path} was built for a different feature encoder")
        weights, mean, std, meta = migrate_policy_arrays(weights, mean, std, meta)
    if mean.shape != (FEATURE_DIM,):
        raise ValueError(f"policy {path} normalization does not match the feature encoder")
    if arch == "gru":
        actor = NumpyGRUActor(weights, mean, std)
    else:
        actor = NumpyActor(weights, mean, std)
    return actor, meta


def random_policy(path, hidden=(128, 128), seed=0, arch="mlp"):
    """A randomly initialized policy file (for tooling tests)."""
    import torch

    from ctf.nets import actor_to_numpy, build_actor

    torch.manual_seed(seed)
    actor = build_actor(arch, hidden)
    save_policy(path, actor_to_numpy(actor), np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM),
                notes="random init")
