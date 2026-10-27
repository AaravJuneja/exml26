"""Initial state for a bins-space student distilled from a goal-space MLP policy.

    python tools/distill_init.py TEACHER.npz --critic runs/b1/checkpoint.pt --out init.pt

The student keeps the teacher's hidden layers and observation normalization. Input
columns are matched by name: the shared base features and `prev.jump` keep their
weights; the teacher's previous-goal columns are dropped (equivalent to holding them
at their normalized mean, 0); the student's previous throttle/turn columns start with
zero weights and identity normalization. The output layer is new (small orthogonal,
as a fresh actor's), since bins heads are not a linear relabelling of goal heads.
With `--critic`, the critic and return normalization come from that goal-space
checkpoint, migrated the same way. Use the result with `ctf.train --init`; the
observation statistics stay frozen.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctf import features as F  # noqa: E402
from ctf.policy_io import read_policy  # noqa: E402


def _columns(old_names):
    index = {name: i for i, name in enumerate(old_names)}
    return [index.get(name) for name in F.FEATURE_NAMES]


def _migrate_input(weight, sources):
    """weight [n_out, old_dim] -> [n_out, new_dim] (zeros for new columns)."""
    weight = torch.as_tensor(weight)
    new = torch.zeros((weight.shape[0], len(sources)), dtype=weight.dtype)
    for j, i in enumerate(sources):
        if i is not None:
            new[:, j] = weight[:, i]
    return new


def build(teacher_path, critic_path=None, seed=0):
    arch, weights, mean, std, meta = read_policy(teacher_path)
    if arch != "mlp" or meta.get("action_space") != "goal":
        raise ValueError("the teacher must be a goal-space MLP policy")
    old_names = meta["feature_names"]
    sources = _columns(old_names)
    actor = {}
    for i, (w, b) in enumerate(weights[:-1]):
        w = torch.as_tensor(w.T.copy())
        actor[f"net.{2 * i}.weight"] = _migrate_input(w, sources) if i == 0 else w
        actor[f"net.{2 * i}.bias"] = torch.as_tensor(b.copy())
    torch.manual_seed(seed)
    out = torch.nn.Linear(weights[-1][0].shape[0], sum((F.N_THROTTLE, F.N_TURN, 2)))
    torch.nn.init.orthogonal_(out.weight, 0.01)
    torch.nn.init.zeros_(out.bias)
    last = 2 * (len(weights) - 1)
    actor[f"net.{last}.weight"] = out.weight.detach().clone()
    actor[f"net.{last}.bias"] = out.bias.detach().clone()
    new_mean = np.zeros(len(sources))
    new_var = np.ones(len(sources))
    for j, i in enumerate(sources):
        if i is not None:
            new_mean[j], new_var[j] = float(mean[i]), float(std[i]) ** 2
    state = {
        "actor": actor, "norm": {"mean": new_mean, "var": new_var, "count": 1e9},
        "ret_norm": {"mean": np.zeros(1), "var": np.ones(1), "count": 1e-4},
        "update": 0, "samples": 0, "norm_frozen": True, "args": {"arch": "mlp"},
        "feature_names": list(F.FEATURE_NAMES),
        "action_tables": {head: list(v) for head, v in F.ACTION_TABLES.items()},
        "distilled_from": str(teacher_path),
    }
    if critic_path:
        ckpt = torch.load(critic_path, map_location="cpu", weights_only=False)
        if list(ckpt["feature_names"]) != list(old_names):
            raise ValueError("critic checkpoint has another feature layout than the teacher")
        critic = dict(ckpt["critic"])
        critic["net.0.weight"] = _migrate_input(critic["net.0.weight"], sources)
        state["critic"] = critic
        state["ret_norm"] = ckpt["ret_norm"]
        state["norm"]["count"] = ckpt["norm"]["count"]
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("teacher")
    p.add_argument("--critic", default="", help="goal-space checkpoint.pt for the critic")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    state = build(args.teacher, args.critic or None, args.seed)
    if "critic" not in state:
        raise SystemExit("--critic is required (ctf.train --init loads actor and critic)")
    torch.save(state, args.out)
    print(f"wrote {args.out}: actor {[tuple(v.shape) for k, v in state['actor'].items()]}")


if __name__ == "__main__":
    main()
