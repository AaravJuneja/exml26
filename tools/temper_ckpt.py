"""Copy a training checkpoint with the MLP actor's logits divided by tau (all heads).

    python tools/temper_ckpt.py IN.pt OUT.pt --tau 0.7

The copied actor samples exactly as the original at temperature tau, so resuming
from it continues PPO on the sharper (deployed) policy. Everything else (critic,
optimizer, counters, RNG) is copied unchanged.
"""

import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--tau", type=float, required=True)
    args = p.parse_args()
    state = torch.load(args.src, map_location="cpu", weights_only=False)
    actor = dict(state["actor"])
    last = max(int(k.split(".")[1]) for k in actor if k.startswith("net."))
    for name in ("weight", "bias"):
        key = f"net.{last}.{name}"
        actor[key] = actor[key] / args.tau
    state["actor"] = actor
    state["actor_temperature"] = args.tau
    torch.save(state, args.dst)
    print(f"wrote {args.dst}: actor logits / {args.tau} (layer net.{last})")


if __name__ == "__main__":
    main()
