"""Offline distillation: collect labelled student states, then fit actors to them.

    python tools/distill_offline.py collect STUDENT.npz --teacher policy:/path/b1.npz@dither \
        --matches 40 --workers 3 --out data.npz
    python tools/distill_offline.py fit data.npz --init runs/d2/checkpoint.pt [--hidden 512 512] \
        --epochs 20 --out fitted.pt

`collect` plays full matches (student acting, sampled; a mix of scripted opponents and
mirror games) with the policy teacher of ctf/distill.py labelling every alive
decision, and stores the raw features and the soft targets (float16).

`fit` trains the actor of a checkpoint state (optionally widened with Net2WiderNet)
on the soft cross-entropy alone, holding out the last 10% of matches, and reports the
per-head KL to the targets. `--out` writes the state with the fitted actor (critic,
normalization and the rest unchanged), usable with `ctf.train --init`.
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctf import features as F  # noqa: E402

OPPONENTS = ("relay", "balanced", "guardian", "rusher", "camper", "chaos", "steady", "baseline",
             "mirror")


def _collect_one(args):
    policy, teacher, seed, opponent, side = args
    from ctf.policy_io import load_policy
    from ctf.rollout import collect

    actor = load_policy(policy)[0]
    data = collect(actor, [{"seed": seed, "side": side, "opponent": opponent}], {}, 6000, seed,
                   teacher=teacher)
    alive = data["alive"]
    return (data["obs"][alive].astype(np.float16), data["teacher_soft"][alive].astype(np.float16))


def cmd_collect(args):
    rng = np.random.default_rng(args.seed)
    jobs = [(args.policy, args.teacher, int(rng.integers(200_000, 2**31 - 1)),
             OPPONENTS[i % len(OPPONENTS)], "red" if rng.random() < 0.5 else "blue")
            for i in range(args.matches)]
    started = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(_collect_one, jobs, chunksize=1)
    match = np.concatenate([np.full(len(p[0]), i, dtype=np.int32) for i, p in enumerate(parts)])
    np.savez(args.out, obs=np.concatenate([p[0] for p in parts]),
             soft=np.concatenate([p[1] for p in parts]), match=match)
    print(f"wrote {args.out}: {len(match)} samples from {len(parts)} matches "
          f"in {time.perf_counter() - started:.0f}s")


def cmd_fit(args):
    import torch

    from ctf.nets import build_actor, log_softmax_heads
    from ctf.train import widen_mlp

    device = torch.device(args.device)
    state = torch.load(args.init, map_location="cpu", weights_only=False)
    weights = state["actor"]
    hidden = [t.shape[0] for k, t in sorted(weights.items()) if k.endswith(".weight")][:-1]
    if args.hidden and list(args.hidden) != hidden:
        weights = widen_mlp(weights, args.hidden, np.random.default_rng(0))
        hidden = list(args.hidden)
    sizes = (F.N_THROTTLE, F.N_TURN, 2)
    actor = build_actor("mlp", tuple(hidden), F.FEATURE_DIM, sizes)
    actor.load_state_dict(weights)
    actor.to(device)
    mean = np.asarray(state["norm"]["mean"], dtype=np.float32)
    std = np.maximum(np.sqrt(np.asarray(state["norm"]["var"], dtype=np.float64)), 1e-2)

    data = np.load(args.data)
    match = data["match"]
    val = match >= np.quantile(np.unique(match), 0.9)
    x = np.clip((data["obs"].astype(np.float32) - mean) / std.astype(np.float32), -5.0, 5.0)
    q = data["soft"].astype(np.float32)
    q[:, :F.N_THROTTLE] /= q[:, :F.N_THROTTLE].sum(1, keepdims=True)
    q[:, F.N_THROTTLE:F.N_THROTTLE + F.N_TURN] /= q[:, F.N_THROTTLE:F.N_THROTTLE + F.N_TURN].sum(
        1, keepdims=True)
    q[:, -2:] /= q[:, -2:].sum(1, keepdims=True)
    xt = torch.as_tensor(x[~val], device=device)
    qt = torch.as_tensor(q[~val], device=device)
    xv = torch.as_tensor(x[val], device=device)
    qv = torch.as_tensor(q[val], device=device)
    bounds = ((0, F.N_THROTTLE), (F.N_THROTTLE, F.N_THROTTLE + F.N_TURN),
              (F.N_THROTTLE + F.N_TURN, F.N_THROTTLE + F.N_TURN + 2))

    def entropy(target):
        return -(target * target.clamp(min=1e-12).log()).sum(-1)

    def report(tag):
        with torch.no_grad():
            out = []
            for xs, qs in ((xt[:200_000], qt[:200_000]), (xv, qv)):
                lp = torch.cat([log_softmax_heads(actor(c), sizes) for c in torch.split(xs, 65536)])
                kls = [float(((qs[:, a:b] * lp[:, a:b]).sum(-1).neg() - entropy(qs[:, a:b])).mean())
                       for a, b in bounds]
                out.append(kls)
        print(f"{tag}: train KL {sum(out[0]):.3f} {[round(k, 3) for k in out[0]]} | "
              f"val KL {sum(out[1]):.3f} {[round(k, 3) for k in out[1]]}", flush=True)

    print(f"fit: hidden {hidden}, {len(xt)} train / {len(xv)} val samples", flush=True)
    report("epoch 0")
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr, eps=1e-5)
    n = len(xt)
    steps = args.epochs * ((n + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.05, 1.0 - s / steps))
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, args.batch):
            idx = perm[start:start + args.batch]
            loss = -(qt[idx] * log_softmax_heads(actor(xt[idx]), sizes)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            sched.step()
        if epoch % args.report_every == 0 or epoch == args.epochs:
            report(f"epoch {epoch}")
    if args.out:
        actor.cpu()
        out = dict(state, actor=actor.state_dict())
        out.pop("optimizer", None)
        out["args"] = dict(state.get("args", {}), arch="mlp", hidden=hidden)
        torch.save(out, args.out)
        print(f"wrote {args.out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("policy")
    c.add_argument("--teacher", required=True)
    c.add_argument("--matches", type=int, default=36)
    c.add_argument("--workers", type=int, default=3)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("data")
    f.add_argument("--init", required=True)
    f.add_argument("--hidden", type=int, nargs="*", default=[])
    f.add_argument("--epochs", type=int, default=10)
    f.add_argument("--batch", type=int, default=4096)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--device", default="mps")
    f.add_argument("--report-every", type=int, default=2)
    f.add_argument("--out", default="")
    args = p.parse_args()
    {"collect": cmd_collect, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
