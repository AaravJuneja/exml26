"""PPO trainer: parameter-shared actor, decentralized critic, full-match rollouts.

    python -m ctf.train --name m2-idle --opponents idle:1 --workers 10 --envs 3

Each update: every worker plays `envs` complete matches with the current
policy; the learner computes gap-aware GAE (players have no callbacks while
dead, so transitions span the respawn gap with discount gamma**gap), then runs
clipped PPO epochs. Logs go to runs/NAME/log.jsonl, policies to runs/NAME/policies/.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch

from ctf import features as F
from ctf.nets import Actor, Critic, RunningNorm, actor_to_numpy, log_prob_entropy
from ctf.policy_io import save_policy
from ctf.rewards import DEFAULT_WEIGHTS
from ctf.rollout import worker_main

DEFAULT_RUNS = Path(os.environ.get("CTF_RUNS", Path(__file__).resolve().parents[2] / "runs"))
EVAL_SEED_LIMIT = 100_000  # training maps never use seeds below this


def gae(rewards, values, alive, gamma, lam):
    """Advantages over [T, N] arrays; dead ticks fold into the previous decision."""
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    next_v = np.zeros(N, dtype=np.float64)
    next_adv = np.zeros(N, dtype=np.float64)
    pend_r = np.zeros(N, dtype=np.float64)
    pend_k = np.zeros(N, dtype=np.float64)
    for t in range(T - 1, -1, -1):
        a = alive[t]
        r = rewards[t].astype(np.float64)
        disc = gamma ** (pend_k + 1.0)
        delta = r + gamma * pend_r + disc * next_v - values[t]
        advantage = delta + disc * lam * next_adv
        adv[t] = np.where(a, advantage, 0.0)
        next_v = np.where(a, values[t], next_v)
        next_adv = np.where(a, advantage, next_adv)
        pend_r = np.where(a, 0.0, r + gamma * pend_r)
        pend_k = np.where(a, 0.0, pend_k + 1.0)
    return adv


def parse_mix(text):
    mix = []
    for part in text.split(","):
        name, _, weight = part.rpartition(":")
        if not name:
            name, weight = weight, "1"
        mix.append((name, float(weight)))
    total = sum(w for _, w in mix)
    return [(n, w / total) for n, w in mix]


class Trainer:
    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.runs) / args.name
        (self.run_dir / "policies").mkdir(parents=True, exist_ok=True)
        torch.manual_seed(args.seed)
        self.rng = np.random.default_rng(args.seed)
        torch.set_num_threads(args.learner_threads)
        self.device = torch.device(args.device)
        self.actor = Actor(tuple(args.hidden)).to(self.device)
        self.critic = Critic(tuple(args.critic_hidden)).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=args.lr, eps=1e-5)
        self.norm = RunningNorm(F.FEATURE_DIM)
        self.ret_norm = RunningNorm(1)
        self.update = 0
        self.samples = 0
        self.mix = parse_mix(args.opponents)
        self.opp_winrate = {}  # opponent spec -> EMA of learner win rate (PFSP difficulty)
        ckpt = self.run_dir / "checkpoint.pt"
        if ckpt.exists() and not args.fresh:
            self.load(ckpt)
        elif args.init:
            self.load(Path(args.init), weights_only=True)

    # ---------------------------------------------------------- checkpoints
    def state(self):
        return {
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(), "norm": self.norm.state(),
            "ret_norm": self.ret_norm.state(), "update": self.update, "samples": self.samples,
            "opp_winrate": self.opp_winrate,
            "np_rng": self.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "args": vars(self.args),
        }

    def save(self):
        tmp = self.run_dir / "checkpoint.tmp"
        torch.save(self.state(), tmp)
        os.replace(tmp, self.run_dir / "checkpoint.pt")

    def load(self, path, weights_only=False):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.norm.load(state["norm"])
        self.ret_norm.load(state["ret_norm"])
        if not weights_only:
            self.optimizer.load_state_dict(state["optimizer"])
            self.update, self.samples = state["update"], state["samples"]
            self.opp_winrate = state.get("opp_winrate", {})
            if "np_rng" in state:
                self.rng.bit_generator.state = state["np_rng"]
                torch.set_rng_state(state["torch_rng"])

    def export_policy(self, path):
        save_policy(path, actor_to_numpy(self.actor), self.norm.mean, self.norm.std,
                    run=self.args.name, update=self.update, samples=self.samples)

    # ------------------------------------------------------------ schedules
    def progress_fraction(self):
        if not self.args.anneal_samples:
            return 0.0
        return min(1.0, self.samples / self.args.anneal_samples)

    def reward_weights(self):
        weights = dict(DEFAULT_WEIGHTS)
        weights.update(json.loads(self.args.reward_weights))
        frac = self.progress_fraction()
        weights["shaping"] *= (1.0 - frac) + frac * self.args.progress_final
        weights["gamma"] = self.args.gamma
        return weights

    def entropy_coef(self):
        frac = self.progress_fraction()
        return self.args.ent_coef * (1.0 - frac) + self.args.ent_final * frac

    def pool(self):
        """Frozen league opponents: recent own snapshots plus any extra policy globs."""
        own = sorted((self.run_dir / "policies").glob("u*.npz"))
        own = own[-self.args.pool_size:] if self.args.pool_size else []
        extra = []
        for pattern in self.args.pool_extra:
            extra += sorted(glob.glob(pattern))
        return [str(path) for path in own] + extra

    def pick_pool_opponent(self, pool):
        """Prioritized fictitious self-play: weight (1 - winrate)^2, floor 0.05."""
        weights = np.array([(1.0 - self.opp_winrate.get("policy:" + path, 0.5)) ** 2 + 0.05
                            for path in pool])
        return "policy:" + pool[int(self.rng.choice(len(pool), p=weights / weights.sum()))]

    def jobs(self, count):
        names = [n for n, _ in self.mix]
        probs = [p for _, p in self.mix]
        pool = self.pool() if "pool" in names else []
        out = []
        for _ in range(count):
            name = names[int(self.rng.choice(len(names), p=probs))]
            if name == "pool":
                name = self.pick_pool_opponent(pool) if pool else "self"
            out.append({
                "seed": int(self.rng.integers(EVAL_SEED_LIMIT, 2**31 - 1)),
                "side": "red" if self.rng.random() < 0.5 else "blue",
                "opponent": name,
            })
        return out

    def record_results(self, stats):
        for s in stats:
            key = s["opponent"]
            value = 1.0 if s["result"] == "win" else 0.0
            old = self.opp_winrate.get(key, 0.5)
            self.opp_winrate[key] = 0.9 * old + 0.1 * value

    # --------------------------------------------------------------- update
    def popart_rescale(self, old_mean, old_std):
        """Rescale the critic's output layer so raw value predictions are unchanged."""
        new_mean, new_std = float(self.ret_norm.mean[0]), float(self.ret_norm.std[0])
        last = self.critic.net[-1]
        with torch.no_grad():
            last.weight.mul_(old_std / new_std)
            last.bias.mul_(old_std).add_(old_mean - new_mean).div_(new_std)

    def ppo_update(self, batch):
        args = self.args
        T, E = batch["rewards"].shape[:2]
        alive = batch["alive"].reshape(T, E * 3)
        raw = batch["obs"].reshape(T, E * 3, -1)
        mean = torch.as_tensor(self.norm.mean, dtype=torch.float32, device=self.device)
        std = torch.as_tensor(self.norm.std, dtype=torch.float32, device=self.device)

        flat_index = np.nonzero(alive.reshape(-1))[0]
        x_all = torch.as_tensor(raw.reshape(T * E * 3, -1)[flat_index], device=self.device)
        x_all = torch.clamp((x_all - mean) / std, -5.0, 5.0)
        with torch.no_grad():
            v_norm = torch.cat([self.critic(chunk) for chunk in torch.split(x_all, 65536)])
        v_raw = (v_norm.cpu().numpy() * self.ret_norm.std[0] + self.ret_norm.mean[0])
        values = np.zeros(T * E * 3, dtype=np.float64)
        values[flat_index] = v_raw
        values = values.reshape(T, E * 3)
        adv = gae(batch["rewards"].reshape(T, E * 3), values, alive, args.gamma, args.lam)
        returns = (adv + values).reshape(-1)[flat_index]
        adv = adv.reshape(-1)[flat_index]
        explained = 1.0 - np.var(returns - v_raw) / (np.var(returns) + 1e-8)
        old_mean, old_std = float(self.ret_norm.mean[0]), float(self.ret_norm.std[0])
        self.ret_norm.update(returns.reshape(-1, 1))
        self.popart_rescale(old_mean, old_std)
        ret_t = torch.as_tensor((returns - self.ret_norm.mean[0]) / self.ret_norm.std[0],
                                dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8),
                                dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(batch["actions"].reshape(T * E * 3, 3)[flat_index].astype(np.int64),
                                device=self.device)
        old_logp = torch.as_tensor(batch["logp"].reshape(-1)[flat_index], device=self.device)

        n = len(flat_index)
        ent_coef = self.entropy_coef()
        stats = {"pi_loss": [], "v_loss": [], "entropy": [], "kl": [], "clipfrac": []}
        stop = False
        for _epoch in range(args.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, args.minibatch):
                idx = perm[start:start + args.minibatch]
                logits = self.actor(x_all[idx])
                logp, entropy = log_prob_entropy(logits, act_t[idx])
                ratio = torch.exp(logp - old_logp[idx])
                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - old_logp[idx])).mean().item()
                if args.target_kl and kl > 1.5 * args.target_kl:
                    stop = True
                    break
                surrogate = torch.min(ratio * adv_t[idx],
                                      torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_t[idx])
                pi_loss = -surrogate.mean()
                v_loss = 0.5 * (self.critic(x_all[idx]) - ret_t[idx]).pow(2).mean()
                loss = pi_loss + args.vf_coef * v_loss - ent_coef * entropy.mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), args.max_grad)
                self.optimizer.step()
                stats["pi_loss"].append(pi_loss.item())
                stats["v_loss"].append(v_loss.item())
                stats["entropy"].append(entropy.mean().item())
                stats["kl"].append(kl)
                stats["clipfrac"].append(((ratio - 1).abs() > args.clip).float().mean().item())
            if stop:
                break
        if self.update < args.norm_updates:
            self.norm.update(raw.reshape(T * E * 3, -1)[flat_index][:: args.norm_stride])
        out = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
        out.update({"explained_var": float(explained), "samples": int(n),
                    "minibatches": len(stats["kl"]), "ent_coef": ent_coef,
                    "adv_raw_std": float(adv.std())})
        return out

    # ------------------------------------------------------------------ run
    def run(self):
        args = self.args
        ctx = mp.get_context("spawn")
        pipes, procs = [], []
        for i in range(args.workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=worker_main, args=(child, i), daemon=True)
            proc.start()
            child.close()
            pipes.append(parent)
            procs.append(proc)
        log_path = self.run_dir / "log.jsonl"
        try:
            while self.update < args.updates:
                t0 = time.perf_counter()
                weights = actor_to_numpy(self.actor)
                mean, std = self.norm.mean.astype(np.float32), self.norm.std.astype(np.float32)
                config = {"reward_weights": self.reward_weights(), "max_ticks": args.max_ticks,
                          "memory_wipe_prob": args.memory_wipe}
                for pipe in pipes:
                    config = dict(config, seed=int(self.rng.integers(2**31)))
                    pipe.send((weights, mean, std, self.jobs(args.envs), config))
                results = [self.receive(pipe, proc) for pipe, proc in zip(pipes, procs)]
                t1 = time.perf_counter()
                batch = {key: np.concatenate([r[key] for r in results], axis=1)
                         for key in ("obs", "actions", "logp", "rewards", "alive")}
                stats = [s for r in results for s in r["stats"]]
                self.record_results(stats)
                learn = self.ppo_update(batch)
                t2 = time.perf_counter()
                self.update += 1
                self.samples += learn["samples"]
                record = self.summarize(stats, learn, t1 - t0, t2 - t1,
                                        sum(r["ticks"] for r in results))
                with open(log_path, "a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps({k: record[k] for k in (
                    "update", "samples_total", "win_rate", "goals_for", "goals_against",
                    "wall_deaths_pp", "pickups", "entropy", "kl", "explained_var",
                    "rollout_s", "update_s", "ticks_per_s")}), flush=True)
                if self.update % args.save_every == 0 or self.update == args.updates:
                    self.save()
                if self.update % args.snapshot_every == 0 or self.update == args.updates:
                    self.export_policy(self.run_dir / "policies" / f"u{self.update:05d}.npz")
        finally:
            for pipe in pipes:
                try:
                    pipe.send(None)
                except (BrokenPipeError, OSError):
                    pass
            for proc in procs:
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.terminate()

    def receive(self, pipe, proc):
        """Wait for a worker's rollout; fail loudly if it died or reported an error."""
        deadline = time.monotonic() + self.args.worker_timeout
        while not pipe.poll(5.0):
            if not proc.is_alive():
                raise RuntimeError(f"rollout worker {proc.pid} died (exit {proc.exitcode})")
            if time.monotonic() > deadline:
                raise TimeoutError(f"rollout worker {proc.pid} timed out")
        result = pipe.recv()
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"rollout worker failed:\n{result['error']}")
        return result

    def summarize(self, stats, learn, rollout_s, update_s, ticks):
        n = len(stats)
        record = {
            "update": self.update, "samples_total": self.samples, "time": time.time(),
            "matches": n,
            "win_rate": sum(s["result"] == "win" for s in stats) / n,
            "draw_rate": sum(s["result"] == "draw" for s in stats) / n,
            "goals_for": sum(s["goals_for"] for s in stats) / n,
            "goals_against": sum(s["goals_against"] for s in stats) / n,
            "pickups": sum(s["pickups"] for s in stats) / n,
            "wall_deaths_pp": sum(s["wall_deaths"] for s in stats) / (3 * n),
            "player_deaths_pp": sum(s["player_deaths"] for s in stats) / (3 * n),
            "enemy_deaths": sum(s["enemy_deaths"] for s in stats) / n,
            "carrier_deaths": sum(s["carrier_deaths"] for s in stats) / n,
            "flag_returns": sum(s["flag_returns"] for s in stats) / n,
            "reward": {k: sum(s["reward"][k] for s in stats) / n for k in stats[0]["reward"]},
            "rollout_s": round(rollout_s, 2), "update_s": round(update_s, 2),
            "ticks_per_s": round(ticks / rollout_s),
            "by_opponent": {},
            **learn,
        }
        def group(name):
            return "pool" if name.startswith("policy:") else name

        for name in sorted({group(s["opponent"]) for s in stats}):
            subset = [s for s in stats if group(s["opponent"]) == name]
            record["by_opponent"][name] = {
                "n": len(subset),
                "win_rate": sum(s["result"] == "win" for s in subset) / len(subset),
                "goal_diff": sum(s["goals_for"] - s["goals_against"] for s in subset) / len(subset),
            }
        return record


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--runs", default=str(DEFAULT_RUNS))
    p.add_argument("--opponents", default="idle:1")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--envs", type=int, default=3, help="matches per worker per update")
    p.add_argument("--max-ticks", type=int, default=6000)
    p.add_argument("--updates", type=int, default=10**9)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--critic-hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.998)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--ent-final", type=float, default=0.003)
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--max-grad", type=float, default=0.5)
    p.add_argument("--anneal-samples", type=float, default=0.0)
    p.add_argument("--progress-final", type=float, default=0.2)
    p.add_argument("--reward-weights", default="{}")
    p.add_argument("--memory-wipe", type=float, default=0.02)
    p.add_argument("--norm-stride", type=int, default=7)
    p.add_argument("--norm-updates", type=int, default=20,
                   help="observation statistics are frozen after this many updates")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--learner-threads", type=int, default=8)
    p.add_argument("--worker-timeout", type=float, default=1800.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--init", default="")
    p.add_argument("--pool-size", type=int, default=20, help="own snapshots kept in the league")
    p.add_argument("--pool-extra", nargs="*", default=[], help="globs of extra league policies")
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args(argv)
    Trainer(args).run()


if __name__ == "__main__":
    main()
