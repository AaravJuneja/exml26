"""PPO trainer: parameter-shared actor, decentralized critic, full-match rollouts.

    python -m ctf.train --name m2-idle --opponents idle:1 --workers 10 --envs 3

Each update: every worker plays `envs` complete matches with the current
policy; the learner computes gap-aware GAE (players have no callbacks while
dead, so transitions span the respawn gap with discount gamma**gap), then runs
clipped PPO epochs. Logs go to runs/NAME/log.jsonl, policies to runs/NAME/policies/.

`--arch gru` swaps the MLP actor for GRUActor (the critic stays a feed-forward
MLP over the same features). GAE is unchanged; the policy loss is computed
over per-player sequences of alive decisions, cut into `--chunk`-step chunks
that start from hidden states stored by the rollout, optionally re-warmed with
`--burn-in` steps of the previous chunk under the current weights.
"""

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch

from ctf import feature_layouts as L
from ctf import features as F
from ctf.nets import Critic, RunningNorm, actor_to_numpy, build_actor, log_prob_entropy
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


def hidden_every(chunk, burn_in):
    """Stride (in decisions since a reset) at which rollouts store hidden states:
    every chunk start and every burn-in start must be a multiple of it."""
    return math.gcd(chunk, burn_in) if burn_in else chunk


def build_sequences(alive, resets, h0_key, chunk, burn_in):
    """Cut every player's alive decisions into chunks for recurrent PPO.

    alive, resets: [T, N] bool (N = units * 3 player columns); h0_key: [K] grid
    keys `t * N + n` of the stored hidden states. Positions refer to the
    flattened alive decisions in row-major (t, n) order, i.e. to
    `np.flatnonzero(alive.reshape(-1))`, the order the trainer uses.

    A chunk starts at every decision whose index since the last reset is a
    multiple of `chunk`, so no chunk contains a reset. Returns a dict:
      main [C, chunk]      positions of the chunk's decisions, -1 padded
      burn [C, burn_in]    positions of the `burn_in` decisions before the chunk
      has_burn [C]         chunk does not start a segment (and burn_in > 0)
      h0_index [C]         index into h0_key of the initial state: the state
                           before the first burn-in decision, else before the
                           chunk's first decision
    """
    T, N = alive.shape
    flat_index = np.flatnonzero(alive.reshape(-1))
    t_of, n_of = np.divmod(flat_index, N)
    order = np.lexsort((t_of, n_of))  # by player, then time
    reset = resets.reshape(-1)[flat_index][order]
    column = n_of[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = column[1:] != column[:-1]
    if np.any(first & ~reset):
        raise ValueError("a player's first decision must start a fresh hidden state")
    seg_start = np.flatnonzero(reset)
    count = np.arange(len(order)) - seg_start[np.cumsum(reset) - 1]
    starts = np.flatnonzero(count % chunk == 0)
    lengths = np.diff(np.append(starts, len(order)))
    offsets = np.arange(chunk)
    valid = offsets[None, :] < lengths[:, None]
    main_order = np.minimum(starts[:, None] + offsets, len(order) - 1)
    main = np.where(valid, order[main_order], -1)
    has_burn = (count[starts] > 0) & (burn_in > 0)
    burn_order = np.where(has_burn[:, None], starts[:, None] - burn_in + np.arange(burn_in), 0)
    burn = order[burn_order]
    init_key = flat_index[order[np.where(has_burn, starts - burn_in, starts)]]
    h0_index = np.minimum(np.searchsorted(h0_key, init_key), max(len(h0_key) - 1, 0))
    if len(starts) and (len(h0_key) == 0 or np.any(h0_key[h0_index] != init_key)):
        raise ValueError("rollout did not store the hidden state at a chunk start "
                         "(hidden_every must divide --chunk and --burn-in)")
    return {"main": main, "burn": burn, "has_burn": has_burn, "h0_index": h0_index}


def sequence_logits(actor, x_all, h0, seqs, chunks):
    """Logits of the valid decisions of `chunks` (index array into seqs) and their positions.

    x_all: normalized features [n, D] in position order; h0: stored states [K, H].
    Burn-in runs without gradient from the stored state before the burn-in steps.
    """
    main = seqs["main"][chunks]
    h = h0[seqs["h0_index"][chunks]]
    has_burn = seqs["has_burn"][chunks]
    if has_burn.any():
        h = h.clone()
        h[has_burn] = actor.final_state(x_all[seqs["burn"][chunks][has_burn]], h[has_burn])
    valid = main >= 0
    logits, _ = actor(x_all[main.clamp(min=0)], h)
    return logits[valid], main[valid]


def merge_results(results, recurrent):
    """Concatenate worker rollouts along the unit axis (stored hidden states re-keyed)."""
    keys = ("obs", "actions", "logp", "rewards", "alive") + (("resets",) if recurrent else ())
    if "teacher_actions" in results[0]:
        keys += ("teacher_actions", "teacher_valid")
    batch = {key: np.concatenate([r[key] for r in results], axis=1) for key in keys}
    if recurrent:
        N = batch["alive"].shape[1] * 3
        keys, offset = [], 0
        for r in results:
            keys.append(r["h0_t"] * N + (r["h0_unit"] + offset) * 3 + r["h0_slot"])
            offset += r["alive"].shape[1]
        h0_key = np.concatenate(keys)
        order = np.argsort(h0_key, kind="stable")
        batch["h0_key"] = h0_key[order]
        batch["h0"] = np.concatenate([r["h0"] for r in results])[order]
    return batch


def widen_mlp(weights, new_hidden, rng, noise=1e-4):
    """Net2WiderNet for a Sequential of Linear/Tanh layers stored as net.{i}.weight/bias.

    Each new hidden unit copies a random existing unit's incoming weights; the
    outgoing weights of every copied unit are divided by its number of copies, so
    the network computes the same function (up to `noise` added to break symmetry).
    """
    linear = sorted({int(k.split(".")[1]) for k in weights if k.startswith("net.")})
    out = dict(weights)
    for layer, target in zip(linear[:-1], new_hidden):
        w_in, b_in = out[f"net.{layer}.weight"], out[f"net.{layer}.bias"]
        nxt = linear[linear.index(layer) + 1]
        w_out = out[f"net.{nxt}.weight"]
        old = w_in.shape[0]
        if target < old:
            raise ValueError(f"cannot shrink layer {layer} from {old} to {target}")
        if target == old:
            continue
        mapping = torch.cat([torch.arange(old),
                             torch.as_tensor(rng.integers(0, old, target - old))]).to(w_in.device)
        counts = torch.bincount(mapping, minlength=old).to(dtype=w_out.dtype, device=w_out.device)
        new_w_in = w_in[mapping].clone()
        new_w_in[old:] += noise * torch.randn_like(new_w_in[old:])
        out[f"net.{layer}.weight"] = new_w_in
        out[f"net.{layer}.bias"] = b_in[mapping].clone()
        out[f"net.{nxt}.weight"] = (w_out / counts)[:, mapping].clone()
    return out


V3_THROTTLE = L.V3_ACTION_TABLES["throttle"]
V3_DIM, V3_BASE = 180, 158  # v3: 158 base features (incl. rays) then 22 previous-action
NORM_CLIP_STD = RunningNorm(1).clip_std


def state_layout(state):
    """(feature names, action tables) of a checkpoint; unnamed ones (v3, v4) by width."""
    return L.resolve_layout(state.get("feature_names"), state.get("action_tables"),
                            dim=len(state["norm"]["mean"]))


def _input_layer(weights, width):
    """Key of the first matrix that reads `width` features (the input layer), or None."""
    for name, tensor in weights.items():
        if name.endswith(".weight") and tensor.dim() == 2 and tensor.shape[1] == width:
            return name
    return None


def migrate_state(state, old_names, old_tables, new_names=None, new_tables=None,
                  allow_lossy=False):
    """Warm-start an MLP checkpoint state built for another feature layout / action tables.

    Input columns are matched by name (new columns: zero weights, identity
    normalization; new previous-action values inherit the nearest old value of their
    head) in the actor's and the critic's input layers; the actor's output rows map to
    the nearest old action value per head with -log(copies) on duplicated rows. See
    ctf/feature_layouts.py. Returns a new state labelled with the new layout.
    """
    new_names = F.FEATURE_NAMES if new_names is None else tuple(new_names)
    new_tables = L.normalize_tables(F.ACTION_TABLES if new_tables is None else new_tables)
    old_names, old_tables = tuple(old_names), L.normalize_tables(old_tables)
    reasons = L.lossy_changes(old_names, old_tables, new_names, new_tables)
    if reasons and not allow_lossy:
        raise ValueError("lossy warm-start migration refused: " + "; ".join(reasons[:5]))
    sources = L.column_sources(old_names, old_tables, new_names, new_tables)
    norm = dict(state["norm"])
    mean, var = np.asarray(norm["mean"], np.float64), np.asarray(norm["var"], np.float64)
    if len(mean) != len(old_names):
        raise ValueError(f"checkpoint has {len(mean)} features, layout has {len(old_names)}")
    std = np.maximum(np.sqrt(var), NORM_CLIP_STD)

    def migrate_input(weights):
        weights = dict(weights)
        key = _input_layer(weights, len(old_names))
        if key is None:
            if weights:
                raise ValueError("no input layer with the old feature width found")
            return weights
        bias_key = key[:-len("weight")] + "bias"
        w, b = weights[key], weights[bias_key]
        new_w, new_b = L.migrate_input(w.detach().cpu().numpy(), b.detach().cpu().numpy(),
                                       mean, std, sources, old_names, old_tables)
        weights[key] = torch.as_tensor(new_w, dtype=w.dtype, device=w.device)
        weights[bias_key] = torch.as_tensor(new_b, dtype=b.dtype, device=b.device)
        return weights

    actor = state["actor"]
    if any(not k.startswith("net.") for k in actor):
        raise ValueError("feature-layout migration supports MLP actors only")
    actor = migrate_input(actor)
    last = max(int(k.split(".")[1]) for k in actor if k.startswith("net."))
    w, b = actor[f"net.{last}.weight"], actor[f"net.{last}.bias"]
    new_w, new_b = L.migrate_output(w.detach().cpu().numpy(), b.detach().cpu().numpy(),
                                    old_tables, new_tables)
    actor[f"net.{last}.weight"] = torch.as_tensor(new_w, dtype=w.dtype, device=w.device)
    actor[f"net.{last}.bias"] = torch.as_tensor(new_b, dtype=b.dtype, device=b.device)
    norm["mean"], norm["var"] = L.migrate_norm(mean, var, sources)
    return dict(state, actor=actor, critic=migrate_input(state["critic"]), norm=norm,
                feature_names=list(new_names),
                action_tables={head: list(v) for head, v in new_tables.items()})


def migrate_v3_to_v4(state):
    """Warm-start a v3 checkpoint (180 features, 12 throttle bins) in the frozen v4 layout.

    v4 inserted the team block after the rays and refined the throttle bins; this is
    `migrate_state` between the two frozen layouts.
    """
    return migrate_state(state, L.V3_FEATURE_NAMES, L.V3_ACTION_TABLES,
                         L.V4_FEATURE_NAMES, L.V4_ACTION_TABLES)


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
        if args.burn_in < 0:  # auto: burn-in refreshes stale stored states (review finding)
            args.burn_in = 16 if args.arch == "gru" else 0
        if args.arch == "gru":
            if args.chunk % max(args.burn_in, 1) and args.burn_in:
                print(f"warning: chunk {args.chunk} not a multiple of burn-in {args.burn_in}; "
                      "hidden states will be stored more often (memory)", flush=True)
            if args.chunk < 1 or not 0 <= args.burn_in <= args.chunk:
                raise ValueError("--arch gru needs --chunk >= 1 and 0 <= --burn-in <= --chunk")
        self.recurrent = args.arch == "gru"
        self.actor = build_actor(args.arch, tuple(args.hidden)).to(self.device)
        self.critic = Critic(tuple(args.critic_hidden)).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=args.lr, eps=1e-5)
        self.norm = RunningNorm(F.FEATURE_DIM)
        self.ret_norm = RunningNorm(1)
        self.update = 0
        self.samples = 0
        self.norm_frozen = False
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
            "opp_winrate": self.opp_winrate, "norm_frozen": self.norm_frozen,
            "np_rng": self.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "args": vars(self.args),
            "feature_names": list(F.FEATURE_NAMES),
            "action_tables": {head: list(v) for head, v in F.ACTION_TABLES.items()},
        }

    def save(self):
        tmp = self.run_dir / "checkpoint.tmp"
        torch.save(self.state(), tmp)
        os.replace(tmp, self.run_dir / "checkpoint.pt")

    def load(self, path, weights_only=False):
        state = torch.load(path, map_location=self.device, weights_only=False)
        arch = state.get("args", {}).get("arch", "mlp")
        if arch != self.args.arch:
            raise ValueError(f"{path} holds a {arch!r} actor but --arch is {self.args.arch!r}")
        names, tables = state_layout(state)
        if not L.same_layout(names, tables, F.FEATURE_NAMES, F.ACTION_TABLES):
            if not weights_only:
                raise ValueError(f"{path} uses another feature layout ({len(names)} features); "
                                 "resume needs the current one (use --init to warm-start "
                                 "with feature migration)")
            if self.args.arch != "mlp":
                raise ValueError("feature-layout migration supports --arch mlp only")
            if "feature_names" not in state and len(names) == V3_DIM:
                state = migrate_v3_to_v4(state)
                names, tables = L.V4_FEATURE_NAMES, L.V4_ACTION_TABLES
            state = migrate_state(state, names, tables)
        if weights_only and self.args.arch == "mlp":
            # Net2WiderNet when the requested network is wider than the checkpoint.
            actor_hidden = [t.shape[0] for k, t in sorted(state["actor"].items())
                            if k.endswith(".weight")][:-1]
            if list(self.args.hidden) != actor_hidden:
                state = dict(state, actor=widen_mlp(state["actor"], self.args.hidden, self.rng))
            critic_hidden = [t.shape[0] for k, t in sorted(state["critic"].items())
                             if k.endswith(".weight")][:-1]
            if list(self.args.critic_hidden) != critic_hidden:
                state = dict(state, critic=widen_mlp(state["critic"], self.args.critic_hidden,
                                                     self.rng))
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.norm.load(state["norm"])
        self.ret_norm.load(state["ret_norm"])
        self.norm_frozen = state.get("norm_frozen", state["update"] >= self.args.norm_updates)
        if not weights_only:
            self.optimizer.load_state_dict(state["optimizer"])
            self.update, self.samples = state["update"], state["samples"]
            self.opp_winrate = state.get("opp_winrate", {})
            if "np_rng" in state:
                self.rng.bit_generator.state = state["np_rng"]
                torch.set_rng_state(state["torch_rng"].cpu())

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
        weights["gamma"] = self.args.shaping_gamma
        weights["offset"] = self.args.shaping_offset
        return weights

    def kick_coef(self):
        """Kickstarting weight: linear decay from --kick-coef to 0 over --kick-samples."""
        if not self.args.teacher or not self.args.kick_coef:
            return 0.0
        if not self.args.kick_samples:
            return self.args.kick_coef
        return self.args.kick_coef * max(0.0, 1.0 - self.samples / self.args.kick_samples)

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
        def winrate(path):  # PFSP difficulty from both decoding modes of a snapshot
            seen = [self.opp_winrate[k + path] for k in ("greedy:", "policy:")
                    if k + path in self.opp_winrate]
            return sum(seen) / len(seen) if seen else 0.5

        weights = np.array([(1.0 - winrate(path)) ** 2 + 0.05 for path in pool])
        path = pool[int(self.rng.choice(len(pool), p=weights / weights.sum()))]
        # Frozen opponents mostly play greedy, as a deployed policy does (sampling is far weaker).
        kind = "greedy:" if self.rng.random() < self.args.pool_greedy else "policy:"
        return kind + path

    def mix_probabilities(self):
        """Base mix, optionally re-weighted toward scripted opponents we still lose to."""
        names = [n for n, _ in self.mix]
        weights = np.array([p for _, p in self.mix], dtype=np.float64)
        if self.args.adaptive_mix:
            for i, name in enumerate(names):
                if name in ("pool", "mirror", "self"):
                    continue
                win = self.opp_winrate.get(name, 0.5)
                weights[i] *= (1.0 - win) ** 2 + self.args.adaptive_floor
        return names, weights / weights.sum()

    def jobs(self, count):
        names, probs = self.mix_probabilities()
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
            last.weight[0].mul_(old_std / new_std)
            last.bias[0].mul_(old_std).add_(old_mean - new_mean).div_(new_std)

    def ppo_update(self, batch):
        args = self.args
        clock = [time.perf_counter()]
        timing = {}

        def lap(name):
            now = time.perf_counter()
            timing[name] = round(now - clock[0], 3)
            clock[0] = now

        T, E = batch["rewards"].shape[:2]
        alive = batch["alive"].reshape(T, E * 3)
        raw = batch["obs"].reshape(T, E * 3, -1)
        mean = torch.as_tensor(self.norm.mean, dtype=torch.float32, device=self.device)
        std = torch.as_tensor(self.norm.std, dtype=torch.float32, device=self.device)

        flat_index = np.nonzero(alive.reshape(-1))[0]
        x_all = torch.as_tensor(raw.reshape(T * E * 3, -1)[flat_index], device=self.device)
        x_all = torch.clamp((x_all - mean) / std, -5.0, 5.0)
        lap("prep_s")
        with torch.no_grad():
            heads = torch.cat([self.critic(chunk) for chunk in torch.split(x_all, 65536)])
        heads = heads.cpu().numpy().astype(np.float64)
        v_raw = heads[:, 0] * self.ret_norm.std[0] + self.ret_norm.mean[0]
        values = np.zeros(T * E * 3, dtype=np.float64)
        values[flat_index] = v_raw
        values = values.reshape(T, E * 3)
        lap("critic_s")
        adv = gae(batch["rewards"].reshape(T, E * 3), values, alive, args.gamma, args.lam)
        returns = (adv + values).reshape(-1)[flat_index]
        adv = adv.reshape(-1)[flat_index]

        # Undiscounted win objective: reward 1 at the final tick of a won match.
        win_values = np.zeros(T * E * 3, dtype=np.float64)
        win_values[flat_index] = heads[:, 1]
        win_values = win_values.reshape(T, E * 3)
        win_rewards = np.zeros((T, E * 3), dtype=np.float32)
        win_rewards[-1] = np.repeat(batch["won"].astype(np.float32), 3)
        win_adv = gae(win_rewards, win_values, alive, 1.0, args.win_lam)
        win_adv = win_adv.reshape(-1)[flat_index]
        # The win head regresses on the actual outcome of each learner unit's match.
        win_returns = np.repeat(batch["won"].astype(np.float64), 3)[None, :].repeat(T, 0)
        win_returns = win_returns.reshape(-1)[flat_index]
        explained = 1.0 - np.var(returns - v_raw) / (np.var(returns) + 1e-8)
        old_mean, old_std = float(self.ret_norm.mean[0]), float(self.ret_norm.std[0])
        self.ret_norm.update(returns.reshape(-1, 1))
        self.popart_rescale(old_mean, old_std)
        ret_t = torch.as_tensor((returns - self.ret_norm.mean[0]) / self.ret_norm.std[0],
                                dtype=torch.float32, device=self.device)
        combined = (adv - adv.mean()) / (adv.std() + 1e-8)
        if args.win_coef:
            combined = combined + args.win_coef * (win_adv - win_adv.mean()) / (win_adv.std() + 1e-8)
            combined = (combined - combined.mean()) / (combined.std() + 1e-8)
        adv_t = torch.as_tensor(combined, dtype=torch.float32, device=self.device)
        win_t = torch.as_tensor(win_returns, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(batch["actions"].reshape(T * E * 3, 3)[flat_index].astype(np.int64),
                                device=self.device)
        old_logp = torch.as_tensor(batch["logp"].reshape(-1)[flat_index], device=self.device)
        kick_coef = self.kick_coef() if "teacher_actions" in batch else 0.0
        if kick_coef:
            teach_t = torch.as_tensor(
                batch["teacher_actions"].reshape(T * E * 3, 3)[flat_index].astype(np.int64),
                device=self.device)
            teach_valid = torch.as_tensor(
                batch["teacher_valid"].reshape(-1)[flat_index].astype(np.float32), device=self.device)

        n = len(flat_index)
        lap("gae_s")
        ent_coef = self.entropy_coef()
        stats = {"pi_loss": [], "v_loss": [], "entropy": [], "kl": [], "clipfrac": [],
                 "kick_ce": [], "kick_agree": []}
        extra = {}
        if self.recurrent:
            seqs = build_sequences(alive, batch["resets"].reshape(T, E * 3), batch["h0_key"],
                                   args.chunk, args.burn_in)
            seqs = {k: torch.as_tensor(v, device=self.device) for k, v in seqs.items()}
            h0 = torch.as_tensor(batch["h0"], device=self.device)
            extra["chunks"] = int(len(seqs["main"]))
        actor_stopped = False
        first = True
        for _epoch in range(args.epochs):
            for idx, logits in self.minibatches(x_all, n, seqs if self.recurrent else None,
                                                h0 if self.recurrent else None):
                logp, entropy = log_prob_entropy(logits, act_t[idx])
                if first:  # weights untouched: recomputed log-probs must match the rollout's
                    extra["logp_drift"] = float((logp - old_logp[idx]).abs().max().item())
                    first = False
                ratio = torch.exp(logp - old_logp[idx])
                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - old_logp[idx])).mean().item()
                if args.target_kl and kl > 1.5 * args.target_kl:
                    actor_stopped = True
                surrogate = torch.min(ratio * adv_t[idx],
                                      torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_t[idx])
                pi_loss = -surrogate.mean()
                out = self.critic(x_all[idx])
                v_loss = 0.5 * (out[:, 0] - ret_t[idx]).pow(2).mean()
                v_loss = v_loss + 0.5 * (out[:, 1] - win_t[idx]).pow(2).mean()
                if actor_stopped:  # past the trust region: keep fitting the critic only
                    loss = args.vf_coef * v_loss
                else:
                    loss = pi_loss + args.vf_coef * v_loss - ent_coef * entropy.mean()
                    if kick_coef:
                        # Kickstarting: cross-entropy toward the shadow teacher's labels on
                        # the states the student visits (DAgger-style), annealed to zero.
                        valid = teach_valid[idx]
                        teach_logp, _ = log_prob_entropy(logits, teach_t[idx])
                        kick_ce = -(teach_logp * valid).sum() / valid.sum().clamp(min=1.0)
                        loss = loss + kick_coef * kick_ce
                        with torch.no_grad():
                            stats["kick_ce"].append(kick_ce.detach())
                            stats["kick_agree"].append(((act_t[idx] == teach_t[idx]).all(-1)
                                                        .float() * valid).sum()
                                                       / valid.sum().clamp(min=1.0))
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), args.max_grad)
                self.optimizer.step()
                with torch.no_grad():
                    stats["pi_loss"].append(pi_loss.detach())
                    stats["v_loss"].append(v_loss.detach())
                    stats["entropy"].append(entropy.mean().detach())
                    stats["kl"].append(torch.as_tensor(kl))
                    stats["clipfrac"].append(((ratio - 1).abs() > args.clip).float().mean())
        if not self.norm_frozen:
            self.norm.update(raw.reshape(T * E * 3, -1)[flat_index][:: args.norm_stride])
            self.norm_frozen = self.update + 1 >= args.norm_updates
        lap("optimize_s")
        out = {k: float(torch.stack([t.float().cpu() for t in v]).mean()) if v else 0.0
               for k, v in stats.items()}
        out.update({"timing": timing, "explained_var": float(explained), "samples": int(n),
                    "minibatches": len(stats["kl"]), "ent_coef": ent_coef,
                    "actor_stopped": actor_stopped, "kick_coef": kick_coef,
                    "win_value_mean": float(heads[:, 1].mean()),
                    "adv_raw_std": float(adv.std()), **extra})
        return out

    def minibatches(self, x_all, n, seqs, h0):
        """Yield (decision positions, actor logits) for one epoch of shuffled minibatches.

        MLP: flat minibatches of `--minibatch` decisions. GRU: minibatches of
        `--minibatch // --chunk` whole chunks, run through the GRU (logits are
        computed lazily, after the previous minibatch's optimizer step).
        """
        if seqs is None:
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.args.minibatch):
                idx = perm[start:start + self.args.minibatch]
                yield idx, self.actor(x_all[idx])
            return
        n_chunks = len(seqs["main"])
        per_batch = max(1, self.args.minibatch // self.args.chunk)
        perm = torch.randperm(n_chunks, device=self.device)
        for start in range(0, n_chunks, per_batch):
            chunks = perm[start:start + per_batch]
            logits, idx = sequence_logits(self.actor, x_all, h0, seqs, chunks)
            yield idx, logits

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
            pending = False
            cycle_start = time.perf_counter()
            while self.update < args.updates:
                t0 = time.perf_counter()
                if not pending:
                    self.dispatch(pipes)
                results = [self.receive(pipe, proc) for pipe, proc in zip(pipes, procs)]
                t1 = time.perf_counter()
                # Pipelining: start the next rollout with the pre-update weights while the
                # learner updates (one update of policy lag; PPO's ratio uses the recorded
                # behaviour log-probs). Only once normalization is frozen, so the statistics
                # used by workers and learner always agree.
                pending = bool(args.pipeline and self.norm_frozen and self.update + 1 < args.updates)
                if pending:
                    self.dispatch(pipes)
                merge_start = time.perf_counter()
                batch = merge_results(results, self.recurrent)
                merge_s = time.perf_counter() - merge_start
                stats = [s for r in results for s in r["stats"]]
                batch["won"] = np.array([s["result"] == "win" for s in stats])
                self.record_results(stats)
                learn = self.ppo_update(batch)
                t2 = time.perf_counter()
                self.update += 1
                self.samples += learn["samples"]
                record = self.summarize(stats, learn, t1 - t0, t2 - t1,
                                        sum(r["ticks"] for r in results))
                now = time.perf_counter()
                record["cycle_s"] = round(now - cycle_start, 2)
                record["ticks_per_s"] = round(sum(r["ticks"] for r in results) / (now - cycle_start))
                record["pipelined"] = pending
                mix_names, mix_probs = self.mix_probabilities()
                record["mix"] = {n: round(float(q), 4) for n, q in zip(mix_names, mix_probs)}
                record["timing"]["merge_s"] = round(merge_s, 3)
                cycle_start = now
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

    def dispatch(self, pipes):
        args = self.args
        weights = actor_to_numpy(self.actor)
        mean, std = self.norm.mean.astype(np.float32), self.norm.std.astype(np.float32)
        config = {"reward_weights": self.reward_weights(), "max_ticks": args.max_ticks,
                  "teacher": args.teacher if self.kick_coef() > 0 else None,
                  "memory_wipe_prob": args.memory_wipe,
                  "hidden_every": hidden_every(args.chunk, args.burn_in)}
        for pipe in pipes:
            pipe.send((weights, mean, std, self.jobs(args.envs),
                       dict(config, seed=int(self.rng.integers(2**31)))))

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
            return "pool" if name.startswith(("policy:", "greedy:")) else name

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
    p.add_argument("--arch", choices=("mlp", "gru"), default="mlp",
                   help="actor: tanh MLP over --hidden, or gru = Linear(H1)+tanh -> GRU(HG) "
                        "with --hidden H1 HG")
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    p.add_argument("--chunk", type=int, default=64,
                   help="gru: decisions per training sequence")
    p.add_argument("--burn-in", type=int, default=-1,
                   help="gru: decisions of the previous chunk replayed (no gradient) to refresh "
                        "the stored start state under the current weights (-1: 16 for gru)")
    p.add_argument("--critic-hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.998)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=16384)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--ent-final", type=float, default=0.003)
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--max-grad", type=float, default=0.5)
    p.add_argument("--anneal-samples", type=float, default=0.0)
    p.add_argument("--win-coef", type=float, default=0.0,
                   help="weight of the undiscounted win advantage added to the PPO advantage")
    p.add_argument("--win-lam", type=float, default=1.0)
    p.add_argument("--shaping-offset", type=float, default=0.0,
                   help="constant added to the potential, in route units (e.g. 360 with "
                        "--shaping-gamma equal to --gamma gives exactly invariant shaping "
                        "without an idle bonus)")
    p.add_argument("--shaping-gamma", type=float, default=1.0,
                   help="discount inside the shaping term; 1.0 avoids a per-tick bonus for idling "
                        "far from the objective (negative potentials)")
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
    p.add_argument("--pipeline", type=int, default=1,
                   help="overlap the next rollout with the learner update (1 update of lag)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--init", default="")
    p.add_argument("--adaptive-mix", type=int, default=0,
                   help="scale scripted opponents' shares by (1 - EMA win rate)^2 + floor")
    p.add_argument("--adaptive-floor", type=float, default=0.2)
    p.add_argument("--teacher", default="", help="scripted teacher for kickstarting (e.g. relay)")
    p.add_argument("--kick-coef", type=float, default=0.0,
                   help="initial weight of the cross-entropy toward the teacher's labels")
    p.add_argument("--kick-samples", type=float, default=0.0,
                   help="learner samples over which the kickstarting weight decays to zero")
    p.add_argument("--pool-greedy", type=float, default=0.8,
                   help="probability that a frozen pool opponent decodes greedily")
    p.add_argument("--pool-size", type=int, default=20, help="own snapshots kept in the league")
    p.add_argument("--pool-extra", nargs="*", default=[], help="globs of extra league policies")
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args(argv)
    Trainer(args).run()


if __name__ == "__main__":
    main()
