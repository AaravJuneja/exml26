"""Numpy-only inference pieces (no torch): head layout, NumpyActor, NumpyGRUActor, RunningNorm."""

import struct

import numpy as np

from ctf.features import FEATURE_DIM, N_THROTTLE, N_TURN, _num

HEAD_SIZES = (N_THROTTLE, N_TURN, 2)
N_LOGITS = sum(HEAD_SIZES)
HEAD_STARTS = (0, N_THROTTLE, N_THROTTLE + N_TURN)
GRU_KEYS = ("W_in", "b_in", "W_ih", "b_ih", "W_hh", "b_hh", "W_out", "b_out")


_PACK_ROW = struct.Struct(f"{FEATURE_DIM}f").pack


def rows_to_float32(rows):
    """np.asarray(rows, dtype=np.float32) for lists of FEATURE_DIM floats, faster.

    struct's 'f' packing is the same C double -> float conversion; anything it
    rejects (wrong length, out of float32 range) goes through numpy instead.
    """
    if not rows:
        return np.asarray(rows, dtype=np.float32)
    try:
        data = bytearray().join([_PACK_ROW(*row) for row in rows])
    except (struct.error, OverflowError, TypeError):
        return np.asarray(rows, dtype=np.float32)
    return np.frombuffer(data, dtype=np.float32).reshape(len(rows), FEATURE_DIM)


def sample_logits(logits, rng, greedy=False):
    """Return (actions int array [N,3], joint log-prob [N]) for a batch of logits.

    Elementwise steps and the (exact) maxima run on all heads at once; every
    per-head sum keeps its own reduction, so results match a head-by-head loop.
    """
    n = len(logits)
    shifted = logits - np.repeat(np.maximum.reduceat(logits, HEAD_STARTS, axis=1), HEAD_SIZES, axis=1)
    exp = np.exp(shifted)
    log_norm = np.log(np.stack([exp[:, a:a + s].sum(axis=1)
                                for a, s in zip(HEAD_STARTS, HEAD_SIZES)], axis=1))
    if not greedy:
        # One draw for all heads: the same numbers, in the same order, as one (N, size)
        # draw per head.
        gumbel = -np.log(-np.log(rng.random(n * N_LOGITS) + 1e-12) + 1e-12)
    actions = np.empty((n, len(HEAD_SIZES)), dtype=np.int64)
    for k, (a, s) in enumerate(zip(HEAD_STARTS, HEAD_SIZES)):
        head = logits[:, a:a + s]
        if greedy:
            actions[:, k] = head.argmax(axis=1)
        else:
            actions[:, k] = (head + gumbel[n * a:n * (a + s)].reshape(n, s)).argmax(axis=1)
    chosen = shifted[np.arange(n)[:, None], actions + HEAD_STARTS] - log_norm  # log-softmax
    logp = np.zeros(n, dtype=np.float32)
    for k in range(len(HEAD_SIZES)):
        logp += chosen[:, k]
    return actions, logp


def memory_is_fresh(memory, observation):
    """True if encoding `observation` starts `memory` from scratch.

    That is a brand-new or reset PlayerMemory, or a tick that went backwards (a
    new match, on which `PlayerMemory.update` resets itself). The recurrent
    hidden state is reset exactly when this holds, so it always lives as long
    as the explicit memory. Call before `encode`.
    """
    if memory.last_tick is None:
        return True
    return int(_num(observation.get("tick"), 0)) < memory.last_tick


class NumpyActor:
    """Batched numpy forward pass identical to Actor (tanh hidden layers)."""

    recurrent = False
    arch = "mlp"

    def __init__(self, weights, obs_mean, obs_std):
        self.weights = weights
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32)
        self.obs_std = np.asarray(obs_std, dtype=np.float32)

    def normalize(self, x):
        return np.clip((x - self.obs_mean) / self.obs_std, -5.0, 5.0)

    def logits(self, x):
        h = self.normalize(np.asarray(x, dtype=np.float32))
        for w, b in self.weights[:-1]:
            h = np.tanh(h @ w + b)
        w, b = self.weights[-1]
        return h @ w + b

    def sample(self, x, rng, greedy=False):
        """Return (actions int array [N,3], joint log-prob [N])."""
        return sample_logits(self.logits(x), rng, greedy)


def _sigmoid(x):
    # Same function as 1 / (1 + exp(-x)) but cannot overflow.
    return 0.5 * np.tanh(0.5 * x) + 0.5


class NumpyGRUActor:
    """Batched numpy forward of GRUActor: tanh input layer -> GRU cell -> linear heads.

    `weights` maps GRU_KEYS to float32 arrays applied as `x @ W + b`; the GRU
    gates are ordered (reset, update, new) as in torch:
        r = s(gi_r + gh_r), z = s(gi_z + gh_z), n = tanh(gi_n + r * gh_n),
        h' = (1 - z) * n + z * h.
    The caller owns the hidden state ([N, hidden_size], zeros at a reset).
    """

    recurrent = True
    arch = "gru"

    def __init__(self, weights, obs_mean, obs_std):
        missing = [key for key in GRU_KEYS if key not in weights]
        if missing:
            raise ValueError(f"GRU weights missing {missing}")
        self.weights = {key: np.asarray(weights[key], dtype=np.float32) for key in GRU_KEYS}
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32)
        self.obs_std = np.asarray(obs_std, dtype=np.float32)
        self.hidden_size = self.weights["W_hh"].shape[0]
        if self.weights["W_hh"].shape != (self.hidden_size, 3 * self.hidden_size):
            raise ValueError("W_hh must be [H, 3H]")

    def normalize(self, x):
        return np.clip((x - self.obs_mean) / self.obs_std, -5.0, 5.0)

    def initial_state(self, n):
        return np.zeros((n, self.hidden_size), dtype=np.float32)

    def step(self, x, h):
        """One decision for a batch: returns (logits [N, 23], new hidden [N, H])."""
        w = self.weights
        H = self.hidden_size
        z_in = np.tanh(self.normalize(np.asarray(x, dtype=np.float32)) @ w["W_in"] + w["b_in"])
        h = np.asarray(h, dtype=np.float32)
        gi = z_in @ w["W_ih"] + w["b_ih"]
        gh = h @ w["W_hh"] + w["b_hh"]
        r = _sigmoid(gi[:, :H] + gh[:, :H])
        z = _sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        h_new = n + z * (h - n)
        return h_new @ w["W_out"] + w["b_out"], h_new

    def step_sample(self, x, h, rng, greedy=False):
        """Return (actions [N,3], joint log-prob [N], new hidden [N,H])."""
        logits, h_new = self.step(x, h)
        actions, logp = sample_logits(logits, rng, greedy)
        return actions, logp, h_new


def make_actor(weights, obs_mean, obs_std):
    """NumpyActor for a list of (W, b) layers, NumpyGRUActor for a dict of GRU weights."""
    if isinstance(weights, dict):
        return NumpyGRUActor(weights, obs_mean, obs_std)
    return NumpyActor(weights, obs_mean, obs_std)


class PolicyMemories:
    """Per-player PlayerMemory plus (for recurrent actors) hidden state, keyed by any id.

    `encode_batch` encodes one observation per key, resetting the hidden state of
    any key whose memory starts fresh; `act` samples and advances the state. The
    hidden state advances only when a key is encoded, i.e. on that player's
    alive decisions, exactly as in a deployed per-player process.
    """

    def __init__(self, actor, keys, encode, memory_factory):
        self.actor = actor
        self.encode = encode
        self.memories = {key: memory_factory() for key in keys}
        self.hidden = ({key: np.zeros(actor.hidden_size, dtype=np.float32) for key in keys}
                       if actor.recurrent else None)

    def encode_batch(self, keys, observations):
        rows = []
        for key, observation in zip(keys, observations):
            memory = self.memories[key]
            if self.hidden is not None and memory_is_fresh(memory, observation):
                self.hidden[key][:] = 0.0
            rows.append(self.encode(observation, memory))
        return rows_to_float32(rows)

    def act(self, keys, x, rng, greedy=False):
        """Sample actions [N,3] for encoded rows `x`; records them in each memory."""
        if self.hidden is None:
            actions, _ = self.actor.sample(x, rng, greedy)
        else:
            h = np.stack([self.hidden[key] for key in keys])
            actions, _, h_new = self.actor.step_sample(x, h, rng, greedy)
            for key, row in zip(keys, h_new):
                self.hidden[key] = row
        for key, (t, r, j) in zip(keys, actions.tolist()):
            self.memories[key].set_action(t, r, j)
        return actions


class RunningNorm:
    """Running mean/variance of features (Chan et al. parallel update)."""

    def __init__(self, dim, clip_std=1e-2):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4
        self.clip_std = clip_std

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        if len(batch) == 0:
            return
        b_mean, b_var, b_count = batch.mean(0), batch.var(0), len(batch)
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean = self.mean + delta * b_count / total
        m2 = self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self):
        return np.maximum(np.sqrt(self.var), self.clip_std)

    def state(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load(self, state):
        self.mean, self.var, self.count = state["mean"], state["var"], state["count"]
