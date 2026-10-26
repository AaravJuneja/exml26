"""Numpy-only inference pieces (no torch): head layout, NumpyActor, RunningNorm."""

import numpy as np

from ctf.features import N_THROTTLE, N_TURN

HEAD_SIZES = (N_THROTTLE, N_TURN, 2)
N_LOGITS = sum(HEAD_SIZES)


class NumpyActor:
    """Batched numpy forward pass identical to Actor (tanh hidden layers)."""

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
        logits = self.logits(x)
        actions, logp = [], np.zeros(len(logits), dtype=np.float32)
        start = 0
        for size in HEAD_SIZES:
            head = logits[:, start:start + size]
            start += size
            shifted = head - head.max(axis=1, keepdims=True)
            log_softmax = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            if greedy:
                index = head.argmax(axis=1)
            else:
                gumbel = -np.log(-np.log(rng.random(head.shape) + 1e-12) + 1e-12)
                index = (head + gumbel).argmax(axis=1)
            actions.append(index)
            logp += log_softmax[np.arange(len(head)), index]
        return np.stack(actions, axis=1).astype(np.int64), logp


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
