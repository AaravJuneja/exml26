"""Actor/critic networks (torch for training, numpy for rollouts and export).

The actor is an MLP over normalized features with three categorical heads:
throttle (N_THROTTLE), turn (N_TURN) and jump (2). The critic is a separate,
wider MLP used only in training. Both see only one player's own features.
"""

import numpy as np
import torch
from torch import nn

from ctf.features import FEATURE_DIM, N_THROTTLE, N_TURN

HEAD_SIZES = (N_THROTTLE, N_TURN, 2)
N_LOGITS = sum(HEAD_SIZES)


def _layer(n_in, n_out, gain):
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Actor(nn.Module):
    def __init__(self, hidden=(128, 128)):
        super().__init__()
        layers, n_in = [], FEATURE_DIM
        for n_out in hidden:
            layers += [_layer(n_in, n_out, 2 ** 0.5), nn.Tanh()]
            n_in = n_out
        layers.append(_layer(n_in, N_LOGITS, 0.01))
        self.net = nn.Sequential(*layers)
        self.hidden = tuple(hidden)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, hidden=(256, 256)):
        super().__init__()
        layers, n_in = [], FEATURE_DIM
        for n_out in hidden:
            layers += [_layer(n_in, n_out, 2 ** 0.5), nn.Tanh()]
            n_in = n_out
        layers.append(_layer(n_in, 1, 1.0))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def split_heads(logits):
    a, b = HEAD_SIZES[0], HEAD_SIZES[0] + HEAD_SIZES[1]
    return logits[..., :a], logits[..., a:b], logits[..., b:]


def log_prob_entropy(logits, actions):
    """Joint log-probability of (throttle, turn, jump) indices and summed entropy."""
    logp, entropy = 0.0, 0.0
    for head, index in zip(split_heads(logits), actions.unbind(-1)):
        log_softmax = torch.log_softmax(head, dim=-1)
        logp = logp + log_softmax.gather(-1, index.long().unsqueeze(-1)).squeeze(-1)
        entropy = entropy - (log_softmax.exp() * log_softmax).sum(-1)
    return logp, entropy


def actor_to_numpy(actor):
    """List of (W, b) float32 arrays for NumpyActor."""
    weights = []
    for module in actor.net:
        if isinstance(module, nn.Linear):
            weights.append((module.weight.detach().cpu().numpy().T.astype(np.float32).copy(),
                            module.bias.detach().cpu().numpy().astype(np.float32).copy()))
    return weights


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
