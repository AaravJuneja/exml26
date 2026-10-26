"""Actor/critic networks (torch for training, numpy for rollouts and export).

The actor is an MLP over normalized features with three categorical heads:
throttle (N_THROTTLE), turn (N_TURN) and jump (2). The critic is a separate,
wider MLP used only in training. Both see only one player's own features.
"""

import numpy as np
import torch
from torch import nn

from ctf.features import FEATURE_DIM
from ctf.npnets import HEAD_SIZES, N_LOGITS, NumpyActor, RunningNorm  # noqa: F401


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
    """Two outputs: [0] normalized shaped return, [1] undiscounted win probability (logit-free)."""

    def __init__(self, hidden=(256, 256)):
        super().__init__()
        layers, n_in = [], FEATURE_DIM
        for n_out in hidden:
            layers += [_layer(n_in, n_out, 2 ** 0.5), nn.Tanh()]
            n_in = n_out
        layers.append(_layer(n_in, 2, 1.0))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
