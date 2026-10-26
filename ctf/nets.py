"""Actor/critic networks (torch for training, numpy for rollouts and export).

The actor is an MLP over normalized features with three categorical heads:
throttle (N_THROTTLE), turn (N_TURN) and jump (2), or heading, speed and jump in
the goal action space (ctf/action_space.py); `GRUActor` is the optional
recurrent variant (`--arch gru`). The critic is a separate, wider MLP used only
in training. Both see only one player's own features.
"""

import numpy as np
import torch
from torch import nn

from ctf.features import FEATURE_DIM
from ctf.npnets import (GOAL_HEAD_SIZES, HEAD_SIZES, N_LOGITS, NumpyActor,  # noqa: F401
                        NumpyGRUActor, RunningNorm)


def _layer(n_in, n_out, gain):
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class Actor(nn.Module):
    def __init__(self, hidden=(128, 128), n_in=FEATURE_DIM, n_logits=N_LOGITS):
        super().__init__()
        layers = []
        for n_out in hidden:
            layers += [_layer(n_in, n_out, 2 ** 0.5), nn.Tanh()]
            n_in = n_out
        layers.append(_layer(n_in, n_logits, 0.01))
        self.net = nn.Sequential(*layers)
        self.hidden = tuple(hidden)

    def forward(self, x):
        return self.net(x)


class GRUActor(nn.Module):
    """Linear(FEATURE_DIM -> H1) + tanh -> GRU(H1 -> Hg) -> Linear(Hg -> N_LOGITS).

    The hidden state advances once per alive decision of a player; see
    NumpyGRUActor for the matching numpy step used in rollouts and export.
    """

    def __init__(self, hidden=(128, 128), n_features=FEATURE_DIM, n_logits=N_LOGITS):
        super().__init__()
        if len(hidden) != 2:
            raise ValueError("GRU actor takes --hidden H1 HG (input layer, GRU size)")
        n_in, n_gru = hidden
        self.inp = _layer(n_features, n_in, 2 ** 0.5)
        self.gru = nn.GRU(n_in, n_gru, batch_first=True)
        for name, param in self.gru.named_parameters():
            if name.startswith("weight_hh"):
                for k in range(3):  # orthogonal recurrent weights, one block per gate
                    nn.init.orthogonal_(param.data[k * n_gru:(k + 1) * n_gru])
            elif name.startswith("weight_ih"):
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)
        self.out = _layer(n_gru, n_logits, 0.01)
        self.hidden = tuple(hidden)
        self.hidden_size = n_gru

    def forward(self, x, h0):
        """x [B, L, FEATURE_DIM] normalized, h0 [B, Hg] -> (logits [B, L, N_LOGITS], h [B, Hg])."""
        out, h = self.gru(torch.tanh(self.inp(x)), h0.unsqueeze(0))
        return self.out(out), h.squeeze(0)

    @torch.no_grad()
    def final_state(self, x, h0):
        """Hidden state after running x [B, L, D] from h0 (burn-in)."""
        _, h = self.gru(torch.tanh(self.inp(x)), h0.unsqueeze(0))
        return h.squeeze(0)


def build_actor(arch, hidden, n_features=FEATURE_DIM, head_sizes=HEAD_SIZES):
    if arch == "gru":
        return GRUActor(hidden, n_features, sum(head_sizes))
    if arch == "mlp":
        return Actor(hidden, n_features, sum(head_sizes))
    raise ValueError(f"unknown actor arch {arch!r}")


class Critic(nn.Module):
    """Two outputs: [0] normalized shaped return, [1] undiscounted win probability (logit-free)."""

    def __init__(self, hidden=(256, 256), n_in=FEATURE_DIM):
        super().__init__()
        layers = []
        for n_out in hidden:
            layers += [_layer(n_in, n_out, 2 ** 0.5), nn.Tanh()]
            n_in = n_out
        layers.append(_layer(n_in, 2, 1.0))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def split_heads(logits, head_sizes=HEAD_SIZES):
    a, b = head_sizes[0], head_sizes[0] + head_sizes[1]
    return logits[..., :a], logits[..., a:b], logits[..., b:]


def log_prob_entropy(logits, actions, head_sizes=HEAD_SIZES):
    """Joint log-probability of the three heads' indices and summed entropy."""
    logp, entropy = 0.0, 0.0
    for head, index in zip(split_heads(logits, head_sizes), actions.unbind(-1)):
        log_softmax = torch.log_softmax(head, dim=-1)
        logp = logp + log_softmax.gather(-1, index.long().unsqueeze(-1)).squeeze(-1)
        entropy = entropy - (log_softmax.exp() * log_softmax).sum(-1)
    return logp, entropy


def _np(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32)


def actor_to_numpy(actor):
    """List of (W, b) float32 arrays for NumpyActor, or a GRU_KEYS dict for NumpyGRUActor."""
    if isinstance(actor, GRUActor):
        gru = actor.gru
        return {
            "W_in": _np(actor.inp.weight).T.copy(), "b_in": _np(actor.inp.bias).copy(),
            "W_ih": _np(gru.weight_ih_l0).T.copy(), "b_ih": _np(gru.bias_ih_l0).copy(),
            "W_hh": _np(gru.weight_hh_l0).T.copy(), "b_hh": _np(gru.bias_hh_l0).copy(),
            "W_out": _np(actor.out.weight).T.copy(), "b_out": _np(actor.out.bias).copy(),
        }
    weights = []
    for module in actor.net:
        if isinstance(module, nn.Linear):
            weights.append((module.weight.detach().cpu().numpy().T.astype(np.float32).copy(),
                            module.bias.detach().cpu().numpy().astype(np.float32).copy()))
    return weights
