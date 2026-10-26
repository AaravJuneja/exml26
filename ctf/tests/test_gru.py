"""Recurrent (GRU) actor: numpy/torch/export parity, rollout <-> learner log-prob consistency."""

import importlib.util
import itertools
import json
import time

import numpy as np
import pytest
import torch

from ctf import features as F
from ctf.export import export
from ctf.nets import GRUActor, actor_to_numpy, log_prob_entropy
from ctf.npnets import GRU_KEYS, HEAD_SIZES, NumpyActor, NumpyGRUActor, PolicyMemories
from ctf.policy_io import load_policy, save_policy
from ctf.rewards import DEFAULT_WEIGHTS
from ctf.rollout import collect
from ctf.sim import Match
from ctf.train import build_sequences, hidden_every, merge_results, sequence_logits

_counter = itertools.count()


def _gru_weights(seed, hidden=(64, 32)):
    """GRU weights with a perturbed head (varied actions, fast spheres that die) and
    a non-trivial recurrent part, plus a non-trivial normalization."""
    torch.manual_seed(seed)
    weights = actor_to_numpy(GRUActor(hidden))
    rng = np.random.default_rng(seed)
    for key in ("W_out", "b_out"):
        weights[key] = weights[key] + rng.normal(0.0, 0.5, weights[key].shape).astype(np.float32)
    weights["b_out"][F.N_THROTTLE - 1] += 1.5
    weights["W_hh"] = weights["W_hh"] * np.float32(1.5)
    mean = rng.normal(0.0, 0.1, F.FEATURE_DIM).astype(np.float32)
    std = rng.uniform(0.3, 2.0, F.FEATURE_DIM).astype(np.float32)
    return weights, mean, std


def _policy(tmp_path, seed):
    weights, mean, std = _gru_weights(seed)
    path = tmp_path / f"gru{seed}.npz"
    save_policy(path, weights, mean, std, notes="test")
    return path


def _load_module(path):
    spec = importlib.util.spec_from_file_location(f"exported_gru_{next(_counter)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numpy_gru_matches_torch():
    torch.manual_seed(0)
    actor = GRUActor((64, 32))
    weights, mean, std = _gru_weights(0)
    with torch.no_grad():  # load the perturbed weights back into torch
        actor.inp.weight.copy_(torch.as_tensor(weights["W_in"].T))
        actor.inp.bias.copy_(torch.as_tensor(weights["b_in"]))
        actor.gru.weight_ih_l0.copy_(torch.as_tensor(weights["W_ih"].T))
        actor.gru.bias_ih_l0.copy_(torch.as_tensor(weights["b_ih"]))
        actor.gru.weight_hh_l0.copy_(torch.as_tensor(weights["W_hh"].T))
        actor.gru.bias_hh_l0.copy_(torch.as_tensor(weights["b_hh"]))
        actor.out.weight.copy_(torch.as_tensor(weights["W_out"].T))
        actor.out.bias.copy_(torch.as_tensor(weights["b_out"]))
    assert all(np.array_equal(actor_to_numpy(actor)[k], weights[k]) for k in GRU_KEYS)
    numpy_actor = NumpyGRUActor(weights, mean, std)
    x = np.random.default_rng(1).normal(0.0, 3.0, (6, 50, F.FEATURE_DIM)).astype(np.float32)
    h = np.random.default_rng(2).normal(0.0, 0.5, (6, 32)).astype(np.float32)
    h0, steps = h.copy(), []
    for t in range(x.shape[1]):
        logits, h = numpy_actor.step(x[:, t], h)
        steps.append(logits)
    xt = torch.clamp((torch.as_tensor(x) - torch.as_tensor(mean)) / torch.as_tensor(std), -5, 5)
    with torch.no_grad():
        logits_t, h_t = actor(xt, torch.as_tensor(h0))
    assert np.abs(logits_t.numpy() - np.stack(steps, 1)).max() < 1e-5
    assert np.abs(h_t.numpy() - h).max() < 1e-5


def test_policy_file_roundtrip_and_backward_compat(tmp_path):
    path = _policy(tmp_path, 1)
    actor, meta = load_policy(path)
    assert isinstance(actor, NumpyGRUActor) and actor.recurrent
    assert meta["arch"] == "gru" and meta["gru_hidden"] == 32
    weights, mean, _ = _gru_weights(1)
    assert all(np.array_equal(actor.weights[k], weights[k]) for k in GRU_KEYS)
    assert np.array_equal(actor.obs_mean, mean)
    # A file written before `arch` existed (meta without it) still loads as an MLP.
    old = tmp_path / "old.npz"
    meta = {"feature_dim": F.FEATURE_DIM, "head_sizes": list(HEAD_SIZES),
            "activation": "tanh", "feature_version": 1}
    w0 = np.ones((F.FEATURE_DIM, sum(HEAD_SIZES)), dtype=np.float32)
    np.savez(old, W0=w0, b0=np.zeros(sum(HEAD_SIZES), np.float32),
             obs_mean=np.zeros(F.FEATURE_DIM, np.float32),
             obs_std=np.ones(F.FEATURE_DIM, np.float32), meta=np.array(json.dumps(meta)))
    actor, meta = load_policy(old)
    assert type(actor) is NumpyActor and meta["arch"] == "mlp" and len(actor.weights) == 1


@pytest.mark.parametrize("chunk,burn_in", [(16, 0), (16, 8), (32, 32)])
def test_learner_recomputes_rollout_log_probs(chunk, burn_in):
    """With unchanged weights, log-probs recomputed through chunked GRU sequences
    (stored start states, optional burn-in) equal the rollout's to 1e-4."""
    weights, mean, std = _gru_weights(3)
    actor = NumpyGRUActor(weights, mean, std)
    jobs = [{"seed": 101, "side": "red", "opponent": "mirror"},
            {"seed": 102, "side": "blue", "opponent": "self"},
            {"seed": 103, "side": "red", "opponent": "baseline"}]
    data = collect(actor, jobs, dict(DEFAULT_WEIGHTS), max_ticks=500, seed=7,
                   memory_wipe_prob=1.0, hidden_every=hidden_every(chunk, burn_in))
    batch = merge_results([data, data], recurrent=True)  # two "workers": exercises re-keying
    T, U = batch["alive"].shape[:2]
    alive = batch["alive"].reshape(T, U * 3)
    resets = batch["resets"].reshape(T, U * 3)
    assert not np.any(resets & ~alive)

    # The data exercises resets mid-sequence (wipes) and respawn gaps without a reset.
    n_resets = resets.sum(0)
    assert np.all(n_resets >= 1) and np.any(n_resets >= 2)
    gaps_without_reset = 0
    for n in range(U * 3):
        ticks = np.flatnonzero(alive[:, n])
        for prev, cur in zip(ticks[:-1], ticks[1:]):
            gaps_without_reset += cur - prev > 30 and not resets[cur, n]
    assert gaps_without_reset > 0

    seqs = build_sequences(alive, resets, batch["h0_key"], chunk, burn_in)
    positions = seqs["main"][seqs["main"] >= 0]
    n = int(alive.sum())
    assert np.array_equal(np.sort(positions), np.arange(n))  # every decision exactly once
    assert seqs["has_burn"].any() == (burn_in > 0)

    torch_actor = GRUActor((64, 32))
    with torch.no_grad():
        for module, (w, b) in ((torch_actor.inp, ("W_in", "b_in")),
                               (torch_actor.out, ("W_out", "b_out"))):
            module.weight.copy_(torch.as_tensor(weights[w].T))
            module.bias.copy_(torch.as_tensor(weights[b]))
        for name, key in (("weight_ih_l0", "W_ih"), ("weight_hh_l0", "W_hh")):
            getattr(torch_actor.gru, name).copy_(torch.as_tensor(weights[key].T))
        for name, key in (("bias_ih_l0", "b_ih"), ("bias_hh_l0", "b_hh")):
            getattr(torch_actor.gru, name).copy_(torch.as_tensor(weights[key]))

    flat = np.flatnonzero(alive.reshape(-1))
    x_all = torch.as_tensor(batch["obs"].reshape(T * U * 3, -1)[flat])
    x_all = torch.clamp((x_all - torch.as_tensor(mean)) / torch.as_tensor(std), -5.0, 5.0)
    acts = torch.as_tensor(batch["actions"].reshape(-1, 3)[flat].astype(np.int64))
    old_logp = batch["logp"].reshape(-1)[flat]
    tseqs = {k: torch.as_tensor(v) for k, v in seqs.items()}
    h0 = torch.as_tensor(batch["h0"])
    got = np.full(n, np.nan)
    order = torch.randperm(len(seqs["main"]))
    with torch.no_grad():
        for start in range(0, len(order), 7):
            logits, idx = sequence_logits(torch_actor, x_all, h0, tseqs, order[start:start + 7])
            logp, _ = log_prob_entropy(logits, acts[idx])
            got[idx.numpy()] = logp.numpy()
    err = np.abs(got - old_logp).max()
    assert err < 1e-4, err
    assert old_logp.std() > 0.5  # a stochastic policy, so the check is meaningful


def _reference_play(model_path, policy_path, seed, ticks):
    """Exported per-player modules against NumpyGRUActor (one row per player per call)."""
    match = Match(seed, max_ticks=ticks)
    modules = {s.id: _load_module(model_path) for s in match.spheres}
    actor, _ = load_policy(policy_path)
    reference = PolicyMemories(actor, [s.id for s in match.spheres], F.encode, F.PlayerMemory)
    stats = {"calls": 0, "respawns": 0, "max_err": 0.0, "state_err": 0.0, "actions": set()}
    last_seen = {}
    while not match.is_over():
        actions = {}
        for sid, observation in match.observe().items():
            action = modules[sid].decide(observation)
            x = reference.encode_batch([sid], [observation])
            h_before = reference.hidden[sid].copy()
            logits, _ = actor.step(x, h_before[None])
            chosen = reference.act([sid], x, None, greedy=True)[0]
            got = modules[sid]._MX_LOGITS
            stats["max_err"] = max(stats["max_err"], float(np.abs(got - logits[0]).max()))
            stats["state_err"] = max(stats["state_err"], float(
                np.abs(modules[sid]._MX_STATE - reference.hidden[sid]).max()))
            expected = (F.THROTTLE_VALUES[chosen[0]], F.TURN_VALUES[chosen[1]], bool(chosen[2]))
            assert action == expected, (observation["tick"], sid)
            stats["actions"].add(expected)
            if sid in last_seen and observation["tick"] - last_seen[sid] > 1:
                stats["respawns"] += 1
            last_seen[sid] = observation["tick"]
            stats["calls"] += 1
            actions[sid] = action
        match.act(actions)
    stats["deaths"] = len(match.death_log())
    return stats, modules


@pytest.fixture(scope="module")
def exported_gru(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("gru_export")
    policy = _policy(tmp_path, 11)
    return policy, export(policy, tmp_path / "model_gru.py")


def test_exported_gru_matches_numpy_actor(exported_gru):
    policy, model = exported_gru
    stats, modules = _reference_play(model, policy, seed=100, ticks=800)
    assert stats["max_err"] < 1e-6 and stats["state_err"] < 1e-6, stats
    assert stats["deaths"] > 0 and stats["respawns"] > 0, stats
    assert len(stats["actions"]) >= 5
    assert "GRU 32" in model.read_text().split("\n", 8)[4]

    # New match (tick goes backwards): memory and hidden state restart from zero.
    module = modules[0]
    assert np.any(module._MX_STATE != 0)
    observation = Match(2, max_ticks=1).observe()[0]
    module.decide(observation)
    actor, _ = load_policy(policy)
    x = np.asarray([F.encode(observation, F.PlayerMemory())], dtype=np.float32)
    logits, h = actor.step(x, actor.initial_state(1))
    assert np.abs(module._MX_LOGITS - logits[0]).max() < 1e-6
    assert np.abs(module._MX_STATE - h[0]).max() < 1e-6

    # The same tick decided twice gives the same answer and advances the state once.
    state = module._MX_STATE.copy()
    again = module.decide(observation)
    assert np.abs(module._MX_LOGITS - logits[0]).max() < 1e-6
    assert np.array_equal(module._MX_STATE, state)
    assert again == module.decide(observation)


def test_exported_gru_recovers_from_bad_input(exported_gru):
    module = _load_module(exported_gru[1])
    for bad in (None, {}, {"tick": "x"}, {"self": {"x": "nan"}}, 42):
        throttle, turn, jump = module.decide(bad)
        assert throttle in F.THROTTLE_VALUES and isinstance(jump, bool)
    module.decide(Match(3, max_ticks=1).observe()[0])
    assert np.all(np.isfinite(module._MX_LOGITS)) and np.all(np.isfinite(module._MX_STATE))


def test_evaluator_plays_gru_npz_like_the_exported_model(exported_gru):
    from ctf import evaluate as E

    policy, model = exported_gru
    reference = E.evaluate(f"{policy}@greedy", "example", [3, 4], max_ticks=900, n_boot=50)[1]
    assert sum(r["a"]["wall_deaths"] + r["a"]["player_deaths"] for r in reference) >= 2
    for spec in (str(model), f"{model}@proc"):
        _, records = E.evaluate(spec, "example", [3, 4], max_ticks=900, n_boot=50)
        assert len(records) == len(reference) == 4
        for got, want in zip(records, reference):
            got, want = E.deterministic_view(got), E.deterministic_view(want)
            for key in ("spec_a", "digest_a"):
                got.pop(key), want.pop(key)
            assert got == want, spec


def test_exported_gru_decide_is_fast(tmp_path):
    torch.manual_seed(5)
    weights = actor_to_numpy(GRUActor((128, 128)))  # production size
    path = tmp_path / "gru128.npz"
    save_policy(path, weights, np.zeros(F.FEATURE_DIM), np.ones(F.FEATURE_DIM))
    module = _load_module(export(path, tmp_path / "gru128.py"))
    match = Match(5, max_ticks=400)
    times = []
    while not match.is_over():
        observation = match.observe().get(0)
        if observation is not None:
            start = time.perf_counter()
            action = module.decide(observation)
            times.append(time.perf_counter() - start)
        match.act({0: action} if observation is not None else {})
    assert np.median(times) < 1e-3, np.median(times)
    assert np.percentile(times, 99) < 3e-3, np.percentile(times, 99)
