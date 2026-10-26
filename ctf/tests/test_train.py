import json

import numpy as np

from ctf.train import gae, main


def _reference_gae(rewards, values, alive, gamma, lam):
    """Slow per-slot version: decisions are alive ticks; dead ticks fold into the previous one."""
    T, N = rewards.shape
    out = np.zeros((T, N))
    for n in range(N):
        steps = [t for t in range(T) if alive[t, n]]
        next_adv = 0.0
        for i in range(len(steps) - 1, -1, -1):
            t = steps[i]
            end = steps[i + 1] if i + 1 < len(steps) else T
            ret = sum(gamma ** (s - t) * rewards[s, n] for s in range(t, end))
            next_v = values[end, n] if end < T else 0.0
            disc = gamma ** (end - t)
            delta = ret + disc * next_v - values[t, n]
            next_adv = delta + disc * lam * next_adv
            out[t, n] = next_adv
    return out


def test_gae_matches_reference_with_respawn_gaps():
    rng = np.random.default_rng(0)
    T, N = 400, 5
    rewards = rng.normal(size=(T, N))
    values = rng.normal(size=(T, N))
    alive = rng.random((T, N)) > 0.2
    for n in range(N):  # long death gaps
        start = rng.integers(0, T - 70)
        alive[start:start + 61, n] = False
    values = np.where(alive, values, 0.0)
    fast = gae(rewards, values, alive, 0.99, 0.95)
    slow = _reference_gae(rewards, values, alive, 0.99, 0.95)
    assert np.allclose(fast, np.where(alive, slow, 0.0), atol=1e-4)


def test_training_smoke(tmp_path):
    main(["--name", "smoke", "--runs", str(tmp_path), "--workers", "2", "--envs", "2",
          "--max-ticks", "200", "--updates", "2", "--minibatch", "256", "--save-every", "1",
          "--snapshot-every", "1", "--opponents", "idle:1,baseline:1,mirror:1,pool:1", "--win-coef", "0.5",
          "--learner-threads", "2"])
    lines = (tmp_path / "smoke" / "log.jsonl").read_text().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[-1])
    assert record["samples"] > 0 and np.isfinite(record["pi_loss"])
    assert (tmp_path / "smoke" / "policies" / "u00002.npz").exists()
    # resume continues from the checkpoint
    main(["--name", "smoke", "--runs", str(tmp_path), "--workers", "1", "--envs", "1",
          "--max-ticks", "100", "--updates", "3", "--minibatch", "256", "--learner-threads", "2"])
    assert len((tmp_path / "smoke" / "log.jsonl").read_text().splitlines()) == 3


def test_training_smoke_gru(tmp_path):
    from ctf.policy_io import load_policy

    main(["--name", "gru", "--runs", str(tmp_path), "--arch", "gru", "--workers", "2",
          "--envs", "2", "--max-ticks", "200", "--updates", "2", "--minibatch", "512",
          "--chunk", "32", "--burn-in", "16", "--memory-wipe", "0.5", "--save-every", "1",
          "--snapshot-every", "1", "--opponents", "idle:1,mirror:1,pool:1",
          "--learner-threads", "2"])
    log = (tmp_path / "gru" / "log.jsonl").read_text()
    records = [json.loads(line) for line in log.splitlines()]
    assert len(records) == 2
    for record in records:
        assert record["samples"] > 0 and np.isfinite(record["pi_loss"]) and record["chunks"] > 0
        assert record["logp_drift"] < 1e-4, record["logp_drift"]
    actor, meta = load_policy(tmp_path / "gru" / "policies" / "u00002.npz")
    assert actor.recurrent and meta["arch"] == "gru"
    # resume continues; an arch mismatch is refused
    main(["--name", "gru", "--runs", str(tmp_path), "--arch", "gru", "--workers", "1",
          "--envs", "1", "--max-ticks", "100", "--updates", "3", "--learner-threads", "2"])
    assert len((tmp_path / "gru" / "log.jsonl").read_text().splitlines()) == 3
    try:
        main(["--name", "gru", "--runs", str(tmp_path), "--workers", "1", "--updates", "4"])
    except ValueError as exc:
        assert "arch" in str(exc)
    else:
        raise AssertionError("loading a gru checkpoint into an mlp actor must fail")


def test_pipelined_training_smoke(tmp_path):
    main(["--name", "pipe", "--runs", str(tmp_path), "--workers", "2", "--envs", "1",
          "--max-ticks", "150", "--updates", "4", "--minibatch", "256", "--norm-updates", "1",
          "--opponents", "baseline:1,mirror:1", "--learner-threads", "2", "--pipeline", "1"])
    records = [json.loads(line) for line in (tmp_path / "pipe" / "log.jsonl").read_text().splitlines()]
    assert len(records) == 4
    assert [r["pipelined"] for r in records] == [False, True, True, False]
    assert all(np.isfinite(r["pi_loss"]) for r in records)


def test_adaptive_mix_shifts_weight_to_harder_opponents(tmp_path):
    from ctf.train import Trainer, main as _main  # noqa: F401
    import argparse
    from ctf import train as T
    parser_args = ["--name", "mix", "--runs", str(tmp_path), "--opponents",
                   "baseline:0.5,relay:0.5,mirror:0.2", "--adaptive-mix", "1"]
    captured = {}

    class Probe(T.Trainer):
        def run(self):
            captured["trainer"] = self

    original = T.Trainer
    T.Trainer = Probe
    try:
        T.main(parser_args)
    finally:
        T.Trainer = original
    trainer = captured["trainer"]
    trainer.opp_winrate = {"baseline": 0.95, "relay": 0.05}
    names, probs = trainer.mix_probabilities()
    p = dict(zip(names, probs))
    assert p["relay"] > 5 * p["baseline"]
    assert abs(sum(probs) - 1) < 1e-9 and p["mirror"] > 0


def test_net2wider_preserves_function():
    import torch

    from ctf.nets import Actor, Critic
    from ctf.train import widen_mlp

    torch.manual_seed(0)
    small, big = Actor((32, 32)), Actor((64, 80))
    wide = widen_mlp(small.state_dict(), (64, 80), np.random.default_rng(0), noise=0.0)
    big.load_state_dict(wide)
    x = torch.randn(50, small.net[0].in_features)
    assert torch.allclose(small(x), big(x), atol=1e-5)
    c_small, c_big = Critic((16, 16)), Critic((48, 48))
    c_big.load_state_dict(widen_mlp(c_small.state_dict(), (48, 48), np.random.default_rng(1), 0.0))
    assert torch.allclose(c_small(x), c_big(x), atol=1e-5)


def test_v3_to_v4_migration_preserves_action_distribution():
    """Probabilities of the original throttle values are preserved when duplicates are merged."""
    import torch

    from ctf import feature_layouts as L
    from ctf.train import V3_DIM, V3_BASE, V3_THROTTLE, migrate_v3_to_v4

    v4_throttle = L.V4_ACTION_TABLES["throttle"]
    v4_dim, team_dim, n_throttle = len(L.V4_FEATURE_NAMES), 8, len(v4_throttle)
    torch.manual_seed(0)
    old = torch.nn.Sequential(torch.nn.Linear(V3_DIM, 16), torch.nn.Tanh(), torch.nn.Linear(16, 23))
    state = {"actor": {f"net.{k}": v.detach().clone() for k, v in
                       {"0.weight": old[0].weight, "0.bias": old[0].bias,
                        "2.weight": old[2].weight, "2.bias": old[2].bias}.items()},
             "critic": {}, "norm": {"mean": np.zeros(V3_DIM), "var": np.ones(V3_DIM), "count": 1.0}}
    new = migrate_v3_to_v4(state)
    assert new["feature_names"] == list(L.V4_FEATURE_NAMES)
    x_old = torch.randn(5, V3_DIM)
    x_old[:, V3_BASE:] = 0
    x_old[:, V3_BASE + 3] = 1.0          # previous throttle -0.03 (v3)
    x_new = torch.zeros(5, v4_dim)
    x_new[:, :V3_BASE] = x_old[:, :V3_BASE]
    x_new[:, V3_BASE + team_dim + v4_throttle.index(-0.034)] = 1.0
    h_old = torch.tanh(x_old @ old[0].weight.T + old[0].bias)
    h_new = torch.tanh(x_new @ new["actor"]["net.0.weight"].T + new["actor"]["net.0.bias"])
    assert torch.allclose(h_old, h_new, atol=1e-5)
    logits_old = h_old @ old[2].weight.T + old[2].bias
    logits_new = h_new @ new["actor"]["net.2.weight"].T + new["actor"]["net.2.bias"]
    p_old = torch.softmax(logits_old[:, :12], -1)
    p_new = torch.softmax(logits_new[:, :n_throttle], -1)
    nearest = [min(range(12), key=lambda i: abs(V3_THROTTLE[i] - v)) for v in v4_throttle]
    merged = torch.zeros_like(p_old)
    for j, i in enumerate(nearest):
        merged[:, i] += p_new[:, j]
    assert torch.allclose(merged, p_old, atol=1e-5)
    assert torch.allclose(logits_old[:, 12:], logits_new[:, n_throttle:], atol=1e-5)
