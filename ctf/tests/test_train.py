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


def test_feature_surgery_preserves_old_policy(tmp_path):
    """A checkpoint built for fewer features plays identically after expansion (new inputs zeroed)."""
    import torch

    from ctf import features as F
    from ctf.nets import Actor
    from ctf.train import expand_features, feature_index_map

    old_dim = F.FEATURE_DIM - 11
    torch.manual_seed(0)
    old = Actor((32, 32)) if not hasattr(Actor, "__wrapped__") else None
    # build an old-width actor by hand
    old_net = torch.nn.Sequential(torch.nn.Linear(old_dim, 32), torch.nn.Tanh(),
                                  torch.nn.Linear(32, 23))
    state = {"actor": {"net.0.weight": old_net[0].weight.detach().clone(),
                       "net.0.bias": old_net[0].bias.detach().clone(),
                       "net.2.weight": old_net[2].weight.detach().clone(),
                       "net.2.bias": old_net[2].bias.detach().clone()},
             "critic": {}, "norm": {"mean": np.zeros(old_dim), "var": np.ones(old_dim), "count": 5.0}}
    new = expand_features(state, old_dim)
    x_old = torch.randn(7, old_dim)
    x_new = torch.randn(7, F.FEATURE_DIM)  # random values in the new feature slots
    x_new[:, feature_index_map(old_dim)] = x_old
    w, b = new["actor"]["net.0.weight"], new["actor"]["net.0.bias"]
    assert torch.allclose(x_old @ old_net[0].weight.T + old_net[0].bias, x_new @ w.T + b, atol=1e-5)
    assert new["norm"]["mean"].shape == (F.FEATURE_DIM,)
    del old


def test_pipelined_training_smoke(tmp_path):
    main(["--name", "pipe", "--runs", str(tmp_path), "--workers", "2", "--envs", "1",
          "--max-ticks", "150", "--updates", "4", "--minibatch", "256", "--norm-updates", "1",
          "--opponents", "baseline:1,mirror:1", "--learner-threads", "2", "--pipeline", "1"])
    records = [json.loads(line) for line in (tmp_path / "pipe" / "log.jsonl").read_text().splitlines()]
    assert len(records) == 4
    assert [r["pipelined"] for r in records] == [False, True, True, False]
    assert all(np.isfinite(r["pi_loss"]) for r in records)
