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
