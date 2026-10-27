"""Team critic (--team-critic) and kill reward: opt-in behaviour of the breakthrough experiments."""

import numpy as np
import torch

from ctf.nets import Critic
from ctf.policies import SCRIPTED, load_decide
from ctf.rewards import RewardTracker
from ctf.sim import Match
from ctf.train import (main, pad_optimizer_state, team_critic_dim, team_input, team_mates,
                       widen_critic_input)


def test_team_mates_pairs_teammates_on_the_same_tick():
    T, units = 4, 2
    N = units * 3
    alive = np.ones((T, N), dtype=bool)
    alive[1, 4] = False  # unit 1, slot 1 dead at t = 1
    flat = np.flatnonzero(alive.reshape(-1))
    mates = team_mates(flat, N)
    pos = {int(f): i for i, f in enumerate(flat)}
    for i, f in enumerate(flat):
        t, n = divmod(int(f), N)
        u, k = divmod(n, 3)
        others = [j for j in range(3) if j != k]
        for col, j in enumerate(others):
            g = t * N + u * 3 + j
            assert mates[i, col] == (pos[g] if alive[t, u * 3 + j] else -1)


def test_widened_critic_computes_the_same_values():
    D = 7
    critic = Critic((16, 16), D)
    team = Critic((16, 16), team_critic_dim(D))
    team.load_state_dict(widen_critic_input(critic.state_dict(), D))
    x = torch.randn(5, D)
    mates = torch.tensor([[1, 2], [0, -1], [-1, -1], [4, 0], [3, 2]])
    with torch.no_grad():
        assert torch.allclose(critic(x), team(team_input(x, mates, torch.arange(5))), atol=1e-6)


def test_pad_optimizer_state_after_widening():
    p = torch.nn.Parameter(torch.zeros(3, 4))
    opt = torch.optim.Adam([p])
    p.grad = torch.ones(3, 4)
    opt.step()
    state = opt.state_dict()
    q = torch.nn.Parameter(torch.zeros(3, 9))
    opt2 = torch.optim.Adam([q])
    opt2.load_state_dict(state)
    pad_optimizer_state(opt2)
    assert opt2.state[q]["exp_avg"].shape == (3, 9)
    assert torch.all(opt2.state[q]["exp_avg"][:, 4:] == 0)
    q.grad = torch.ones(3, 9)
    opt2.step()


def test_team_critic_training_smoke(tmp_path):
    main(["--name", "tc", "--runs", str(tmp_path), "--workers", "1", "--envs", "2",
          "--max-ticks", "150", "--updates", "2", "--minibatch", "256", "--save-every", "1",
          "--opponents", "baseline:1,mirror:1", "--team-critic", "1", "--learner-threads", "1",
          "--action-space", "goal"])
    # warm start a team critic from the per-player checkpoint, and resume it
    main(["--name", "plain", "--runs", str(tmp_path), "--workers", "1", "--envs", "1",
          "--max-ticks", "100", "--updates", "1", "--minibatch", "256", "--save-every", "1",
          "--learner-threads", "1", "--action-space", "goal"])
    main(["--name", "tc2", "--runs", str(tmp_path), "--workers", "1", "--envs", "1",
          "--max-ticks", "100", "--updates", "1", "--minibatch", "256", "--save-every", "1",
          "--learner-threads", "1", "--action-space", "goal", "--team-critic", "1",
          "--init", str(tmp_path / "plain" / "checkpoint.pt")])
    import shutil
    (tmp_path / "tc3").mkdir()
    shutil.copy(tmp_path / "plain" / "checkpoint.pt", tmp_path / "tc3" / "checkpoint.pt")
    main(["--name", "tc3", "--runs", str(tmp_path), "--workers", "1", "--envs", "1",
          "--max-ticks", "100", "--updates", "2", "--minibatch", "256", "--save-every", "1",
          "--learner-threads", "1", "--action-space", "goal", "--team-critic", "1"])


def test_kill_reward_credits_the_killer():
    def play(weights):
        match = Match(11, max_ticks=1500, teammates_field=True)
        tracker = RewardTracker(match, "red", weights)
        decide = {s.id: load_decide(SCRIPTED["relay"]) for s in match.spheres}
        total = 0.0
        while not match.is_over():
            obs = match.observe()
            tracker.before()
            match.act({sid: decide[sid](o) for sid, o in obs.items()})
            total += sum(tracker.after())
        return match, tracker, total

    match, plain, total0 = play({})
    assert "kill" not in plain.totals and "kills" not in plain.counts
    match, tracker, total1 = play({"kill": 0.5})
    enemy_kills = sum(1 for m in match.death_log() if m["team"] == "blue" and m["reason"] == "player")
    assert 0 < tracker.counts["kills"] <= enemy_kills
    assert abs((total1 - total0) - 0.5 * tracker.counts["kills"]) < 1e-6
