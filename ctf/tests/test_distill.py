import json
import math

import numpy as np

from ctf import action_space as A
from ctf import features as F
from ctf.controller import control, goal_step
from ctf.distill import GOAL_H, GOAL_S, PolicyTeacher, control_batch, snap_indices
from ctf.npnets import NumpyActor
from ctf.policy_io import random_policy


def test_control_batch_matches_scalar_controller():
    rng = np.random.default_rng(0)
    for _ in range(60):
        me = {"yaw": float(rng.uniform(-7, 7)), "vx": float(rng.normal(0, 8)),
              "vy": float(rng.normal(0, 8))}
        if rng.random() < 0.2:
            me["vx"], me["vy"] = 0.0, 0.0
        for reverse in (False, True):
            u, turn, rev = control_batch(me["yaw"], me["vx"], me["vy"], GOAL_H[0], GOAL_S[0],
                                         reverse)
            for g in range(GOAL_H.shape[1]):
                su, sturn, srev = control(me, float(GOAL_H[0, g]), float(GOAL_S[0, g]), reverse)
                assert abs(su - u[g]) < 1e-9 and abs(sturn - turn[g]) < 1e-9
                assert srev == bool(rev[g])


def test_snap_matches_nearest_bins():
    values = np.concatenate([np.linspace(-1.2, 1.2, 2001), np.array(F.THROTTLE_VALUES),
                             np.array(F.TURN_VALUES)])
    ti, ri = snap_indices(values, values)
    for v, t, r in zip(values, ti, ri):
        assert t == A._nearest_index(F.THROTTLE_VALUES, v)
        assert r == A._nearest_index(F.TURN_VALUES, v)


def test_capped_snap_respects_the_speed_limit_when_it_can():
    from ctf.distill import snap_capped

    rng = np.random.default_rng(1)
    changed = 0
    for _ in range(200):
        yaw, vx, vy = rng.uniform(-4, 4), rng.normal(0, 7), rng.normal(0, 7)
        u, turn, _ = control_batch(yaw, vx, vy, GOAL_H[0], GOAL_S[0], False)
        near_t, near_r = snap_indices(u, turn)
        cap_t, cap_r = snap_capped(yaw, vx, vy, u, turn, GOAL_S[0])
        assert np.array_equal(near_r, cap_r)
        assert np.all(np.abs(cap_t - near_t) <= 1)
        moved = cap_t != near_t
        changed += int(moved.sum())
        # A moved bin is the other neighbour of the controller's throttle.
        lo = np.minimum(np.array(F.THROTTLE_VALUES)[cap_t], np.array(F.THROTTLE_VALUES)[near_t])
        hi = np.maximum(np.array(F.THROTTLE_VALUES)[cap_t], np.array(F.THROTTLE_VALUES)[near_t])
        assert np.all(~moved | ((lo <= u + 1e-12) & (u <= hi + 1e-12)))
    assert changed > 0


def test_dither_weights_are_unbiased():
    from ctf.distill import bins_weights

    rng = np.random.default_rng(2)
    u = rng.uniform(-1, 1, 500)
    turn = rng.uniform(-1, 1, 500)
    t_lo, t_hi, w_t, r_lo, r_hi, w_r = bins_weights("dither", 0.0, 0.0, 0.0, u, turn, 5.0)
    T, R = np.array(F.THROTTLE_VALUES), np.array(F.TURN_VALUES)
    assert np.allclose(T[t_lo] * (1 - w_t) + T[t_hi] * w_t, u)
    assert np.allclose(R[r_lo] * (1 - w_r) + R[r_hi] * w_r, turn)
    for mode in ("dcap", "dlethal", "near", "cap"):
        out = bins_weights(mode, 0.3, 9.0, 6.0, u, turn, 10.85)
        assert all(np.all((w >= 0) & (w <= 1)) for w in (out[2], out[5]))


def test_dlethal_never_rounds_a_safe_push_into_a_lethal_one():
    from ctf.distill import LETHAL_GUARD, bins_weights
    from ctf.controller import CTL_DT, CTL_FRICTION, CTL_TURN_BASE, CTL_TURN_PENALTY

    rng = np.random.default_rng(4)
    T = np.array(F.THROTTLE_VALUES)
    moved = 0
    for _ in range(300):
        yaw, speed, angle = rng.uniform(-3, 3), rng.uniform(5, 11), rng.uniform(-3, 3)
        vx, vy = speed * math.sin(angle), speed * math.cos(angle)
        u, turn, _ = control_batch(yaw, vx, vy, GOAL_H[0], GOAL_S[0], False)
        t_lo, t_hi, w_t, *_ = bins_weights("dlethal", yaw, vx, vy, u, turn, GOAL_S[0])
        _, _, w_d, *_ = bins_weights("dither", yaw, vx, vy, u, turn, GOAL_S[0])
        moved += int((w_t != w_d).sum())
        rate = CTL_TURN_BASE * (1 - CTL_TURN_PENALTY * min(speed / 18.0, 1.0)) * CTL_DT
        new_yaw = yaw + np.clip(turn, -1, 1) * rate

        def post(push):
            return CTL_FRICTION * np.hypot(vx + 6 * push * np.sin(new_yaw),
                                           vy + 6 * push * np.cos(new_yaw))

        safe = post(u) < LETHAL_GUARD
        for index, weight in ((t_lo, 1 - w_t), (t_hi, w_t)):
            lethal = post(T[index]) >= LETHAL_GUARD
            other = np.where(index == t_lo, t_hi, t_lo)
            avoidable = post(T[other]) < LETHAL_GUARD
            assert not np.any(safe & lethal & avoidable & (weight > 0))
    assert moved > 0


def test_snap_spec_round_trip():
    from ctf.evaluate import parse_spec

    path = "teams/exml_rl_pure/policy.npz"
    spec = parse_spec(path + "@sample,seed=3,snap=cap")
    assert spec.snap == "cap" and not spec.greedy and spec.sample_seed == 3
    assert parse_spec(str(spec)) == spec
    assert parse_spec(path + "@snap").snap == "near"
    assert parse_spec(path).snap == ""


def _goal_actor(tmp_path, seed=0):
    path = tmp_path / f"goal{seed}.npz"
    random_policy(path, hidden=(16, 16), seed=seed, space="goal")
    from ctf.policy_io import load_policy
    return load_policy(path)[0], path


def _observation(tick, yaw, vx, vy):
    from ctf.sim import Match
    from ctf.rollout import documented

    match = Match(5, max_ticks=50)
    sid = match.spheres[0].id
    doc = documented(match.observe()[sid])
    doc = dict(doc, tick=tick, self=dict(doc["self"], yaw=yaw, vx=vx, vy=vy))
    return doc


def test_policy_teacher_soft_targets(tmp_path):
    actor, _ = _goal_actor(tmp_path)
    teacher = PolicyTeacher(actor, snap="cap", commit="sample")
    rng = np.random.default_rng(0)
    memories = [teacher.memory() for _ in range(3)]
    docs = [_observation(0, 0.3, 4.0, -2.0), _observation(0, -2.0, 0.0, 0.0),
            _observation(0, 1.0, -9.0, 3.0)]
    soft, hard = teacher.label(memories, docs, rng)
    nt, nr = F.N_THROTTLE, F.N_TURN
    assert soft.shape == (3, nt + nr + 2)
    for head in (soft[:, :nt], soft[:, nt:nt + nr], soft[:, nt + nr:]):
        assert np.allclose(head.sum(1), 1.0, atol=1e-5) and head.min() >= 0.0
    assert hard[:, 0].max() < nt and hard[:, 1].max() < nr
    # The committed goal drives the shadow memory exactly as goal_step would.
    for memory, doc in zip(memories, docs):
        twin = F.PlayerMemory(goal=True)
        twin.ctrl = (None, False, False)
        throttle, _, _ = goal_step(twin, doc, memory.prev_heading, memory.prev_speed,
                                   memory.prev_jump, F.HEADING_VALUES, F.SPEED_VALUES)
        assert memory.ctrl == twin.ctrl
        assert math.isclose(memory.prev_push, throttle, abs_tol=1e-9)


def test_policy_teacher_deterministic_labels_are_the_snapped_controller(tmp_path):
    actor, _ = _goal_actor(tmp_path)
    w = [(np.zeros_like(a), np.zeros_like(b)) for a, b in actor.weights]
    bias = np.full(w[-1][1].shape, -50.0, dtype=np.float32)
    h, s = 17, 4  # one goal and no jump with probability ~1
    bias[h] = bias[F.N_HEADING + s] = bias[F.N_HEADING + F.N_SPEED] = 50.0
    w[-1] = (w[-1][0], bias)
    sure = NumpyActor(w, actor.obs_mean, actor.obs_std, "goal")
    teacher = PolicyTeacher(sure, snap="near")
    memory = teacher.memory()
    doc = _observation(3, 0.7, 5.0, 1.0)
    soft, hard = teacher.label([memory], [doc], np.random.default_rng(0))
    twin = F.PlayerMemory(goal=True)
    throttle, turn, _ = goal_step(twin, doc, h, s, 0, F.HEADING_VALUES, F.SPEED_VALUES)
    assert hard[0].tolist() == [A._nearest_index(F.THROTTLE_VALUES, throttle),
                                A._nearest_index(F.TURN_VALUES, turn), 0]
    assert soft[0].max() > 0.999


def test_posterior_commit_explains_the_executed_action(tmp_path):
    from ctf.distill import bins_weights

    actor, _ = _goal_actor(tmp_path)
    teacher = PolicyTeacher(actor)  # dither, posterior
    rng = np.random.default_rng(3)

    def bins_of(me, h, s, draw):
        u, turn, _ = control(me, math.radians(F.HEADING_VALUES[h]), F.SPEED_VALUES[s], False)
        t_lo, t_hi, w_t, r_lo, r_hi, w_r = bins_weights("dither", me["yaw"], me["vx"], me["vy"],
                                                        u, turn, F.SPEED_VALUES[s])
        if draw is None:
            return t_lo, t_hi, w_t, r_lo, r_hi, w_r
        return (int(t_hi if draw.random() < w_t else t_lo),
                int(r_hi if draw.random() < w_r else r_lo))

    for k in range(30):
        memory = teacher.memory()
        doc = _observation(k, float(rng.uniform(-3, 3)), float(rng.normal(0, 6)),
                           float(rng.normal(0, 6)))
        teacher.label([memory], [doc], rng)
        me = doc["self"]
        ti, ri = bins_of(me, int(rng.integers(F.N_HEADING)), int(rng.integers(F.N_SPEED)), rng)
        teacher.executed(memory, (ti, ri, 1), F.THROTTLE_VALUES[ti], rng)
        assert memory.prev_push == F.THROTTLE_VALUES[ti] and memory.prev_jump
        assert memory.distill_pending is None
        t_lo, t_hi, w_t, r_lo, r_hi, w_r = bins_of(me, memory.prev_heading, memory.prev_speed,
                                                   None)
        pt = (1 - w_t) * (t_lo == ti) + w_t * (t_hi == ti)
        pr = (1 - w_r) * (r_lo == ri) + w_r * (r_hi == ri)
        assert pt * pr > 0  # the committed goal could have produced the executed bins


def test_policy_teacher_rollout_and_training(tmp_path):
    import torch

    from ctf.nets import Actor, actor_to_numpy
    from ctf.rollout import collect
    from ctf.train import main

    _, path = _goal_actor(tmp_path)
    torch.manual_seed(0)
    actor = NumpyActor(actor_to_numpy(Actor((16, 16))), np.zeros(F.FEATURE_DIM),
                       np.ones(F.FEATURE_DIM))
    data = collect(actor, [{"seed": 9, "side": "red", "opponent": "relay"}], {}, 150, 0,
                   memory_wipe_prob=1.0, teacher=f"policy:{path}")
    assert data["teacher_valid"].sum() == data["alive"].sum()
    soft = data["teacher_soft"][data["teacher_valid"]]
    assert np.allclose(soft[:, :F.N_THROTTLE].sum(1), 1.0, atol=1e-5)
    plain = collect(actor, [{"seed": 9, "side": "red", "opponent": "relay"}], {}, 150, 0,
                    memory_wipe_prob=1.0)
    assert np.array_equal(plain["actions"], data["actions"])  # the teacher never acts
    main(["--name", "distill", "--runs", str(tmp_path), "--workers", "1", "--envs", "1",
          "--max-ticks", "150", "--updates", "2", "--minibatch", "256", "--learner-threads", "2",
          "--opponents", "baseline:1,mirror:1", "--teacher", f"policy:{path}",
          "--kick-coef", "1.0", "--kick-samples", "1e9"])
    records = [json.loads(line) for line in
               (tmp_path / "distill" / "log.jsonl").read_text().splitlines()]
    assert all(r["kick_ce"] > 0 and r["kick_kl"] >= -1e-4 for r in records)
    assert all(0.0 <= r["kick_agree_mode"] <= 1.0 for r in records)


def test_distill_init_keeps_hidden_layers(tmp_path):
    import importlib.util
    from pathlib import Path

    import torch

    from ctf.nets import Critic

    actor, path = _goal_actor(tmp_path)
    critic = Critic((8, 8), F.GOAL_FEATURE_DIM)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"critic": critic.state_dict(), "feature_names": list(F.GOAL_FEATURE_NAMES),
                "ret_norm": {"mean": np.zeros(1), "var": np.ones(1), "count": 5.0},
                "norm": {"count": 7.0}}, ckpt)
    tool = Path(__file__).resolve().parents[2] / "tools" / "distill_init.py"
    spec = importlib.util.spec_from_file_location("distill_init", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = module.build(path, ckpt)
    w0 = state["actor"]["net.0.weight"].numpy()
    assert w0.shape == (16, F.FEATURE_DIM)
    goal_index = {n: i for i, n in enumerate(F.GOAL_FEATURE_NAMES)}
    for j, name in enumerate(F.FEATURE_NAMES):
        if name in goal_index:
            assert np.array_equal(w0[:, j], actor.weights[0][0][goal_index[name]])
        else:
            assert not w0[:, j].any() and name.startswith("prev.")
    assert state["actor"]["net.4.weight"].shape == (F.N_THROTTLE + F.N_TURN + 2, 16)
    assert state["critic"]["net.0.weight"].shape == (8, F.FEATURE_DIM)
