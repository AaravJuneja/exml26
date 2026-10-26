"""The optional goal action space: controller, projection, layout, export, rollout, training."""

import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ctf import action_space as A
from ctf import feature_layouts as L
from ctf import features as F
from ctf.controller import control, goal_step
from engine.constants import OBSTACLE_LETHAL_SPEED
from engine.sphere import Sphere

_counter = itertools.count()
TOP = 17.64


def _me(sphere):
    return {"yaw": sphere.yaw, "vx": sphere.vx, "vy": sphere.vy, "speed": sphere.speed}


def _drive(sphere, heading_abs, speed, ticks, reverse=False, trace=None):
    """Hold an absolute heading (degrees) and speed for `ticks` engine steps."""
    for _ in range(ticks):
        relative = math.radians(heading_abs) - sphere.yaw
        throttle, turn, reverse = control(_me(sphere), relative, speed, reverse)
        assert -1.0 <= throttle <= 1.0 and -1.0 <= turn <= 1.0
        sphere.apply_action(throttle, turn, False)
        sphere.integrate(0.05)
        if trace is not None:
            trace.append(sphere.speed)
    return reverse


def _direction(sphere):
    return math.degrees(math.atan2(sphere.vx, sphere.vy))


def _angle_error(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ------------------------------------------------------------------ controller
@pytest.mark.parametrize("heading", [0.0, 30.0, 90.0, -135.0, 180.0])
@pytest.mark.parametrize("speed", [4.0, 10.85, TOP])
def test_controller_reaches_speed_and_heading_in_the_open(heading, speed):
    sphere = Sphere(0, "a", 0.0, 0.0)
    trace = []
    _drive(sphere, heading, speed, 80, trace=trace)
    assert abs(sphere.speed - speed) < 0.05, sphere.speed
    assert _angle_error(_direction(sphere), heading) < 2.5
    # Never above the target (+ rounding): 10.85 stays below the lethal 11.
    assert max(trace) <= speed + 1e-6
    if speed < OBSTACLE_LETHAL_SPEED:
        assert max(trace) < OBSTACLE_LETHAL_SPEED


def test_controller_accelerates_fast_from_rest():
    sphere = Sphere(0, "a", 0.0, 0.0)
    trace = []
    _drive(sphere, 0.0, TOP, 6, trace=trace)
    assert trace[0] > 5.5 and sphere.speed > 16.0  # full throttle along the facing


def test_controller_turns_around():
    sphere = Sphere(0, "a", 0.0, 0.0)
    reverse = _drive(sphere, 0.0, TOP, 60)
    trace = []
    _drive(sphere, 180.0, TOP, 40, reverse, trace)
    assert _angle_error(_direction(sphere), 180.0) < 1.0 and abs(sphere.speed - TOP) < 0.05
    assert max(trace) <= TOP + 1e-6
    # It brakes through zero rather than orbiting: the speed drops low on the way.
    assert min(trace) < 3.0


def test_controller_turns_sideways_without_overshooting_speed():
    sphere = Sphere(0, "a", 0.0, 0.0)
    reverse = _drive(sphere, 0.0, 10.5, 60)
    trace = []
    _drive(sphere, 90.0, 10.5, 60, reverse, trace)
    assert _angle_error(_direction(sphere), 90.0) < 2.5 and max(trace) <= 10.5 + 1e-6


def test_controller_stops():
    sphere = Sphere(0, "a", 0.0, 0.0)
    reverse = _drive(sphere, 45.0, TOP, 60)
    trace = []
    _drive(sphere, 0.0, 0.0, 15, reverse, trace)
    assert trace[4] < 0.5 and sphere.speed < 0.05
    x, y = sphere.x, sphere.y
    _drive(sphere, 0.0, 0.0, 20)
    assert math.hypot(sphere.x - x, sphere.y - y) < 0.05


def test_controller_turns_in_place_toward_heading_at_zero_speed():
    sphere = Sphere(0, "a", 0.0, 0.0)
    _drive(sphere, 90.0, 0.0, 15)
    assert abs(math.degrees(sphere.yaw) - 90.0) < 1.0 and sphere.speed < 1e-9


def test_controller_survives_bad_input():
    for me in ({}, {"yaw": "x", "vx": None}, {"yaw": float("nan"), "vx": float("inf")}):
        throttle, turn, reverse = control(me, 0.5, 10.0)
        assert math.isfinite(throttle) and math.isfinite(turn) and isinstance(reverse, bool)


def test_goal_step_records_and_repeats_a_tick_identically():
    memory = F.PlayerMemory(goal=True)
    observation = {"tick": 5, "self": {"yaw": 0.0, "vx": 0.0, "vy": 0.0, "speed": 0.0}}
    h, s = F.HEADING_VALUES.index(180.0), F.SPEED_VALUES.index(TOP)
    first = goal_step(memory, observation, h, s, 1, F.HEADING_VALUES, F.SPEED_VALUES)
    assert memory.ctrl == (5, False, True) and first[0] == -1.0  # straight back: reverse thrust
    again = goal_step(memory, observation, h, s, 1, F.HEADING_VALUES, F.SPEED_VALUES)
    assert memory.ctrl == (5, False, True)
    assert first == again and first[2] is True
    assert (memory.prev_heading, memory.prev_speed, memory.prev_jump) == (h, s, True)
    assert memory.prev_push == first[0]


# ------------------------------------------------------------------ projection
def test_project_action_reads_intent_from_thrust():
    me = {"yaw": 0.3, "vx": 0.0, "vy": 0.0}
    relative, speed = A.project_action(me, 1.0, 0.0)
    assert abs(relative) < 1e-9 and speed > 15.0
    relative, speed = A.project_action({"yaw": 0.0, "vx": 0.0, "vy": 17.0}, -1.0, 0.0)
    assert speed < 2.0
    relative, speed = A.project_action({"yaw": 0.0, "vx": 0.0, "vy": 17.0}, -1.0, 0.0, ticks=1.0)
    assert abs(relative) < 1e-9 and 10.0 < speed < 11.5


def test_teacher_goal_prefers_the_commanded_goal():
    class Memory:
        goal = (1.0, 10.5)

    def decide(observation):
        return 0.0, 0.0, False

    decide.__globals__["M"] = Memory  # module-level M of a _core bot
    try:
        relative, speed = A.teacher_goal(decide, (0.0, 0.0, False), {"self": {"yaw": 0.25}})
    finally:
        del decide.__globals__["M"]
    assert abs(relative - 0.75) < 1e-12 and speed == 10.5
    h, s = A.goal_indices(math.radians(177.0), 11.0)
    assert F.HEADING_VALUES[h] == 180.0 and F.SPEED_VALUES[s] == 10.85
    assert F.HEADING_VALUES[A.heading_index(math.radians(-179.0))] == 180.0


def test_relay_records_its_goal_and_labels_cover_alive_players():
    from ctf.npnets import NumpyActor
    from ctf.rollout import collect

    rng = np.random.default_rng(0)
    d, n = A.GOAL.feature_dim, sum(A.GOAL.head_sizes)
    actor = NumpyActor([(rng.normal(0, 0.1, (d, n)).astype(np.float32), np.zeros(n, np.float32))],
                       np.zeros(d), np.ones(d), space="goal")
    data = collect(actor, [{"seed": 9, "side": "red", "opponent": "relay"},
                           {"seed": 10, "side": "blue", "opponent": "mirror"}], {}, 150, 0,
                   teacher="relay")
    assert data["obs"].shape[-1] == F.GOAL_FEATURE_DIM
    assert data["teacher_valid"].sum() == data["alive"].sum()
    labels = data["teacher_actions"][data["teacher_valid"]]
    for k, size in enumerate(A.GOAL.head_sizes):
        assert labels[:, k].max() < size and labels[:, k].min() >= 0
        assert data["actions"][data["alive"]][:, k].max() < size
    # relay starts by driving at speed, not idling
    assert np.mean(labels[:, 1] > 0) > 0.8


# ------------------------------------------------------------------ layout
def test_goal_layout_is_distinct_and_only_the_action_block_differs():
    assert len(F.GOAL_FEATURE_NAMES) == F.GOAL_FEATURE_DIM
    base = F.FEATURE_DIM - F.ACTION_DIM
    assert F.GOAL_FEATURE_NAMES[:base] == F.FEATURE_NAMES[:base]
    assert not L.same_layout(F.FEATURE_NAMES, F.ACTION_TABLES,
                             F.GOAL_FEATURE_NAMES, F.GOAL_ACTION_TABLES)
    assert L.heads_of(L.normalize_tables(F.GOAL_ACTION_TABLES)) == L.GOAL_HEADS
    with pytest.raises(ValueError, match="action spaces"):
        L.column_sources(F.FEATURE_NAMES, F.ACTION_TABLES,
                         F.GOAL_FEATURE_NAMES, F.GOAL_ACTION_TABLES)
    assert A.of_tables(json.loads(json.dumps(F.GOAL_ACTION_TABLES))) is A.GOAL


def test_goal_memory_encodes_the_previous_goal():
    from ctf.sim import Match

    observation = Match(3, max_ticks=1).observe()[0]
    bins, goal = F.PlayerMemory(), F.PlayerMemory(goal=True)
    row_bins, row_goal = F.encode(observation, bins), F.encode(observation, goal)
    base = F.FEATURE_DIM - F.ACTION_DIM
    assert row_bins[:base] == row_goal[:base]
    named = dict(zip(F.GOAL_FEATURE_NAMES, row_goal))
    assert named["prev.heading=0.0"] == 1.0 and named["prev.speed=0.0"] == 1.0
    goal.set_goal(2, 5, True, 0.7)
    goal.cached = None
    named = dict(zip(F.GOAL_FEATURE_NAMES, F.encode(observation, goal)))
    assert named["prev.heading=%r" % F.HEADING_VALUES[2]] == 1.0
    assert named["prev.speed=%r" % F.SPEED_VALUES[5]] == 1.0 and named["prev.jump"] == 1.0
    assert sum(v for k, v in named.items() if k.startswith("prev.")) == 3.0
    goal.reset()
    assert goal.goal and goal.prev_push == 0.0 and goal.ctrl == (None, False, False)


# ------------------------------------------------------------------ export
def _goal_policy(tmp_path, seed, hidden=64):
    """Random goal-space MLP with non-trivial normalization, mostly choosing fast goals."""
    from ctf.policy_io import load_policy, save_policy

    rng = np.random.default_rng(seed)
    d, n = F.GOAL_FEATURE_DIM, sum(A.GOAL.head_sizes)
    sizes = [d, hidden, hidden, n]
    weights = [(rng.normal(0.0, 1.0 / np.sqrt(a), (a, b)).astype(np.float32),
                rng.normal(0.0, 0.1, b).astype(np.float32)) for a, b in zip(sizes, sizes[1:])]
    w, b = weights[-1]
    b = b + rng.normal(0.0, 0.5, n).astype(np.float32)
    b[F.N_HEADING + F.N_SPEED - 1] += 3.0  # mostly fast and straight, so spheres crash
    b[F.IDLE_HEADING] += 1.0
    weights[-1] = (w * np.float32(2.0), b)
    mean = rng.normal(0.0, 0.1, d)
    std = rng.uniform(0.3, 2.0, d)
    path = tmp_path / f"goal{seed}.npz"
    save_policy(path, weights, mean, std, space="goal", notes="test")
    actor, meta = load_policy(path)
    assert actor.space == "goal" and meta["action_space"] == "goal"
    return path


def _load_module(path):
    spec = importlib.util.spec_from_file_location(f"goal_export_{next(_counter)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exported_goal_model_decides_like_the_reference(tmp_path):
    from ctf.export import CONTROLLER_BEGIN, CONTROLLER_END, export
    from ctf.npnets import PolicyMemories
    from ctf.policy_io import load_policy
    from ctf.rollout import documented
    from ctf.sim import Match
    import ctf.controller

    policy = _goal_policy(tmp_path, 5)
    model = export(policy, tmp_path / "goal_model.py")
    text = model.read_text()
    inlined = text.split(CONTROLLER_BEGIN, 1)[1].split(CONTROLLER_END, 1)[0]
    assert inlined == Path(ctf.controller.__file__).read_text().rstrip("\n") + "\n"
    actor, _ = load_policy(policy)
    match = Match(21, max_ticks=900)
    modules = {s.id: _load_module(model) for s in match.spheres}
    reference = PolicyMemories(actor, [s.id for s in match.spheres], F.encode)
    calls, ties, chosen, deaths = 0, 0, set(), 0
    while not match.is_over():
        observations = match.observe()
        ids = sorted(observations)
        docs = [documented(observations[sid]) for sid in ids]
        x = reference.encode_batch(ids, docs)
        indices, engine = reference.act_engine(ids, x, None, True, docs)
        actions = {}
        for sid, doc, want, index in zip(ids, docs, engine, indices.tolist()):
            got = modules[sid].decide(doc)
            ref = actor.logits(x[ids.index(sid):ids.index(sid) + 1])[0]
            assert np.max(np.abs(modules[sid]._MX_LOGITS - ref)) < 1e-5
            if got != want:  # only a near-tie may flip a greedy choice
                start, gaps = 0, []
                for size in A.GOAL.head_sizes:
                    head = np.sort(ref[start:start + size])
                    gaps.append(head[-1] - head[-2])
                    start += size
                assert min(gaps) < 1e-5, (doc["tick"], sid, got, want)
                ties += 1
            calls += 1
            chosen.add(tuple(index[:2]))
            actions[sid] = got
        match.act(actions)
    deaths = len(match.death_log())
    assert calls > 4000 and ties <= calls // 1000
    assert len(chosen) >= 5 and deaths > 0


def test_exported_goal_gru_decides_like_the_reference(tmp_path):
    from ctf.export import export
    from ctf.npnets import PolicyMemories
    from ctf.policy_io import load_policy, save_policy
    from ctf.rollout import documented
    from ctf.sim import Match

    rng = np.random.default_rng(1)
    d, n, h = F.GOAL_FEATURE_DIM, sum(A.GOAL.head_sizes), 16
    shapes = {"W_in": (d, 32), "b_in": (32,), "W_ih": (32, 3 * h), "b_ih": (3 * h,),
              "W_hh": (h, 3 * h), "b_hh": (3 * h,), "W_out": (h, n), "b_out": (n,)}
    weights = {k: rng.normal(0.0, 0.3, s).astype(np.float32) for k, s in shapes.items()}
    path = tmp_path / "goal_gru.npz"
    save_policy(path, weights, np.zeros(d), np.ones(d), space="goal")
    actor, _ = load_policy(path)
    assert actor.recurrent and actor.space == "goal"
    model = export(path, tmp_path / "goal_gru.py")
    match = Match(3, max_ticks=300)
    modules = {s.id: _load_module(model) for s in match.spheres}
    reference = PolicyMemories(actor, [s.id for s in match.spheres], F.encode)
    while not match.is_over():
        observations = match.observe()
        ids = sorted(observations)
        docs = [documented(observations[sid]) for sid in ids]
        x = reference.encode_batch(ids, docs)
        _, engine = reference.act_engine(ids, x, None, True, docs)
        actions = {sid: modules[sid].decide(doc) for sid, doc in zip(ids, docs)}
        assert [actions[sid] for sid in ids] == engine
        match.act(actions)


def test_exported_goal_model_survives_bad_input(tmp_path, capsys):
    from ctf.export import export
    from ctf.sim import Match

    module = _load_module(export(_goal_policy(tmp_path, 6), tmp_path / "goal_bad.py"))
    for bad in (None, {}, {"tick": "x"}, {"self": None, "tick": None},
                {"tick": 3, "self": {"x": float("inf"), "yaw": float("nan")}}, 42):
        throttle, turn, jump = module.decide(bad)
        assert math.isfinite(throttle) and math.isfinite(turn) and isinstance(jump, bool)
    throttle, turn, jump = module.decide(Match(3, max_ticks=1).observe()[0])
    assert -1.0 <= throttle <= 1.0 and -1.0 <= turn <= 1.0
    capsys.readouterr()


def test_goal_policy_evaluates(tmp_path):
    from ctf import evaluate as E

    policy = _goal_policy(tmp_path, 7)
    summary, records = E.evaluate(f"{policy}@greedy", "idle", [3], max_ticks=300, n_boot=10)
    assert len(records) == 2 and all(r["a"]["wall_deaths"] >= 0 for r in records)


# ------------------------------------------------------------------ training
def test_goal_training_smoke_with_and_without_teacher(tmp_path):
    from ctf.policy_io import load_policy
    from ctf.train import main

    main(["--name", "goal", "--runs", str(tmp_path), "--action-space", "goal", "--workers", "2",
          "--envs", "1", "--max-ticks", "200", "--updates", "2", "--minibatch", "256",
          "--save-every", "1", "--snapshot-every", "1", "--learner-threads", "2",
          "--opponents", "baseline:1,mirror:1,pool:1"])
    main(["--name", "goalkick", "--runs", str(tmp_path), "--action-space", "goal",
          "--workers", "2", "--envs", "1", "--max-ticks", "200", "--updates", "2",
          "--minibatch", "256", "--learner-threads", "2", "--opponents", "relay:1,self:1",
          "--teacher", "relay", "--kick-coef", "1.0"])
    for name in ("goal", "goalkick"):
        records = [json.loads(line) for line in
                   (tmp_path / name / "log.jsonl").read_text().splitlines()]
        assert len(records) == 2
        assert all(r["samples"] > 0 and np.isfinite(r["pi_loss"]) for r in records)
        assert all(r["logp_drift"] < 1e-4 for r in records)
    kick = [json.loads(line) for line in (tmp_path / "goalkick" / "log.jsonl").read_text().splitlines()]
    assert all(r["kick_coef"] > 0 and r["kick_ce"] > 0 for r in kick)
    actor, meta = load_policy(tmp_path / "goal" / "policies" / "u00002.npz")
    assert actor.space == "goal" and meta["action_space"] == "goal"
    assert len(actor.obs_mean) == F.GOAL_FEATURE_DIM
    # resume continues; resuming it in the bins space is refused
    main(["--name", "goal", "--runs", str(tmp_path), "--action-space", "goal", "--workers", "1",
          "--envs", "1", "--max-ticks", "100", "--updates", "3", "--minibatch", "256",
          "--learner-threads", "2"])
    assert len((tmp_path / "goal" / "log.jsonl").read_text().splitlines()) == 3
    with pytest.raises(ValueError):
        main(["--name", "goal", "--runs", str(tmp_path), "--workers", "1", "--updates", "4"])
    with pytest.raises(ValueError, match="action spaces"):
        main(["--name", "goal2", "--runs", str(tmp_path), "--workers", "1", "--updates", "1",
              "--init", str(tmp_path / "goal" / "checkpoint.pt")])
