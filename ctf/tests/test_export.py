import ast
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from ctf import features as F
from ctf.export import FEATURES_BEGIN, FEATURES_END, export
from ctf.nets import HEAD_SIZES
from ctf.policy_io import load_policy, random_policy, save_policy
from ctf.sim import Match

PY39 = os.environ.get("EXML_PY39", ".venv39/bin/python")
_counter = itertools.count()


def _policy(tmp_path, seed, hidden=(128, 128)):
    """Random policy with a perturbed last layer and non-trivial normalization."""
    raw = tmp_path / f"raw{seed}.npz"
    random_policy(raw, hidden=hidden, seed=seed)
    actor, _ = load_policy(raw)
    rng = np.random.default_rng(seed)
    weights = [(w.copy(), b.copy()) for w, b in actor.weights]
    w, b = weights[-1]
    weights[-1] = (w + rng.normal(0.0, 0.4, w.shape).astype(np.float32),
                   b + rng.normal(0.0, 0.5, b.shape).astype(np.float32))
    # Bias throttle towards full speed so that spheres hit walls and die.
    weights[-1][1][F.N_THROTTLE - 1] += 1.5
    mean = rng.normal(0.0, 0.1, F.FEATURE_DIM)
    std = rng.uniform(0.3, 2.0, F.FEATURE_DIM)
    path = tmp_path / f"policy{seed}.npz"
    save_policy(path, weights, mean, std, notes="test")
    return path


def _load_module(path):
    spec = importlib.util.spec_from_file_location(f"exported_{next(_counter)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_greedy(logits):
    out, start = [], 0
    for size in HEAD_SIZES:
        out.append(int(np.argmax(logits[start:start + size])))
        start += size
    return out


def _near_tie(logits, tol=1e-5):
    start = 0
    for size in HEAD_SIZES:
        head = np.sort(logits[start:start + size])
        start += size
        if head[-1] - head[-2] < tol:
            return True
    return False


def _play(model_path, policy_path, seed, ticks):
    """Drive a Match with exported players and check them against the reference."""
    match = Match(seed, max_ticks=ticks)
    modules = {s.id: _load_module(model_path) for s in match.spheres}
    actor, _ = load_policy(policy_path)
    memories = {s.id: F.PlayerMemory() for s in match.spheres}
    last_seen = {}
    stats = {"calls": 0, "respawns": 0, "ties": 0, "max_err": 0.0, "actions": set()}
    while not match.is_over():
        observations = match.observe()
        actions = {}
        for sid, observation in observations.items():
            action = modules[sid].decide(observation)
            row = F.encode(observation, memories[sid])
            ref = actor.logits(np.asarray([row]))[0]
            got = modules[sid]._MX_LOGITS
            stats["max_err"] = max(stats["max_err"], float(np.max(np.abs(got - ref))))
            t, r, j = _reference_greedy(ref)
            memories[sid].set_action(t, r, j)
            expected = (F.THROTTLE_VALUES[t], F.TURN_VALUES[r], bool(j))
            if action != expected:
                assert _near_tie(ref), (observation["tick"], sid, action, expected)
                stats["ties"] += 1
            assert isinstance(action[2], bool)
            stats["actions"].add(expected)
            if sid in last_seen and observation["tick"] - last_seen[sid] > 1:
                stats["respawns"] += 1
            last_seen[sid] = observation["tick"]
            stats["calls"] += 1
            actions[sid] = action
        match.act(actions)
    stats["deaths"] = len(match.death_log())
    return stats


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("export")
    out = []
    for seed in (11, 12, 13):
        policy = _policy(tmp_path, seed)
        model = export(policy, tmp_path / f"model{seed}.py")
        out.append((policy, model))
    return out


def test_logits_and_actions_match_reference(exported):
    total = {"calls": 0, "respawns": 0, "deaths": 0, "ties": 0}
    for i, (policy, model) in enumerate(exported):
        stats = _play(model, policy, seed=100 + i, ticks=700)
        assert stats["max_err"] < 1e-5, stats["max_err"]
        assert len(stats["actions"]) >= 5, "policy too deterministic to be a useful test"
        for key in total:
            total[key] += stats[key]
    assert total["calls"] >= 9000
    assert total["deaths"] > 0 and total["respawns"] > 0, total
    assert total["ties"] <= total["calls"] // 1000


def test_features_are_inlined_verbatim_and_standalone(exported):
    text = Path(exported[0][1]).read_text()
    source = Path(F.__file__).read_text()
    inlined = text.split(FEATURES_BEGIN, 1)[1].split(FEATURES_END, 1)[0]
    assert inlined == source.rstrip("\n") + "\n"
    imported = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= {"os", "base64", "sys", "zlib", "math", "numpy"}, imported


def test_thread_env_is_set_before_numpy(exported):
    text = Path(exported[0][1]).read_text()
    assert text.index("VECLIB_MAXIMUM_THREADS") < text.index("import numpy")
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS")}
    script = (
        "import importlib.util, os, sys\n"
        "spec = importlib.util.spec_from_file_location('m', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "print(os.environ['OMP_NUM_THREADS'], os.environ['VECLIB_MAXIMUM_THREADS'],"
        " 'torch' in sys.modules, 'ctf' in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", script, str(exported[0][1])], env=env,
                            capture_output=True, text=True, timeout=60, check=True)
    assert result.stdout.split() == ["1", "1", "False", "False"]


def test_memory_resets_on_new_match(exported):
    policy, model = exported[0]
    module = _load_module(model)
    match = Match(1, max_ticks=60)
    while not match.is_over():
        module.decide(match.observe()[0])
        match.act({})
    assert module._MEMORY.last_tick == 59
    observation = Match(2, max_ticks=1).observe()[0]
    module.decide(observation)
    assert module._MEMORY.last_tick == 0 and module._MEMORY.prev_tick == 0
    actor, _ = load_policy(policy)
    ref = actor.logits(np.asarray([F.encode(observation, F.PlayerMemory())]))[0]
    assert np.max(np.abs(module._MX_LOGITS - ref)) < 1e-5


def test_decide_survives_bad_input(exported, capsys):
    module = _load_module(exported[0][1])
    valid_throttle, valid_turn = set(F.THROTTLE_VALUES), set(F.TURN_VALUES)
    for bad in (None, {}, {"tick": "x"}, {"self": None, "tick": None}, {"self": {"x": "nan"}},
                {"tick": 3, "self": {"x": float("inf"), "yaw": float("nan")},
                 "visible_enemies": [{"id": 9, "x": None}], "death_messages": [{}]}, 42):
        throttle, turn, jump = module.decide(bad)
        assert throttle in valid_throttle and turn in valid_turn and isinstance(jump, bool)
    # A normal observation still works afterwards.
    action = module.decide(Match(3, max_ticks=1).observe()[0])
    assert action[0] in valid_throttle
    assert np.all(np.isfinite(module._MX_LOGITS))
    capsys.readouterr()


def test_sample_decode(tmp_path):
    policy = _policy(tmp_path, 21)
    model = export(policy, tmp_path / "sample.py", decode="sample")
    observation = Match(4, max_ticks=1).observe()[0]

    def run(n):
        module = _load_module(model)
        out = []
        for tick in range(n):
            observation["tick"] = tick
            out.append(module.decide(observation))
        return out, module

    first, module = run(400)
    second, _ = run(400)
    assert first == second, "sampling must be seeded from the player id"
    assert len(set(first)) > 5
    assert module._mx_rng is not None
    # Empirical jump frequency follows the softmax of the jump head.
    jump = module._MX_LOGITS[F.N_THROTTLE + F.N_TURN:]
    p_jump = float(np.exp(jump[1]) / np.exp(jump).sum())
    freq = np.mean([a[2] for a in first[200:]])
    assert abs(freq - p_jump) < 0.15


def test_warm_decide_is_fast(exported):
    module = _load_module(exported[0][1])
    match = Match(5, max_ticks=300)
    times = []
    while not match.is_over():
        observations = match.observe()
        observation = observations.get(0)
        if observation is not None:
            start = time.perf_counter()
            action = module.decide(observation)
            times.append(time.perf_counter() - start)
        match.act({0: action} if observation is not None else {})
    assert np.median(times) < 1e-3, np.median(times)
    assert np.percentile(times, 99) < 3e-3, np.percentile(times, 99)


@pytest.mark.skipif(not Path(PY39).exists(), reason="Python 3.9 env not available")
def test_runs_under_python39(exported, tmp_path):
    policy, model = exported[1]
    match = Match(6, max_ticks=200)
    module = _load_module(model)
    trace, expected = [], []
    while not match.is_over():
        observation = match.observe().get(0)
        if observation is not None:
            trace.append(observation)
            expected.append(list(module.decide(observation)))
        match.act({0: tuple(expected[-1])} if observation is not None else {})
    observations = tmp_path / "trace.json"
    observations.write_text(json.dumps(trace))
    script = (
        "import importlib.util, json, sys\n"
        "spec = importlib.util.spec_from_file_location('m', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "trace = json.load(open(sys.argv[2]))\n"
        "print(sys.version_info[:2]); print(json.dumps([list(m.decide(o)) for o in trace]))\n"
    )
    result = subprocess.run([PY39, "-c", script, str(model), str(observations)],
                            capture_output=True, text=True, timeout=120, check=True)
    version, actions = result.stdout.strip().splitlines()
    assert version == "(3, 9)"
    assert json.loads(actions) == expected
    assert "Warning" not in result.stderr and "error" not in result.stderr, result.stderr
