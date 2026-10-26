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


def _min_gap(logits):
    gaps, start = [], 0
    for size in HEAD_SIZES:
        head = np.sort(logits[start:start + size])
        start += size
        gaps.append(head[-1] - head[-2])
    return float(min(gaps))


def _play(model_path, policy_path, seed, ticks, strict=False):
    """Drive a Match with exported players and check them against the reference.

    The reference is NumpyActor's greedy decode (`argmax`, first maximum on
    exact ties). With `strict`, every action must match exactly.
    """
    match = Match(seed, max_ticks=ticks)
    modules = {s.id: _load_module(model_path) for s in match.spheres}
    actor, _ = load_policy(policy_path)
    memories = {s.id: F.PlayerMemory() for s in match.spheres}
    last_seen = {}
    stats = {"calls": 0, "respawns": 0, "ties": 0, "max_err": 0.0, "actions": set(),
             "min_gap": np.inf, "max_logit": 0.0}
    while not match.is_over():
        observations = match.observe()
        actions = {}
        for sid, observation in observations.items():
            action = modules[sid].decide(observation)
            row = F.encode(observation, memories[sid])
            ref = actor.logits(np.asarray([row]))[0]
            got = modules[sid]._MX_LOGITS
            stats["max_err"] = max(stats["max_err"], float(np.max(np.abs(got - ref))))
            stats["min_gap"] = min(stats["min_gap"], _min_gap(ref))
            stats["max_logit"] = max(stats["max_logit"], float(np.max(np.abs(ref))))
            t, r, j = _reference_greedy(ref)
            assert (t, r, j) == tuple(actor.sample(np.asarray([row]), None, greedy=True)[0][0])
            memories[sid].set_action(t, r, j)
            expected = (F.THROTTLE_VALUES[t], F.TURN_VALUES[r], bool(j))
            if action != expected:
                assert not strict and _near_tie(ref), (observation["tick"], sid, action, expected)
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


def _rescaled(tmp_path, source, name, scale, bias=None):
    """Copy of `source` with the last layer scaled (and its bias optionally replaced)."""
    actor, _ = load_policy(source)
    weights = [(w.copy(), b.copy()) for w, b in actor.weights]
    w, b = weights[-1]
    weights[-1] = (w * np.float32(scale), b * np.float32(scale) if bias is None else bias)
    path = tmp_path / f"{name}.npz"
    save_policy(path, weights, actor.obs_mean, actor.obs_std, notes=name)
    return path


def test_near_ties_match_numpy_actor_argmax(tmp_path):
    # Logits ~1e-4 apart, as in a barely trained or very uncertain head. The exported
    # forward pass is not bit-identical to NumpyActor (matvec vs matmul rounding), but
    # its error scales with the logits, so greedy actions still agree exactly.
    policy = _rescaled(tmp_path, _policy(tmp_path, 31), "flat", 1e-4)
    model = export(policy, tmp_path / "flat.py")
    stats = _play(model, policy, seed=7, ticks=400, strict=True)
    assert stats["max_logit"] < 1e-2 and stats["min_gap"] < 1e-6, stats
    assert len(stats["actions"]) >= 5 and stats["ties"] == 0


def test_exact_ties_pick_the_first_maximum(tmp_path):
    # All-equal logits: numpy argmax (exported and NumpyActor alike) picks index 0.
    bias = np.zeros(sum(HEAD_SIZES), dtype=np.float32)
    policy = _rescaled(tmp_path, _policy(tmp_path, 32), "tied", 0.0, bias=bias)
    model = export(policy, tmp_path / "tied.py")
    stats = _play(model, policy, seed=8, ticks=50, strict=True)
    assert stats["max_logit"] == 0.0
    assert stats["actions"] == {(F.THROTTLE_VALUES[0], F.TURN_VALUES[0], False)}


def test_sampled_decode_matches_evaluator_distribution(tmp_path):
    """export(decode="sample") draws by inverse CDF from the player's own stream;
    the evaluator's `.npz@sample` uses Gumbel-max with its own stream. Games are
    therefore not bit-identical, but per-head action frequencies agree."""
    policy = _rescaled(tmp_path, _policy(tmp_path, 33), "soft", 0.5)
    module = _load_module(export(policy, tmp_path / "soft.py", decode="sample"))
    actor, _ = load_policy(policy)
    observation = Match(4, max_ticks=1).observe()[0]
    row = np.asarray(F.encode(observation, F.PlayerMemory()), dtype=np.float32)
    logits = module._mx_forward(row).copy()
    module._mx_rng = np.random.default_rng(0)
    n = 20000
    exported = np.array([[module._mx_sample(head) for head in module._MX_HEADS]
                         for _ in range(n)])
    evaluator, _ = actor.sample(np.repeat(row[None], n, axis=0), np.random.default_rng(1))
    start, informative = 0, 0
    for k, size in enumerate(HEAD_SIZES):
        head = logits[start:start + size].astype(np.float64)
        start += size
        p = np.exp(head - head.max())
        p /= p.sum()
        informative += p.max() < 0.9
        f_exp = np.bincount(exported[:, k], minlength=size) / n
        f_eval = np.bincount(evaluator[:, k], minlength=size) / n
        # 4.5 binomial standard deviations per action (n = 20000 per estimate).
        tol = 4.5 * np.sqrt(p * (1 - p) / n) + 1e-3
        assert np.all(np.abs(f_exp - p) < tol), (k, f_exp, p)
        assert np.all(np.abs(f_eval - p) < tol), (k, f_eval, p)
    assert informative >= 2, "logits too peaked for a useful distribution test"


def test_exported_model_evaluates_like_npz_greedy(exported):
    """The exported file, evaluated as a submission (documented observations, one module
    per player in-process, or one worker process per player with @proc), plays exactly
    the same games as the policy's `.npz@greedy` spec."""
    from ctf import evaluate as E

    policy, model = exported[0]
    reference = E.evaluate(f"{policy}@greedy", "example", [3, 4], max_ticks=1200,
                           n_boot=50)[1]
    # Eventful games, so a diverging trajectory would show up in the records.
    assert sum(r["a"]["wall_deaths"] + r["b"]["player_deaths"] for r in reference) >= 3
    for spec in (str(model), f"{model}@proc"):
        assert E.parse_spec(spec).documented_only
        _, records = E.evaluate(spec, "example", [3, 4], max_ticks=1200, n_boot=50)
        assert len(records) == len(reference) == 4
        for got, want in zip(records, reference):
            got, want = E.deterministic_view(got), E.deterministic_view(want)
            for key in ("spec_a", "digest_a"):
                got.pop(key), want.pop(key)
            assert got == want, spec


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


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")


def test_thread_env_is_set_before_numpy(exported):
    text = Path(exported[0][1]).read_text()
    assert text.index("VECLIB_MAXIMUM_THREADS") < text.index("import numpy")
    script = (
        "import importlib.util, os, sys\n"
        "if sys.argv[2] == 'preimport': import numpy\n"
        "spec = importlib.util.spec_from_file_location('m', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        f"print(*[os.environ.get(v) for v in {THREAD_VARS!r}],"
        " 'torch' in sys.modules, 'ctf' in sys.modules)\n"
        "try:\n"
        "    import threadpoolctl\n"
        "    blas = [p['num_threads'] for p in threadpoolctl.threadpool_info()"
        " if p['user_api'] == 'blas']\n"
        "    print(*(blas or ['none']))\n"
        "except ImportError:\n"
        "    print('none')\n"
    )
    clean = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}

    def run(env, mode):
        result = subprocess.run([sys.executable, "-c", script, str(exported[0][1]), mode],
                                env=env, capture_output=True, text=True, timeout=60, check=True)
        return [line.split() for line in result.stdout.strip().splitlines()]

    # Unset or inherited (e.g. a tournament host exporting OMP_NUM_THREADS=8): forced to 1.
    for env in (clean, {**clean, **{v: "8" for v in THREAD_VARS}}):
        env_line, blas_line = run(env, "fresh")
        assert env_line == ["1"] * len(THREAD_VARS) + ["False", "False"]
        assert blas_line == ["none"] or set(blas_line) == {"1"}, blas_line
    # numpy already imported by the host: too late to matter, so the host env is untouched.
    env_line, _ = run({**clean, **{v: "8" for v in THREAD_VARS}}, "preimport")
    assert env_line[:len(THREAD_VARS)] == ["8"] * len(THREAD_VARS)


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
