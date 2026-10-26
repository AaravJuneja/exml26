"""Feature registry and name-based warm-start migration (ctf/feature_layouts.py)."""

import importlib.util
import json
import sys

import numpy as np
import pytest
import torch

from ctf import feature_layouts as L
from ctf import features as F
from ctf.policies import ROOT
from ctf.sim import Match
from ctf.train import NORM_CLIP_STD, migrate_state

FIXTURES = ROOT / "ctf" / "tests" / "fixtures"


def _prev_names(tables):
    return (tuple("prev.throttle=%r" % v for v in tables["throttle"])
            + tuple("prev.turn=%r" % v for v in tables["turn"]) + ("prev.jump",))


def _layout(base, tables):
    names = tuple(n for n in base if not n.startswith("prev.")) + _prev_names(tables)
    return names, L.normalize_tables(tables)


def _is_action(name):
    return name.startswith("prev.") and name != "prev.jump"


def _state(names, tables, seed, hidden=(32, 24), critic_hidden=(40,)):
    """Random MLP checkpoint state with non-trivial normalization (rare one-hots clip at 5)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dim, n_logits = len(names), sum(len(tables[h]) for h in L.HEADS) + 2

    def mlp(sizes):
        out = {}
        for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            out[f"net.{2 * k}.weight"] = torch.randn(b, a) * 0.5
            out[f"net.{2 * k}.bias"] = torch.randn(b) * 0.3
        return out

    mean, var = rng.normal(0.0, 0.5, dim), rng.uniform(0.05, 3.0, dim)
    for j, name in enumerate(names):
        if name.startswith("prev."):
            mean[j] = rng.uniform(0.001, 0.3)
            var[j] = mean[j] * (1 - mean[j])
    return {"actor": mlp((dim, *hidden, n_logits)), "critic": mlp((dim, *critic_hidden, 2)),
            "norm": {"mean": mean, "var": var, "count": 10.0}}


def _forward(weights, x, mean, var):
    std = np.maximum(np.sqrt(var), NORM_CLIP_STD)
    h = torch.clamp((torch.as_tensor(x, dtype=torch.float32)
                     - torch.as_tensor(mean, dtype=torch.float32))
                    / torch.as_tensor(std, dtype=torch.float32), -5.0, 5.0)
    layers = sorted({int(k.split(".")[1]) for k in weights})
    for i, k in enumerate(layers):
        h = h @ weights[f"net.{k}.weight"].T + weights[f"net.{k}.bias"]
        if i + 1 < len(layers):
            h = torch.tanh(h)
    return h


def _inputs(names, rng, n, throttle, turn):
    """Raw rows: random values and the given previous action (one-hot) plus a random jump."""
    x = rng.normal(0.0, 1.0, (n, len(names)))
    for j, name in enumerate(names):
        if name.startswith("prev."):
            x[:, j] = 0.0
    x[:, names.index("prev.throttle=%r" % throttle)] = 1.0
    x[:, names.index("prev.turn=%r" % turn)] = 1.0
    x[:, names.index("prev.jump")] = rng.integers(0, 2, n)
    return x


def _translate(x_old, old_names, new_names, fill):
    """Same rows in the new layout (by name); columns new to it get `fill`."""
    index = {name: i for i, name in enumerate(old_names)}
    x = np.zeros((len(x_old), len(new_names)))
    for j, name in enumerate(new_names):
        if name in index:
            x[:, j] = x_old[:, index[name]]
        elif not _is_action(name):
            x[:, j] = fill(len(x_old))
    return x


V4_NAMES, V4_TABLES = L.V4_FEATURE_NAMES, L.normalize_tables(L.V4_ACTION_TABLES)
# Synthetic next layouts: WIDE adds two features and finer throttle and turn bins;
# MOVED also drops one feature and moves another.
WIDE_TABLES = {"throttle": tuple(sorted(V4_TABLES["throttle"] + (0.07, 0.5, -0.6))),
               "turn": tuple(sorted(V4_TABLES["turn"] + (0.4, -0.85)))}
_base = [n for n in V4_NAMES if not n.startswith("prev.")]
_base.insert(10, "new.a")
_base.insert(40, "new.b")
WIDE_NAMES, WIDE_TABLES = _layout(_base, WIDE_TABLES)
_base = [n for n in _base if n not in ("game.tick", "self.speed")] + ["self.speed"]
MOVED_NAMES = _layout(_base, WIDE_TABLES)[0]


def _head_probs(logits, tables):
    out, start = [], 0
    for head in L.HEADS:
        size = len(tables[head])
        out.append(torch.softmax(logits[:, start:start + size], -1))
        start += size
    out.append(torch.softmax(logits[:, start:], -1))
    return out


@pytest.mark.parametrize("fill", ["zero", "random"])
def test_migration_preserves_logits_and_probabilities(fill):
    rng = np.random.default_rng(1)
    state = _state(V4_NAMES, V4_TABLES, seed=2)
    new = migrate_state(state, V4_NAMES, V4_TABLES, MOVED_NAMES, WIDE_TABLES)
    assert new["feature_names"] == list(MOVED_NAMES)
    filler = (lambda n: np.zeros(n)) if fill == "zero" else (lambda n: rng.normal(0, 3, n))
    old_mean = state["norm"]["mean"]
    for throttle in (-0.05, 0.034, 1.0):
        for turn in (-1.0, 0.3):
            x_old = _inputs(V4_NAMES, rng, 32, throttle, turn)
            x_old[:, V4_NAMES.index("game.tick")] = old_mean[V4_NAMES.index("game.tick")]
            x_new = _translate(x_old, V4_NAMES, MOVED_NAMES, filler)
            for net in ("actor", "critic"):
                out_old = _forward(state[net], x_old, old_mean, state["norm"]["var"])
                out_new = _forward(new[net], x_new, new["norm"]["mean"], new["norm"]["var"])
                if net == "critic":
                    assert torch.allclose(out_old, out_new, atol=1e-4)
                    continue
                p_old, p_new = _head_probs(out_old, V4_TABLES), _head_probs(out_new, WIDE_TABLES)
                for k, head in enumerate(L.HEADS):
                    old_v, new_v = V4_TABLES[head], WIDE_TABLES[head]
                    nearest = [min(range(len(old_v)), key=lambda i: abs(old_v[i] - v))
                               for v in new_v]
                    merged = torch.zeros_like(p_old[k])
                    for j, i in enumerate(nearest):
                        merged[:, i] += p_new[k][:, j]
                    assert torch.allclose(merged, p_old[k], atol=1e-5), head
                    # Logits of the original values: unchanged, minus log(copies) when split.
                    copies = np.bincount(nearest, minlength=len(old_v))
                    off_old = sum(len(V4_TABLES[h]) for h in L.HEADS[:k])
                    off_new = sum(len(WIDE_TABLES[h]) for h in L.HEADS[:k])
                    for i, v in enumerate(old_v):
                        got = out_new[:, off_new + new_v.index(v)]
                        want = out_old[:, off_old + i] - float(np.log(copies[i]))
                        assert torch.allclose(got, want, atol=1e-4), (head, v)
                assert torch.allclose(p_new[-1], p_old[-1], atol=1e-5)


def test_new_previous_action_value_acts_like_its_nearest_old_value():
    rng = np.random.default_rng(3)
    state = _state(V4_NAMES, V4_TABLES, seed=4)
    new = migrate_state(state, V4_NAMES, V4_TABLES, WIDE_NAMES, WIDE_TABLES)
    mean = state["norm"]["mean"]
    x_old = _inputs(V4_NAMES, rng, 16, 0.06, 0.3)   # nearest to 0.07, 0.4
    x_new = _translate(x_old, V4_NAMES, WIDE_NAMES, np.zeros)
    for name, value in (("throttle", 0.06), ("turn", 0.3)):
        x_new[:, WIDE_NAMES.index("prev.%s=%r" % (name, value))] = 0.0
    x_new[:, WIDE_NAMES.index("prev.throttle=0.07")] = 1.0
    x_new[:, WIDE_NAMES.index("prev.turn=0.4")] = 1.0
    w_old, w_new = state["actor"], new["actor"]
    h_old = torch.tanh(_forward({k: w_old[k] for k in ("net.0.weight", "net.0.bias")}, x_old,
                                mean, state["norm"]["var"]))
    h_new = torch.tanh(_forward({k: w_new[k] for k in ("net.0.weight", "net.0.bias")}, x_new,
                                new["norm"]["mean"], new["norm"]["var"]))
    assert torch.allclose(h_old, h_new, atol=1e-5)


def _assert_states_close(a, b):
    for net in ("actor", "critic"):
        assert a[net].keys() == b[net].keys()
        for key in a[net]:
            assert torch.allclose(a[net][key], b[net][key], atol=1e-5), (net, key)
    assert np.allclose(a["norm"]["mean"], b["norm"]["mean"])
    assert np.allclose(a["norm"]["var"], b["norm"]["var"])


@pytest.mark.parametrize("target", ["current", "wide"])
def test_migration_round_trips(target):
    names, tables = ((F.FEATURE_NAMES, F.ACTION_TABLES) if target == "current"
                     else (WIDE_NAMES, WIDE_TABLES))
    state = _state(V4_NAMES, V4_TABLES, seed=5)
    there = migrate_state(state, V4_NAMES, V4_TABLES, names, tables)
    back = migrate_state(there, names, tables, V4_NAMES, V4_TABLES)
    assert back["feature_names"] == list(V4_NAMES)
    _assert_states_close(state, back)
    # Migrating to the same layout changes nothing.
    _assert_states_close(there, migrate_state(there, names, tables, names, tables))


def test_policy_migration_matches_checkpoint_migration_and_round_trips():
    state = _state(V4_NAMES, V4_TABLES, seed=6)
    layers = [(state["actor"][f"net.{k}.weight"].numpy().T, state["actor"][f"net.{k}.bias"].numpy())
              for k in (0, 2, 4)]
    std = np.maximum(np.sqrt(state["norm"]["var"]), NORM_CLIP_STD)
    new_layers, mean, new_std = L.migrate_mlp(layers, state["norm"]["mean"], std, V4_NAMES,
                                              V4_TABLES, WIDE_NAMES, WIDE_TABLES)
    reference = migrate_state(state, V4_NAMES, V4_TABLES, WIDE_NAMES, WIDE_TABLES)
    for k, (w, b) in zip((0, 2, 4), new_layers):
        assert np.allclose(w.T, reference["actor"][f"net.{k}.weight"].numpy(), atol=1e-6)
        assert np.allclose(b, reference["actor"][f"net.{k}.bias"].numpy(), atol=1e-6)
    assert np.allclose(mean, reference["norm"]["mean"])
    back, mean_b, std_b = L.migrate_mlp(new_layers, mean, new_std, WIDE_NAMES, WIDE_TABLES,
                                        V4_NAMES, V4_TABLES)
    for (w0, b0), (w1, b1) in zip(layers, back):
        assert np.allclose(w0, w1, atol=1e-5) and np.allclose(b0, b1, atol=1e-5)
    assert np.allclose(mean_b, state["norm"]["mean"]) and np.allclose(std_b, std)


def _probe_trainer(argv):
    from ctf import train as T
    captured = {}

    class Probe(T.Trainer):
        def run(self):
            captured["trainer"] = self

    original = T.Trainer
    T.Trainer = Probe
    try:
        T.main(argv)
    finally:
        T.Trainer = original
    return captured["trainer"]


def _v4_checkpoint(path, seed=7):
    state = _state(V4_NAMES, V4_TABLES, seed, hidden=(32, 32), critic_hidden=(48, 48))
    state.update(ret_norm={"mean": np.zeros(1), "var": np.ones(1), "count": 1.0}, update=40,
                 samples=123, norm_frozen=True, args={"arch": "mlp"})
    torch.save(state, path)
    return state


def test_v4_checkpoint_warm_starts_in_the_current_layout(tmp_path):
    old = _v4_checkpoint(tmp_path / "v4.pt")
    trainer = _probe_trainer(["--name", "mig", "--runs", str(tmp_path), "--init",
                              str(tmp_path / "v4.pt"), "--hidden", "32", "32",
                              "--critic-hidden", "48", "48"])
    state = trainer.state()
    assert state["feature_names"] == list(F.FEATURE_NAMES)
    assert L.normalize_tables(state["action_tables"]) == L.normalize_tables(F.ACTION_TABLES)
    rng = np.random.default_rng(8)
    x_old = _inputs(V4_NAMES, rng, 20, 0.037, -0.1)
    x_new = _translate(x_old, V4_NAMES, F.FEATURE_NAMES, lambda n: np.zeros(n))
    for net, module in (("actor", trainer.actor), ("critic", trainer.critic)):
        want = _forward(old[net], x_old, old["norm"]["mean"], old["norm"]["var"])
        with torch.no_grad():
            got = module(torch.clamp(torch.as_tensor(
                (x_new - trainer.norm.mean) / trainer.norm.std, dtype=torch.float32), -5, 5))
        assert torch.allclose(want, got, atol=1e-4), net
    # A saved current checkpoint resumes; a v4 one cannot be resumed (only --init).
    trainer.save()
    _probe_trainer(["--name", "mig", "--runs", str(tmp_path), "--hidden", "32", "32",
                    "--critic-hidden", "48", "48"])
    (tmp_path / "old").mkdir()
    _v4_checkpoint(tmp_path / "old" / "checkpoint.pt")
    with pytest.raises(ValueError, match="feature layout"):
        _probe_trainer(["--name", "old", "--runs", str(tmp_path), "--hidden", "32", "32",
                        "--critic-hidden", "48", "48"])


def test_v3_checkpoint_still_warm_starts(tmp_path):
    from ctf.train import V3_DIM
    names, tables = L.V3_FEATURE_NAMES, L.V3_ACTION_TABLES
    state = _state(names, tables, 9, hidden=(16, 16), critic_hidden=(16, 16))
    state.update(ret_norm={"mean": np.zeros(1), "var": np.ones(1), "count": 1.0}, update=1,
                 samples=1, args={"arch": "mlp"})
    assert len(state["norm"]["mean"]) == V3_DIM
    torch.save(state, tmp_path / "v3.pt")
    trainer = _probe_trainer(["--name", "v3", "--runs", str(tmp_path), "--init",
                              str(tmp_path / "v3.pt"), "--hidden", "16", "16",
                              "--critic-hidden", "16", "16"])
    assert len(trainer.norm.mean) == F.FEATURE_DIM


def test_v4_policy_file_migrates_on_load(tmp_path):
    from ctf.npnets import NumpyActor
    from ctf.policy_io import load_policy, read_policy

    spec = importlib.util.spec_from_file_location("ctf_tools_migrate_policy",
                                                  ROOT / "tools" / "migrate_policy.py")
    tool = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no tools/__pycache__
    try:
        spec.loader.exec_module(tool)
    finally:
        sys.dont_write_bytecode = previous
    migrate_file = tool.main

    v4 = FIXTURES / "policy_mlp_v4.npz"
    _, weights, mean, std, meta = read_policy(v4)
    assert "feature_names" not in meta and meta["feature_version"] == 4
    with pytest.raises(ValueError, match="different feature encoder"):
        load_policy(v4, migrate=False)
    actor, meta = load_policy(v4)
    assert meta["migrated_from"]["feature_version"] == 4
    assert meta["feature_names"] == list(F.FEATURE_NAMES)
    migrate_file(v4, tmp_path / "v5.npz")
    actor_file, meta_file = load_policy(tmp_path / "v5.npz", migrate=False)
    assert meta_file["feature_names"] == list(F.FEATURE_NAMES) and meta_file["run"] == meta["run"]
    rng = np.random.default_rng(10)
    x_old = _inputs(V4_NAMES, rng, 64, 0.025, 0.6).astype(np.float32)
    x_new = _translate(x_old, V4_NAMES, F.FEATURE_NAMES, lambda n: np.zeros(n))
    want = NumpyActor(weights, mean, std).logits(x_old)
    assert np.allclose(actor.logits(x_new), want, atol=1e-4)
    assert np.allclose(actor_file.logits(x_new), want, atol=1e-4)


def test_policy_meta_records_names_and_tables(tmp_path):
    from ctf.policy_io import random_policy, read_policy
    random_policy(tmp_path / "p.npz", hidden=(8,))
    meta = read_policy(tmp_path / "p.npz")[-1]
    assert meta["feature_names"] == list(F.FEATURE_NAMES)
    assert meta["action_tables"] == json.loads(json.dumps(F.ACTION_TABLES))
    assert meta["feature_version"] == F.FEATURE_VERSION


def test_feature_names_label_their_columns():
    assert len(F.FEATURE_NAMES) == F.FEATURE_DIM == len(set(F.FEATURE_NAMES))
    assert set(L.V4_FEATURE_NAMES) <= set(F.FEATURE_NAMES)
    match = Match(3, max_ticks=5)
    observation = match.observe()[0]
    memory = F.PlayerMemory()
    memory.set_action(F.THROTTLE_VALUES.index(0.3), F.TURN_VALUES.index(-0.6), True)
    observation["tick"] = 120
    row = dict(zip(F.FEATURE_NAMES, F.encode(observation, memory)))
    assert row["prev.throttle=0.3"] == row["prev.turn=-0.6"] == row["prev.jump"] == 1.0
    assert sum(v for k, v in row.items() if k.startswith("prev.throttle=")) == 1.0
    assert row["game.tick"] == 120 / 6000.0
    assert row["self.speed"] == observation["self"]["speed"] / 18.0
    assert row["nbump.age"] == row["nbump.dist"] == 1.0  # no contact remembered yet
