"""Tests for tools/release.py (tiny seeds and short matches; latency skipped or faked)."""

import importlib.util
import json

import numpy as np
import pytest

from ctf.export import render
from ctf.features import FEATURE_DIM
from ctf.npnets import HEAD_SIZES
from ctf.policies import ROOT
from ctf.policy_io import save_policy

_spec = importlib.util.spec_from_file_location("release_tool", ROOT / "tools" / "release.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

TICKS = 300
QUICK = ["--seeds", "5000", "--opponents", "baseline", "--max-ticks", str(TICKS),
         "--workers", "1", "--py39", "", "--quiet"]


def _random_policy(path, seed, hidden=(32, 32), **meta):
    """Random MLP policy without torch, with provenance meta as training writes it."""
    rng = np.random.default_rng(seed)
    sizes = [FEATURE_DIM, *hidden, sum(HEAD_SIZES)]
    weights = [(rng.normal(0, 1 / np.sqrt(n), (n, m)).astype(np.float32),
                rng.normal(0, 0.1, m).astype(np.float32)) for n, m in zip(sizes, sizes[1:])]
    save_policy(path, weights, np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM), **meta)
    return path


@pytest.fixture(scope="module")
def policy(tmp_path_factory):
    path = tmp_path_factory.mktemp("policies") / "u00042.npz"
    return _random_policy(path, 7, run="unit", update=42, samples=1234)


@pytest.fixture(scope="module")
def good_source(policy):
    return render(policy)


# -------------------------------------------------------------------- audit
def test_audit_passes_exported_model(good_source):
    audit = R.audit_source(good_source)
    assert audit["pass"], audit["problems"]
    assert "numpy" in audit["imports"]
    # The inlined encoder mentions teammates only in a comment, which is allowed.
    assert all(m["word"] == "teammates" for m in audit["comment_mentions"])


GOOD = "import math\nimport numpy as np\n\ndef decide(observation):\n    return 0.0, 0.0, False\n"


@pytest.mark.parametrize("source, check", [
    ("import torch\n" + GOOD, "imports_ok"),
    ("from sklearn import tree\n" + GOOD, "imports_ok"),
    ("import subprocess\n" + GOOD, "imports_ok"),
    (GOOD + "def f():\n    import requests\n", "imports_ok"),
    (GOOD + "x = __import__('torch')\n", "no_dangerous_calls"),
    (GOOD + "y = open('/tmp/x', 'w')\n", "no_dangerous_calls"),
    (GOOD + "import os\nos.system('ls')\n", "no_dangerous_calls"),
    (GOOD.replace("return", "mates = observation.get('teammates')\n    return"),
     "no_undocumented_fields"),
    (GOOD.replace("return", "m = observation['sonar_recieve']\n    return"),
     "no_undocumented_fields"),
    (GOOD.replace("return", "m = observation.teammates\n    return"), "no_undocumented_fields"),
    (GOOD.replace("decide", "act"), "defines_decide"),
    (GOOD.replace("decide(observation)", "decide(observation, extra)"), "defines_decide"),
    ("class X:\n    def decide(self, observation):\n        pass\n", "defines_decide"),
    (GOOD + "match = 1\nmatch x:\n    case 1:\n        pass\n", "py39_syntax"),
    (GOOD + "def broken(:\n", "parses"),
])
def test_audit_catches_bad_file(source, check):
    audit = R.audit_source(source)
    assert not audit["pass"]
    assert audit["checks"][check] is False, audit


def test_audit_size_limit_and_comments(tmp_path, good_source):
    assert not R.audit_source(good_source, max_kib=1)["checks"]["size_ok"]
    commented = GOOD + "# teammates and sonar_recieve are never read\n"
    assert R.audit_source(commented)["pass"]
    path = tmp_path / "bad.py"
    path.write_bytes(b"\xff\xfe" + GOOD.encode())
    assert not R.audit_file(path)["pass"]


def test_seed_guard():
    assert R.check_seeds([5000, 5031]) == []
    assert R.check_seeds([1000]) == ["evaluation"]
    assert R.check_seeds([2099, 100000]) == ["training", "evaluation"]
    with pytest.raises(SystemExit):
        R.main(["unused.npz", "--name", "x", "--seeds", "1000-1001"])


def test_parse_latency_output():
    text = 'noise\n{\n  "verdict": {\n    "all_pass": true\n  }\n}\n'
    assert R.parse_latency_output(text)["verdict"]["all_pass"] is True
    with pytest.raises(ValueError):
        R.parse_latency_output("no json here")


# ------------------------------------------------------------- end to end
def test_quick_release_end_to_end(tmp_path, policy):
    teams, docs = tmp_path / "teams", tmp_path / "RELEASE.md"
    code = R.main([str(policy), "--name", "quick", "--teams-dir", str(teams), "--docs", str(docs),
                   "--skip-latency", "--compare", str(policy), *QUICK])
    out = teams / "quick"
    assert code == 2  # INCOMPLETE: latency skipped and matches shortened
    report = json.loads((out / "release_report.json").read_text())
    assert report["status"] == "INCOMPLETE"
    assert report["checks"] == {"audit": True, "parity": True, "latency": None,
                                "gauntlet_no_errors": True}
    assert report["policy"]["run"] == "unit" and report["policy"]["update"] == 42
    assert report["policy"]["samples"] == 1234
    assert report["policy"]["sha256"] == R.sha256(policy) == R.sha256(out / "policy.npz")
    assert report["policy"]["model_sha256"] == R.sha256(out / "model.py")
    assert len(report["parity"]["games"]) == 2  # one parity seed (the gauntlet's) x 2 sides
    rows = report["strength"]["rows"]
    assert [r["opponent"] for r in rows] == ["baseline"] and rows[0]["n"] == 2
    # Comparing a policy with its own export: every paired difference is zero.
    delta = report["compare"]["rows"][0]
    assert delta["goal_diff_delta"] == 0 and delta["match_points_delta"] == 0
    readme = (out / "README.md").read_text()
    assert R.sha256(out / "model.py") in readme and "| update | 42 |" in readme
    text = docs.read_text()
    assert "## quick: INCOMPLETE" in text and "| baseline |" in text

    # Releasing the same name again replaces its docs section instead of appending.
    R.main([str(policy), "--name", "quick", "--teams-dir", str(teams), "--docs", str(docs),
            "--skip-latency", *QUICK, "--opponents", "idle"])
    text = docs.read_text()
    assert text.count("<!-- release:quick begin -->") == 1 and "| idle |" in text


def test_latency_gates_release(tmp_path, policy, monkeypatch):
    """A full-length release passes only if every latency run passes (latency faked)."""
    calls = []

    def fake(model, ticks, stress, seed, workdir=None):
        calls.append(stress)
        return {"ticks": ticks, "stress_procs": stress, "pass": stress == 0,
                "verdict": {"callback_max_le_20ms": stress == 0, "all_pass": stress == 0},
                "callback": {"p50_ms": 1.0, "p99_ms": 2.0, "max_ms": 25.0, "over_20ms": 1}}

    monkeypatch.setattr(R, "run_latency", fake)
    monkeypatch.setattr(R, "FULL_TICKS", TICKS)  # treat the short matches as full length
    args = [str(policy), "--name", "gate", "--teams-dir", str(tmp_path), "--docs", "",
            "--latency-ticks", str(TICKS), "--stress", "4", "--parity-opponent", "idle", *QUICK]
    assert R.main(args) == 1
    report = json.loads((tmp_path / "gate" / "release_report.json").read_text())
    assert calls == [0, 4] and report["status"] == "FAIL" and report["checks"]["latency"] is False

    monkeypatch.setattr(R, "run_latency", lambda *a, **k: {**fake(*a, **k), "pass": True})
    assert R.main(args) == 0


def test_parity_catches_wrong_weights(tmp_path, policy):
    """A model.py exported from different weights fails parity, traced to a real mismatch."""
    from ctf.features import N_THROTTLE, THROTTLE_VALUES, TURN_VALUES
    from ctf.policy_io import read_policy

    # Constant full throttle while turning: wall deaths on seed 5000 whatever the features
    # (two random policies can both play an uneventful game with identical records).
    other = _random_policy(tmp_path / "other.npz", 8)
    _, weights, mean, std, _ = read_policy(other)
    w, b = weights[-1]
    b = np.zeros_like(b)
    b[THROTTLE_VALUES.index(1.0)] = b[N_THROTTLE + TURN_VALUES.index(0.3)] = 5.0
    weights[-1] = (np.zeros_like(w), b)
    save_policy(other, weights, mean, std)
    model = tmp_path / "model.py"
    model.write_text(render(other))
    assert R.first_divergence(model, other, "baseline", 5000, "red", TICKS) is None
    result = R.parity(model, policy, "baseline", [5000], TICKS, workers=1)
    assert not result["pass"]
    differing = [g for g in result["games"] if not g["identical"]]
    assert differing
    for game in differing:
        assert game["status"] == "MISMATCH" and game["deployed_equals_inprocess"]
        assert game["divergence"]["explained"] is False and game["divergence"]["heads"]


def test_bad_model_stops_release(tmp_path, policy, monkeypatch):
    original = R.export_release

    def export_then_break(policy_path, out_dir):
        info = original(policy_path, out_dir)
        path = R.Path(info["model"])
        path.write_text(path.read_text() + "\n_X = {}.get('teammates')\n")
        return info

    monkeypatch.setattr(R, "export_release", export_then_break)
    code = R.main([str(policy), "--name", "bad", "--teams-dir", str(tmp_path), "--docs", "",
                   "--skip-latency", *QUICK])
    report = json.loads((tmp_path / "bad" / "release_report.json").read_text())
    assert code == 1 and report["checks"]["audit"] is False
    assert "parity" not in report and "strength" not in report
