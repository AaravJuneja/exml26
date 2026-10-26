"""Tests for the match replay recorder (tools/replay.py) and the viewer's parser."""

import importlib.util
import json
import shutil
import subprocess
import sys

import pytest

from ctf import evaluate as E
from ctf.features import THROTTLE_VALUES, TURN_VALUES
from ctf.policies import ROOT

TICKS = 300
SEED = 1000


def _load_replay_module():
    spec = importlib.util.spec_from_file_location("ctf_tools_replay", ROOT / "tools" / "replay.py")
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no tools/__pycache__
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


R = _load_replay_module()


@pytest.fixture(scope="module")
def recorded():
    replay, match = R.record_match("baseline", "example", SEED, "red", max_ticks=TICKS)
    return json.loads(R.dumps(replay)), match


@pytest.fixture(scope="module")
def policy_path(tmp_path_factory):
    from ctf.policy_io import random_policy

    path = tmp_path_factory.mktemp("policy") / "random.npz"
    random_policy(path, hidden=(32, 32), seed=3)
    return str(path)


def test_structure_and_tick_counts(recorded):
    replay, match = recorded
    meta = replay["meta"]
    assert replay["version"] == R.REPLAY_VERSION
    assert meta["ticks"] == match.tick == TICKS and meta["every"] == 1
    assert [f["t"] for f in replay["frames"]] == list(range(TICKS + 1))
    n_spheres = len(replay["spheres"])
    assert n_spheres == 2 * E.PLAYERS_PER_TEAM
    assert [s["id"] for s in replay["spheres"]] == list(range(n_spheres))
    for frame in replay["frames"]:
        assert len(frame["s"]) == n_spheres
        assert all(len(s) == len(R.SPHERE_FIELDS) for s in frame["s"])
        assert len(frame["f"]) == len(frame["sc"]) == len(meta["teams"])
        assert "a" not in frame  # scripted players record no actions by default
    final = replay["frames"][-1]
    assert dict(zip(meta["teams"], final["sc"])) == meta["score"] == match.score
    m = replay["map"]
    assert len(m["obstacles"]) == len(match.obstacles)
    assert len(m["ramps"]) == len(match.ramps)
    assert all(len(r["c"]) == 4 for r in m["ramps"])
    assert set(m["homes"]) == set(meta["teams"])


def test_matches_evaluate(recorded):
    replay, _ = recorded
    record = E.run_match("baseline", "example", SEED, "red", max_ticks=TICKS)
    meta = replay["meta"]
    assert meta["score"] == record["score"] and meta["result"] == record["result"]
    events = replay["events"]
    for role, side in (("a", "red"), ("b", "blue")):
        team = [e for e in events if e["team"] == side]
        assert sum(e["type"] == "pickup" for e in team) == record[role]["pickups"]
        assert sum(e["type"] == "capture" for e in team) == record[role]["captures"]
        for reason in ("wall", "player"):
            assert (sum(e["type"] == "death" and e["reason"] == reason for e in team)
                    == record[role][f"{reason}_deaths"])


def test_deaths_match_death_log(recorded):
    replay, match = recorded
    deaths = [e for e in replay["events"] if e["type"] == "death"]
    log = match.death_log()
    assert deaths, "seed should produce at least one death in 300 ticks"
    assert len(deaths) == len(log)
    for event, message in zip(deaths, log):
        assert (event["tick"], event["team"], event["player"], event["reason"]) == (
            message["tick"], message["team"], message["player"], message["reason"])
        assert event["x"] == pytest.approx(message["x"], abs=0.006)
        assert event["y"] == pytest.approx(message["y"], abs=0.006)
        sphere = replay["spheres"][event["id"]]
        assert (sphere["team"], sphere["player"]) == (event["team"], event["player"])
        # dead in the frame right after the step in which it died
        assert replay["frames"][event["tick"] + 1]["s"][event["id"]][5] == 0
    ticks = [e["tick"] for e in replay["events"]]
    assert ticks == sorted(ticks)


def test_subsampling_keeps_final_tick():
    replay, _ = R.record_match("baseline", "example", SEED, "red", max_ticks=TICKS, every=7)
    ticks = [f["t"] for f in replay["frames"]]
    assert ticks == list(range(0, TICKS, 7)) + [TICKS]


def test_policy_actions(policy_path):
    replay, _ = R.record_match(policy_path, "baseline", SEED, "blue", max_ticks=120)
    policy_ids = {s["id"] for s in replay["spheres"] if s["team"] == "blue"}
    assert {s["id"] for s in replay["spheres"] if s["actions"]} == policy_ids
    seen = 0
    for frame in replay["frames"][:-1]:
        for sid, action in enumerate(frame["a"]):
            if sid not in policy_ids or not frame["s"][sid][5]:
                assert action is None
                continue
            throttle, turn, jump = action
            assert any(abs(throttle - v) < 1e-3 for v in THROTTLE_VALUES)
            assert any(abs(turn - v) < 1e-3 for v in TURN_VALUES)
            assert jump in (0, 1)
            seen += 1
    assert seen > 0


def test_cli_and_viewer(tmp_path):
    out, html = tmp_path / "r.json", tmp_path / "r.html"
    R.main(["baseline", "idle", "--seed", "3", "--ticks", "60", "--every", "2",
            "--out", str(out), "--html", str(html)])
    replay = json.loads(out.read_text())
    assert replay["meta"]["ticks"] == 60 and len(replay["frames"]) == 31
    page = html.read_text()
    assert "<!--REPLAY_DATA-->" not in page and 'id="replay-data"' in page

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available for the viewer parse check")
    source = (ROOT / "docs" / "viewer.html").read_text()
    script = source.split("<script>", 1)[1].split("</script>", 1)[0]
    module = tmp_path / "viewer.js"
    module.write_text(script)
    check = (f"const V = require({json.dumps(str(module))});"
             f"const R = V.parseReplay(require({json.dumps(str(out))}));"
             "console.log(R.frames.length, R.lastTick, R.events.length);")
    result = subprocess.run([node, "-e", check], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["31", "60", str(len(replay["events"]))]
