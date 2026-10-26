"""Training opponents under opponents/: contract, validity and a short match each."""

import ast
import math
import pickle
import sys
import time

import pytest

from ctf.policies import ROOT, load_decide
from ctf.sim import DOCUMENTED_KEYS, Match
from opponents import build

BOTS = sorted(build.STYLES)
FORBIDDEN = ("teammates", "sonar_recieve")


class StrictObservation(dict):
    """Observation that refuses any key outside the documented contract."""

    def _check(self, key):
        if key not in DOCUMENTED_KEYS:
            raise KeyError(f"undocumented observation key {key!r}")

    def __getitem__(self, key):
        self._check(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._check(key)
        return dict.get(self, key, default)

    def __contains__(self, key):
        self._check(key)
        return dict.__contains__(self, key)


def _strict(observation):
    return StrictObservation({k: observation[k] for k in DOCUMENTED_KEYS})


def _valid(action):
    assert isinstance(action, tuple) and len(action) in (3, 4)
    throttle, turn, jump = action[:3]
    assert isinstance(throttle, float) and math.isfinite(throttle) and -1.0 <= throttle <= 1.0
    assert isinstance(turn, float) and math.isfinite(turn) and -1.0 <= turn <= 1.0
    assert isinstance(jump, bool)
    if len(action) == 4 and action[3] is not None:
        assert len(pickle.dumps(action[3])) < 512


def test_generated_models_are_current():
    for name in BOTS:
        path = ROOT / "opponents" / name / "model.py"
        assert path.read_text() == build.render(name), f"run python -m opponents.build ({name})"


@pytest.mark.parametrize("name", BOTS)
def test_standalone_stdlib_and_no_forbidden_keys(name):
    source = (ROOT / "opponents" / name / "model.py").read_text()
    for key in FORBIDDEN:
        assert key not in source
    tree = ast.parse(source)
    names = {"decide"} & {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert names == {"decide"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            mods = [node.module]
        else:
            continue
        for mod in mods:
            assert mod.split(".")[0] in sys.stdlib_module_names, mod


@pytest.mark.parametrize("name", BOTS)
def test_short_match_valid_actions(name):
    """Both seats against the baseline: strict documented observations, valid actions, fast."""
    for seat in ("red", "blue"):
        match = Match(2024, max_ticks=500, teammates_field=True)
        path = ROOT / "opponents" / name / "model.py"
        mine = {s.id: load_decide(path) for s in match.spheres if s.team == seat}
        other = {s.id: load_decide(ROOT / "teams/baseline/model.py")
                 for s in match.spheres if s.team != seat}
        calls, spent, worst = 0, 0.0, 0.0
        while not match.is_over():
            actions = {}
            for sid, obs in match.observe().items():
                if sid in mine:
                    t0 = time.perf_counter()
                    action = mine[sid](_strict(obs))
                    dt = time.perf_counter() - t0
                    spent += dt
                    worst = max(worst, dt)
                    calls += 1
                    _valid(action)
                    actions[sid] = action
                else:
                    actions[sid] = other[sid](obs)
            match.act(actions)
        assert calls > 0
        assert spent / calls < 0.5e-3, f"{name}: {1e3 * spent / calls:.3f} ms per decide"
        assert worst < 25e-3, f"{name}: worst decide {1e3 * worst:.1f} ms"


@pytest.mark.parametrize("name", BOTS)
def test_survives_odd_observations(name):
    """Missing flags/enemies, respawn gaps and a new match in the same process."""
    decide = load_decide(ROOT / "opponents" / name / "model.py")
    me = {"id": 4, "team": "blue", "player": 2, "x": -91.5, "y": 0.0, "z": 1.0, "vx": 0.0,
          "vy": 0.0, "vz": 0.0, "speed": 0.0, "yaw": 0.0, "grounded": True, "alive": True,
          "carrying": None}
    flag = {"team": "blue", "x": -90.0, "y": 0.0, "carried_by": -1, "at_home": True}
    base = {"self": me, "visible_enemies": [], "own_flag": flag, "other_flags": [],
            "score": {"blue": 0}, "sonar_receive": [{"from_id": 3, "from_player": 1,
                                                     "value": "garbage"}],
            "death_messages": [{"player": 1, "team": "blue", "tick": 0, "x": -60.0,
                                "y": 3.0, "reason": "wall"}]}
    for tick in (0, 1, 2, 80, 81, 5, 6):
        _valid(decide(StrictObservation(dict(base, tick=tick))))


def test_chaos_team_agrees_on_draw():
    """Unseeded chaos: player 1's seed reaches its mates by sonar at spawn."""
    match = Match(7, max_ticks=8)
    decide = {s.id: load_decide(ROOT / "opponents" / "chaos" / "model.py") for s in match.spheres}
    while not match.is_over():
        observations = match.observe()
        match.act({sid: decide[sid](_strict(obs)) for sid, obs in observations.items()})
    for team in ("red", "blue"):
        seeds = {decide[s.id].__globals__["M"].team_seed for s in match.spheres if s.team == team}
        assert len(seeds) == 1
