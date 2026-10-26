"""Bit-for-bit parity between ctf.sim.Match and the reference engine.game.Game."""

import pytest

import engine.game as game_module
from ctf.policies import ROOT, load_decide
from ctf.sim import Match

WIGGLE = ROOT / "ctf" / "tests" / "wiggle_policy.py"


class _Recorder:
    """Stand-in for rpc ModelWorker: in-process, records every observation."""

    log = []

    def __init__(self, team, model_path, sphere_id=None):
        self.sphere_id = sphere_id
        self._decide = load_decide(model_path)

    def wait_ready(self, timeout=None):
        return True

    def get_action(self, observation, timeout):
        _Recorder.log.append((observation["tick"], self.sphere_id, observation))
        return self._decide(observation)

    def close(self):
        pass


class _NullTelemetry:
    file_path = None

    def record_map(self, *args, **kwargs):
        pass

    def record_tick(self, *args, **kwargs):
        pass


class _Done:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _SyncPool:
    def submit(self, fn, *args):
        return _Done(fn(*args))


def _reference(seed, red, blue, ticks, monkeypatch):
    monkeypatch.setattr(game_module, "ModelWorker", _Recorder)
    monkeypatch.setattr(game_module, "TelemetryRecorder", _NullTelemetry)
    _Recorder.log = []
    teams = [{"name": "red", "model_path": str(red)}, {"name": "blue", "model_path": str(blue)}]
    game = game_module.Game(teams, spheres_per_team=3, seed=seed, max_ticks=ticks)
    game._pool.shutdown(wait=False)
    game._pool = _SyncPool()
    while not game.is_over():
        game.step()
    return game, list(_Recorder.log)


def _candidate(seed, red, blue, ticks, broadphase):
    match = Match(seed, max_ticks=ticks, broadphase=broadphase, teammates_field=True)
    decide = {s.id: load_decide(red if s.team == "red" else blue) for s in match.spheres}
    log = []
    while not match.is_over():
        observations = match.observe()
        actions = {}
        for sid, observation in observations.items():
            log.append((observation["tick"], sid, observation))
            actions[sid] = decide[sid](observation)
        match.act(actions)
    return match, log


def _state(spheres, flags, score):
    return (
        [(s.x, s.y, s.z, s.vx, s.vy, s.vz, s.yaw, s.alive) for s in spheres],
        [(f.x, f.y, f.carrier.id if f.carrier else -1) for f in flags],
        dict(score),
    )


@pytest.mark.parametrize("broadphase", [False, True])
@pytest.mark.parametrize(
    "seed,red,blue",
    [
        (3, WIGGLE, WIGGLE),
        (11, ROOT / "teams/baseline/model.py", ROOT / "teams/example_team/model.py"),
        (29, ROOT / "teams/example_team/model2.py", ROOT / "teams/baseline/model.py"),
    ],
)
def test_match_reproduces_reference_game(seed, red, blue, broadphase, monkeypatch):
    ticks = 1500
    game, reference_log = _reference(seed, red, blue, ticks, monkeypatch)
    match, candidate_log = _candidate(seed, red, blue, ticks, broadphase)
    assert len(reference_log) == len(candidate_log)
    for expected, actual in zip(reference_log, candidate_log):
        assert expected == actual
    assert _state(game.spheres, game.flags, game.score) == _state(
        match.spheres, match.flags, match.score
    )
    assert game._death_log == match.death_log()
