"""Loading of scripted `decide` modules with one independent instance per player."""

import importlib.util
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_counter = itertools.count()

SCRIPTED = {
    "baseline": "teams/baseline/model.py",
    "example": "teams/example_team/model.py",
    "example2": "teams/example_team/model2.py",
}


def load_decide(path):
    """Fresh module instance, so module-level state is private to one player."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    spec = importlib.util.spec_from_file_location(f"ctf_policy_{next(_counter)}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decide


def idle(observation):
    return 0.0, 0.0, False


class ScriptedTeam:
    """Three independent instances of a scripted model, keyed by sphere id."""

    def __init__(self, name, sphere_ids):
        self.name = name
        if name == "idle":
            self._decide = {sid: idle for sid in sphere_ids}
        else:
            path = SCRIPTED.get(name, name)
            self._decide = {sid: load_decide(path) for sid in sphere_ids}

    def act(self, observations):
        actions = {}
        for sid, decide in self._decide.items():
            observation = observations.get(sid)
            if observation is None:
                continue
            try:
                actions[sid] = decide(observation)
            except Exception:
                actions[sid] = (0.0, 0.0, False, None)
        return actions
