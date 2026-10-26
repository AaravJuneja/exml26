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
    **{name: f"opponents/{name}/model.py"
       for name in ("rusher", "guardian", "camper", "relay", "balanced", "chaos", "steady")},
}


_CODE_CACHE = {}


def load_decide(path):
    """Fresh module instance, so module-level state is private to one player.

    The source is compiled once per process (keyed by path, size and mtime) and
    executed into a new module namespace for every instance.
    """
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    code = _CODE_CACHE.get(key)
    if code is None:
        code = compile(path.read_bytes(), str(path), "exec")
        _CODE_CACHE[key] = code
    spec = importlib.util.spec_from_file_location(f"ctf_policy_{next(_counter)}", path)
    module = importlib.util.module_from_spec(spec)
    exec(code, module.__dict__)
    return module.decide


def _silent(*args, **kwargs):
    pass


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
            for decide in self._decide.values():
                decide.__globals__["print"] = _silent

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
