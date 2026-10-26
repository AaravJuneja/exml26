"""Recorded fingerprints of the rollout hot path, for exact-equivalence tests.

Scenarios:
* `scripted`: full matches between scripted teams (every bot in the training mix),
  with features encoded for all six players, memory wipes at fixed ticks and one
  memory set carried into a second match (tick going backwards). Records the
  complete observation history, every feature row and every scripted action.
* `rollout`: `ctf.rollout.collect` with a trained MLP policy (and a random GRU)
  against several opponents, including mirror self-play and memory wipes.
  Records obs, actions, logp, rewards, alive, stats (and the GRU's hidden states).

Streams are hashed in chunks of `CHUNK` ticks (exact `repr` of every value, so
float bits, types and key order all count); a mismatch names the tick range.
Floating-point library results differ between machines (and BLAS thread
counts), so fingerprints are stored per platform:

    python -m ctf.tests.equivalence --write      # record this platform's fixture
    python -m ctf.tests.equivalence              # compare against it
"""

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

from ctf import features as F
from ctf.policies import ROOT, ScriptedTeam
from ctf.rollout import documented
from ctf.sim import Match

DATA = Path(__file__).resolve().parent / "fixtures"
FIXTURE = DATA / "equivalence.json"
POLICY = DATA / "policy_mlp_v5.npz"
CHUNK = 250

# (seed, red, blue, ticks); the last scenario continues with the same memories and bots
SCRIPTED = [
    (101, "relay", "baseline", 2400),
    (202, "rusher", "camper", 2000),
    (303, "guardian", "balanced", 2000),
    (404, "chaos", "steady", 2000),
    (505, "example", "example2", 1500),
    (506, str(ROOT / "ctf/tests/wiggle_policy.py"), "idle", 1200),
]
CONTINUE = {506: 505}  # scenario 506 reuses the memories and bots of 505 (memory carry-over)
WIPE_EVERY = 397       # memory wipe of one player every WIPE_EVERY ticks

ROLLOUTS = {
    "mlp": dict(ticks=1500, seed=11, wipe=0.5, hidden_every=0, jobs=[
        {"seed": 7001, "side": "red", "opponent": "relay"},
        {"seed": 7002, "side": "blue", "opponent": "baseline"},
        {"seed": 7003, "side": "red", "opponent": "mirror"},
        {"seed": 7004, "side": "blue", "opponent": "self"},
        {"seed": 7005, "side": "red", "opponent": "guardian"},
        {"seed": 7006, "side": "blue", "opponent": "idle"},
    ]),
    "gru": dict(ticks=600, seed=12, wipe=0.5, hidden_every=4, jobs=[
        {"seed": 7101, "side": "blue", "opponent": "mirror"},
        {"seed": 7102, "side": "red", "opponent": "camper"},
        {"seed": 7103, "side": "red", "opponent": "self"},
    ]),
}


def platform_key():
    # BLAS results (hence logits and log-probs) depend on its thread count.
    threads = os.environ.get("OPENBLAS_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS") or "default"
    return (f"{sys.platform}-{platform.machine()}-py{sys.version_info[0]}.{sys.version_info[1]}"
            f"-threads{threads}")


class Stream:
    """Chunked sha256 of a sequence of reprs, chunk = tick // CHUNK."""

    def __init__(self):
        self.chunks = []
        self._hash = None
        self._chunk = -1

    def add(self, tick, value):
        chunk = tick // CHUNK
        if chunk != self._chunk:
            self._flush()
            self._chunk = chunk
            self._hash = hashlib.sha256()
        self._hash.update(repr(value).encode())
        self._hash.update(b"\n")

    def _flush(self):
        if self._hash is not None:
            self.chunks.append(self._hash.hexdigest()[:16])
            self._hash = None

    def result(self):
        self._flush()
        return self.chunks


def _nearest(values, value):
    return min(range(len(values)), key=lambda i: abs(values[i] - value))


def _team(spec, ids, seed):
    team = ScriptedTeam(spec, ids)
    for decide in team._decide.values():
        if "SEED" in decide.__globals__:
            decide.__globals__["SEED"] = seed  # chaos: reproducible draws
    return team


def play_scripted(seed, red, blue, ticks, state=None):
    match = Match(seed, max_ticks=ticks, teammates_field=True)
    if state is None:
        teams = {name: _team(spec, [s.id for s in match.spheres if s.team == name], seed)
                 for name, spec in zip(match.team_names, (red, blue))}
        memories = {s.id: F.PlayerMemory() for s in match.spheres}
    else:
        teams, memories = state
    obs_stream, feat_stream, act_stream = Stream(), Stream(), Stream()
    n_feat = 0
    while not match.is_over():
        t = match.tick
        observations = match.observe()
        obs_stream.add(t, sorted(observations.items()))
        for sid, observation in observations.items():
            memory = memories[sid]
            if t % WIPE_EVERY == WIPE_EVERY - 1 and sid == (t // WIPE_EVERY) % 6:
                memory.reset()
            row = F.encode(documented(observation), memory)
            feat_stream.add(t, (sid, row))
            if t % 7 == 0:
                feat_stream.add(t, F.encode(documented(observation), memory))  # cached path
            n_feat += 1
        actions = {}
        for team in teams.values():
            actions.update(team.act(observations))
        act_stream.add(t, sorted(actions.items(), key=lambda kv: kv[0]))
        for sid in observations:
            action = actions.get(sid)
            if isinstance(action, (tuple, list)) and len(action) >= 3:
                memories[sid].set_action(_nearest(F.THROTTLE_VALUES, float(action[0])),
                                         _nearest(F.TURN_VALUES, float(action[1])),
                                         bool(action[2]))
        match.act(actions)
    log = match.death_log()
    events = {
        "rows": n_feat,
        "deaths": len(log),
        "wall_deaths": sum(1 for m in log if m["reason"] == "wall"),
        "score": dict(match.score),
        "bumps": sum(len(m.bumps) for m in memories.values()),
    }
    result = {"observations": obs_stream.result(), "features": feat_stream.result(),
              "actions": act_stream.result(), "events": events}
    return result, (teams, memories)


def run_scripted():
    out, states = {}, {}
    for seed, red, blue, ticks in SCRIPTED:
        state = states.get(CONTINUE.get(seed))
        result, states[seed] = play_scripted(seed, red, blue, ticks, state)
        out[f"{seed}:{Path(red).stem if '/' in red else red}-{blue}"] = result
    return out


def gru_actor(seed=3, hidden=32):
    from ctf.npnets import NumpyGRUActor
    rng = np.random.default_rng(seed)
    d, n_out = F.FEATURE_DIM, F.N_THROTTLE + F.N_TURN + 2
    shapes = {"W_in": (d, 64), "b_in": (64,), "W_ih": (64, 3 * hidden), "b_ih": (3 * hidden,),
              "W_hh": (hidden, 3 * hidden), "b_hh": (3 * hidden,), "W_out": (hidden, n_out),
              "b_out": (n_out,)}
    weights = {k: (rng.normal(0.0, 0.3, s)).astype(np.float32) for k, s in shapes.items()}
    weights["b_out"][F.THROTTLE_VALUES.index(1.0)] += 2.0
    return NumpyGRUActor(weights, np.zeros(d), np.ones(d))


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def run_rollouts():
    from ctf.policy_io import load_policy
    from ctf.rollout import _POLICY_CACHE, collect
    out = {}
    for name, cfg in ROLLOUTS.items():
        actor = load_policy(POLICY)[0] if name == "mlp" else gru_actor()
        jobs = [dict(j) for j in cfg["jobs"]]
        if name == "mlp":
            jobs.append({"seed": 7007, "side": "red", "opponent": f"policy:{POLICY}"})
            jobs.append({"seed": 7008, "side": "blue", "opponent": f"greedy:{POLICY}"})
        _POLICY_CACHE.clear()
        data = collect(actor, jobs, {}, cfg["ticks"], cfg["seed"], cfg["wipe"],
                       cfg["hidden_every"])
        record = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if key == "obs":  # per tick chunk, to localise a mismatch
                    record[key] = [_digest(value[i:i + CHUNK])
                                   for i in range(0, len(value), CHUNK)]
                else:
                    record[key] = _digest(value)
        stats = [{k: v for k, v in s.items() if k != "opponent"} for s in data["stats"]]
        record["stats"] = hashlib.sha256(repr(stats).encode()).hexdigest()[:16]
        record["summary"] = {
            "goals": [s["goals_for"] for s in data["stats"]],
            "alive_frac": round(float(data["alive"].mean()), 4),
            "wall_deaths": sum(s["wall_deaths"] for s in data["stats"]),
            "pickups": sum(s["pickups"] for s in data["stats"]),
        }
        out[name] = record
    return out


def compute():
    return {"scripted": run_scripted(), "rollout": run_rollouts()}


def load():
    if not FIXTURE.exists():
        return {}
    return json.loads(FIXTURE.read_text())


def main():
    import time
    start = time.perf_counter()
    current = compute()
    print(f"computed in {time.perf_counter() - start:.1f}s")
    for name, result in current["scripted"].items():
        print(name, result["events"])
    for name, result in current["rollout"].items():
        print(name, result["summary"])
    key = platform_key()
    if "--write" in sys.argv:
        stored = load()
        stored[key] = current
        DATA.mkdir(exist_ok=True)
        FIXTURE.write_text(json.dumps(stored, indent=1, sort_keys=True) + "\n")
        print("wrote", key)
        return
    expected = load().get(key)
    if expected is None:
        print("no fixture for", key)
        sys.exit(1)
    print("MATCH" if expected == current else "MISMATCH")
    sys.exit(0 if expected == current else 1)


if __name__ == "__main__":
    main()
