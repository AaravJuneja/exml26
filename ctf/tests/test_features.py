import math
import time

from ctf import features as F
from ctf.policies import ROOT, load_decide
from ctf.sim import Match


def _play(seed=5, ticks=600, policy="ctf/tests/wiggle_policy.py"):
    match = Match(seed, max_ticks=ticks)
    decide = {s.id: load_decide(ROOT / policy) for s in match.spheres}
    memories = {s.id: F.PlayerMemory() for s in match.spheres}
    rows = []
    while not match.is_over():
        observations = match.observe()
        actions = {}
        for sid, observation in observations.items():
            rows.append((sid, F.encode(observation, memories[sid])))
            actions[sid] = decide[sid](observation)
        match.act(actions)
    return match, memories, rows


def test_dimension_and_finite():
    _, _, rows = _play()
    assert rows
    for _, row in rows:
        assert len(row) == F.FEATURE_DIM
        assert all(math.isfinite(v) for v in row)
        assert max(abs(v) for v in row) <= 10.0


def _rotate(observation, team):
    """Rotate every documented position/velocity by 180 degrees and relabel the team."""
    import copy
    o = copy.deepcopy(observation)

    def rot(d):
        for kx, ky in (("x", "y"), ("vx", "vy")):
            if kx in d:
                d[kx], d[ky] = -d[kx], -d[ky]
        if "yaw" in d:
            d["yaw"] = d["yaw"] + math.pi
    rot(o["self"])
    o["self"]["team"] = team
    for e in o["visible_enemies"]:
        rot(e)
    rot(o["own_flag"])
    for f in o["other_flags"]:
        rot(f)
    for m in o["death_messages"]:
        rot(m)
    return o


def test_canonical_frame_is_exactly_rotation_invariant():
    match, _, _ = (None, None, None)
    match = Match(7, max_ticks=400)
    decide = {s.id: load_decide(ROOT / "ctf/tests/wiggle_policy.py") for s in match.spheres}
    mem_a, mem_b = F.PlayerMemory(), F.PlayerMemory()
    checked = 0
    while not match.is_over():
        observations = match.observe()
        if 0 in observations:
            original = observations[0]
            fa = F.encode(original, mem_a)
            fb = F.encode(_rotate(original, "blue"), mem_b)
            assert max(abs(a - b) for a, b in zip(fa, fb)) < 1e-9
            mem_a.set_action(3, 2, 0)
            mem_b.set_action(3, 2, 0)
            checked += 1
        match.act({sid: decide[sid](o) for sid, o in observations.items()})
    assert checked > 200
    assert F.encode(match.observe()[0], F.PlayerMemory())[0] < 0  # own home on canonical -x


def test_encode_is_idempotent_within_a_tick():
    match = Match(9, max_ticks=5)
    observation = match.observe()[0]
    memory = F.PlayerMemory()
    first = F.encode(observation, memory)
    assert F.encode(observation, memory) == first


def test_flag_velocity_resets_on_carrier_change():
    memory = F.PlayerMemory()
    base = {"tick": 0, "self": {"x": -80.0, "y": 0.0, "speed": 0.0, "player": 1, "team": "blue",
                                "id": 3},
            "visible_enemies": [], "other_flags": [], "score": {}, "sonar_receive": [],
            "death_messages": []}
    F.encode(dict(base, tick=0, own_flag={"x": -90.0, "y": 0.0, "carried_by": -1, "at_home": True}),
             memory)
    F.encode(dict(base, tick=1, own_flag={"x": -60.0, "y": 0.0, "carried_by": 0, "at_home": False}),
             memory)
    F.encode(dict(base, tick=2, own_flag={"x": -59.0, "y": 0.0, "carried_by": 0, "at_home": False}),
             memory)
    assert memory.flag_velocity == (0.0, 0.0)  # first displacement after pickup is catch-up
    F.encode(dict(base, tick=3, own_flag={"x": -58.0, "y": 0.0, "carried_by": 0, "at_home": False}),
             memory)
    assert abs(memory.flag_velocity[0] - 20.0) < 1e-9
    F.encode(dict(base, tick=4, own_flag={"x": -89.0, "y": 0.0, "carried_by": 1, "at_home": False}),
             memory)
    assert memory.flag_velocity == (0.0, 0.0)


def test_bumps_and_wall_deaths_are_remembered():
    match, memories, _ = _play(seed=3, ticks=1500)
    wall_deaths = sum(1 for m in match.death_log() if m["reason"] == "wall")
    assert wall_deaths > 0
    assert any(memory.bumps for memory in memories.values())
    for memory in memories.values():
        assert len(memory.bumps) <= F.BUMP_CAPACITY


def test_memory_resets_on_new_match():
    memory = F.PlayerMemory()
    match = Match(1, max_ticks=50)
    for _ in range(50):
        observations = match.observe()
        F.encode(observations[0], memory)
        match.act({})
    assert memory.last_tick == 49
    F.encode(Match(2, max_ticks=1).observe()[0], memory)
    assert memory.last_tick == 0 and memory.prev_tick == 0


def test_encode_is_fast():
    match, memories, _ = _play(seed=4, ticks=300)
    observations = match.observe() if not match.is_over() else None
    match = Match(4, max_ticks=10)
    observation = match.observe()[0]
    memory = F.PlayerMemory()
    for i in range(80):
        memory._add_bump(i * 0.7 - 30, (i % 9) * 3.0 - 10)
    start = time.perf_counter()
    for _ in range(2000):
        F.encode(observation, memory)
    per_call = (time.perf_counter() - start) / 2000
    assert per_call < 300e-6, per_call
