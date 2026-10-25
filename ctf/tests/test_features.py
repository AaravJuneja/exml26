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


def test_canonical_frame_is_side_symmetric():
    """Red at home facing +y and blue at home facing +y see mirrored-by-rotation worlds."""
    match = Match(7, max_ticks=1)
    observations = match.observe()
    red = next(s for s in match.spheres if s.team == "red" and s.player == 2)
    blue = next(s for s in match.spheres if s.team == "blue" and s.player == 2)
    fr = F.encode(observations[red.id], F.PlayerMemory())
    fb = F.encode(observations[blue.id], F.PlayerMemory())
    # Own home is always on the canonical -x side.
    assert fr[0] < 0 and fb[0] < 0
    # Own-home and enemy-home body-frame distances agree for both sides.
    home_block = F.SELF_DIM + F.GAME_DIM + 14
    assert fr[home_block] == fb[home_block] == 1.0
    assert abs(fr[home_block + 3] - fb[home_block + 3]) < 0.05


def test_bumps_and_wall_deaths_are_remembered():
    match, memories, _ = _play(seed=3, ticks=1500)
    wall_deaths = sum(1 for m in match.death_log() if m["reason"] == "wall")
    assert wall_deaths > 0
    assert any(memory.bumps for memory in memories.values())
    for memory in memories.values():
        for bx, by in memory.bumps:
            for home in (memory.own_home, memory.enemy_home):
                assert math.hypot(bx - home[0], by - home[1]) >= F.HOME_CLEAR


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
