from ctf.policies import ROOT, load_decide
from ctf.rewards import DEFAULT_WEIGHTS, RewardTracker
from ctf.sim import Match

WIGGLE = ROOT / "ctf/tests/wiggle_policy.py"


def _run(seed, ticks, weights=None, prefill_log=False):
    match = Match(seed, max_ticks=ticks)
    if prefill_log:  # start with the engine's 200-entry log already full
        match._death_log.extend({"player": 1, "team": "old", "tick": -1000, "x": 0.0, "y": 0.0,
                                 "reason": "wall"} for _ in range(200))
    tracker = RewardTracker(match, "red", weights or {})
    decide = {s.id: load_decide(WIGGLE) for s in match.spheres}
    true_deaths = 0
    totals = [0.0, 0.0, 0.0]
    while not match.is_over():
        observations = match.observe()
        tracker.before()
        alive_before = {s.id: s.alive for s in match.spheres}
        match.act({sid: decide[sid](o) for sid, o in observations.items()})
        true_deaths += sum(1 for s in match.spheres
                           if s.team == "red" and alive_before[s.id] and not s.alive)
        for k, r in enumerate(tracker.after()):
            totals[k] += r
    return match, tracker, true_deaths, totals


def test_deaths_counted_past_engine_log_cap():
    match, tracker, true_deaths, _ = _run(3, 3000, prefill_log=True)
    assert len(match.death_log()) == 200  # the engine log stays truncated throughout
    counted = tracker.counts["wall_deaths"] + tracker.counts["player_deaths"]
    assert counted == true_deaths > 20


def test_shaping_is_exact_discounted_telescoping():
    """sum_t gamma^t F_t == gamma^T * 0 - Phi(s_0) for every player, through deaths and captures."""
    gamma = 0.99
    match = Match(5, max_ticks=1500)
    tracker = RewardTracker(match, "red", {"gamma": gamma, "score": 0.0, "win": 0.0, "pickup": 0.0,
                                           "wall_death": 0.0, "player_death": 0.0,
                                           "carrier_death": 0.0, "flag_return": 0.0})
    decide = {s.id: load_decide(WIGGLE) for s in match.spheres}
    discounted = [0.0, 0.0, 0.0]
    phi0 = None
    t = 0
    while not match.is_over():
        observations = match.observe()
        tracker.before()
        if phi0 is None:
            phi0 = list(tracker._phi)
        match.act({sid: decide[sid](o) for sid, o in observations.items()})
        for k, r in enumerate(tracker.after()):
            discounted[k] += gamma ** t * r
        t += 1
    for k in range(3):
        assert abs(discounted[k] + phi0[k]) < 1e-9


def test_pickup_does_not_jump_potential():
    match = Match(1, max_ticks=10)
    tracker = RewardTracker(match, "red", {"gamma": 1.0})
    sphere = tracker.spheres[0]
    flag = tracker.enemy_flag
    sphere.x, sphere.y = flag.x + 2.0, flag.y
    before = tracker._potentials()[0]
    flag.carrier, sphere.carrying_flag = sphere, flag
    after = tracker._potentials()[0]
    assert abs(after - before) < 0.5 / 180 * 4.5
