"""Tests for the evaluation harness (short matches; full-length runs live on the training machine)."""

import json
import multiprocessing
import os

import numpy as np
import pytest

from ctf import evaluate as E
from ctf.features import THROTTLE_VALUES, TURN_VALUES
from ctf.policies import ROOT
from ctf.sim import DOCUMENTED_KEYS, Match

WIGGLE = str(ROOT / "ctf" / "tests" / "wiggle_policy.py")
TICKS = 400


@pytest.fixture(scope="module")
def policy_path(tmp_path_factory):
    from ctf.policy_io import random_policy

    path = tmp_path_factory.mktemp("policy") / "random.npz"
    random_policy(path, hidden=(32, 32), seed=3)
    return str(path)


# ------------------------------------------------------------------ parsing
def test_parse_seeds():
    assert E.parse_seeds("1000-1003") == [1000, 1001, 1002, 1003]
    assert E.parse_seeds("1,5-7", "9") == [1, 5, 6, 7, 9]
    assert E.parse_seeds(["3", 1, range(4, 6)], "3") == [3, 1, 4, 5]
    with pytest.raises(ValueError):
        E.parse_seeds("9-3")
    assert E._seed_text([1, 2, 3, 7, 9, 10]) == "1-3,7,9-10"


def test_parse_spec(policy_path):
    assert E.parse_spec("idle").kind == "idle"
    base = E.parse_spec("baseline")
    assert base.kind == "scripted" and not base.documented_only and base.label == "baseline"
    assert E.parse_spec("baseline@doc").documented_only
    assert E.parse_spec(str(ROOT / "teams/baseline/model.py")) == base
    wiggle = E.parse_spec(WIGGLE)
    assert wiggle.documented_only and wiggle.label == "tests/wiggle_policy"
    assert E.parse_spec(WIGGLE + "@full").documented_only is False
    proc = E.parse_spec(WIGGLE + "@proc")
    assert proc.process and proc.documented_only and proc.label == "tests/wiggle_policy@proc"
    assert E.parse_spec(str(proc)) == proc and not wiggle.process
    assert E.parse_spec("baseline@proc").label == "baseline@proc"
    with pytest.raises(ValueError):
        E.parse_spec(policy_path + "@proc")
    npz = E.parse_spec(policy_path)
    assert npz.kind == "npz" and npz.greedy and npz.documented_only
    sampled = E.parse_spec(policy_path + "@sample,seed=7")
    assert not sampled.greedy and sampled.sample_seed == 7
    assert E.parse_spec(str(sampled)) == sampled
    with pytest.raises(ValueError):
        E.parse_spec(policy_path + "@full")
    with pytest.raises(ValueError):
        E.parse_spec("nonsense")
    with pytest.raises(FileNotFoundError):
        E.parse_spec("missing/model.py")


# --------------------------------------------------------------- run_match
def test_match_record_fields():
    record = E.run_match("baseline", "example", 11, "red", max_ticks=TICKS)
    assert record["ticks"] == TICKS and record["a_side"] == "red" and record["b_side"] == "blue"
    assert record["score_a"] == record["score"]["red"] == record["a"]["captures"]
    assert record["score_b"] == record["score"]["blue"] == record["b"]["captures"]
    assert record["result"] in ("win", "draw", "loss")
    assert record["goal_diff"] == record["score_a"] - record["score_b"]
    assert record["wall_time"] > 0
    json.dumps(record)


def test_determinism(policy_path):
    for a, b in (("baseline", "example2"), (policy_path + "@sample,seed=2", WIGGLE)):
        first = E.run_match(a, b, 5, "blue", max_ticks=TICKS)
        second = E.run_match(a, b, 5, "blue", max_ticks=TICKS)
        assert E.deterministic_view(first) == E.deterministic_view(second)
    sampled = [E.deterministic_view(E.run_match(policy_path + f"@sample,seed={s}", WIGGLE, 5,
                                                "red", max_ticks=TICKS)) for s in (1, 2)]
    greedy = [E.deterministic_view(E.run_match(policy_path, WIGGLE, 5, "red", max_ticks=TICKS))
              for _ in range(2)]
    assert greedy[0] == greedy[1]
    assert sampled[0]["a"] != sampled[1]["a"] or sampled[0]["score"] != sampled[1]["score"]


def test_side_swap_really_swaps():
    # A on red vs B on blue is the same game whichever spec is called "A".
    forward = E.run_match("baseline", WIGGLE, 7, "red", max_ticks=TICKS)
    mirror = E.run_match(WIGGLE, "baseline", 7, "blue", max_ticks=TICKS)
    assert forward["score"] == mirror["score"]
    assert forward["a"] == {**mirror["b"]} and forward["b"] == mirror["a"]
    assert forward["result"] == {"win": "loss", "loss": "win", "draw": "draw"}[mirror["result"]]

    # Swapping sides changes which engine team A controls.
    swapped = E.run_match("baseline", WIGGLE, 7, "blue", max_ticks=TICKS)
    assert swapped["a"]["side"] == "blue" and swapped["score_a"] == swapped["score"]["blue"]
    assert E.deterministic_view(swapped) != E.deterministic_view(forward)

    # The idle team really is the one standing still.
    idle_red = E.run_match("idle", "baseline", 3, "red", max_ticks=1500)
    idle_blue = E.run_match("idle", "baseline", 3, "blue", max_ticks=1500)
    for record in (idle_red, idle_blue):
        assert record["a"]["pickups"] == 0 and record["a"]["wall_deaths"] == 0
        assert record["b"]["pickups"] > 0


def test_policy_sees_documented_keys_and_sets_memory(policy_path, monkeypatch):
    seen = []
    real_encode = E.encode

    def checking_encode(observation, memory):
        seen.append(set(observation))
        return real_encode(observation, memory)

    monkeypatch.setattr(E, "encode", checking_encode)
    spec = E.parse_spec(policy_path + "@sample,seed=1")
    match = Match(4, max_ticks=50, teammates_field=True)
    ids = [s.id for s in match.spheres if s.team == "red"]
    team = E._make_team(spec, ids, 4, 0)
    while not match.is_over():
        observations = match.observe()
        mine = {sid: o for sid, o in observations.items() if sid in ids}
        assert "teammates" in next(iter(mine.values()))
        actions = team.act(mine)
        assert set(actions) == set(mine)
        for sid, (throttle, turn, jump) in actions.items():
            memory = team.memories[sid]
            assert THROTTLE_VALUES[memory.prev_throttle] == throttle
            assert TURN_VALUES[memory.prev_turn] == turn
            assert memory.prev_jump == jump
        match.act(actions)
    assert seen and all(keys == set(DOCUMENTED_KEYS) for keys in seen)


def test_process_isolated_players(tmp_path):
    # @proc (one rpc worker process per player) plays the same game as in-process modules,
    # also from inside the evaluator's process pool, and cleans its workers up.
    jobs = [(spec, "baseline", 9, "blue", 80) for spec in (WIGGLE, WIGGLE + "@proc")]
    plain, proc = sorted((E.deterministic_view(r) for r in E.run_jobs(jobs, workers=2)),
                         key=lambda r: r["spec_a"].endswith("proc"))
    assert proc["spec_a"].endswith("proc")
    plain.pop("spec_a"), proc.pop("spec_a")
    assert plain == proc
    probe = tmp_path / "probe_model.py"
    probe.write_text(
        "import os\n"
        "def decide(observation):\n"
        "    with open(os.path.join(%r, str(os.getpid())), 'a') as f:\n"
        "        f.write('%%s %%d\\n' %% (sorted(observation) == sorted(%r),"
        " observation['self']['id']))\n"
        "    return 1.0, 0.0, False\n" % (str(tmp_path / "seen"), list(E.DOCUMENTED_KEYS)))
    (tmp_path / "seen").mkdir()
    record = E.run_match(str(probe) + "@proc", "baseline", 2, "red", max_ticks=20)
    assert record["a"]["errors"] == 0 and record["timing"]["decide_a"]["calls"] == 60
    seen = {int(p.name): set(p.read_text().splitlines()) for p in (tmp_path / "seen").iterdir()}
    assert len(seen) == 3 and os.getpid() not in seen  # one process per player
    assert sorted(len(lines) for lines in seen.values()) == [1, 1, 1]  # one sphere each
    assert all(line.startswith("True ") for lines in seen.values() for line in lines)
    assert not multiprocessing.active_children()


def test_scripted_doc_mode_strips_teammates(tmp_path):
    probe = tmp_path / "probe_model.py"
    probe.write_text(
        "def decide(observation):\n"
        "    if 'teammates' in observation or 'sonar_recieve' in observation:\n"
        "        raise RuntimeError('undocumented key')\n"
        "    return 0.0, 0.0, False\n")
    record = E.run_match(str(probe), "baseline", 2, "red", max_ticks=30)
    assert record["a"]["errors"] == 0
    record = E.run_match(str(probe) + "@full", "idle", 2, "red", max_ticks=30)
    assert record["a"]["errors"] > 0


# ------------------------------------------------------------- aggregation
def _record(seed, side, score_a, score_b, **team):
    stats = {"captures": score_a, "pickups": 2 * score_a, "wall_deaths": 3, "player_deaths": 0,
             "unknown_deaths": 0, "carrier_deaths": 1, "first_capture_tick": 100 if score_a else None,
             "errors": 0}
    stats.update(team)
    other = {**stats, "captures": score_b, "pickups": score_b, "wall_deaths": 0,
             "first_capture_tick": None}
    result = "win" if score_a > score_b else "loss" if score_a < score_b else "draw"
    return {"seed": seed, "a_side": side, "score_a": score_a, "score_b": score_b,
            "result": result, "players": 3, "ticks": 6000, "a": stats, "b": other}


def test_summary_maths():
    records = [_record(1, "red", 3, 1), _record(1, "blue", 0, 0),
               _record(2, "red", 1, 2), _record(2, "blue", 2, 0)]
    s = E.summarize(records, n_boot=500)
    assert (s["wins"], s["draws"], s["losses"]) == (2, 1, 1)
    assert s["n_matches"] == 4 and s["n_seeds"] == 2
    assert s["win_rate"] == 0.5 and s["match_points"] == 2 and s["match_points_per_match"] == 0.5
    assert s["draw_adjusted_score"] == 0.625
    assert s["goal_diff_mean"] == pytest.approx((2 + 0 - 1 + 2) / 4)
    assert s["goals_for_mean"] == 1.5 and s["goals_against_mean"] == 0.75
    assert s["per_side"]["red"] == {"n": 2, "wins": 1, "draws": 0, "losses": 1,
                                    "win_rate": 0.5, "goal_diff_mean": 0.5}
    assert s["per_side"]["blue"]["goal_diff_mean"] == 1.0
    assert s["a"]["captures_per_match"] == 1.5
    assert s["a"]["pickups_per_match"] == 3.0 and s["a"]["conversion"] == 0.5
    assert s["a"]["wall_deaths_per_player_per_match"] == 1.0
    assert s["b"]["wall_deaths_per_player_per_match"] == 0.0
    assert s["a"]["matches_with_capture"] == 3
    low, high = s["goal_diff_ci"]
    assert low <= s["goal_diff_mean"] <= high
    # Two seeds: the bootstrap can only produce the two per-seed means and their average.
    assert {low, high} <= {0.5, 0.75, 1.0}


def test_bootstrap_keeps_seed_pairs_together():
    # Red always wins 5-0 and blue always loses 0-5: each seed nets to exactly zero.
    records = []
    for seed in range(40):
        records += [_record(seed, "red", 5, 0), _record(seed, "blue", 0, 5)]
    s = E.summarize(records, n_boot=2000)
    assert s["goal_diff_mean"] == 0.0 and s["goal_diff_ci"] == [0.0, 0.0]
    assert s["win_rate_ci"] == [0.5, 0.5]
    # An unpaired, game-level bootstrap would be wide here; ours must not be.
    shuffled = list(reversed(records))
    assert E.summarize(shuffled, n_boot=2000)["goal_diff_ci"] == [0.0, 0.0]


def test_bootstrap_ratio_weights_by_games():
    # Seed 1 has two games (+2 each), seed 2 one game (-4): mean is 0 per game, not -1.
    records = [_record(1, "red", 2, 0), _record(1, "blue", 2, 0), _record(2, "red", 0, 4)]
    mean, ci, p_positive = E.paired_bootstrap(records, lambda r: r["score_a"] - r["score_b"],
                                              n_boot=4000)
    assert mean == 0.0
    assert ci[0] == -4.0 and ci[1] == 2.0
    assert 0.2 < p_positive < 0.3  # only the {1,1} resample (p = 1/4) is positive


def _vs(records, spec_b="baseline@full", digest_b=None, max_ticks=6000):
    return [{**r, "spec_b": spec_b, "digest_b": digest_b, "max_ticks": max_ticks}
            for r in records]


def test_compare_is_paired():
    x = [_record(s, side, 2, 1) for s in range(10) for side in ("red", "blue")]
    y = [_record(s, side, 1, 1) for s in range(10) for side in ("red", "blue")]
    c = E.compare(_vs(x), _vs(y), n_boot=500)
    assert c["n_pairs"] == 20 and c["n_seeds"] == 10
    assert c["n_dropped_x"] == 0 and c["n_dropped_y"] == 0
    assert c["goal_diff_delta"] == 1.0 and c["goal_diff_delta_ci"] == [1.0, 1.0]
    assert c["match_points_delta"] == 1.0 and c["p_match_points_delta_positive"] == 1.0


def test_compare_validates_record_sets():
    x = _vs([_record(s, side, 2, 1) for s in range(4) for side in ("red", "blue")])
    y = _vs([_record(s, side, 1, 1) for s in range(4) for side in ("red", "blue")])

    def fails(xs, ys, match):
        with pytest.raises(ValueError, match=match):
            E.compare(xs, ys, n_boot=50)

    fails(x, _vs(y, spec_b="example@full"), "different opponents")
    fails(x, _vs(y, spec_b="/a/other/pol.npz@greedy"), "different opponents")
    fails(x, y[:4] + _vs(y[4:], spec_b="idle"), "Y mixes opponents")
    fails(x, _vs(y, max_ticks=3000), r"max_ticks")
    fails(x[:1] + x, y, "X has duplicate")
    fails(x, [], "empty")
    fails(x[:1], y, "no seed has both sides")
    # Same opponent file recorded on two machines: compared by directory/name + digest.
    xs, ys = _vs(x, "/srv/runs/opp/pol.npz@greedy", "aa"), _vs(y, "/home/runs/opp/pol.npz@greedy")
    assert E.compare(xs, ys, n_boot=50)["n_pairs"] == 8  # Y has no digest: not checked
    fails(xs, _vs(y, "/home/runs/opp/pol.npz@greedy", "bb"), "opponent file differs")
    fails(xs, _vs(y, "/home/runs/opp/pol.npz@sample,seed=0", "aa"), "different opponents")

    # Incomplete seeds are dropped (seed 0 lacks blue in X, seed 3 lacks red in Y).
    partial_x = [r for r in x if (r["seed"], r["a_side"]) != (0, "blue")]
    partial_y = [r for r in y if (r["seed"], r["a_side"]) != (3, "red")]
    with pytest.warns(UserWarning, match="dropped 3 X and 3 Y records from 2 incomplete"):
        c = E.compare(partial_x, partial_y, n_boot=50)
    assert (c["n_pairs"], c["n_seeds"], c["n_dropped_x"], c["n_dropped_y"]) == (4, 2, 3, 3)
    with pytest.raises(ValueError, match="incomplete seeds"):
        E.compare(partial_x, y, n_boot=50, strict=True)


def test_policy_cache_reloads_overwritten_file(tmp_path):
    from ctf.policy_io import load_policy, random_policy

    path = tmp_path / "ckpt.npz"
    random_policy(path, hidden=(16,), seed=1)
    spec = E.parse_spec(str(path))
    first, digest = E._loaded(spec)
    assert E._loaded(spec)[0] is first and digest == E.file_digest(spec)
    random_policy(path, hidden=(16,), seed=2)  # overwritten in place, same size
    second, new_digest = E._loaded(spec)
    assert new_digest != digest and new_digest == E.file_digest(spec)
    expected, _ = load_policy(path)
    assert np.array_equal(second.weights[0][0], expected.weights[0][0])
    assert not np.array_equal(second.weights[0][0], first.weights[0][0])
    assert sum(key[1] == spec.path for key in E._CACHE) == 1  # stale entry evicted
    record = E.run_match(str(path), "idle", 1, max_ticks=5)
    assert record["digest_a"] == new_digest and record["digest_b"] is None


# -------------------------------------------------------------- end to end
def test_evaluate_end_to_end(tmp_path, policy_path):
    out = tmp_path / "eval" / "x.jsonl"
    summary, records = E.evaluate(policy_path, "baseline", "20-21", workers=2, out=out,
                                  max_ticks=150, n_boot=200)
    assert summary["n_matches"] == 4 and summary["n_seeds"] == 2
    assert [(r["seed"], r["a_side"]) for r in records] == [
        (20, "red"), (20, "blue"), (21, "red"), (21, "blue")]
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines == records
    assert not (tmp_path / "eval" / "x.jsonl.partial").exists()
    assert json.loads(out.with_suffix(".summary.json").read_text())["n_matches"] == 4
    assert summary["throughput"]["matches_per_min"] > 0
    serial, serial_records = E.evaluate(policy_path, "baseline", [20, 21], workers=1,
                                        max_ticks=150, n_boot=200)
    assert [E.deterministic_view(r) for r in serial_records] == [
        E.deterministic_view(r) for r in records]
    assert serial["goal_diff_ci"] == summary["goal_diff_ci"]


def test_gauntlet_and_cli(tmp_path, capsys):
    result = E.gauntlet("baseline", ["idle", WIGGLE], seeds="1", workers=1, max_ticks=60,
                        n_boot=100, out_dir=tmp_path)
    assert [r["meta"]["label_b"] for r in result["results"]] == ["idle", "tests/wiggle_policy"]
    assert all(r["n_matches"] == 2 for r in result["results"])
    assert (tmp_path / "baseline_vs_idle.jsonl").exists()
    assert "baseline" in capsys.readouterr().out

    out = tmp_path / "cli.jsonl"
    summary = E.main(["example", "idle", "--seeds", "3", "--workers", "1", "--max-ticks", "60",
                      "--boot", "100", "--out", str(out), "--quiet"])
    assert summary["n_matches"] == 2 and len(out.read_text().splitlines()) == 2
    again = E.main(["--summarize", str(out), "--boot", "100"])
    assert again["wins"] == summary["wins"]
    rr = E.main(["--round-robin", "idle", "example", "baseline", "--seeds", "3", "--workers",
                 "1", "--max-ticks", "40", "--boot", "50", "--quiet"])
    assert len(rr["results"]) == 3
    assert np.isfinite(rr["throughput"]["ticks_per_s"])
