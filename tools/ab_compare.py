"""Paired A/B comparison of two policies on identical games.

    python tools/ab_compare.py A.npz B.npz [--opponents relay balanced ...]
                               [--seeds 3000-3063] [--workers 4] [--out DIR]

Both policies play the same seeds, both sides, against each opponent (64 seeds x 2
sides = 128 games per opponent by default). Reports each policy's win rate and goal
difference, the paired B - A deltas with seed-clustered bootstrap 95% CIs, wall deaths
per player and the median capture -> next-pickup delay. Evaluation seeds 3000-3999 are
reserved for these comparisons (training uses >= 100000; auto-eval 1000-1099;
release 5000-5099).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctf.evaluate import compare, evaluate, parse_seeds, summarize  # noqa: E402

DEFAULT_OPPONENTS = ["relay", "balanced", "guardian", "rusher", "camper", "baseline"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--opponents", nargs="+", default=DEFAULT_OPPONENTS)
    p.add_argument("--seeds", default="3000-3063")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="")
    args = p.parse_args()
    seeds = parse_seeds(args.seeds)
    def spec(path):  # .npz policies decode greedily; exported model.py files are played as-is
        return path + "@greedy" if path.endswith(".npz") else path

    a_spec, b_spec = spec(args.a), spec(args.b)

    rows, report = [], {"a": args.a, "b": args.b, "seeds": args.seeds, "opponents": {}}
    for opponent in args.opponents:
        _, rec_a = evaluate(a_spec, opponent, seeds, workers=args.workers, n_boot=2000)
        _, rec_b = evaluate(b_spec, opponent, seeds, workers=args.workers, n_boot=2000)
        sa, sb = summarize(rec_a, n_boot=2000), summarize(rec_b, n_boot=2000)
        delta = compare(rec_b, rec_a, n_boot=4000)
        entry = {
            "a": {"win": sa["win_rate"], "gd": sa["goal_diff_mean"],
                  "wall_pp": sa["a"]["wall_deaths_per_player_per_match"],
                  "repickup": sa["a"].get("median_repickup_delay")},
            "b": {"win": sb["win_rate"], "gd": sb["goal_diff_mean"],
                  "wall_pp": sb["a"]["wall_deaths_per_player_per_match"],
                  "repickup": sb["a"].get("median_repickup_delay")},
            "delta_gd": delta["goal_diff_delta"], "delta_gd_ci": delta["goal_diff_delta_ci"],
            "delta_win": delta["match_points_delta"], "delta_win_ci": delta["match_points_delta_ci"],
            "n": delta["n_pairs"],
        }
        report["opponents"][opponent] = entry
        rows.append((opponent, entry))
        print(f"{opponent:9s} A win {entry['a']['win']:.2f} gd {entry['a']['gd']:+.2f} | "
              f"B win {entry['b']['win']:.2f} gd {entry['b']['gd']:+.2f} | "
              f"dGD {entry['delta_gd']:+.2f} [{entry['delta_gd_ci'][0]:+.2f}, "
              f"{entry['delta_gd_ci'][1]:+.2f}] dWin {entry['delta_win']:+.3f} "
              f"[{entry['delta_win_ci'][0]:+.3f}, {entry['delta_win_ci'][1]:+.3f}] | "
              f"wall {entry['a']['wall_pp']:.2f}/{entry['b']['wall_pp']:.2f} | "
              f"repickup {entry['a']['repickup']}/{entry['b']['repickup']}", flush=True)
    mean_dgd = sum(e["delta_gd"] for _, e in rows) / len(rows)
    mean_dwin = sum(e["delta_win"] for _, e in rows) / len(rows)
    report["mean_delta_gd"], report["mean_delta_win"] = mean_dgd, mean_dwin
    print(f"mean over opponents: dGD {mean_dgd:+.2f}  dWin {mean_dwin:+.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
