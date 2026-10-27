"""Paired comparison of several policies against a reference, with cached records.

    python tools/ab_multi.py --ref REF_SPEC --cand SPEC [SPEC ...] [--opponents relay balanced]
                             [--seeds 3000-3063] [--workers 3] --cache DIR [--out report.json]

Same statistics as tools/ab_compare.py (seed-clustered paired bootstrap of B - A over
both sides of every seed), but every (spec, opponent) record set is stored in
DIR/<slug>.jsonl and reused, so a reference is played once for many candidates.
Specs are evaluate specs (e.g. path.npz@sample,seed=1).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctf.evaluate import compare, evaluate, file_digest, load_records, parse_seeds, parse_spec, summarize  # noqa: E402,E501


def records_for(spec, opponent, seeds, workers, cache):
    spec = parse_spec(spec)
    key = f"{spec}|{file_digest(spec)}|{opponent}|{seeds[0]}-{seeds[-1]}|{len(seeds)}"
    path = Path(cache) / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".jsonl")
    if path.exists():
        return load_records(path)
    _, records = evaluate(spec, opponent, seeds, workers=workers, out=path, n_boot=200)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--cand", nargs="+", required=True)
    p.add_argument("--opponents", nargs="+", default=["relay", "balanced"])
    p.add_argument("--seeds", default="3000-3063")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", default="")
    args = p.parse_args()
    seeds = parse_seeds(args.seeds)
    report = {"ref": args.ref, "seeds": args.seeds, "cands": {}}
    for opponent in args.opponents:
        ref = records_for(args.ref, opponent, seeds, args.workers, args.cache)
        s = summarize(ref, n_boot=1000)
        print(f"{opponent:9s} REF win {s['win_rate']:.2f} gd {s['goal_diff_mean']:+.2f}  {args.ref}",
              flush=True)
        for cand in args.cand:
            recs = records_for(cand, opponent, seeds, args.workers, args.cache)
            sc = summarize(recs, n_boot=1000)
            d = compare(recs, ref, n_boot=4000)
            entry = {"win": sc["win_rate"], "gd": sc["goal_diff_mean"],
                     "dgd": d["goal_diff_delta"], "dgd_ci": d["goal_diff_delta_ci"],
                     "dwin": d["match_points_delta"], "dwin_ci": d["match_points_delta_ci"],
                     "ref_win": s["win_rate"], "ref_gd": s["goal_diff_mean"]}
            report["cands"].setdefault(cand, {})[opponent] = entry
            print(f"{opponent:9s}   win {entry['win']:.2f} gd {entry['gd']:+.2f} dGD {entry['dgd']:+.2f} "
                  f"[{entry['dgd_ci'][0]:+.2f}, {entry['dgd_ci'][1]:+.2f}] dWin {entry['dwin']:+.3f} "
                  f"[{entry['dwin_ci'][0]:+.3f}, {entry['dwin_ci'][1]:+.3f}]  {cand}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
