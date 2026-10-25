"""Watch a training run and gauntlet new snapshots on held-out seeds.

    python tools/auto_eval.py --run m2b --every 50 --opponents idle baseline --workers 3

Appends one JSON line per evaluated snapshot to RUNS/RUN/eval.jsonl.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctf.evaluate import gauntlet  # noqa: E402
from ctf.train import DEFAULT_RUNS  # noqa: E402


def compact(summary):
    a = summary.get("a", {})
    return {
        "win_rate": summary["win_rate"], "win_ci": summary["win_rate_ci"],
        "draw_rate": summary["draw_rate"], "goal_diff": summary["goal_diff_mean"],
        "goal_diff_ci": summary["goal_diff_ci"], "goals_for": summary["goals_for_mean"],
        "goals_against": summary["goals_against_mean"],
        "wall_deaths_pp": a.get("wall_deaths_per_player_per_match"),
        "conversion": a.get("conversion"), "n": summary["n_matches"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--runs", default=str(DEFAULT_RUNS))
    p.add_argument("--every", type=int, default=50)
    p.add_argument("--opponents", nargs="+", default=["idle", "baseline"])
    p.add_argument("--seeds", default="1000-1023")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--interval", type=float, default=60.0)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.runs) / args.run
    out = run_dir / "eval.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["update"] for line in out.read_text().splitlines() if line}
    while True:
        snaps = []
        for path in sorted((run_dir / "policies").glob("u*.npz")):
            update = int(re.findall(r"\d+", path.stem)[0])
            if update % args.every == 0 and update not in done:
                snaps.append((update, path))
        done.update(update for update, _ in snaps[:-1])  # skip stale ones; evaluate the newest
        for update, path in snaps[-1:]:
            started = time.time()
            result = gauntlet(str(path), opponents=args.opponents, seeds=args.seeds,
                              workers=args.workers, n_boot=2000, show=False)
            record = {"update": update, "policy": str(path), "time": time.time(),
                      "seconds": round(time.time() - started, 1),
                      "vs": {s["meta"]["label_b"]: compact(s) for s in result["results"]}}
            with open(out, "a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps({"update": update, **{k: round(v["win_rate"], 3)
                                                   for k, v in record["vs"].items()}}), flush=True)
            done.add(update)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
