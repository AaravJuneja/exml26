"""Build the model gallery data for docs/viewer.html and index.html.

    python tools/gallery.py                 # history + latest-model replays + milestones
    python tools/gallery.py --latest-only   # history + latest-model replays only
    python tools/gallery.py --no-replays    # history + release info only (fast)
    python tools/gallery.py --run m5a --snapshot 300 --opponents relay,guardian --seed 1001

Everything is written to docs/gallery/data/ (git-ignored) as plain <script> files, so the
pages work when opened straight from disk (file://, where fetch() of JSON is blocked):

    gallery.js            window.GALLERY = {runs, replays, latest, milestones, release, headline}
    replays/<id>.js       window.REPLAYS["<id>"] = <tools/replay.py JSON>

Model history comes from each run's eval.jsonl (held-out gauntlet every 50 updates) and
log.jsonl (training stats, binned down to ~CURVE_POINTS points), copied from runs/ into
runs_cache/. Replays are recorded with *each run's own frozen code copy* (runs/<run>/code),
because policies from older runs use older feature layouts. At most 3 replay processes run
at once; existing replays are reused unless --force is given.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CACHE = ROOT / "runs_cache"
OUT = ROOT / "docs" / "gallery" / "data"
STATUS = ROOT / "docs" / "tracker_status.json"
RELEASE = ROOT / "teams" / "exml_rl" / "release_report.json"

DEFAULT_RUNS = ["m2e", "m3b", "m3d", "m3e-c", "m4c", "m5a", "m5b"]
DEFAULT_LATEST = ["m5b", "m5a"]
DEFAULT_OPPONENTS = ["relay", "balanced", "baseline", "guardian"]
DEFAULT_MILESTONES = ["m3b", "m3d", "m4c", "m5a"]
OPPONENT_ORDER = ["relay", "balanced", "guardian", "baseline", "rusher", "camper", "chaos",
                  "steady", "example", "example2"]
CURVE_POINTS = 120
CURVE_KEYS = {"win": "win_rate", "gf": "goals_for", "ga": "goals_against",
              "wall": "wall_deaths_pp", "ent": "entropy", "pick": "pickups"}
MAX_JOBS = 3


def log(msg):
    print(msg, flush=True)


def read_jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def write_js(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def js_assign(target, obj):
    payload = json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")
    return f"{target} = {payload};\n"


# ------------------------------------------------------------------ run state
def inventory_of(runs):
    """{run: {"policies": [updates], "code": bool}} for runs that exist under runs/."""
    inv = {}
    for run in runs:
        run_dir = RUNS / run
        if not run_dir.is_dir():
            continue
        names = [p.name for p in (run_dir / "policies").glob("u*.npz")]
        updates = sorted(int(n[1:-4]) for n in names if n[1:-4].isdigit())
        inv[run] = {"policies": updates, "code": (run_dir / "code" / "tools" / "replay.py").exists()}
    return inv


def copy_logs(runs):
    for run in runs:
        dest = CACHE / run
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("log.jsonl", "eval.jsonl"):
            src = RUNS / run / name
            if src.exists():
                shutil.copy2(src, dest / name)


# ------------------------------------------------------------------ history
def nearest_samples(log_rows, update):
    best = None
    for row in log_rows:
        if row.get("update", -1) <= update:
            best = row
        else:
            break
    return best.get("samples_total") if best else None


def curve(log_rows, points=CURVE_POINTS):
    rows = [r for r in log_rows if "update" in r]
    if not rows:
        return None
    size = max(1, -(-len(rows) // points))
    out = {"u": [], "s": [], "t": [], **{k: [] for k in CURVE_KEYS}}
    for i in range(0, len(rows), size):
        chunk = rows[i:i + size]
        last = chunk[-1]
        out["u"].append(last["update"])
        out["s"].append(last.get("samples_total", 0))
        out["t"].append(round(last.get("time", 0)))
        for key, src in CURVE_KEYS.items():
            vals = [r[src] for r in chunk if isinstance(r.get(src), (int, float))]
            out[key].append(round(sum(vals) / len(vals), 4) if vals else None)
    return out


def eval_rows(run, log_rows):
    rows = []
    for r in read_jsonl(CACHE / run / "eval.jsonl"):
        vs = {}
        for opp, v in r.get("vs", {}).items():
            lo, hi = (v.get("win_ci") or [None, None])[:2]
            vs[opp] = {"w": v.get("win_rate"), "lo": lo, "hi": hi, "d": v.get("draw_rate"),
                       "gd": v.get("goal_diff"), "gf": v.get("goals_for"),
                       "ga": v.get("goals_against"), "wall": v.get("wall_deaths_pp"),
                       "n": v.get("n")}
        rows.append({"u": r["update"], "t": round(r.get("time", 0)),
                     "s": nearest_samples(log_rows, r["update"]), "vs": vs})
    rows.sort(key=lambda e: e["u"])
    return rows


def build_runs(runs, inventory):
    status = {}
    if STATUS.exists():
        status = {r["name"]: r for r in json.loads(STATUS.read_text()).get("runs", [])}
    out = []
    for run in runs:
        log_rows = read_jsonl(CACHE / run / "log.jsonl")
        evals = eval_rows(run, log_rows)
        if run not in inventory and not log_rows and not evals:
            continue
        inv = inventory.get(run, {"policies": [], "code": False})
        last = log_rows[-1] if log_rows else {}
        info = status.get(run, {})
        out.append({
            "name": run, "description": info.get("description", ""), "state": info.get("state", ""),
            "policies": inv["policies"], "has_code": inv["code"], "evals": evals,
            "curve": curve(log_rows),
            "last": {"update": last.get("update"), "samples": last.get("samples_total"),
                     "time": round(last["time"]) if "time" in last else None},
        })
    return out


# ------------------------------------------------------------------ replays
def replay_id(run, update, opponent, seed, side):
    return f"{run}-u{update:05d}-{opponent}-s{seed}-{side}"


def record_replay(task, ticks, every, force):
    """Record one replay with the run's code copy and wrap it as JS."""
    rid, run, update, opp, seed, side = (task[k] for k in ("id", "run", "update", "opponent",
                                                          "seed", "side"))
    js_path = OUT / "replays" / f"{rid}.js"
    raw_dir = CACHE / "gallery"
    raw = raw_dir / f"{rid}.json"
    if not force and js_path.exists() and raw.exists():
        meta = json.loads(raw.read_text())["meta"]
        if meta.get("every") == every and meta.get("max_ticks") == ticks:
            return summarize(task, raw, js_path, cached=True)
    policy = RUNS / run / "policies" / f"u{update:05d}.npz"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "tools/replay.py", str(policy), opp, "--seed", str(seed),
           "--side", side, "--ticks", str(ticks), "--every", str(every), "--out", str(raw)]
    started = time.time()
    result = subprocess.run(cmd, cwd=RUNS / run / "code", capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  FAILED {rid}: {(result.stderr or result.stdout).strip()[-400:]}")
        return None
    replay = json.loads(raw.read_text())
    a_side = replay["meta"]["a_side"]
    label = f"{run} u{update}"
    replay["meta"]["label_a"] = label
    replay["meta"]["labels"][a_side] = label
    for sphere in replay["spheres"]:
        if sphere["team"] == a_side:
            sphere["label"] = label
    text = json.dumps(replay, separators=(",", ":")).replace("</", "<\\/")
    write_js(js_path, f"window.REPLAYS = window.REPLAYS || {{}};\n"
                      f"window.REPLAYS[{json.dumps(rid)}] = {text};\n")
    log(f"  {rid}: {result.stdout.strip().split('  ->')[0]} ({time.time() - started:.0f} s)")
    return summarize(task, raw, js_path)


def summarize(task, raw, js_path, cached=False):
    meta = json.loads(raw.read_text())["meta"]
    replay = json.loads(raw.read_text())
    counts = {}
    for e in replay["events"]:
        key = e["type"] if e["type"] != "death" else f"death_{e['reason']}"
        counts[key] = counts.get(key, 0) + 1
    if cached:
        log(f"  {task['id']}: cached ({meta['score_a']}-{meta['score_b']} {meta['result']})")
    return {**task, "file": f"replays/{js_path.name}", "bytes": js_path.stat().st_size,
            "score_a": meta["score_a"], "score_b": meta["score_b"], "result": meta["result"],
            "ticks": meta["ticks"], "every": meta["every"], "events": counts}


def pick_snapshot(updates, wanted):
    if not updates:
        return None
    if wanted in (None, "latest"):
        return updates[-1]
    wanted = int(wanted)
    if wanted not in updates:
        raise SystemExit(f"snapshot u{wanted} not found (have {updates[0]}..{updates[-1]})")
    return wanted


# ------------------------------------------------------------------ release + headline
def release_info():
    if not RELEASE.exists():
        return None
    r = json.loads(RELEASE.read_text())
    pol = r.get("policy", {})
    strength = r.get("strength", {})
    rows = [{"opponent": x["opponent"], "n": x["n"], "wins": x["wins"], "draws": x["draws"],
             "losses": x["losses"], "w": x["win_rate"], "ci": x.get("win_rate_ci"),
             "gd": x["goal_diff"], "gf": x.get("goals_for"), "ga": x.get("goals_against"),
             "wall": x.get("wall_deaths_pp"), "p99": x.get("decide_p99_ms_max")}
            for x in strength.get("rows", [])]
    lat = r.get("latency", {})
    lat_runs = (lat.get("runs") or lat.get("final_measurement", {}).get("runs")
                or lat.get("final_remeasurement") or [])
    latency = []
    for run in lat_runs:
        cb, dec = run.get("callback", {}), run.get("inprocess_decide", {})
        latency.append({"stress": run.get("stress_procs"), "pass": run.get("pass"),
                        "cb_p50": cb.get("p50_ms"), "cb_p99": cb.get("p99_ms"),
                        "cb_max": cb.get("max_ms"), "over10": cb.get("over_10ms"),
                        "n": cb.get("n"), "dec_p99": dec.get("p99_ms"),
                        "timeouts": run.get("timeouts_or_respawns"),
                        "verdict": run.get("verdict", {})})
    return {"name": r.get("name"), "date": r.get("date"), "status": r.get("status"),
            "checks": r.get("checks", {}), "run": pol.get("run"), "update": pol.get("update"),
            "samples": pol.get("samples"), "arch": pol.get("arch"),
            "sha": (pol.get("sha256") or "")[:12], "model_kib": pol.get("model_kib"),
            "feature_version": pol.get("meta", {}).get("feature_version"),
            "averaged": pol.get("averaged_snapshots") or len(pol.get("meta", {}).get("averaged_from") or []) or None,
            "temperature": pol.get("meta", {}).get("temperature"),
            "seeds": strength.get("seeds"), "mean_w": strength.get("mean_win_rate"),
            "total": strength.get("total"), "rows": rows, "latency": latency}


def headline(runs, latest):
    best = {}
    for run in runs:
        for ev in run["evals"]:
            for opp, v in ev["vs"].items():
                if v["w"] is None:
                    continue
                cur = best.get(opp)
                if cur is None or (v["w"], v["gd"] or 0) > (cur["w"], cur["gd"] or 0):
                    best[opp] = {"w": v["w"], "gd": v["gd"], "run": run["name"], "u": ev["u"]}
    times = [r["last"]["time"] for r in runs if r["last"]["time"]]
    times += [e["t"] for r in runs for e in r["evals"] if e["t"]]
    latest_eval = None
    if latest:
        run = next((r for r in runs if r["name"] == latest["run"]), None)
        if run and run["evals"]:
            latest_eval = {"run": run["name"], **run["evals"][-1]}
    return {"best": best, "updated": max(times) if times else None, "latest_eval": latest_eval}


# ------------------------------------------------------------------ main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="runs for the model history")
    p.add_argument("--run", help="run for the latest-model replays (default: m5b, else m5a)")
    p.add_argument("--snapshot", default="latest", help="update number or 'latest'")
    p.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS))
    p.add_argument("--milestones", default=",".join(DEFAULT_MILESTONES),
                   help="runs (RUN or RUN:UPDATE) replayed against --milestone-opponent")
    p.add_argument("--milestone-opponent", default="relay")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--side", choices=("red", "blue"), default="red", help="side of the model")
    p.add_argument("--ticks", type=int, default=6000)
    p.add_argument("--every", type=int, default=2, help="keep every Nth tick (2 halves size)")
    p.add_argument("--jobs", type=int, default=MAX_JOBS, help=f"parallel replays (max {MAX_JOBS})")
    p.add_argument("--latest-only", action="store_true", help="skip milestone replays")
    p.add_argument("--no-replays", action="store_true", help="only refresh history/release")
    p.add_argument("--no-fetch", action="store_true", help="use runs_cache/ logs as they are")
    p.add_argument("--force", action="store_true", help="re-record existing replays")
    args = p.parse_args(argv)

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    milestone_specs = [m.strip() for m in args.milestones.split(",") if m.strip()]
    milestone_runs = [m.split(":")[0] for m in milestone_specs]
    candidates = [args.run] if args.run else DEFAULT_LATEST
    all_runs = list(dict.fromkeys(runs + candidates + milestone_runs))

    log(f"inventory of {RUNS.relative_to(ROOT)}/ ...")
    inventory = inventory_of(all_runs)
    for run in all_runs:
        inv = inventory.get(run)
        log(f"  {run}: " + (f"{len(inv['policies'])} snapshots"
                            f"{'' if inv['code'] else ', no code copy (no replays)'}"
                            if inv else "missing, skipped"))
    if not args.no_fetch:
        log("copying logs ...")
        copy_logs([r for r in runs if r in inventory])

    latest_run = next((r for r in candidates if r in inventory and inventory[r]["policies"]), None)
    latest = None
    tasks = []
    if latest_run:
        update = pick_snapshot(inventory[latest_run]["policies"], args.snapshot)
        latest = {"run": latest_run, "update": update, "replays": []}
        for opp in [o.strip() for o in args.opponents.split(",") if o.strip()]:
            rid = replay_id(latest_run, update, opp, args.seed, args.side)
            latest["replays"].append(rid)
            tasks.append({"id": rid, "run": latest_run, "update": update, "opponent": opp,
                          "seed": args.seed, "side": args.side, "group": "latest"})
    milestones = []
    if not args.latest_only:
        for spec in milestone_specs:
            run, _, snap = spec.partition(":")
            inv = inventory.get(run)
            if not inv or not inv["policies"] or not inv["code"]:
                log(f"  milestone {run}: skipped (no snapshots or code copy)")
                continue
            update = pick_snapshot(inv["policies"], snap or "latest")
            rid = replay_id(run, update, args.milestone_opponent, args.seed, args.side)
            milestones.append(rid)
            if all(t["id"] != rid for t in tasks):
                tasks.append({"id": rid, "run": run, "update": update,
                              "opponent": args.milestone_opponent, "seed": args.seed,
                              "side": args.side, "group": "milestone"})
        if latest:  # the latest model vs the same opponent closes the progression
            rid = replay_id(latest["run"], latest["update"], args.milestone_opponent,
                            args.seed, args.side)
            if any(t["id"] == rid for t in tasks) and rid not in milestones:
                milestones.append(rid)

    # keep earlier replays (other snapshots, milestones) that are still on disk
    replays = {}
    old = OUT / "gallery.js"
    if old.exists():
        try:
            text = old.read_text(encoding="utf-8")
            prev = json.loads(text[text.index("=") + 1:].strip().rstrip(";"))
            for rid, info in prev.get("replays", {}).items():
                if (OUT / info["file"]).exists():
                    replays[rid] = info
            if args.latest_only and not milestones:
                milestones = [m for m in prev.get("milestones", []) if m in replays]
        except (ValueError, KeyError) as exc:
            log(f"  could not read previous manifest: {exc}")

    if not args.no_replays and tasks:
        jobs = max(1, min(MAX_JOBS, args.jobs))
        log(f"recording {len(tasks)} replays ({jobs} at a time, every {args.every}) ...")
        with ThreadPoolExecutor(jobs) as pool:
            results = list(pool.map(lambda t: record_replay(t, args.ticks, args.every, args.force),
                                    tasks))
        for res in results:
            if res:
                replays[res["id"]] = res
    if latest:
        latest["replays"] = [r for r in latest["replays"] if r in replays]
    milestones = [m for m in milestones if m in replays]

    history = build_runs(runs, inventory)
    seen = {o for r in history for e in r["evals"] for o in e["vs"]}
    opponents = [o for o in OPPONENT_ORDER if o in seen] + sorted(seen - set(OPPONENT_ORDER))
    gallery = {
        "generated": time.strftime("%Y-%m-%d %H:%M"), "generated_ts": round(time.time()),
        "seed": args.seed, "opponents": opponents, "latest": latest, "milestones": milestones,
        "runs": history, "replays": replays, "release": release_info(),
        "headline": headline(history, latest),
    }
    write_js(OUT / "gallery.js", js_assign("window.GALLERY", gallery))
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    log(f"wrote {OUT / 'gallery.js'} · {len(replays)} replays · {total / 1e6:.1f} MB in "
        f"{OUT.relative_to(ROOT)}")
    log(f"open {ROOT / 'index.html'} or {ROOT / 'docs' / 'viewer.html'}")


if __name__ == "__main__":
    sys.exit(main())
