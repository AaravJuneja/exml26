"""Release pipeline: one command from a trained policy `.npz` to a verified submission.

    python tools/release.py POLICY.npz --name exml_rl [--seeds 5000-5031] [--workers N]
                            [--compare OTHER.npz] [--stress 6] [--skip-latency]

Steps (all results go to teams/NAME/release_report.json and a section of docs/RELEASE.md):

1. export   greedy `model.py`, a copy of the policy (`policy.npz`) and a README with provenance
2. audit    static checks of model.py: stdlib + numpy imports only, no undocumented fields
            (`teammates`, `sonar_recieve`) in code or strings, top-level `decide(observation)`,
            Python 3.9 syntax, no dynamic imports / exec / file or process access, size;
            plus, when a Python 3.9 interpreter is available, a load-and-decide smoke test
3. parity   model.py@proc (one worker process per player, as deployed) gives match records
            identical to POLICY.npz@greedy on 2 seeds x 2 sides against one scripted opponent.
            The reference decides its three players as one float32 batch, the file one row
            at a time; BLAS rounds these ~1e-6 apart, which can flip an exact near-tie. A
            differing game is traced to its first divergent decision and still passes if
            that is such a near-tie (gap < 1e-5, model.py equal to the single-row reference)
            and the deployed record equals model.py played in-process (--strict-parity: no)
4. latency  tools/latency.py on a full match with the real engine, normally and with --stress
            busy-loop processes; every verdict must pass (`all_pass`)
5. strength held-out gauntlet of model.py (documented observations) against ten scripted
            opponents on both sides of seeds that are never used for training (>= 100000)
            or evaluation (1000-2099); --compare adds paired differences for a second policy

The release PASSes only if audit, parity and both latency runs pass and the gauntlet has no
model errors. Shortened runs (--skip-latency, --latency-ticks or --max-ticks below a full
match) can at best be INCOMPLETE. Exit code: 0 PASS, 1 FAIL, 2 INCOMPLETE.
"""

import argparse
import ast
import datetime
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import time
import subprocess
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FULL_TICKS = 6000
DEFAULT_SEEDS = "5000-5031"
DEFAULT_OPPONENTS = ("baseline", "example", "example2", "relay", "balanced", "guardian",
                     "rusher", "camper", "chaos", "steady")
DEFAULT_PARITY_OPPONENT = "baseline"
DEFAULT_STRESS = 6
DEFAULT_MAX_KIB = 4096
# Seeds reserved elsewhere: training draws seeds >= 100000, evaluation uses 1000-2099.
RESERVED_SEEDS = (("training", 100000, None), ("evaluation", 1000, 2099))
FORBIDDEN_WORDS = ("teammates", "sonar_recieve")
DENIED_MODULES = frozenset((
    "asyncio", "builtins", "ctypes", "ftplib", "http", "importlib", "multiprocessing", "pty",
    "shutil", "signal", "smtplib", "socket", "ssl", "subprocess", "telnetlib", "threading",
    "urllib", "webbrowser"))
DENIED_CALLS = frozenset(("__import__", "breakpoint", "compile", "eval", "exec", "input", "open"))
DENIED_OS_ATTRS = frozenset((
    "system", "popen", "fork", "forkpty", "kill", "killpg", "remove", "unlink", "rmdir",
    "removedirs", "rename", "replace", "makedirs", "mkdir", "chdir", "execv", "execve", "execl",
    "execle", "execlp", "execvp", "execvpe", "spawnl", "spawnv", "spawnve", "posix_spawn"))
PY39_DEFAULT = ".venv39/bin/python"
DOCS_TITLE = "# Releases"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def _log(message):
    print(message, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ export
def provenance(policy_path):
    """Run / update / samples of a policy file, from its meta (falling back to the path)."""
    from ctf.policy_io import load_policy

    policy_path = Path(policy_path).resolve()
    _, meta = load_policy(policy_path)
    run = meta.get("run")
    if run is None and policy_path.parent.name == "policies":
        run = policy_path.parent.parent.name
    update = meta.get("update")
    if update is None:
        found = re.fullmatch(r"u(\d+)", policy_path.stem)
        update = int(found.group(1)) if found else None
    return {"source": str(policy_path), "run": run, "update": update,
            "samples": meta.get("samples"), "arch": meta.get("arch", "mlp"),
            "sha256": sha256(policy_path), "meta": meta}


def export_release(policy_path, out_dir, decode="greedy"):
    """Write model.py (`decode`), policy.npz and README.md into out_dir; return info.

    With decode="sample" a greedy twin `.parity_greedy_model.py` is also written: the
    parity check runs on it (same network, features and controller; sampled play cannot
    match record-for-record), while the gauntlet and latency use the deployed file."""
    from ctf.export import export

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_path = Path(policy_path).resolve()
    info = provenance(policy_path)
    copy = out_dir / "policy.npz"
    if copy.resolve() != policy_path:
        shutil.copyfile(policy_path, copy)
    model = export(copy, out_dir / "model.py", decode=decode)
    twin = model
    if decode != "greedy":
        twin = export(copy, out_dir / ".parity_greedy_model.py", decode="greedy")
    info.update({
        "policy_copy": str(copy), "model": str(model), "model_sha256": sha256(model),
        "parity_model": str(twin),
        "model_kib": round(model.stat().st_size / 1024, 1), "decode": decode,
        "date": _now(),
    })
    if sha256(copy) != info["sha256"]:
        raise RuntimeError("policy copy does not match the source")
    write_readme(out_dir / "README.md", info, out_dir.name)
    return info


def write_readme(path, info, name):
    lines = [
        f"# {name}",
        "",
        "Generated by `tools/release.py`; do not edit by hand. The submission is the single",
        f"file `model.py` (stdlib + numpy, {info.get('decode', 'greedy')} decoding). `policy.npz` is the source policy,",
        "`release_report.json` the verification results.",
        "",
        "| field | value |",
        "|---|---|",
        f"| run | {info['run']} |",
        f"| update | {info['update']} |",
        f"| samples | {info['samples']} |",
        f"| architecture | {info['arch']} |",
        f"| source | `{info['source']}` |",
        f"| policy.npz sha256 | `{info['sha256']}` |",
        f"| model.py sha256 | `{info['model_sha256']}` |",
        f"| model.py size | {info['model_kib']} KiB |",
        f"| decode | {info['decode']} |",
        f"| date (UTC) | {info['date']} |",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------- audit
def _stdlib_modules():
    names = getattr(sys, "stdlib_module_names", None)
    if names:
        return frozenset(names)
    return frozenset((  # Python < 3.10: the modules a policy file could plausibly need
        "abc", "array", "base64", "bisect", "collections", "copy", "dataclasses", "enum",
        "functools", "hashlib", "heapq", "itertools", "json", "math", "operator", "os",
        "random", "re", "statistics", "string", "struct", "sys", "time", "types", "typing",
        "warnings", "zlib"))


def _forbidden_mentions(source):
    """(line, word, where) for every forbidden word in code or string literals.

    Comments are ignored (the inlined feature encoder mentions teammates in prose);
    names, attributes and string literals (dictionary keys) are not."""
    found, comments = [], []
    pattern = re.compile(r"\b(" + "|".join(FORBIDDEN_WORDS) + r")\b")
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        for match in pattern.finditer(token.string):
            if token.type == tokenize.COMMENT:
                comments.append({"line": token.start[0], "word": match.group(1)})
            else:
                found.append({"line": token.start[0], "word": match.group(1),
                              "token": tokenize.tok_name[token.type]})
    return found, comments


def audit_source(source, size_bytes=None, max_kib=DEFAULT_MAX_KIB):
    """Static checks of a model.py source text; returns a dict with `pass` and `problems`."""
    problems = []
    checks = {}
    size_bytes = len(source.encode("utf-8")) if size_bytes is None else size_bytes
    checks["size_ok"] = 0 < size_bytes <= max_kib * 1024
    if not checks["size_ok"]:
        problems.append(f"size {size_bytes / 1024:.1f} KiB outside (0, {max_kib}] KiB")

    try:
        tree = ast.parse(source)
        checks["parses"] = True
    except SyntaxError as exc:
        checks["parses"] = False
        problems.append(f"syntax error: {exc}")
        return {"pass": False, "checks": checks, "problems": problems,
                "size_kib": round(size_bytes / 1024, 1)}
    try:
        ast.parse(source, feature_version=(3, 9))
        checks["py39_syntax"] = True
    except SyntaxError as exc:
        checks["py39_syntax"] = False
        problems.append(f"not Python 3.9 syntax: {exc}")

    stdlib = _stdlib_modules()
    imports, bad_imports = set(), []
    bad_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [("." * node.level) + (node.module or "")]
        else:
            names = []
        for name in names:
            top = name.split(".")[0]
            imports.add(name)
            if not top or (top != "numpy" and top not in stdlib) or top in DENIED_MODULES:
                bad_imports.append(f"line {node.lineno}: import {name}")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in DENIED_CALLS:
                bad_calls.append(f"line {node.lineno}: {func.id}()")
        if (isinstance(node, ast.Attribute) and node.attr in DENIED_OS_ATTRS
                and isinstance(node.value, ast.Name) and node.value.id == "os"):
            bad_calls.append(f"line {node.lineno}: os.{node.attr}")
    checks["imports_ok"] = not bad_imports
    problems += [f"disallowed {item}" for item in bad_imports]
    checks["no_dangerous_calls"] = not bad_calls
    problems += [f"disallowed {item}" for item in bad_calls]

    mentions, comment_mentions = _forbidden_mentions(source)
    checks["no_undocumented_fields"] = not mentions
    problems += [f"line {m['line']}: mentions {m['word']!r} ({m['token']})" for m in mentions]

    decides = [node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "decide"]
    ok = False
    if decides:
        args = decides[-1].args
        positional = args.posonlyargs + args.args
        required = len(positional) - len(args.defaults)
        ok = (len(positional) >= 1 or args.vararg is not None) and required <= 1 \
            and not any(d is None for d in args.kw_defaults)
    checks["defines_decide"] = ok
    if not ok:
        problems.append("no top-level decide(observation) taking one argument")

    return {"pass": all(checks.values()), "checks": checks, "problems": problems,
            "imports": sorted(imports), "comment_mentions": comment_mentions,
            "size_kib": round(size_bytes / 1024, 1)}


def audit_file(path, max_kib=DEFAULT_MAX_KIB):
    path = Path(path)
    data = path.read_bytes()
    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return {"pass": False, "checks": {"utf8": False}, "problems": [f"not UTF-8: {exc}"],
                "size_kib": round(len(data) / 1024, 1)}
    return audit_source(source, len(data), max_kib)


_SMOKE = r"""
import importlib.util, math, pickle, sys
spec = importlib.util.spec_from_file_location("team_model", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
observations = pickle.loads(sys.stdin.buffer.read())
out = []
for observation in observations:
    action = module.decide(observation)
    out.append([float(action[0]), float(action[1]), bool(action[2])])
sys.stdout.write(repr(out))
"""


def _smoke_observations(seed, ticks=40):
    """Documented observations of one sphere over a short idle match."""
    from ctf.sim import DOCUMENTED_KEYS, Match

    match = Match(seed, max_ticks=ticks)
    sid = match.spheres[0].id
    out = []
    while not match.is_over():
        observation = match.observe()[sid]
        out.append({key: observation[key] for key in DOCUMENTED_KEYS})
        match.act({})
    return out


def smoke_test(model_path, python, seed):
    """Load model.py in `python` (e.g. 3.9) and call decide on real observations.

    Returns the result dict; outputs are compared with the current interpreter
    (informational: different numpy builds may break exact logit ties)."""
    observations = _smoke_observations(seed)
    payload = pickle.dumps(observations, protocol=4)
    result = {"python": str(python)}
    try:
        version = subprocess.run([python, "-c", "import sys, numpy; print(sys.version.split()[0],"
                                  " numpy.__version__)"], capture_output=True, text=True,
                                 timeout=60, check=True).stdout.split()
        result["version"], result["numpy"] = version[0], version[1]
        proc = subprocess.run([python, "-c", _SMOKE, str(model_path)], input=payload,
                              capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        result.update({"pass": False, "error": repr(exc)})
        return result
    if proc.returncode != 0 or proc.stderr.strip():
        result.update({"pass": False, "error": proc.stderr.decode(errors="replace")[-800:]})
        return result
    actions = ast.literal_eval(proc.stdout.decode())
    here = subprocess.run([sys.executable, "-c", _SMOKE, str(model_path)], input=payload,
                          capture_output=True, timeout=300)
    valid = all(len(a) == 3 and all(-1.0 <= v <= 1.0 for v in a[:2]) for a in actions)
    result.update({"pass": valid and len(actions) == len(observations), "calls": len(actions),
                   "identical_to_current_python":
                       here.returncode == 0 and ast.literal_eval(here.stdout.decode()) == actions})
    return result


# ------------------------------------------------------------------ parity
_PARITY_IGNORED = ("spec_a", "digest_a", "wall_time", "timing")


NEAR_TIE_TOL = 1e-5  # as in ctf/tests/test_export.py


def _record_diff(x, y):
    vx = {k: v for k, v in x.items() if k not in _PARITY_IGNORED}
    vy = {k: v for k, v in y.items() if k not in _PARITY_IGNORED}
    return sorted(k for k in set(vx) | set(vy) if vx.get(k) != vy.get(k))


def first_divergence(model_path, policy_path, opponent, seed, side, max_ticks, tol=NEAR_TIE_TOL):
    """Replay a (seed, side) game driven by the reference policy and find the first
    decision where model.py (in-process module instances) picks a different action.

    The reference evaluates its three players as one batch; the exported file does one
    row at a time, and float32 BLAS may round the two differently (~1e-6). The
    divergence is `explained` when model.py's logits equal the reference's single-row
    logits within `tol` and every head it disagrees on is a near-tie (top-two gap
    below `tol`) in the batched reference."""
    import numpy as np

    from ctf import action_space
    from ctf import evaluate as E
    from ctf.sim import Match

    mine_spec, ref_spec = E.parse_spec(str(model_path)), E.parse_spec(f"{policy_path}@greedy")
    opp_spec = E.parse_spec(opponent)
    full = any(s.kind == "scripted" and not s.documented_only for s in (mine_spec, opp_spec))
    match = Match(seed, team_names=E.SIDES, spheres_per_team=E.PLAYERS_PER_TEAM,
                  max_ticks=max_ticks, teammates_field=full)
    other = E.SIDES[1 - E.SIDES.index(side)]
    team_of = {s.id: s.team for s in match.spheres}
    ids = [s.id for s in match.spheres if s.team == side]
    mine = E._make_team(mine_spec, ids, seed, 0)
    ref = E._make_team(ref_spec, ids, seed, 0)
    opp = E._make_team(opp_spec, [s.id for s in match.spheres if s.team == other], seed, 1)

    actor = ref.actor
    method = "step" if actor.recurrent else "logits"
    original = getattr(actor, method)
    seen = {}

    def capture(*inputs):
        out = original(*inputs)
        seen["inputs"] = [np.array(a) for a in inputs]
        seen["logits"] = np.array(out[0] if actor.recurrent else out)
        return out

    setattr(actor, method, capture)
    try:
        while not match.is_over():
            observations = match.observe()
            own = {sid: o for sid, o in observations.items() if team_of[sid] == side}
            a, b = mine.act(own), ref.act(own)
            bad = [sid for sid in ids if sid in a and tuple(a[sid][:3]) != tuple(b[sid][:3])]
            if bad:
                sid = bad[0]
                row = [k for k in ref.ids if k in own].index(sid)
                batched = seen["logits"][row]
                single = original(*[x[row:row + 1] for x in seen["inputs"]])
                single = np.asarray(single[0] if actor.recurrent else single)[0]
                # Each player has its own module instance, so its logits buffer still
                # holds this decision.
                module = np.asarray(mine.decide[sid].__globals__.get("_MX_LOGITS", np.nan))
                heads, start, near = [], 0, True
                space = action_space.get(actor.space)
                for name, size in zip(space.head_names, space.head_sizes):
                    part, m_part = batched[start:start + size], module[start:start + size]
                    choice = int(m_part.argmax()) if m_part.shape == part.shape else -1
                    gap = float(part.max() - part[choice]) if choice >= 0 else float("inf")
                    if choice != int(part.argmax()):
                        heads.append({"head": name, "reference": int(part.argmax()),
                                      "model": choice, "logit_gap": gap})
                        near = near and gap < tol
                    start += size
                matches_single = module.shape == single.shape and \
                    float(np.abs(module - single).max()) <= tol
                return {"tick": match.tick, "sphere": sid, "heads": heads,
                        "model_vs_single_row_max_abs": float(np.abs(module - single).max())
                        if module.shape == single.shape else None,
                        "batched_vs_single_row_max_abs": float(np.abs(batched - single).max()),
                        "explained": bool(near and matches_single and heads)}
            actions = dict(b)
            actions.update(opp.act({sid: o for sid, o in observations.items()
                                    if team_of[sid] == other}))
            match.act(actions)
    finally:
        delattr(actor, method)  # the actor is cached per process: drop the instance hook
        for team in (mine, ref, opp):
            team.close()
    return None


def parity(model_path, policy_path, opponent, seeds, max_ticks, workers, progress=None,
           strict=False):
    """model.py@proc vs POLICY.npz@greedy: identical match records on every (seed, side).

    A game whose records differ is traced to its first divergent decision. Unless
    `strict`, it still passes when that divergence is a float32 near-tie (see
    `first_divergence`) and the deployed record equals the record of the same file
    played in-process, i.e. the worker processes add nothing of their own."""
    from ctf.evaluate import SIDES, run_jobs

    deployed, reference = f"{model_path}@proc", f"{policy_path}@greedy"
    jobs = [(spec, opponent, seed, side, max_ticks)
            for spec in (deployed, reference) for seed in seeds for side in SIDES]
    by_key = {}
    for record in run_jobs(jobs, workers, progress):
        kind = "deployed" if record["spec_a"].endswith("@doc,proc") else "reference"
        by_key[(kind, record["seed"], record["a_side"])] = record
    games = []
    for seed in seeds:
        for side in SIDES:
            x, y = by_key[("deployed", seed, side)], by_key[("reference", seed, side)]
            differ = _record_diff(x, y)
            game = {"seed": seed, "side": side, "identical": not differ,
                    "status": "identical" if not differ else "MISMATCH",
                    "differs_in": differ, "score": f"{x['score_a']}-{x['score_b']}",
                    "reference_score": f"{y['score_a']}-{y['score_b']}",
                    "deployed_errors": x["a"]["errors"],
                    "deployed_decide_p99_ms": x["timing"]["decide_a"]["p99_ms"]}
            if differ:
                _log(f"      seed {seed} {side}: records differ in {differ}; tracing")
                game["divergence"] = first_divergence(model_path, policy_path, opponent, seed,
                                                      side, max_ticks)
                inproc = next(run_jobs([(str(model_path), opponent, seed, side, max_ticks)]))
                game["deployed_equals_inprocess"] = not _record_diff(x, inproc)
                if (game["divergence"] or {}).get("explained") and \
                        game["deployed_equals_inprocess"]:
                    game["status"] = "near_tie"
            games.append(game)
    ok = {"identical"} if strict else {"identical", "near_tie"}
    return {"pass": all(g["status"] in ok for g in games), "strict": strict,
            "near_tie_tol": NEAR_TIE_TOL, "opponent": opponent, "seeds": list(seeds),
            "max_ticks": max_ticks, "deployed": deployed, "reference": reference,
            "n_identical": sum(g["identical"] for g in games), "games": games}


# ----------------------------------------------------------------- latency
def parse_latency_output(text):
    """The JSON summary printed by tools/latency.py (last top-level object in stdout)."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line == "{"]
    if not starts:
        raise ValueError("no JSON summary in latency output")
    return json.loads("\n".join(lines[starts[-1]:]))


def run_latency(model_path, ticks, stress, seed, workdir=None, timeout=3600):
    cmd = [sys.executable, str(ROOT / "tools" / "latency.py"), "--model", str(model_path),
           "--ticks", str(ticks), "--seed", str(seed), "--stress", str(stress),
           "--inprocess-ticks", str(min(1000, ticks))]
    if workdir is not None:
        cmd += ["--workdir", str(workdir)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    try:
        summary = parse_latency_output(proc.stdout)
    except ValueError as exc:
        return {"ticks": ticks, "stress_procs": stress, "pass": False,
                "error": f"{exc}; exit {proc.returncode}: {proc.stderr[-800:]}"}
    verdict = summary.get("verdict", {})
    callback = summary.get("engine", {}).get("callback", {})
    return {"ticks": ticks, "stress_procs": stress, "pass": bool(verdict.get("all_pass")),
            "verdict": verdict, "callback": callback,
            "inprocess_decide": summary.get("inprocess", {}).get("decide", {}),
            "timeouts_or_respawns": summary.get("engine", {}).get("timeouts_or_respawns"),
            "scores": summary.get("engine", {}).get("scores"),
            "cpu_count": summary.get("cpu_count"), "python": summary.get("python"),
            "summary": summary}


TRAINING_PATTERN = "ctf.train|auto_eval|tools/ab_compare|tools/gallery"


def pause_training(enabled):
    """SIGSTOP local training/evaluation jobs so latency is measured on a quiet machine."""
    if not enabled:
        return False
    subprocess.run(["pkill", "-STOP", "-f", TRAINING_PATTERN], check=False)
    for pid in _children_of_pattern():
        subprocess.run(["kill", "-STOP", pid], check=False)
    time.sleep(3)


def resume_training(paused):
    if not paused:
        return
    for pid in _children_of_pattern():
        subprocess.run(["kill", "-CONT", pid], check=False)
    subprocess.run(["pkill", "-CONT", "-f", TRAINING_PATTERN], check=False)


def _children_of_pattern():
    """Worker processes (multiprocessing children) of the paused jobs."""
    out = subprocess.run(["pgrep", "-f", TRAINING_PATTERN], capture_output=True, text=True)
    pids = []
    for parent in out.stdout.split():
        kids = subprocess.run(["pgrep", "-P", parent], capture_output=True, text=True)
        pids += kids.stdout.split()
    return pids


def machine_load():
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


# ---------------------------------------------------------------- strength
def check_seeds(seeds):
    """Names of the reserved ranges (training / evaluation) that `seeds` overlap."""
    hits = []
    for name, low, high in RESERVED_SEEDS:
        if any(s >= low and (high is None or s <= high) for s in seeds):
            hits.append(name)
    return hits


def _compact(summary):
    a, b = summary.get("a", {}), summary.get("b", {})
    return {
        "opponent": summary["meta"]["label_b"],
        "n": summary["n_matches"], "wins": summary["wins"], "draws": summary["draws"],
        "losses": summary["losses"], "win_rate": summary["win_rate"],
        "win_rate_ci": summary["win_rate_ci"], "goal_diff": summary["goal_diff_mean"],
        "goal_diff_ci": summary["goal_diff_ci"], "goals_for": summary["goals_for_mean"],
        "goals_against": summary["goals_against_mean"],
        "per_side": {side: {"wins": p["wins"], "draws": p["draws"], "losses": p["losses"]}
                     for side, p in summary["per_side"].items()},
        "wall_deaths_pp": a.get("wall_deaths_per_player_per_match"),
        "conversion": a.get("conversion"), "errors": a.get("errors", 0),
        "opponent_errors": b.get("errors", 0), "decide_p99_ms_max": a.get("decide_p99_ms_max"),
        "records": summary["meta"].get("out"),
    }


def strength(spec, opponents, seeds, max_ticks, workers, out_dir, progress=None):
    from ctf.evaluate import gauntlet

    result = gauntlet(spec, opponents, seeds, workers, out_dir=out_dir, max_ticks=max_ticks,
                      progress=progress, show=False)
    rows = [_compact(s) for s in result["results"]]
    n = sum(r["n"] for r in rows)
    return {"spec": result["spec"], "seeds": _seed_text(seeds), "max_ticks": max_ticks,
            "opponents": list(opponents), "rows": rows,
            "mean_win_rate": round(sum(r["win_rate"] for r in rows) / len(rows), 4)
            if rows else None,
            "total": {"n": n, "wins": sum(r["wins"] for r in rows),
                      "draws": sum(r["draws"] for r in rows),
                      "losses": sum(r["losses"] for r in rows)},
            "errors": sum(r["errors"] for r in rows),
            "throughput": result["throughput"]}


def compare_policy(baseline_rows, other_path, opponents, seeds, max_ticks, workers, out_dir,
                   progress=None):
    """Gauntlet of OTHER.npz@greedy on the same games, paired with the release's records."""
    from ctf.evaluate import compare, load_records

    other = strength(f"{Path(other_path).resolve()}@greedy", opponents, seeds, max_ticks,
                     workers, out_dir, progress)
    rows = []
    for mine, theirs in zip(baseline_rows, other["rows"]):
        delta = compare(load_records(mine["records"]), load_records(theirs["records"]),
                        strict=True)
        rows.append({"opponent": mine["opponent"], "release_win_rate": mine["win_rate"],
                     "other_win_rate": theirs["win_rate"], **delta})
    return {"other": str(Path(other_path).resolve()), "other_sha256": sha256(other_path),
            "other_strength": other, "rows": rows}


def _seed_text(seeds):
    from ctf.evaluate import _seed_text as text

    return text(list(seeds))


# ------------------------------------------------------------------ report
def _pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def _ci(ci, scale=100.0, digits=0):
    if not ci or ci[0] is None:
        return "-"
    return f"[{ci[0] * scale:.{digits}f}, {ci[1] * scale:.{digits}f}]"


def strength_table(rows):
    lines = ["| opponent | W-D-L | win % | 95% CI | GD | GD 95% CI | GF / GA | wall deaths/pl | errors |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['opponent']} | {r['wins']}-{r['draws']}-{r['losses']} | {_pct(r['win_rate'])}"
            f" | {_ci(r['win_rate_ci'])} | {r['goal_diff']:+.2f} | {_ci(r['goal_diff_ci'], 1, 2)}"
            f" | {r['goals_for']:.2f} / {r['goals_against']:.2f} | {r['wall_deaths_pp']} |"
            f" {r['errors']} |")
    return lines


def compare_table(rows):
    lines = ["| opponent | release win % | other win % | Δ match pts | 95% CI | Δ GD | 95% CI |"
             " P(Δ GD > 0) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['opponent']} | {_pct(r['release_win_rate'])} | {_pct(r['other_win_rate'])}"
            f" | {r['match_points_delta']:+.3f} | {_ci(r['match_points_delta_ci'], 1, 3)}"
            f" | {r['goal_diff_delta']:+.2f} | {_ci(r['goal_diff_delta_ci'], 1, 2)}"
            f" | {r['p_goal_diff_delta_positive']} |")
    return lines


def _mark(ok):
    return {True: "PASS", False: "FAIL", None: "skipped"}[ok]


def markdown_section(report):
    name, info = report["name"], report["policy"]
    lines = [f"<!-- release:{name} begin -->", f"## {name}: {report['status']}", "",
             f"Released {report['date']} from `{info['source']}` "
             f"(run `{info['run']}`, update {info['update']}, samples {info['samples']}).",
             f"`model.py` sha256 `{info['model_sha256']}` ({info['model_kib']} KiB), "
             f"`policy.npz` sha256 `{info['sha256']}`.", "",
             "| check | result |", "|---|---|"]
    for key, value in report["checks"].items():
        lines.append(f"| {key} | {_mark(value)} |")
    if report.get("incomplete"):
        lines += ["", "Incomplete: " + "; ".join(report["incomplete"]) + "."]

    audit = report["audit"]
    lines += ["", "### Audit", "",
              f"{_mark(audit['pass'])}: {audit['size_kib']} KiB, imports "
              f"{', '.join('`%s`' % i for i in audit.get('imports', []))}."]
    lines += [f"- problem: {p}" for p in audit.get("problems", [])]
    if audit.get("comment_mentions"):
        lines.append("- comments mentioning undocumented names (allowed): lines "
                     + ", ".join(str(m["line"]) for m in audit["comment_mentions"]))
    smoke = audit.get("py39")
    if smoke:
        if smoke.get("pass") is None:
            lines.append(f"- Python 3.9 smoke test skipped: {smoke.get('reason')}")
        else:
            lines.append(f"- Python {smoke.get('version', '?')} / numpy {smoke.get('numpy', '?')}"
                         f" load + decide: {_mark(smoke['pass'])} "
                         f"({smoke.get('calls', 0)} calls, identical to current Python: "
                         f"{smoke.get('identical_to_current_python')})"
                         + (f" error: {smoke['error'][:200]}" if smoke.get("error") else ""))

    par = report.get("parity")
    lines += ["", "### Parity", ""]
    if par:
        lines.append(f"{_mark(par['pass'])}: `model.py@proc` vs `policy.npz@greedy` against "
                     f"{par['opponent']}, {par['max_ticks']} ticks; {par['n_identical']}/"
                     f"{len(par['games'])} records identical"
                     + (" (strict)." if par.get("strict") else
                        f", other games must be float32 near-ties (gap < {par['near_tie_tol']})"
                        " with the deployed record equal to the in-process one."))
        lines += ["", "| seed | side | score (deployed / reference) | result | deployed p99 ms |",
                  "|---|---|---|---|---|"]
        for g in par["games"]:
            result = g["status"]
            div = g.get("divergence")
            if div:
                heads = ", ".join(f"{h['head']} {h['reference']}->{h['model']} gap "
                                  f"{h['logit_gap']:.1e}" for h in div["heads"])
                result += (f": first divergence tick {div['tick']} sphere {div['sphere']} "
                           f"({heads}); deployed = in-process: {g['deployed_equals_inprocess']}")
            elif g.get("differs_in"):
                result += ": differs in " + ", ".join(g["differs_in"]) + " (not reproduced)"
            lines.append(f"| {g['seed']} | {g['side']} | {g['score']} / {g['reference_score']} "
                         f"| {result} | {g['deployed_decide_p99_ms']} |")
    else:
        lines.append("skipped")

    lat = report.get("latency") or {}
    lines += ["", "### Latency (real engine, six worker processes)", ""]
    if lat.get("runs"):
        lines.append(f"Load average before: {lat.get('load_before')}.")
        lines += ["", "| stress procs | ticks | callback p50 / p99 / max ms | over 20 ms |"
                  " timeouts | in-process p99 ms | verdict |", "|---|---|---|---|---|---|---|"]
        for run in lat["runs"]:
            cb = run.get("callback", {})
            failed = [k for k, v in run.get("verdict", {}).items() if not v and k != "all_pass"]
            verdict = "PASS" if run["pass"] else "FAIL " + (", ".join(failed) or run.get("error", ""))
            lines.append(f"| {run['stress_procs']} | {run['ticks']} | {cb.get('p50_ms')} / "
                         f"{cb.get('p99_ms')} / {cb.get('max_ms')} | {cb.get('over_20ms')} | "
                         f"{run.get('timeouts_or_respawns')} | "
                         f"{run.get('inprocess_decide', {}).get('p99_ms')} | {verdict} |")
    else:
        lines.append("skipped")

    st = report.get("strength")
    lines += ["", "### Strength (held-out gauntlet, documented observations)", ""]
    if st:
        total = st["total"]
        lines.append(f"`model.py` on seeds {st['seeds']}, both sides, {st['max_ticks']} ticks: "
                     f"{total['wins']}-{total['draws']}-{total['losses']} overall, mean win "
                     f"rate {_pct(st['mean_win_rate'])}%, model errors {st['errors']}.")
        lines += [""] + strength_table(st["rows"])
    else:
        lines.append("skipped")
    cmp_ = report.get("compare")
    if cmp_:
        lines += ["", f"### Paired comparison with `{Path(cmp_['other']).name}`", "",
                  f"Release minus `{cmp_['other']}` (greedy) on the same games; "
                  "CIs resample seeds.", ""] + compare_table(cmp_["rows"])
    lines += [f"<!-- release:{name} end -->", ""]
    return "\n".join(lines)


_DOCS_INTRO = f"""{DOCS_TITLE}

Generated by `tools/release.py` (see its docstring); one section per release name,
replaced when that name is released again.

    python tools/release.py runs/RUN/policies/uNNNNN.npz --name exml_rl --workers 3
    python tools/release.py NEW.npz --name exml_rl_next --compare OLD.npz

A release PASSes only when the static audit, the `model.py@proc` vs `.npz@greedy` parity
check, both full-match latency runs (normal and `--stress`) and the gauntlet (no model
errors) all pass. Gauntlet seeds default to 5000-5031, which training (>= 100000) and
evaluation (1000-2099) never use.

Parity: the `.npz` reference decides its three players as one float32 batch, `model.py`
one row at a time, and BLAS rounds the two up to ~1e-6 apart. That can flip an argmax
whose top two logits are closer than that, after which the games drift apart. A differing
game is therefore traced to its first divergent decision; it passes (`near_tie`) only if
the flipped heads are within 1e-5 in the reference, `model.py`'s logits equal the
single-row reference, and the deployed (`@proc`) record equals `model.py` in-process.
`--strict-parity` requires bit-identical records instead.
"""


def update_docs(path, report):
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else _DOCS_INTRO
    section = markdown_section(report)
    begin, end = f"<!-- release:{report['name']} begin -->", f"<!-- release:{report['name']} end -->"
    if begin in text and end in text:
        head, rest = text.split(begin, 1)
        tail = rest.split(end, 1)[1].lstrip("\n")
        text = head + section + tail
    else:
        text = text.rstrip("\n") + "\n\n" + section
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# -------------------------------------------------------------------- main
def release(args):
    from ctf.evaluate import parse_seeds

    progress = None if args.quiet else _log
    seeds = parse_seeds(args.seeds)
    parity_seeds = parse_seeds(args.parity_seeds) if args.parity_seeds else seeds[:2]
    reserved = check_seeds(seeds + parity_seeds)
    if reserved and not args.allow_reserved_seeds:
        raise SystemExit(f"seeds overlap the {' and '.join(reserved)} seed ranges; "
                         "use held-out seeds (e.g. 5000-5031) or --allow-reserved-seeds")
    out_dir = Path(args.teams_dir) / args.name
    report = {"name": args.name, "date": _now(), "argv": sys.argv[1:] if args.cli else None,
              "host": os.uname().nodename if hasattr(os, "uname") else None,
              "python": sys.version.split()[0]}
    incomplete = []

    _log(f"[1/5] export {args.policy} -> {out_dir}/model.py")
    info = export_release(args.policy, out_dir, args.decode)
    report["policy"] = {k: v for k, v in info.items() if k != "meta"}
    report["policy"]["meta"] = info["meta"]
    model = Path(info["model"])

    _log("[2/5] audit")
    audit = audit_file(model, args.max_kib)
    python39 = args.py39 if args.py39 is not None else os.environ.get("EXML_PY39", PY39_DEFAULT)
    if python39 and Path(python39).exists():
        audit["py39"] = smoke_test(model, python39, parity_seeds[0])
        audit["pass"] = audit["pass"] and audit["py39"]["pass"]
    else:
        # Optional extra: the contract names no Python version, so a skip is not incomplete.
        audit["py39"] = {"pass": None, "reason": f"no interpreter at {python39!r}"}
    report["audit"] = audit
    _log(f"      {_mark(audit['pass'])} {audit['problems'] or ''}")
    go_on = audit["pass"] or args.keep_going

    if go_on:
        _log(f"[3/5] parity against {args.parity_opponent} on seeds {parity_seeds}")
        report["parity"] = parity(Path(info["parity_model"]), info["policy_copy"],
                                  args.parity_opponent, parity_seeds,
                                  args.max_ticks, args.workers, progress, args.strict_parity)
        _log(f"      {_mark(report['parity']['pass'])} "
             f"{[g['status'] for g in report['parity']['games']]}")
        go_on = report["parity"]["pass"] or args.keep_going

    lat = {"runs": [], "load_before": machine_load()}
    if args.skip_latency:
        incomplete.append("latency skipped")
    elif go_on:
        paused = args.pause_training
        lat["training_paused"] = paused
        try:
            pause_training(paused)
            for stress in (0, args.stress):
                _log(f"[4/5] latency: {args.latency_ticks} ticks, stress {stress}")
                run = run_latency(model, args.latency_ticks, stress, args.latency_seed,
                                  args.latency_workdir)
                _log(f"      {_mark(run['pass'])} callback {run.get('callback', {})}")
                lat["runs"].append(run)
        finally:
            resume_training(paused)
        if args.latency_ticks < FULL_TICKS:
            incomplete.append(f"latency ran {args.latency_ticks} of {FULL_TICKS} ticks")
    report["latency"] = lat

    if go_on:
        _log(f"[5/5] gauntlet: {len(args.opponents)} opponents x {len(seeds)} seeds x 2 sides")
        report["strength"] = strength(str(model), args.opponents, seeds, args.max_ticks,
                                      args.workers, out_dir / "gauntlet", progress)
        if args.compare:
            _log(f"[5/5] compare: {args.compare}")
            report["compare"] = compare_policy(
                report["strength"]["rows"], args.compare, args.opponents, seeds, args.max_ticks,
                args.workers, out_dir / "compare" / Path(args.compare).stem, progress)
    if args.max_ticks < FULL_TICKS:
        incomplete.append(f"matches ran {args.max_ticks} of {FULL_TICKS} ticks")

    latency_ok = None
    if lat["runs"]:
        latency_ok = all(run["pass"] for run in lat["runs"])
    report["checks"] = {
        "audit": audit["pass"],
        "parity": report["parity"]["pass"] if "parity" in report else None,
        "latency": latency_ok,
        "gauntlet_no_errors": report["strength"]["errors"] == 0 if "strength" in report else None,
    }
    report["incomplete"] = incomplete
    if any(v is False for v in report["checks"].values()):
        report["status"] = "FAIL"
    elif any(v is None for v in report["checks"].values()) or incomplete:
        report["status"] = "INCOMPLETE"
    else:
        report["status"] = "PASS"

    report_path = out_dir / "release_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.docs:
        update_docs(args.docs, report)
    print_summary(report, report_path)
    return report


def print_summary(report, report_path):
    print(f"release {report['name']}: {report['status']}")
    for key, value in report["checks"].items():
        print(f"  {key:<20} {_mark(value)}")
    for note in report["incomplete"]:
        print(f"  incomplete: {note}")
    if report.get("strength"):
        print("\n".join(strength_table(report["strength"]["rows"])))
    if report.get("compare"):
        print("\n".join(compare_table(report["compare"]["rows"])))
    print(f"  report: {report_path}")


def build_parser():
    p = argparse.ArgumentParser(prog="tools/release.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("policy", help="trained policy .npz")
    p.add_argument("--name", required=True, help="team directory name under --teams-dir")
    p.add_argument("--seeds", nargs="+", default=[DEFAULT_SEEDS], help="gauntlet seeds")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    p.add_argument("--compare", metavar="OTHER.npz", help="paired gauntlet vs a second policy")
    p.add_argument("--parity-seeds", nargs="+", help="default: the first two gauntlet seeds")
    p.add_argument("--parity-opponent", default=DEFAULT_PARITY_OPPONENT)
    p.add_argument("--strict-parity", action="store_true",
                   help="require bit-identical records (no float32 near-tie allowance)")
    p.add_argument("--max-ticks", type=int, default=FULL_TICKS, help="ticks per parity/gauntlet match")
    p.add_argument("--latency-ticks", type=int, default=FULL_TICKS)
    p.add_argument("--latency-seed", type=int, default=1)
    p.add_argument("--latency-workdir", help="scratch dir for tools/latency.py (default runs/)")
    p.add_argument("--stress", type=int, default=DEFAULT_STRESS,
                   help="busy-loop processes for the second latency run")
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument("--decode", choices=("greedy", "sample"), default="greedy",
                   help="decoding of the deployed model.py (parity runs on a greedy twin)")
    p.add_argument("--pause-training", action="store_true",
                   help="SIGSTOP training/eval jobs on this machine during the latency step")
    p.add_argument("--max-kib", type=float, default=DEFAULT_MAX_KIB)
    p.add_argument("--py39", help=f"Python 3.9 for the load smoke test (default $EXML_PY39 or "
                                  f"{PY39_DEFAULT}; '' to skip)")
    p.add_argument("--teams-dir", default=str(ROOT / "teams"))
    p.add_argument("--docs", default=str(ROOT / "docs" / "RELEASE.md"),
                   help="markdown file to update ('' to skip)")
    p.add_argument("--allow-reserved-seeds", action="store_true")
    p.add_argument("--keep-going", action="store_true",
                   help="run every step even after the audit or parity fails")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.cli = argv is None
    report = release(args)
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
