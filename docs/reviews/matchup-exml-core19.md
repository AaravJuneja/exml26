# Matchup: exml_rl vs evvls-dumb/exml-core19

Date: 2026-10-27, run on the local laptop. Scratch files, raw records and summaries are in `/tmp/core19/`: `rl.jsonl.partial` + `rl_b.jsonl` (main release), `pure.jsonl`, and `*_summary.json`.

## Their submission

- Repo: https://github.com/evvls-dumb/exml-core19 (commit `88343db`). It has no README. `manifest.json` names the root `model.py` as the submission (`entrypoint: decide(observation)`, selection `reference100-pace`, dated 2026-10-25), and its sha256 `b14f5d1b…8e93` matches the manifest. The `training/` folder contains the training source and data only.
- **Approach:** a hierarchical, partly learned policy.
  - **Movement controller:** an MLP (93→128→128→22) that outputs discrete throttle, turn and jump. It was trained by behavioural cloning plus DAgger on a scripted point-to-goal teacher.
  - **Tactical policy:** a GRU (96→64, hidden size 64) trained with PPO against scripted opponents and a small league. Every 5 ticks it picks one of 8 options: direct, left/right lane, return, intercept, screen, stage, escape. Rewards are captures plus a win bonus.
  - **Scripted layers on top:** role assignment (deliver / intercept / screen / attack), route memory built only from paths they have travelled (Dijkstra away from places where wall deaths happened), detours when stuck, and a "pace" mode that holds speed at about 10.2–10.6.
  - **Inference:** NumPy only; weights are embedded as zlib + base85 JSON.

## Security review: safe to run

- Imports: `math`, `dataclasses`, `heapq`, `base64`, `json`, `zlib`, `itertools`, `collections`, `numpy`.
- No file I/O, `os`/`sys`, subprocesses, network, `eval`/`exec`/`compile`, pickle or `importlib`.
- The embedded blob decodes with `json.loads` (data only) to `{controller, actor, config}`: float arrays plus 5 config scalars, about 1.4 MB. There are no strings beyond the key names.
- The model doesn't touch the engine or other players.
- Dependencies: numpy only (already in our venv). Nothing was installed.

## Smoke test (official path)

`main.py 7` was run from a scratch copy, with red = `teams/exml_rl/model.py` and blue = their absolute path. It loaded, ran to tick 6000, and printed `score: {'red': 11, 'blue': 3}`. No errors were printed. The RPC worker respawns silently on a timeout, so timeouts are not logged on this path. Their manifest reports a 1.46 ms maximum RPC response.

## Results (`ctf.evaluate`, seeds 6000–6031 × both sides = 64 games, workers=2)

| our model | W-D-L | win rate (95% CI) | goal diff (95% CI) | GF / GA | our wall deaths/player | our player deaths/player | their player deaths/player | their errors |
|---|---|---|---|---|---|---|---|---|
| `exml_rl` (release) | **64-0-0** | 1.00 [1.00, 1.00] | **+11.0** [9.75, 12.23] | 11.94 / 0.94 | 0.33 | 2.36 | 17.08 | 0 |
| `exml_rl_pure` | 62-1-1 | 0.97 [0.92, 1.00] | +8.78 [7.20, 10.38] | 10.27 / 1.48 | 0.23 | 3.37 | 16.64 | 0 |

- **Release:** both sides 32-0-0 (GD +10.7 as red, +11.3 as blue). Closest games: 7-4 (6021 red), 7-4 (6026 blue). Conversion 0.69 for us vs 0.40 for them. They average 2.3 pickups per match; their median first capture is at tick 2079 (ours 841).
- **Pure:** the only loss was 2-5 (6011 red) and the draw 2-2 (6020 red).
- The release run was killed by an external SIGTERM after 36 games. I resumed it with the missing seeds 6018–6031 and merged the results (same files, digest-checked).
- **Latency:** neither side raised exceptions. Their decide p99 was at most 5.5 ms (release run) and 8.8 ms (pure run), under the 30 ms deadline. Single-call maximums of 0.1–3.7 s (both sides) are host stalls: the laptop was loaded by a concurrent evaluation.

## Observations

- **Speed decides the game.** Their pace layer keeps them at about 10.6, just under the speed-11 wall-death threshold. They get **zero wall deaths**, but the rules eliminate the slower sphere in any opposing collision, so each of their players dies about 17 times per match against our faster spheres. They rarely reach our flag (about 2.3 pickups per match) and almost never get it home.
- **Their defence doesn't stop us.** There is no dedicated defender, and the intercept role only reacts after our flag has left home. We still lose about 4.5 carriers per match. I didn't break this down by cause; that is the main thing left to look at (`tools/diag_match.py`).
- **Release vs pure:** the release scores about 1.7 more goals per match, concedes about 0.5 fewer and has fewer player deaths. That matches its gauntlet ranking.
