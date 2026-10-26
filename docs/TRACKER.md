# Detailed work tracker

Working log. The summary is `docs/tracker.html`.

## Ground rules
- **Submission:** real ML. The network chooses actions. Engineered inputs and preprocessing are fine, and hybrid designs are fine if they win. The goal is the best possible tournament result.
- **Points:** a draw gives 0 points, so we optimise **wins**.
- **numpy:** confirmed available in the tournament. Python version is not a concern; 3.14 is used everywhere.
- **30 ms limit:** wall-clock time from the parent's send to the reply becoming readable (`rpc/worker.py` `conn.poll(0.03)`). The 6 workers run concurrently. Targets: warm `decide` p99 ≤ 3 ms; loaded 6-worker callback p99 ≤ 10 ms, max ≤ 20 ms.
- **Workflow:**
  - push to `main` at milestones;
  - update `tracker.html` at milestones and every ~3 h;
  - each phase is reviewed, with capped review rounds.

## Repo housekeeping (26 Oct)
- **`engine/`, `rpc/`, `main.py`** are byte-identical to the distributed `game-env` (the other session's optional hooks removed); 133 tests pass.
- **Working rule:** all heavy work (tests, evaluations, replays, training) runs on the training machine.

## Infrastructure
| Item | Status | Notes |
|---|---|---|
| `ctf/sim.py` | done | Exact `Game` replica with observe/act split, plus an exact broad phase (obstacles, ramps, ground, line of sight). 6 parity tests against the real `Game`, with recorded observations bit-identical |
| `ctf/features.py` | done | 158 features: self 21, game 4, flags 22, enemies 3×11, bumps 12×3, danger 4×5, previous action 22. Pure Python, about 9 µs per encode |
| `ctf/nets.py`, `ctf/policy_io.py` | done | tanh MLP actor 158→128→128→(12+9+2); critic 256×256; numpy actor; `.npz` policy format |
| `ctf/rewards.py` | done | score ±1, pickup 0.2, wall death −0.3, player death −0.1, carrier death −0.25, flag return +0.2, progress 0.5 per 180 units (annealed) |
| `ctf/rollout.py`, `ctf/train.py` | done | Full-match lockstep workers; GAE across respawn gaps (tested against a reference); PPO with KL early stop, value normalisation, observation normalisation; checkpoint and resume |
| `ctf/export.py` + `tools/latency.py` | done | Inlines `features.py` verbatim; base64 weights; bit-exact against `NumpyActor`; callback p99 about 1 ms under stress; Python 3.9 OK |
| `ctf/evaluate.py` | done | Paired-seed bootstrap; ~565 matches/min on 10 workers; `docs/eval/baseline_gauntlet.json` |
| League | done | `self`, `pool` (PFSP (1−wr)²+0.05 over the last 20 snapshots plus extra globs) and `mirror` (both teams learn) |
| `tools/auto_eval.py` | done | Gauntlets every 50th snapshot into `runs/RUN/eval.jsonl`; shown in the tracker |
| Replay viewer | done | `tools/replay.py` (`--html` embeds the replay); `docs/viewer.html` |
| Scripted opponents | done | relay, balanced, rusher, guardian, camper, chaos, steady; documented-only, generated from `opponents/_core.py`; round robin in `docs/eval/opponents_round_robin.json` |
| Evaluation/export hardening | done | `compare` validation, digest cache, `@proc` per-process exported-model evaluation, forced 1-thread BLAS, near-tie/sampling parity |
| Review round 3 (final for Phases 1–2) | done | Fixed: persist `norm_frozen` across `--init`; Monte Carlo win-head targets (`--win-lam 1`); the critic keeps training after the actor KL stop; `--shaping-offset` for exact PBRS; docs. Note: a mirror job is 2 learner units, so mirror probability p gives about 2p/(1+p) of the learner data |
| `opponents/*` | in progress | rusher, guardian, camper, relay, balanced |
| Phase 1 review | done | Critical: death-log cap (200) broke death penalties; progress shaping could be farmed by dying. Both fixed: deaths deduplicated by key; potential-based route shaping applied across deaths. Major fixes: wall deaths never home-filtered; bump inferred from pushing at rest; flag velocity keyed to the carrier; numpy-only `npnets`; idempotent encode |
| Phase 2 review plus round 2 | done | Critical: teammate carrier position leaked through rewards (the engine assigns the carrier before flag coordinates refresh); now uses public flag coordinates. Major fixes: PopArt critic rescaling; exact PBRS on every tick with terminal Φ=0 (tested by an exact discounted-telescoping identity); observation stats frozen after 20 updates; worker health/timeouts/error propagation; flag-velocity baseline after a carrier change; latency verdict fails on load or model errors; KL check before the step; RNG state in checkpoints |
| Engine ramp fix | done | Per-ramp `height_at` check. `engine/` is byte-identical to `game-env`; also applied to the planning workspaces. m2b/m2c were restarted on the fixed physics |

## Measurements
- **Training machine physics:** 92k ticks/s across 8 processes with the broad phase (bit-identical).
- **Training throughput** (idle opponents, 10 workers × 3 matches): about 517k samples per update; rollout 3.2 s plus update 4.4 s; about **240M learner samples per hour**.

## Runs
| Run | Config | Status / result |
|---|---|---|
| probe1 | 2 updates, idle | Throughput probe only |
| m2a | PPO from scratch vs idle (old rewards) | Stopped at ~16M samples because of the reward bugs; had reached 0.3 goals per match |
| m2b, m2c | Fixed rewards; old physics (m2b) or pre-round-2 code (m2c) | Stopped and superseded |
| m2d | Round-2 fixes, exact discounted PBRS | Stopped: shaping ≈ +30 per match from the (γ−1)Φ idle bonus; 0 pickups at 16M samples |
| m2e | Undiscounted shaping differences | **M2 gate passed**: 8.0 goals per match vs idle, 0.78 wall deaths per player at 150M samples; held-out 96% vs idle, 4% vs baseline |
| m3a | From m2e, scripted mix plus mirror, win-coef 0.3 | Stopped after round 3 (norm re-opened by `--init`; bootstrapped win head) |
| m3b | Same mix, round-3 fixes | Stopped at u~160. Held-out u150: baseline 44%, relay 6%, others 0–3% |
| m3c | Features v2 (clear-ground map + rays), warm-start from m3b via surgery | Stopped at u~115. Held-out u50: baseline **78%**, relay 3%, balanced 3%, guardian 3%, camper 9% |
| m3d | Features v3 fixes, from m3c weights; GPU learner, pipelined, minibatch 16384; from u716 with adaptive mix and the faster rollouts | running. Held-out: u300 baseline 94%; u450 baseline 97%, camper 25%, relay 9%, balanced 6% |

## Throughput notes
- **Pipelining alone:** about 10% faster, because the learner on 6–8 CPU threads competed with the workers.
- **GPU:** 10 ms vs 107 ms per 16k-sample step under load. Removing per-step `.item()` syncs and using minibatch 16384 cut the update to about 4.5 s. Rollouts are now the bottleneck: about 27k ticks/s ≈ 290M samples/h.

- **Rollout optimisation:** `collect()` is 1.43× faster (median 2.09 s → 1.46 s). Changes: features 60 → 42 µs; a broad-phase cell cache in `sim`; bounding-box line-of-sight rejection; a single-draw sampler. Equivalence is pinned by hash fixtures (`ctf/tests/test_equivalence.py`); the original code was kept in `perf_orig/`. Scripted bots are now compiled once per process.
- **Current cycle:** about 6 s with the learner update (about 5 s) as the bottleneck; about 30k ticks/s.

## Strategy reset (26 Oct, surveys in docs/reviews/)
- **Plateau confirmed by two independent reviews.**
- **Gap is behaviour:** relay projected to our bins (sonarless) still beats us ~70%.
- **Wall deaths:** ours 1–1.7 per player; relay's team ~0.04.
- **Real throughput:** ~52M samples/h per job when 4 jobs ran; consolidated to 2 jobs.
- **Ranked moves:** (1) kickstarting (k1, running); (2) plan B (heading/speed + controller, being built); (3) paired 128-game A/B methodology (`tools/ab_compare.py`); (4) a diverse league plus an exploiter; (5) a team critic.

## Experiments
| Experiment | Result |
|---|---|
| Greedy vs sampled decoding (u720, seeds 2000–2015) | Greedy: baseline 91%, relay 28%, balanced 25%. Sampled: 78%, 6%, 9%. **Use greedy.** |
| lr 1e-3 vs 3e-4 (m3e-b vs m3e-a, 5 workers each, from m3d) | 1e-3 is worse: relay training goal difference −5.9 → −8.4. Stopped. |
| Capacity: Net2Wider 256×256 (m3e-c vs m3e-a) | About equal over 300 updates (C slightly better vs balanced/camper). Adopted 256×256 as the mainline (cheap at inference). |
| Defence potential 0.3 (m3e-d vs m3e-c, from the same checkpoint) | Training relay goal difference: D −5.4 → −4.7 vs C −5.2 → −6.1. Held-out about equal or slightly better. Kept, gated off for our own carrier. |
| v4 bundle; entropy/λ A/B (m4a: 0.003/0.95 vs m4b: 0.0005/0.99) | **m4b better**. Training at updates 161–222: baseline 100%, rusher 43%, balanced −3.6 (m4a: 83%, 24%, −4.7). Held-out: guardian/rusher/camper 33–44% (m4a 25–33%). Relay about equal. |
| m4c: entropy 0.0001, λ 0.995 (from m4b u222) vs m4b | **m4c slightly better**: held-out guardian/rusher/camper 48–56% vs 40–52%; relay/balanced similar (6–19%) |
| v5 features (m5a, from m4c u~205) vs m4c | About equal or slightly better (balanced −2.7 vs −3.3; camper/guardian ~0.5). Held-out relay 17–21%. **Adopted v5.** |
| Support potential 0.15 (m5b vs m5a, from m5a u210) | **Tie**: both flat at 15–20% wins vs relay/balanced. Kept (harmless). |
| Kickstarting from relay (k1 vs m5b, from m5b u~240) | running |
| _(old note)_ Support potential Motivation: reset → re-pickup median 195 ticks (us) vs 51 (relay), m4c u280 replay seed 1003 (won 8–5) |
| Release pipeline (`tools/release.py`) | Candidate m4a u70: audit/parity/latency PASS (p99 1.43 ms, max 5.6 ms; stress p99 1.2, max 10.2). Gauntlet seeds 5000–5031: 41% overall |

## Strategy review (summary)
Rankings: (1) capture-reset readiness: teammate carrier identity, respawn timers, waiter behaviour; (2) velocity control and symmetric fine bins; (3) targeted denial rather than indiscriminate pursuit; (4) a focused curriculum plus a real league; (5) lower entropy and higher λ before more aggressive PPO; (6) targeted memory (trails, bump normals, stuck duration) before learned sonar. Items 1, 2, 4 and part of 3 are in v4; item 5 is the m4a/m4b A/B.

## Diagnosis notes
- **m3b vs relay (seed 1003): lost 2–19.** Relay players average 11–13 u/s (above 11 for 54–73% of the time); ours average 6 u/s (above 11 for 4–9%). We had 13 player deaths and relay 4; pickups 6 vs 20. Cause: no information about safe ground, so we could not sprint. Fix: features v2.

## Backlog from the reviews
- Long-horizon win objective: a separate undiscounted win-value head (γ=1, terminal win indicator), with its advantage added to the PPO advantage at weight β.
- Evaluation `compare`: validate opponent, digest, horizon and full side pairs.
- Policy caches: key them by file digest.
- Export: test inherited thread environment variables, sampled-RNG parity, and promote the exported artifact itself.
- Features: confidence/expiry for inferred contacts; teammate-stop false terrain.

## Next
1. Read the Phase 1 review and fix its findings.
2. M2 gate: at least 8 captures per match against idle on held-out seeds, and at most 2 wall deaths per player.
3. Integrate the exporter and evaluation harness, then commit.
4. Opponent-mix training (M3); self-play league (M4).
5. Ideas backlog:
   - GRU variant
   - learned sonar
   - critic that sees the whole team
   - larger network if the latency budget allows
   - stochastic vs greedy decoding at evaluation
   - reward for a win bonus
   - PBT over hyperparameters
