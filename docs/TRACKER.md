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
| Replay viewer | in progress | `tools/replay.py` and `docs/viewer.html` |
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
| m2d | Round-2 fixes, exact PBRS, PopArt, frozen normalisation; vs idle | running |

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
