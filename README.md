# ExML 2026 Capture the Flag: learned agent

A reinforcement-learning agent for the ExML Capture-the-Flag event. Three
copies of one trained neural network play as a team. Every action comes from
the network; there is no scripted controller in the submission.

**Start here: open [`index.html`](index.html)** in a browser. It links to the
progress tracker, the model gallery and replay viewer, the plan and the latest
release report.

## Submission

Two releases, each one stdlib + numpy file exposing `decide(observation)`:

| Release | Architecture | Held-out (640 games) |
|---|---|---|
| [`teams/exml_rl/model.py`](teams/exml_rl/model.py) (**main**) | Learned desired-velocity policy: the PPO network chooses heading, speed and jump every tick; a deterministic physics actuator (`ctf/controller.py`, inlined) turns that into throttle/turn. Kickstarted from a scripted teacher, then RL with a self-play league. Sampled decoding. | **72% wins** (baseline/example 100%, guardian 63%, rusher 73%, relay 28%) |
| [`teams/exml_rl_pure/model.py`](teams/exml_rl_pure/model.py) | Pure end-to-end PPO network outputting throttle/turn/jump directly. Greedy decoding. | 54% wins |

Audit, deployed-file parity, latency and gauntlet results are in each folder's
`release_report.json`. A distilled pure end-to-end version of the main policy is in
progress.

## Layout

| Path | What |
|---|---|
| `engine/`, `rpc/`, `main.py`, `teams.json` | The official game, byte-identical to the distributed `game-env`. Never modified. |
| `ctf/` | Our code: `sim.py` (exact fast simulator), `features.py` (observation encoder and memory, also inlined into `model.py`), `rewards.py`, `rollout.py`, `train.py` (PPO and league), `nets.py`/`npnets.py`, `evaluate.py` (paired-seed gauntlets), `export.py` (single-file `model.py`), `policy_io.py`, `feature_layouts.py` (warm-start migration). Tests are in `ctf/tests/`. |
| `opponents/` | Strong scripted training opponents (relay, balanced, rusher, guardian, camper, chaos, steady). Training only, never submitted. |
| `teams/` | Provided bots (`baseline`, `example_team`) and our release (`exml_rl`). |
| `tools/` | `release.py`, `replay.py`, `gallery.py`, `auto_eval.py`, `latency.py`, `tracker.py`, `migrate_policy.py`. |
| `bench/` | Throughput benchmarks (`sim_speed.py`, `rollout_speed.py`, `broadphase.py`). |
| `docs/` | `tracker.html` and `TRACKER.md` (progress), `viewer.html` (replays and model history), `RELEASE.md`, `eval/`, `reviews/` and `spec/` (official event documents). |

## Common commands

```sh
# train
python -m ctf.train --name RUN [...]

# evaluate a policy (held-out seeds, both sides, bootstrap CIs)
python -m ctf.evaluate POLICY.npz --gauntlet --seeds 1000-1031 --workers 8

# release: export + audit + parity + latency + gauntlet -> teams/exml_rl/
python tools/release.py POLICY.npz --name exml_rl

# refresh the tracker and the model gallery
tools/update_tracker.sh
python tools/gallery.py

# tests
python -m pytest -q ctf/tests
```
