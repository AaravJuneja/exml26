# ExML 2026 Capture the Flag: learned agent

A reinforcement-learning agent for the ExML Capture-the-Flag event. Three
copies of one trained neural network play as a team, each in its own process,
using only the documented observation fields. There are two submissions: the
main one, where the network chooses a desired heading, speed and jump every tick
and a fixed physics actuator turns that into throttle/turn; and a pure
end-to-end network that outputs throttle/turn/jump directly with no hand-written
controller.

**Start here: open [`index.html`](index.html)** in a browser. It links to the
progress tracker, the model gallery and replay viewer, the plan and the latest
release report.

## Submission

**Ready-to-submit files are in [`submission/`](submission/):** `submission/hybrid/model.py`
(strongest) and `submission/pure/model.py` (no hand-written controller). They are byte-identical
to the verified releases below.

Two releases, each one stdlib + numpy file exposing `decide(observation)`:

| Release | Architecture | Held-out (640 games) |
|---|---|---|
| [`teams/exml_rl/model.py`](teams/exml_rl/model.py) (**main**) | Learned desired-velocity policy: the PPO network chooses heading, speed and jump every tick; a deterministic physics actuator (`ctf/controller.py`, inlined) turns that into throttle/turn. Kickstarted from a scripted teacher, then RL with a self-play league. Weights averaged over 94 snapshots of the run (updates 160–2030, stochastic weight averaging), sampled at temperature 0.7. | **87% wins** (baseline/example/steady 100%, rusher/camper 92%, guardian/chaos 84%, balanced 67%, relay 52%) |
| [`teams/exml_rl_pure/model.py`](teams/exml_rl_pure/model.py) | Pure end-to-end network outputting throttle/turn/jump directly (no hand-written controller): the plan-B policy distilled into a bins-action student (soft labels through the controller, b1's hidden layers as initialisation) and trained with PPO plus a distillation term. Re-distilled from the averaged main policy as teacher, weights averaged over snapshots 350–450, sampled at temperature 0.7. | **80% wins** (baseline/example 100%, steady 98%, rusher 89%, chaos 88%, camper 78%, guardian 73%, balanced 41%, relay 38%) |

Audit, deployed-file parity, latency and gauntlet results are in each folder's
`release_report.json`. "Held-out" means 32 map seeds never used in training or
model selection, each played from both sides (64 games per opponent). Earlier
releases are in git history.

## How it was built

1. **Simulator and features.** `ctf/sim.py` is an exact, faster re-implementation of
   the official engine (parity-tested against it). `ctf/features.py` encodes one
   player's documented observation plus its own memory into 220 inputs (209 for the pure network).
2. **Main policy (plan B).** A PPO actor-critic (MLP 256×256 actor, 384×384 critic)
   picks a heading bin, a speed bin and jump each tick. `ctf/controller.py`, a fixed
   physics actuator, steers the sphere there. Training starts with kickstarting
   (imitating a strong scripted teacher) and then continues as RL against a league of
   scripted opponents, past snapshots and self-play mirrors, with shaped rewards for
   pickups, captures, defence and deaths.
3. **Inference-time improvements, each confirmed by paired A/B on fresh seeds.**
   Sampling at temperature 0.7 (the last layer scaled by 1/0.7) and stochastic
   weight averaging: the released weights are the average of 94 snapshots of the
   training run (`tools/average_policies.py`).
4. **Pure network.** A bins-action student (throttle/turn/jump bins) is distilled
   from the main policy. A shadow teacher labels every student state with soft
   targets pushed through the controller. It is then trained with PPO plus the
   distillation term, re-distilled from the averaged main policy, weight-averaged
   and sampled at temperature 0.7 (`ctf/distill.py`).
5. **Release discipline.** Every candidate goes through the same steps: a paired
   128-games-per-opponent A/B against the current release, a confirmation on a second
   fresh seed set, then `tools/release.py` (audit of the deployed file, parity between
   the deployed `model.py` and the training-side policy, latency through the real
   engine and RPC workers, and a 640-game held-out gauntlet). `docs/TRACKER.md` logs
   every experiment, including the ones that failed.

## Layout

| Path | What |
|---|---|
| `engine/`, `rpc/`, `main.py`, `teams.json` | The official game, byte-identical to the distributed `game-env`. Never modified. |
| `ctf/` | Our code: `sim.py` (exact fast simulator), `features.py` (observation encoder and memory, also inlined into `model.py`), `rewards.py`, `rollout.py`, `train.py` (PPO and league), `nets.py`/`npnets.py`, `evaluate.py` (paired-seed gauntlets), `export.py` (single-file `model.py`), `policy_io.py`, `feature_layouts.py` (warm-start migration), `action_space.py` and `controller.py` (plan-B goal space and actuator), `distill.py` (policy teacher for the pure network). Tests are in `ctf/tests/`. |
| `opponents/` | Strong scripted training opponents (relay, balanced, rusher, guardian, camper, chaos, steady). Training only, never submitted. |
| `submission/` | The two ready-to-submit `model.py` files (`hybrid/`, `pure/`), each with a short README. |
| `teams/` | Provided bots (`baseline`, `example_team`) and our two releases (`exml_rl`, `exml_rl_pure`) with full verification reports. |
| `tools/` | `release.py` (export and verification), `final_check.sh` (fresh-clone verification), `ab_compare.py` / `ab_multi.py` (paired A/B), `temper.py` (temperature), `average_policies.py` (weight averaging), `diag_match.py` (behaviour diagnosis), `distill_*.py`, `replay.py`, `gallery.py`, `auto_eval.py`, `latency.py`, `tracker.py`, `migrate_policy.py`. |
| `bench/` | Throughput benchmarks (`sim_speed.py`, `rollout_speed.py`, `broadphase.py`). |
| `docs/` | `tracker.html` and `TRACKER.md` (progress), `viewer.html` (replays and model history), `RELEASE.md`, `eval/`, `reviews/` and `spec/` (official event documents). |

## Common commands

```sh
# train
python -m ctf.train --name RUN [...]

# evaluate a policy (held-out seeds, both sides, bootstrap CIs)
python -m ctf.evaluate POLICY.npz --gauntlet --seeds 1000-1031 --workers 8

# release: export + audit + parity + latency + gauntlet -> teams/NAME/
python tools/release.py POLICY.npz --name exml_rl --decode sample

# temperature and weight averaging
python tools/temper.py IN.npz OUT.npz --tau 0.7
python tools/average_policies.py OUT.npz runs/b1/policies/u0*.npz

# paired A/B of several candidates against a reference
python tools/ab_multi.py --ref A.npz@sample,seed=1 --cand B.npz@sample,seed=1 --opponents relay balanced --seeds 3000-3063 --cache DIR

# fresh-clone verification of what is on GitHub
bash tools/final_check.sh

# refresh the tracker and the model gallery
tools/update_tracker.sh
python tools/gallery.py

# tests
python -m pytest -q ctf/tests
```
