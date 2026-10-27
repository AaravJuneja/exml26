# ExML 2026 Capture the Flag agent

A reinforcement learning agent for the ExML Capture the Flag event at Exun 2026. Three
copies of one trained network play as a team, each in its own process, using only the
documented observation fields.

There are two submissions:

- **Hybrid** ([`submission/hybrid/model.py`](submission/hybrid/model.py)): the network picks
  a heading, a speed and jump every tick, and a fixed physics controller turns that into
  throttle and turn.
- **Pure** ([`submission/pure/model.py`](submission/pure/model.py)): the network outputs
  throttle, turn and jump directly, with no hand-written controller.

Each is a single file that needs only the standard library and numpy and exposes
`decide(observation)`. They are byte-identical to the releases in `teams/exml_rl/` and
`teams/exml_rl_pure/`.

Open [`index.html`](index.html) in a browser for the progress tracker, the model gallery,
the replay viewer and the latest release report.

## Results

Held-out means 32 map seeds never used in training or model selection, each played from
both sides: 64 games per opponent, 640 in total.

| Release | Held-out wins | Per opponent |
|---|---|---|
| [`teams/exml_rl`](teams/exml_rl/model.py) (hybrid) | **87%** | baseline, example, steady 100%; rusher, camper 92%; guardian, chaos 84%; balanced 67%; relay 52% |
| [`teams/exml_rl_pure`](teams/exml_rl_pure/model.py) (pure) | **80%** | baseline, example 100%; steady 98%; rusher 89%; chaos 88%; camper 78%; guardian 73%; balanced 41%; relay 38% |

Each folder's `release_report.json` holds the audit, parity, latency and gauntlet results.
Earlier releases are in the git history.

## How it was built

1. **Simulator and features.** `ctf/sim.py` is an exact, faster reimplementation of the
   official engine, parity-tested against it. `ctf/features.py` encodes one player's
   observation plus its own memory into 220 inputs (209 for the pure network).
2. **Hybrid policy.** A PPO actor-critic (256x256 actor, 384x384 critic) picks a heading
   bin, a speed bin and jump each tick, and `ctf/controller.py` steers the sphere there.
   Training starts by imitating a strong scripted teacher (kickstarting), then continues
   with RL against scripted opponents, past snapshots and self-play, with shaped rewards
   for pickups, captures, defence and deaths.
3. **Inference.** Two changes, each confirmed by a paired A/B test on fresh seeds:
   sampling at temperature 0.7, and averaging the weights of 94 training snapshots
   (`tools/average_policies.py`).
4. **Pure policy.** A student that outputs throttle, turn and jump bins is distilled from
   the hybrid policy, trained further with PPO plus a distillation term, then weight
   averaged and sampled at temperature 0.7 (`ctf/distill.py`).
5. **Release checks.** Every candidate gets a paired A/B test (128 games per opponent)
   against the current release and a confirmation on a second seed set. Then
   `tools/release.py` audits the deployed file, checks parity between `model.py` and the
   training policy, measures latency through the real engine and RPC workers, and plays a
   640-game held-out gauntlet. `docs/TRACKER.md` logs every experiment, including the
   ones that failed.

## Layout

| Path | Contents |
|---|---|
| `engine/`, `rpc/`, `main.py`, `teams.json` | The official game, byte-identical to the distributed `game-env`. |
| `ctf/` | Simulator, features, rewards, rollouts, PPO and league training (`train.py`), networks, evaluation, export to a single `model.py`, action space and controller, distillation. Tests are in `ctf/tests/`. |
| `opponents/` | Scripted opponents used only for training: relay, balanced, rusher, guardian, camper, chaos and steady. |
| `submission/` | The two ready-to-submit `model.py` files. |
| `teams/` | The provided bots (`baseline`, `example_team`) and the two releases with their reports. |
| `tools/` | Release and verification, paired A/B tests, temperature and weight averaging, diagnostics, distillation, replays, the gallery and the tracker. |
| `bench/` | Throughput benchmarks. |
| `docs/` | Tracker, replay viewer, release log, evaluations, experiment notes and the official event spec (`docs/spec/`). |

## Usage

The submissions need numpy. Training and tests also need torch and pytest.

```sh
# train
python -m ctf.train --name RUN [...]

# evaluate a policy on held-out seeds, both sides, with bootstrap CIs
python -m ctf.evaluate POLICY.npz --gauntlet --seeds 1000-1031 --workers 8

# release: export, audit, parity, latency and gauntlet into teams/NAME/
python tools/release.py POLICY.npz --name exml_rl --decode sample

# temperature and weight averaging
python tools/temper.py IN.npz OUT.npz --tau 0.7
python tools/average_policies.py OUT.npz runs/b1/policies/u0*.npz

# paired A/B of candidates against a reference
python tools/ab_multi.py --ref A.npz@sample,seed=1 --cand B.npz@sample,seed=1 --opponents relay balanced --seeds 3000-3063 --cache DIR

# verify a fresh clone of the repository
bash tools/final_check.sh

# refresh the tracker and the model gallery from runs/
tools/update_tracker.sh
python tools/gallery.py

# tests
python -m pytest -q ctf/tests
```

## AI use

AI was used for research, gathering information, exploring ideas, understanding concepts, and supporting the learning and research process.

