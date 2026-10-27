# Submissions

Two alternative submissions; each directory holds one self-contained `model.py`.

| Directory | What | Held-out wins |
|---|---|---|
| [`hybrid/`](hybrid/) | Learned PPO policy choosing heading/speed/jump + a fixed physics actuator (inlined). **Strongest.** | **87%** (beats relay 52%) |
| [`pure/`](pure/) | End-to-end neural network outputting throttle/turn/jump directly; no hand-written controller. | **80%** |

Both files are byte-identical to the verified releases in `teams/exml_rl/` and
`teams/exml_rl_pure/` (see the README in each directory for hashes and results).
