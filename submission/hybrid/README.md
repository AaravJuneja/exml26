# Hybrid: learned policy + fixed physics actuator

**Submit `model.py`** — a single file (Python standard library + numpy) exposing
`decide(observation)`. It uses only the documented observation fields, keeps
per-player memory inside its own process, and needs no other files.

A PPO-trained neural network chooses a desired heading, speed and jump every tick; a small deterministic physics actuator (inlined) converts that into throttle/turn. Trained by kickstarting from a scripted teacher, then reinforcement learning against a league of scripted opponents, past snapshots and self-play. The weights are the average of 94 training snapshots (stochastic weight averaging), sampled at temperature 0.7.

| | |
|---|---|
| model.py sha256 | `eb54b99bacc6eef2d84bc2b58dcb9546fa57841bb4519457d9b98fdf46b9aed2` |
| Same file as | [`teams/exml_rl/model.py`](../../teams/exml_rl/model.py) (full verification report: [`teams/exml_rl/release_report.json`](../../teams/exml_rl/release_report.json)) |
| Checks | static audit (documented fields only, allowed imports, no dangerous calls), Python 3.9 load test, deployed-file parity, latency through the real engine and RPC workers, malformed-observation fuzzing, 640-game held-out gauntlet |
| Latency (quiet machine) | model decide() under 1 ms; engine callback p99 under 1 ms (limit 30 ms) |

Held-out results (32 unseen map seeds × both sides = 64 games per opponent): **87% (558/640) wins**

| opponent | win rate | goal difference per match |
|---|---|---|
| relay | 52% | +1.38 |
| balanced | 67% | +1.86 |
| guardian | 84% | +5.12 |
| rusher | 92% | +4.80 |
| camper | 92% | +5.48 |
| chaos | 84% | +5.44 |
| steady | 100% | +9.55 |
| baseline / example / example2 | 100% | +13.3 to +13.7 |
