# Pure: end-to-end neural network

**Submit `model.py`** — a single file (Python standard library + numpy) exposing
`decide(observation)`. It uses only the documented observation fields, keeps
per-player memory inside its own process, and needs no other files.

A single neural network maps the observation (plus its own memory features) directly to throttle, turn and jump. There is no hand-written controller. It was distilled from the hybrid policy (soft targets), trained further with PPO, re-distilled from the weight-averaged hybrid policy, weight-averaged over 11 snapshots and sampled at temperature 0.7.

| | |
|---|---|
| model.py sha256 | `9923790d4df2e3b2188ab838d31d4b7ff3432ef0e8e7087af2ab153dde3395f5` |
| Same file as | [`teams/exml_rl_pure/model.py`](../../teams/exml_rl_pure/model.py) (full verification report: [`teams/exml_rl_pure/release_report.json`](../../teams/exml_rl_pure/release_report.json)) |
| Checks | static audit (documented fields only, allowed imports, no dangerous calls), Python 3.9 load test, deployed-file parity, latency through the real engine and RPC workers, malformed-observation fuzzing, 640-game held-out gauntlet |
| Latency (quiet machine) | model decide() under 1 ms; engine callback p99 under 1 ms (limit 30 ms) |

Held-out results (32 unseen map seeds × both sides = 64 games per opponent): **80% (515/640) wins**

| opponent | win rate | goal difference per match |
|---|---|---|
| relay | 38% | −0.81 |
| balanced | 41% | −0.55 |
| guardian | 73% | +3.11 |
| rusher | 89% | +3.77 |
| camper | 78% | +3.52 |
| chaos | 88% | +5.16 |
| steady | 98% | +8.17 |
| baseline / example / example2 | 100% | +12.7 to +13.0 |
