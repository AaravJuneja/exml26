# Simulator throughput — 25 October 2026

`python -m bench.sim_speed --ticks 2000 --procs 1,2,4,6`. Unmodified `engine/`,
in-process callbacks, no telemetry, no RPC, no real-time sleep. The machine was
shared with two other jobs (load average 8.2 on 6 cores), so these are lower
bounds.

| Setting | Ticks/s |
|---|---:|
| 1 process, idle policy | 904 |
| 1 process, random policy | 761 |
| 1 process, `example_team/model.py` | 1,194 (many spheres dead, so fewer to simulate) |
| 2 processes, random | 1,105 aggregate |
| 4 processes, random | 1,790 aggregate |
| 6 processes, random | 1,869 aggregate |

Best case: about **1,870 ticks/s**, which is:
- **~40M agent transitions per hour** (6 spheres per tick), or ~20M for one learning team
- **~1,100 full 6,000-tick games per hour**

These exclude network inference and learning updates; expect a 2–3× drop once
those are included. Scaling flattens above 4 processes on this contended machine.

## Training machine (4P+6E cores, 16 GB, Python 3.13)

Same benchmark, idle policy. Load average was already ~4.8 before the run.

| Processes | Aggregate ticks/s |
|---:|---:|
| 1 | 2,798 |
| 4 | 10,788 |
| 6 | 12,883 |
| 8 | 14,726 |
| 10 | 16,056 |

That gives about **16,000 ticks/s**: ~170M transitions per hour for one learning team, or
~9,600 full games per hour. This is ~8.6× the WSL laptop, so all training runs go
on the training machine.

## Exact broad-phase filter (`bench/broadphase.py`)

Each sphere is resolved alone against only nearby obstacles and ramps (bounding
circle plus a 3.5-unit margin, original order preserved). The unmodified
engine functions do the work; only their input lists shrink.

**Parity:** `--parity 10 --ticks 6000` gave bit-identical position, velocity,
alive state and score on every tick of 10 full games with a fast,
collision-heavy deterministic policy (64–109 deaths per game).

| Training machine, `wiggle` policy, 6,000 ticks | Ticks/s |
|---|---:|
| 1 process, unfiltered | ~3,000 |
| 1 process, filtered | ~15,800 (5.4×) |
| 4 processes, filtered | 65,400 |
| 8 processes, filtered | **92,200** |
| 10 processes, filtered | 87,100 |

That is ~55,000 full games per hour of raw simulation. Network inference,
feature building and PPO updates will be the bottleneck from here, not physics.
