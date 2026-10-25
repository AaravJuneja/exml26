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
