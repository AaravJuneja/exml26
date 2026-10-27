# Breakthrough experiments log (relay / balanced)

Working log of the breakthrough experiments. Incumbent: b1 (goal action space), released
snapshot u960; diagnosis on u1080. All judgements use `tools/ab_compare.py` (paired, seeds
3000–3063, both sides = 128 games per opponent, sampled decoding on both arms) unless stated.

## 1. Diagnosis: b1 u1080 (sampled) vs relay and balanced

Tool: `tools/diag_match.py` (new). Seeds 4000–4031 × both sides = 64 games per opponent,
full simulator state used for analysis only.

| per match | us vs relay | relay | us vs balanced | balanced |
|---|---|---|---|---|
| goals | 9.27 | 10.80 | 9.11 | 9.06 |
| pickups | 11.9 | 12.9 | 11.9 | 11.2 |
| conversion (captures / pickups) | 0.778 | 0.838 | 0.768 | 0.811 |
| carrier killed by player / by wall | 2.52 / 0.25 | 2.41 / 0.02 | 2.70 / 0.31 | 1.98 / 0.03 |
| carry time to capture (median ticks) | 328 | 324 | 325 | 322 |
| flag idle after a capture, ticks/match (sum) | 1156 | 818 | | |
| idle after capture, median / mean / share > 200 ticks | 46 / 136 / 27% | 24 / 84 / 13% | 52 / 138 | 103 / 166 |
| player deaths (wall deaths) | 19.6 (0.42) | 14.8 (0.02) | 19.8 (0.47) | 18.6 (0.08) |
| non-carrier speed (enemy zone / own zone / mid) | 8.2 / 8.6 / 8.6 | 9.8 / 10.6 / 10.4 | 8.3 / 8.8 / 8.6 | 10.1 / 10.4 / 10.3 |
| carrier speed | 11.2 | 11.2 | 11.1 | 11.3 |

Who kills whom vs relay (per match): relay's **waiter at our flag home** kills 5.05 of our
players (4.4 within 10 units of our home, **2.7 within 40 ticks of respawning**: spawn kills;
victim median speed 5.8 vs killer 12.0). Our waiter at relay's home kills only 2.1 relay
players (1.1 spawn kills). Relay's carrier kills 3.4 of ours (median speed 16.8 when it does).

**Gaps (in goals):**
1. *Cycle restart*: our flag sits idle 340 ticks/match longer after our captures (our waiter
   is missing or dead in 27% of resets vs 13% for relay) ≈ 1 capture per match.
2. *Combat at the homes*: we die 4.8 more times per match; the extra deaths are
   home-zone duels and spawn kills by relay's waiter; our non-carriers move ~2 u/s slower
   than relay's everywhere, and the slower sphere dies in a collision.
3. *Conversion*: 6 points lower (incl. 0.25 carrier wall deaths/match) ≈ 0.7 goals per match.
   Carry speed and carry time are already equal to relay's.

## 2. Experiments

### T: decoding temperature (no training)

Hypothesis: sampled decoding beats greedy (tracker), but full-temperature sampling still
puts mass on poor heading/speed choices (our non-carriers are ~2 u/s slower than relay's);
a sharper but still stochastic policy should be stronger. `tools/temper.py` divides the
last layer by tau, so sampling the copy is sampling at temperature tau; the exported
`model.py` needs no code change. Reference: u1080 sampled (tau 1). `tools/ab_multi.py`
(new; caches records so the reference is played once).

| opponent | ref tau 1: win / GD | tau 0.3 dGD | tau 0.5 dGD | **tau 0.7**: win / GD | tau 0.7 dGD [95% CI] | dWin [95% CI] |
|---|---|---|---|---|---|---|
| relay | 23% / −2.45 | +1.16 [+0.14, +2.19] | +1.41 [+0.44, +2.38] | 35% / −0.71 | **+1.73 [+0.55, +2.95]** | +0.125 [+0.008, +0.242] |
| balanced | 34% / −0.56 | +0.57 [−0.23, +1.35] | +0.62 [−0.29, +1.52] | 41% / +0.36 | **+0.92 [+0.11, +1.75]** | +0.070 [−0.031, +0.172] |
| guardian | 72% / +2.46 | | | 67% / +2.27 | −0.19 [−0.82, +0.43] | −0.047 [−0.148, +0.055] |
| rusher | 70% / +2.53 | | | 77% / +2.97 | +0.44 [+0.00, +0.88] | +0.070 [−0.008, +0.148] |
| camper | 69% / +2.39 | | | 70% / +2.53 | +0.14 [−0.47, +0.74] | +0.008 [−0.078, +0.102] |
| chaos | 70% / +2.88 | | | 80% / +3.85 | **+0.98 [+0.25, +1.68]** | +0.109 [+0.000, +0.219] |

**Result: tau 0.7 is a free, significant gain** on relay, balanced and chaos, neutral
elsewhere (guardian −0.19, n.s.). Mean dGD over the six opponents +0.67. Applies to any
snapshot; all later training comparisons are judged at tau 0.7 as well as tau 1.

**Per-head temperature on b1 u1540** (reference: u1540 at tau 0.7 on all heads, seeds 3000–3063;
taus are heading / speed / jump):

| candidate | relay dGD [CI] | balanced dGD [CI] |
|---|---|---|
| 0.7 / **0.5** / 0.7 | +0.29 [−0.37, +0.96] (win 43%, GD −0.12) | **+1.10 [+0.35, +1.83]** (win 52%, GD +0.78) |
| 0.5 / 0.7 / 0.7 | −0.17 [−1.09, +0.78] | +0.35 [−0.59, +1.32] |
| 0.7 / 0.7 / 1.0 | −0.24 [−1.09, +0.58] | +0.73 [−0.12, +1.52] |
| 0.85 all | **−0.85 [−1.65, −0.07]** | −0.23 [−1.02, +0.59] |

A sharper *speed* head is what helps (consistent with the diagnosis: our non-carriers were
~2 u/s slower than relay's); a sharper heading head does not.

### Training experiments

Protocol: every arm resumes from the *same* copy of b1's checkpoint (u1340, 644M samples;
`runs/xbase`), with b1's exact command line (schedules, league pool = b1's last 20
snapshots, opponent EMA, optimizer state) and one changed flag; 3 workers × 5 envs = the
same 15 matches per update as b1's 5 × 3. **Control = b1's own continuation** from u1340
(same code path, same batch), compared at equal update numbers.

| id | hypothesis | change (one) | result |
|---|---|---|---|
| x1 | A MAPPO critic that also sees both teammates' features (training only) assigns credit for coordination (waiter, home duels) better | `--team-critic 1` (critic input = own + teammates' features in player order + alive flags; widened from b1's critic with zero weights so it starts as the same function; Adam moments zero-padded) | **No gain.** Explained variance not higher (0.661 vs b1 0.681 over u1341–1540). Held-out at u1540, both at tau 0.7: relay dGD **−0.88 [−1.78, +0.08]**, dWin −0.18 [−0.27, −0.09]; balanced −0.27 [−1.12, +0.61]. x1 u1540 ≈ the u1340 starting point (relay −1.28 vs −1.02); b1 itself moved +0.62 (n.s.) / +14 pp wins over the same 200 updates. Stopped. |
| x3 | If tau 0.7 is the better policy, keep training *that* policy on-policy (teammates and league see the deployed behaviour) | resume from the checkpoint with the actor's last layer divided by 0.7 (`tools/temper_ckpt.py`) | **Clearly worse.** Training stats start strong (first 100 updates: relay −0.8, balanced +0.3, guardian +3.5) then decay (relay −2.7 / −3.1); entropy climbs 0.94 → 1.1 and clip fraction runs at 0.15–0.29. Held-out u1540 (native) vs b1 u1540 at tau 0.7: relay **−2.16 [−3.12, −1.19]**, balanced **−1.64 [−2.47, −0.78]**. Temperature is a deployment-time setting only; PPO on the sharpened policy destroys the gain. Stopped. |
| x2 | Duels are asymmetric in the reward (death −0.1, a kill earns nothing unless it is the carrier), so the policy under-values winning home-zone duels / denying the waiter | `"kill": 0.1` reward weight: an enemy eliminated by collision (the enemy's own documented death message) credits our closest living sphere (`ctf/rewards.py`) | **Neutral.** Enemy kills per match unchanged in training (16.9 → 17.0 vs b1 17.1 → 17.4); held-out u1540 at tau 0.7 vs b1: relay −0.07 [−1.05, +0.93], balanced +0.25 [−0.67, +1.12]. Also a de-facto replicate of the b1 continuation: run-to-run spread over 200 updates is of the order of ±1 GD, so x1's −0.88 is at most weak evidence of harm. Stopped. |

## 3. Conclusions (27 Oct, ~03:50 UTC)

- **What moves the needle: decoding temperature.** Sampling at tau 0.7 instead of 1 is a
  significant gain on u1080 (relay +1.73, balanced +0.92, chaos +0.98 GD; others neutral).
  On u1540 a sharper speed head (heading 0.7 / speed 0.5 / jump 0.7) adds balanced +1.10
  [+0.35, +1.83] and relay +0.29 (n.s.) on top of uniform 0.7; tau 0.85 is worse than 0.7.
  b1 u1540 at tau 0.7 scores 44% / GD −0.41 vs relay and 38% / −0.32 vs balanced
  (u1080 at tau 1: 23% / −2.45 and 34% / −0.56 on the same seeds).
- **What does not (at 200 updates ≈ 65M samples from b1 u1340):** team critic (x1),
  kill reward (x2), and continuing PPO on a tempered policy (x3, clearly harmful).
  Run-to-run spread of a b1 continuation is about ±1 GD at this horizon, so small
  training-side effects cannot be resolved without replicated arms or much longer runs.
- **Pending (training machine went offline ~01:50 UTC):** `runs/xev5` refines the speed temperature
  (0.7/0.35/0.7 and 0.6/0.5/0.7 vs 0.7/0.5/0.7); results in `runs/xeval/ev5.json`.
  Not yet done: a confirmation of the chosen per-head temperatures on *fresh* seeds
  (e.g. 3100–3163, all six opponents) to remove the selection bias of the sweep.

**Recommendation:** release the newest good b1 snapshot (u1540 tested here) through
`tools/temper.py --tau 0.7,0.5,0.7` (or `--tau 0.7` if the fresh-seed confirmation does
not hold up), after the usual release gauntlet and parity checks; keep training at tau 1.
Snapshots: `runs/xeval/b1_u01540_h07s05j07.npz` (per-head) and
`runs/xeval/b1_u01540_t0.7.npz` (uniform).
