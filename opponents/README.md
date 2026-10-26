# Training opponents

Scripted, training-only opponents for the learned agent. Each `opponents/<name>/model.py`
is a standalone, standard-library-only `decide(observation)` file. It reads only the
documented observation keys, so it never uses `teammates` or `sonar_recieve`. Sonar between
teammates is used where noted.

The files are generated. Edit `_core.py` (shared engine) or `STYLES` in `build.py`, then run:

```
python -m opponents.build            # regenerate all models
python -m opponents.build --check    # CI: fail if stale (also checked by ctf/tests/test_opponents.py)
python -m opponents.arena match balanced baseline --seed 2000 [--trace 100]
python -m opponents.arena panel guardian relay,balanced --seeds 2100-2107
python -m opponents.arena rr --seeds 2000-2015 --procs 10 --out docs/eval/opponents_round_robin.json
```

## Shared mechanics (`_core.py`)

- **Speed governor.** Ground that is not known to be clear is driven below the lethal
  speed. The limit is `vsafe`, at most 10.85, and the lethal speed is 11. Ground is known
  to be clear inside the 16-unit obstacle-free disks around both homes, on the bot's own
  grounded trail, on trails of visible grounded enemies, and on sonar-shared segments. On
  that ground the bot drives at up to 17.64. It looks ahead about 2.5 + 0.16·v along both
  its heading and its velocity.
- **Bumps.** A low-speed contact zeroes the velocity. The push-out direction gives the
  obstacle normal. From it the bot estimates an obstacle circle and merges overlapping
  estimates. It then slides along the wall with a persistent side, and later plans tangent
  detours around known circles. A teammate's `wall` death adds a circle too.
- **Velocity-vector control.** The bot thrusts forward or in reverse along the velocity
  error, so it never orbits a target. A carrier reverses straight out after a pickup or
  capture instead of turning. Thrust is clamped so it can never push the speed above the
  target.
- **Combat.** The bot rams visible enemies when it can be faster at contact, aiming at the
  predicted contact point. Defenders accept speed ties inside their home disk. When a
  contact is predicted with a faster enemy, the bot jumps. Optionally it also jumps on
  respawn, which counters spawn-camping.

## Styles

| bot | design |
|---|---|
| `rusher` | Three attackers on three lanes; 10.85 on unknown ground, 17.64 on known ground; jump-dodges; no defence. |
| `guardian` | Two guards orbit the flag at about 10 while facing inward, so their vision sweeps the home disk. They ram intruders at full speed, accept ties, and chase the stolen-flag coordinates along the carrier's trail. One raider. |
| `camper` | Two players shuttle in front of the enemy spawn and flag, killing fresh respawns and arriving carriers. One of them snipes the flag when it is clear. A third player harasses midfield and intercepts. |
| `relay` | Three cyclers. While one carries, another hovers over the enemy flag home and re-takes it the tick it resets. Sonar shares safe segments, bump circles and "waiting" claims; a duplicate waiter escorts instead. |
| `balanced` | Two relay cyclers that also defend when the flag is taken nearby, plus a flex player that intercepts carriers or joins the relay. Shared memory, trail following and fast known-ground driving. |
| `chaos` | Seeded, randomised team plan: composition, lanes, speed envelope, aggression, tie policy, jump rate and steering noise. Player 1's draw is shared by sonar at spawn. Set the module's `SEED` for reproducibility; `arena` sets it to the match seed. |
| `steady` | Low speed envelope: never above 10. Two attackers with escort, plus a guard. |
