# ExML | Exun 2026–27 — Model and Environment Specification

This local copy contains the model-relevant contract and is intended to be read without fetching external documentation. See [`ExML Prompt _ Exun 2026.md`](ExML%20Prompt%20_%20Exun%202026.md) for the short task and match summary.

## Model requirements

- The model must be written in Python and expose `decide(observation)`.
- One submitted model is instantiated for all three players on a team.
- The model runs locally in the embedded environment; it is not externalized.
- The callback must finish within **30 ms**. An exception, load failure, invalid action, or timeout produces an idle action for that tick. A timeout restarts that player’s worker and clears its in-memory state.
- Throttle and turn outside `[-1.0, 1.0]` are clamped, but valid in-range outputs are expected.
- The documentation’s disqualification warning reads: “Tampering with client outside of library exports and modifying specified function will result in immediate disqualification.” The wording about modifying the specified function is ambiguous alongside the required `decide` deliverable. Do not tamper with client/platform code or access anything outside the authorized model interface.

## Physics and movement rules

- Arena: 200 × 200 units, centered at `(0, 0)`; sphere maximum horizontal speed: 18 units.
- Collision with an obstacle or raised ramp edge at speed 11 or greater eliminates a sphere. Lower-speed collisions stop or push the sphere away.
- When opposing spheres collide, the slower sphere is eliminated; equal speeds eliminate both. Teammates bounce and do not eliminate each other.
- Jump works only while grounded and rises roughly 2 units above the current support surface.
- Eliminated spheres respawn at their original spawn positions after 3 seconds (60 ticks).
- Arena boundaries stop horizontal movement but do not eliminate a sphere.

## Action interface

Return `(throttle, turn, jump)` or `(throttle, turn, jump, sonar_message)`:

- `throttle`: float from −1.0 to 1.0; 1.0 accelerates forward and −1.0 accelerates backward.
- `turn`: float from −1.0 to 1.0; controls turning direction.
- `jump`: boolean; jumps if grounded.
- `sonar_message`: any small, picklable Python value or `None`. A message is sent to eligible nearby teammates.

## Observation schema

The environment calls `decide` each tick with a dictionary containing:

| Key | Type and meaning |
| --- | --- |
| `tick` | `int`; current tick, at 20 ticks per second. |
| `self` | `dict`; controlled sphere state. |
| `visible_enemies` | `list[dict]`; visible living enemies, with the same fields as `self`. |
| `own_flag` | `dict`; state of the team’s flag. |
| `other_flags` | `list[dict]`; opposing-team flag states. |
| `score` | `dict[str, int]`; current score of both teams. |
| `sonar_receive` | `list[dict]`; sonar messages received from teammates. |
| `death_messages` | `list[dict]`; recent deaths from the controlled sphere’s team. |

### Sphere state (`self` and visible enemies)

- `id`: global sphere ID (`int`).
- `team`: team name (`str`).
- `player`: player number within the team (`int`).
- `x`, `y`, `z`: coordinates (`float`).
- `vx`, `vy`, `vz`: velocity components (`float`).
- `speed`: horizontal speed (`float`).
- `yaw`: facing direction in radians; 0 faces positive Y.
- `grounded`, `alive`: booleans.
- `carrying`: opponent team name or `None`.

Visible enemies are alive and within 40 units, inside the 70-degree forward viewing cone, and not hidden behind sufficiently tall terrain. Visibility is partial; an empty list does not establish that no enemy is nearby.

### Flag and message fields

- Flag records contain `team`, `x`, `y`, `carried_by` (carrier’s global sphere ID or `-1` if not carried), and `at_home`.
- Sonar records contain `from_id`, `from_player`, and `value`.
- Death records contain `player`, `team`, `tick`, `x`, `y`, and `reason` (`"wall"`, `"player"`, or `"unknown"`).
- A non-`None` sonar message reaches living teammates within 10 units with unobstructed line of sight on the following tick.

## Flag rules

- Each team owns one flag at its home position.
- Move within 2.5 units of an opponent’s flag to collect it.
- Carry it within 2.5 units of your own flag’s home position to score one point. Your own flag does not need to be at home.
- If a flag carrier is eliminated, the carried flag returns to its home position.

## Information and coordination limits

- The interface does not provide obstacle polygons, ramp geometry, map seed, spawn positions, or unseen enemy locations. Participants must not access platform state to obtain these.
- Each sphere runs in its own worker process. Module-level variables persist only for that sphere and cannot communicate with teammates.
- Teammates communicate only through sonar and death messages; there is no documented teammate-state field.
- Use only the documented observation fields. The local Python backend’s `teammates` field and misspelled `sonar_recieve` alias are not part of this interface.
- Do not read full-state telemetry or other hidden platform state as model inputs, labels, rewards, or route hints.

## Environment reference

The event documentation says the final-run environment is available with the distributed project and describes a testing workflow using `teams/<team_name>/model.py`, `tools/godot_bridge.py`, and the release build. In the checked-in research directory, `env/` and `tools/godot_bridge.py` are absent; do not assume those artifacts were inspected here. The event prompt says the same backend will be used in the final simulation.
