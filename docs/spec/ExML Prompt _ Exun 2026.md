# ExML | Exun 2026-27: Model Task

## Task

Develop a Python model for the primitive Capture the Flag game. Neural networks and reinforcement learning are examples of permitted machine-learning methods, not exclusive requirements. Fine-tuning may take advantage of the physics model and environment dynamics. This repository may be used for development/testing, and the same backend is intended for the final simulation.

## Match format and deliverable

- Standard round-robin: each team is matched against every other team, and match results determine standings.
- Each game runs for 6,000 ticks at roughly 20 ticks per second.
- Submit one model; three instances of that same model control the three players on a team.
- The model exposes `decide(observation)` and runs locally in the provided environment.

## Scoring and judging

- Safely capture an enemy flag and return it to home-base: **one point** for the home team.
- Judging criteria include points, model complexity, and architecture. The prompt supplies no formula or weights for complexity/architecture.
- The prompt refers to a separate environment/release workflow (`env/`, `tools/godot_bridge.py`, and a release build). Those paths are not in this checkout; the references are retained here as environment context, not as work instructions for this research-only branch.

The full model and observation contract is in [`ExML Documentation _ Exun 2026.md`](ExML%20Documentation%20_%20Exun%202026.md).
