"""Per-tick training rewards for the learning team.

Every term is computable from documented observation content (score, the
player's own position/carrying state, flag records and death messages). The
simulator objects are read only because they hold exactly those values.
"""

import math

COMPONENTS = ("score", "pickup", "death", "carrier_death", "flag_return", "progress")

DEFAULT_WEIGHTS = {
    "score": 1.0,          # per point, own minus enemy, to every learner
    "pickup": 0.2,         # this player starts carrying the enemy flag
    "wall_death": 0.3,     # penalty
    "player_death": 0.1,   # penalty
    "carrier_death": 0.25, # extra penalty for dying while carrying
    "flag_return": 0.2,    # enemy carrier of our flag eliminated (team)
    "progress": 0.5,       # scale of objective-distance progress (per 180 units)
}


class RewardTracker:
    def __init__(self, match, team, weights):
        self.match = match
        self.team = team
        self.enemy = next(name for name in match.team_names if name != team)
        self.weights = dict(weights)
        self.spheres = sorted((s for s in match.spheres if s.team == team), key=lambda s: s.player)
        self.own_flag = next(f for f in match.flags if f.team == team)
        self.enemy_flag = next(f for f in match.flags if f.team == self.enemy)
        self._deaths_seen = len(match.death_log())
        self.totals = {name: 0.0 for name in COMPONENTS}
        self.counts = {"pickups": 0, "wall_deaths": 0, "player_deaths": 0,
                       "carrier_deaths": 0, "flag_returns": 0}

    def _objective_distance(self, sphere):
        if sphere.carrying_flag is not None:
            hx, hy = self.own_flag.home_x, self.own_flag.home_y
            return math.hypot(sphere.x - hx, sphere.y - hy)
        carrier = self.enemy_flag.carrier
        if carrier is not None and carrier.team == self.team:
            return None
        return math.hypot(sphere.x - self.enemy_flag.x, sphere.y - self.enemy_flag.y)

    def before(self):
        """Snapshot taken after observe() and before act()."""
        m = self.match
        self._score = (m.score[self.team], m.score[self.enemy])
        self._own_flag_carrier = self.own_flag.carrier
        self._state = [
            (s.alive, s.carrying_flag is not None, self._objective_distance(s) if s.alive else None)
            for s in self.spheres
        ]

    def after(self):
        """Rewards for the transition just simulated, one value per learner slot."""
        w = self.weights
        m = self.match
        own, enemy = m.score[self.team], m.score[self.enemy]
        score = w["score"] * ((own - self._score[0]) - (enemy - self._score[1]))
        rewards = [score] * len(self.spheres)
        self.totals["score"] += score * len(self.spheres)

        if (self._own_flag_carrier is not None and self.own_flag.carrier is None
                and enemy == self._score[1]):
            bonus = w["flag_return"]
            rewards = [r + bonus for r in rewards]
            self.totals["flag_return"] += bonus * len(self.spheres)
            self.counts["flag_returns"] += 1

        log = m.death_log()
        new_deaths = log[self._deaths_seen:] if len(log) >= self._deaths_seen else log
        self._deaths_seen = len(log)
        died = {}
        for message in new_deaths:
            if message["team"] == self.team:
                died[message["player"]] = message["reason"]

        for k, sphere in enumerate(self.spheres):
            was_alive, was_carrying, distance = self._state[k]
            if not was_alive:
                continue
            reason = died.get(sphere.player)
            if reason is not None:
                penalty = w["wall_death"] if reason == "wall" else w["player_death"]
                rewards[k] -= penalty
                self.totals["death"] -= penalty
                self.counts["wall_deaths" if reason == "wall" else "player_deaths"] += 1
                if was_carrying:
                    rewards[k] -= w["carrier_death"]
                    self.totals["carrier_death"] -= w["carrier_death"]
                    self.counts["carrier_deaths"] += 1
                continue
            carrying = sphere.carrying_flag is not None
            if carrying and not was_carrying:
                rewards[k] += w["pickup"]
                self.totals["pickup"] += w["pickup"]
                self.counts["pickups"] += 1
            if distance is not None and carrying == was_carrying:
                new_distance = self._objective_distance(sphere)
                if new_distance is not None:
                    progress = w["progress"] * (distance - new_distance) / 180.0
                    rewards[k] += progress
                    self.totals["progress"] += progress
        return rewards
