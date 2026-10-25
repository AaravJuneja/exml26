"""Per-tick training rewards for the learning team.

Every term is computable from documented information: the score, the player's
own state (dead players use their spawn point, the first position they ever
observed), flag records with carrier ids, observed flag homes, and death
messages. Simulator objects are read only because they hold exactly those values.

Shaping is exactly potential-based, F = gamma * Phi(after) - Phi(before), applied
on every tick (deaths, respawns and captures included) with Phi = 0 at the end of
the match, so it cannot change which policy is optimal; it only densifies credit.
The potential is minus the remaining route length for the player:
  * carrying the enemy flag: distance to own home;
  * a teammate carries it: the public flag position's distance to own home;
  * otherwise: distance to the enemy flag plus that flag's distance to own home.
Positions come only from documented fields: own position, public flag
coordinates (which lag a fresh carrier by one tick, exactly as observed), the
observed home and the player's own spawn point.
"""

import math

COMPONENTS = ("score", "win", "pickup", "death", "carrier_death", "flag_return", "shaping")

DEFAULT_WEIGHTS = {
    "score": 1.0,          # per point, own minus enemy, to every learner
    "win": 1.0,            # terminal, to every learner, only for a win
    "pickup": 0.1,         # this player starts carrying the enemy flag
    "wall_death": 0.3,     # penalty
    "player_death": 0.1,   # penalty
    "carrier_death": 0.25, # extra penalty for dying while carrying
    "flag_return": 0.2,    # the enemy carrier of our flag is eliminated (team)
    "shaping": 0.5,        # potential scale: Phi = -shaping * route_length / 180
    "gamma": 0.998,        # must equal the trainer's discount for exact PBRS
}


class RewardTracker:
    def __init__(self, match, team, weights):
        self.match = match
        self.team = team
        self.enemy = next(name for name in match.team_names if name != team)
        self.weights = dict(DEFAULT_WEIGHTS, **weights)
        self.spheres = sorted((s for s in match.spheres if s.team == team), key=lambda s: s.player)
        self.own_flag = next(f for f in match.flags if f.team == team)
        self.enemy_flag = next(f for f in match.flags if f.team == self.enemy)
        self.home = (self.own_flag.home_x, self.own_flag.home_y)
        self.spawns = [(s.start_x, s.start_y) for s in self.spheres]
        self._seen_deaths = set()
        self.totals = {name: 0.0 for name in COMPONENTS}
        self.counts = {"pickups": 0, "wall_deaths": 0, "player_deaths": 0,
                       "carrier_deaths": 0, "flag_returns": 0}

    def _potentials(self):
        hx, hy = self.home
        flag = self.enemy_flag
        carrier = flag.carrier
        scale = -self.weights["shaping"] / 180.0
        out = []
        for k, sphere in enumerate(self.spheres):
            x, y = (sphere.x, sphere.y) if sphere.alive else self.spawns[k]
            if carrier is not None and carrier.team == self.team:
                if carrier is sphere:
                    route = math.hypot(x - hx, y - hy)
                else:
                    route = math.hypot(flag.x - hx, flag.y - hy)
            else:
                route = math.hypot(x - flag.x, y - flag.y) + math.hypot(flag.x - hx, flag.y - hy)
            out.append(scale * route)
        return out

    def before(self):
        """Snapshot taken after observe() and before act()."""
        m = self.match
        self._score = (m.score[self.team], m.score[self.enemy])
        self._own_carrier = self.own_flag.carrier.id if self.own_flag.carrier else None
        self._state = [(s.alive, s.carrying_flag is not None) for s in self.spheres]
        self._phi = self._potentials()

    def _new_deaths(self):
        """Own-team deaths logged during the last act() (death log entries carry the tick)."""
        tick = self.match.tick - 1
        found = {}
        for message in reversed(self.match.death_log()):
            if message["tick"] < tick:
                break
            key = (message["team"], message["player"], message["tick"])
            if message["team"] == self.team and key not in self._seen_deaths:
                self._seen_deaths.add(key)
                found[message["player"]] = message["reason"]
        return found

    def after(self):
        """Rewards for the transition just simulated, one value per learner slot."""
        w = self.weights
        m = self.match
        own, enemy = m.score[self.team], m.score[self.enemy]
        scored = own - self._score[0]
        score = w["score"] * (scored - (enemy - self._score[1]))
        rewards = [score] * len(self.spheres)
        self.totals["score"] += score * len(self.spheres)

        own_carrier = self.own_flag.carrier.id if self.own_flag.carrier else None
        if (self._own_carrier is not None and own_carrier != self._own_carrier
                and enemy == self._score[1]):
            rewards = [r + w["flag_return"] for r in rewards]
            self.totals["flag_return"] += w["flag_return"] * len(self.spheres)
            self.counts["flag_returns"] += 1

        died = self._new_deaths()
        for k, sphere in enumerate(self.spheres):
            was_alive, was_carrying = self._state[k]
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
            elif sphere.carrying_flag is not None and not was_carrying:
                rewards[k] += w["pickup"]
                self.totals["pickup"] += w["pickup"]
                self.counts["pickups"] += 1

        terminal = m.tick >= m.max_ticks
        phi = [0.0] * len(self.spheres) if terminal else self._potentials()
        gamma = w["gamma"]
        for k in range(len(self.spheres)):
            shaping = gamma * phi[k] - self._phi[k]
            rewards[k] += shaping
            self.totals["shaping"] += shaping

        if terminal and own > enemy:
            rewards = [r + w["win"] for r in rewards]
            self.totals["win"] += w["win"] * len(self.spheres)
        return rewards
