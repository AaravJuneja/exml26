"""Rollout collection: full matches played in lockstep by one worker process.

Each worker owns several `Match` instances. Every tick it encodes all alive
learner players, runs one batched numpy forward pass, steps opponents and
physics, and records per-tick rewards. Matches always run to the end
(6,000 ticks), so every trajectory ends at a true terminal.
"""

import time

import numpy as np

from ctf import features as F
from ctf.nets import NumpyActor
from ctf.policies import ScriptedTeam
from ctf.policy_io import load_policy
from ctf.rewards import COMPONENTS, RewardTracker
from ctf.sim import DOCUMENTED_KEYS, Match


def documented(observation):
    return {key: observation[key] for key in DOCUMENTED_KEYS}


class PolicyTeam:
    """A frozen learned opponent: batched NumpyActor over three private memories."""

    def __init__(self, actor, sphere_ids, rng, greedy=False):
        self.actor = actor
        self.ids = list(sphere_ids)
        self.memories = {sid: F.PlayerMemory() for sid in self.ids}
        self.rng = rng
        self.greedy = greedy

    def act(self, observations):
        ids = [sid for sid in self.ids if sid in observations]
        if not ids:
            return {}
        x = np.asarray([F.encode(documented(observations[sid]), self.memories[sid]) for sid in ids],
                       dtype=np.float32)
        actions, _ = self.actor.sample(x, self.rng, greedy=self.greedy)
        out = {}
        for sid, (t, r, j) in zip(ids, actions):
            self.memories[sid].set_action(int(t), int(r), int(j))
            out[sid] = (F.THROTTLE_VALUES[t], F.TURN_VALUES[r], bool(j))
        return out


_POLICY_CACHE = {}


def make_opponent(spec, sphere_ids, rng):
    """spec: scripted name/path, or 'policy:/path/file.npz' (sampled), 'greedy:/path' ."""
    if spec.startswith(("policy:", "greedy:")):
        kind, path = spec.split(":", 1)
        if path not in _POLICY_CACHE:
            _POLICY_CACHE[path] = load_policy(path)[0]
        return PolicyTeam(_POLICY_CACHE[path], sphere_ids, rng, greedy=kind == "greedy")
    return ScriptedTeam(spec, sphere_ids)


def collect(actor, jobs, reward_weights, max_ticks, seed, memory_wipe_prob=0.0):
    """Play every job to completion. jobs: [{'seed', 'side', 'opponent'}]."""
    rng = np.random.default_rng(seed)
    n_env = len(jobs)
    T, D = max_ticks, F.FEATURE_DIM
    obs = np.zeros((T, n_env, 3, D), dtype=np.float32)
    actions = np.zeros((T, n_env, 3, 3), dtype=np.int8)
    logp = np.zeros((T, n_env, 3), dtype=np.float32)
    rewards = np.zeros((T, n_env, 3), dtype=np.float32)
    alive = np.zeros((T, n_env, 3), dtype=bool)

    matches, trackers, learner_ids, memories, opponents, wipes = [], [], [], [], [], []
    for job in jobs:
        match = Match(job["seed"], max_ticks=T, teammates_field=True)
        learner_team = match.team_names[0] if job["side"] == "red" else match.team_names[1]
        mine = sorted((s for s in match.spheres if s.team == learner_team), key=lambda s: s.player)
        theirs = [s.id for s in match.spheres if s.team != learner_team]
        matches.append(match)
        trackers.append(RewardTracker(match, learner_team, reward_weights))
        learner_ids.append([s.id for s in mine])
        memories.append([F.PlayerMemory() for _ in mine])
        opponents.append(make_opponent(job["opponent"], theirs, rng))
        wipes.append([int(rng.integers(1, T)) if rng.random() < memory_wipe_prob else -1
                      for _ in mine])

    started = time.perf_counter()
    for t in range(T):
        rows, env_index, slot_index, all_obs = [], [], [], []
        for e, match in enumerate(matches):
            observations = match.observe()
            all_obs.append(observations)
            trackers[e].before()
            for k, sid in enumerate(learner_ids[e]):
                observation = observations.get(sid)
                if observation is None:
                    continue
                if wipes[e][k] == t:
                    memories[e][k].reset()
                rows.append(F.encode(documented(observation), memories[e][k]))
                env_index.append(e)
                slot_index.append(k)
        learner_actions = [dict() for _ in matches]
        if rows:
            x = np.asarray(rows, dtype=np.float32)
            chosen, chosen_logp = actor.sample(x, rng)
            ei, si = np.asarray(env_index), np.asarray(slot_index)
            obs[t, ei, si] = x
            actions[t, ei, si] = chosen
            logp[t, ei, si] = chosen_logp
            alive[t, ei, si] = True
            for i, (e, k) in enumerate(zip(env_index, slot_index)):
                a_t, a_r, a_j = int(chosen[i, 0]), int(chosen[i, 1]), int(chosen[i, 2])
                memories[e][k].set_action(a_t, a_r, a_j)
                learner_actions[e][learner_ids[e][k]] = (
                    F.THROTTLE_VALUES[a_t], F.TURN_VALUES[a_r], bool(a_j))
        for e, match in enumerate(matches):
            step_actions = opponents[e].act(all_obs[e])
            step_actions.update(learner_actions[e])
            match.act(step_actions)
            rewards[t, e] = trackers[e].after()

    stats = []
    for job, match, tracker in zip(jobs, matches, trackers):
        own, enemy = match.score[tracker.team], match.score[tracker.enemy]
        enemy_deaths = sum(1 for m in match.death_log() if m["team"] == tracker.enemy)
        stats.append({
            **job,
            "goals_for": own,
            "goals_against": enemy,
            "result": "win" if own > enemy else "loss" if own < enemy else "draw",
            "reward": {name: tracker.totals[name] for name in COMPONENTS},
            "enemy_deaths": enemy_deaths,
            **tracker.counts,
        })
    return {
        "obs": obs, "actions": actions, "logp": logp, "rewards": rewards, "alive": alive,
        "stats": stats, "seconds": time.perf_counter() - started, "ticks": T * n_env,
    }


def worker_main(conn, worker_id):
    """Process loop: receive (weights, mean, std, jobs, config), send collected data."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    while True:
        message = conn.recv()
        if message is None:
            break
        weights, mean, std, jobs, config = message
        actor = NumpyActor(weights, mean, std)
        data = collect(actor, jobs, config["reward_weights"], config["max_ticks"],
                       config["seed"], config.get("memory_wipe_prob", 0.0))
        conn.send(data)
