"""Rollout collection: full matches played in lockstep by one worker process.

Each worker owns several `Match` instances. Every tick it encodes all alive
learner players, runs one batched numpy forward pass, steps opponents and
physics, and records per-tick rewards. Matches always run to the end
(6,000 ticks), so every trajectory ends at a true terminal.

Recurrent actors keep one hidden state per player. It advances only on that
player's alive decisions (a deployed player gets no callbacks while dead) and
is zeroed exactly when its PlayerMemory starts fresh: at match start and on a
simulated memory wipe (a worker restart). It is NOT reset on respawn, because a
deployed process keeps its module memory across deaths. Training data records
the reset flags and, every `hidden_every` decisions after a reset, the hidden
state before the decision, so the learner can rebuild exact sequences.
"""

import time

import numpy as np

from ctf import features as F
from ctf.npnets import PolicyMemories, make_actor, memory_is_fresh, rows_to_float32
from ctf.policies import SCRIPTED, ScriptedTeam, _silent, load_decide
from ctf.policy_io import load_policy
from ctf.rewards import COMPONENTS, RewardTracker
from ctf.sim import DOCUMENTED_KEYS, Match


def documented(observation):
    return {key: observation[key] for key in DOCUMENTED_KEYS}


class PolicyTeam:
    """A frozen learned opponent: batched actor over three private memories (and hidden states)."""

    def __init__(self, actor, sphere_ids, rng, greedy=False):
        self.actor = actor
        self.ids = list(sphere_ids)
        self.state = PolicyMemories(actor, self.ids, F.encode, F.PlayerMemory)
        self.memories = self.state.memories
        self.rng = rng
        self.greedy = greedy

    def act(self, observations):
        ids = [sid for sid in self.ids if sid in observations]
        if not ids:
            return {}
        x = self.state.encode_batch(ids, [documented(observations[sid]) for sid in ids])
        actions = self.state.act(ids, x, self.rng, greedy=self.greedy)
        return {sid: (F.THROTTLE_VALUES[t], F.TURN_VALUES[r], bool(j))
                for sid, (t, r, j) in zip(ids, actions.tolist())}


_POLICY_CACHE = {}


def make_opponent(spec, sphere_ids, rng, current=None):
    """spec: scripted name/path, 'self' (current policy, sampled),
    'policy:/path/file.npz' (sampled) or 'greedy:/path/file.npz'."""
    if spec == "self":
        return PolicyTeam(current, sphere_ids, rng)
    if spec.startswith(("policy:", "greedy:")):
        kind, path = spec.split(":", 1)
        if path not in _POLICY_CACHE:
            _POLICY_CACHE[path] = load_policy(path)[0]
        return PolicyTeam(_POLICY_CACHE[path], sphere_ids, rng, greedy=kind == "greedy")
    return ScriptedTeam(spec, sphere_ids)


def _nearest_index(table, value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    value = -1.0 if value < -1.0 else 1.0 if value > 1.0 else value
    best, best_d = 0, abs(table[0] - value)
    for i in range(1, len(table)):
        d = abs(table[i] - value)
        if d < best_d:
            best, best_d = i, d
    return best


def _silenced(decide):
    decide.__globals__["print"] = _silent
    return decide


def teacher_label(decide, observation):
    """A scripted teacher's action for this documented observation, as bin indices."""
    try:
        action = decide(observation)
        return (_nearest_index(F.THROTTLE_VALUES, action[0]),
                _nearest_index(F.TURN_VALUES, action[1]), 1 if action[2] else 0)
    except Exception:
        return None


def collect(actor, jobs, reward_weights, max_ticks, seed, memory_wipe_prob=0.0, hidden_every=0,
            teacher=None):
    """Play every job to completion. jobs: [{'seed', 'side', 'opponent'}].

    Opponent 'mirror' means the current policy plays both teams and both teams are
    learners: that job yields two learner units sharing one match.

    With `teacher` (a scripted bot name or path), every learner player has its own
    shadow instance of the teacher, which sees the same documented observation and
    labels the action it would take (rounded to the action bins): `teacher_actions`
    [T, units, 3, 3] and `teacher_valid` [T, units, 3]. The teacher never acts.

    For a recurrent actor the result also has `resets` [T, units, 3] (hidden state
    zeroed before this decision) and the stored states `h0` [K, H] with their
    grid positions `h0_t`, `h0_unit`, `h0_slot` [K]: the state before every
    decision whose index since the last reset is a multiple of `hidden_every`.
    """
    rng = np.random.default_rng(seed)
    T, D = max_ticks, F.FEATURE_DIM

    matches, opponents = [], []
    units = []  # dicts: match index, tracker, sphere ids, memories, wipe ticks, job
    for job in jobs:
        match = Match(job["seed"], max_ticks=T, teammates_field=True)
        m_index = len(matches)
        matches.append(match)
        red, blue = match.team_names
        if job["opponent"] == "mirror":
            teams = [red, blue]
            opponents.append(None)
        else:
            team = red if job["side"] == "red" else blue
            teams = [team]
            theirs = [s.id for s in match.spheres if s.team != team]
            opponents.append(make_opponent(job["opponent"], theirs, rng, current=actor))
        for team in teams:
            mine = sorted((s for s in match.spheres if s.team == team), key=lambda s: s.player)
            units.append({
                "match": m_index,
                "tracker": RewardTracker(match, team, reward_weights),
                "ids": [s.id for s in mine],
                "memories": [F.PlayerMemory() for _ in mine],
                "wipes": [int(rng.integers(1, T)) if rng.random() < memory_wipe_prob else -1
                          for _ in mine],
                "job": dict(job, side="red" if team == red else "blue"),
                "teachers": ([_silenced(load_decide(SCRIPTED.get(teacher, teacher)))
                              for _ in mine] if teacher else None),
            })

    n_unit = len(units)
    obs = np.zeros((T, n_unit, 3, D), dtype=np.float32)
    actions = np.zeros((T, n_unit, 3, 3), dtype=np.int8)
    logp = np.zeros((T, n_unit, 3), dtype=np.float32)
    rewards = np.zeros((T, n_unit, 3), dtype=np.float32)
    alive = np.zeros((T, n_unit, 3), dtype=bool)
    if teacher:
        teacher_actions = np.zeros((T, n_unit, 3, 3), dtype=np.int8)
        teacher_valid = np.zeros((T, n_unit, 3), dtype=bool)
    recurrent = actor.recurrent
    if recurrent:
        if hidden_every < 1:
            raise ValueError("a recurrent actor needs hidden_every >= 1")
        hidden = np.zeros((n_unit, 3, actor.hidden_size), dtype=np.float32)
        since_reset = np.zeros((n_unit, 3), dtype=np.int64)
        resets = np.zeros((T, n_unit, 3), dtype=bool)
        stored = []  # (t, unit, slot, hidden state before the decision)

    started = time.perf_counter()
    for t in range(T):
        all_obs = [match.observe() for match in matches]
        rows, unit_index, slot_index = [], [], []
        for u, unit in enumerate(units):
            unit["tracker"].before()
            observations = all_obs[unit["match"]]
            for k, sid in enumerate(unit["ids"]):
                observation = observations.get(sid)
                if observation is None:
                    continue
                memory = unit["memories"][k]
                if unit["wipes"][k] == t:
                    memory.reset()
                if recurrent:
                    if memory_is_fresh(memory, observation):
                        hidden[u, k] = 0.0
                        since_reset[u, k] = 0
                        resets[t, u, k] = True
                    if since_reset[u, k] % hidden_every == 0:
                        stored.append((t, u, k, hidden[u, k].copy()))
                    since_reset[u, k] += 1
                doc = documented(observation)
                rows.append(F.encode(doc, unit["memories"][k]))
                if teacher:
                    label = teacher_label(unit["teachers"][k], doc)
                    if label is not None:
                        teacher_actions[t, u, k] = label
                        teacher_valid[t, u, k] = True
                unit_index.append(u)
                slot_index.append(k)
        step_actions = [dict() for _ in matches]
        if rows:
            x = rows_to_float32(rows)
            ui, si = np.asarray(unit_index), np.asarray(slot_index)
            if recurrent:
                chosen, chosen_logp, hidden[ui, si] = actor.step_sample(x, hidden[ui, si], rng)
            else:
                chosen, chosen_logp = actor.sample(x, rng)
            if len(rows) == 3 * n_unit:  # everyone alive: rows are in (unit, slot) order
                obs[t] = x.reshape(n_unit, 3, D)
                actions[t] = chosen.reshape(n_unit, 3, 3)
                logp[t] = chosen_logp.reshape(n_unit, 3)
                alive[t] = True
            else:
                obs[t, ui, si] = x
                actions[t, ui, si] = chosen
                logp[t, ui, si] = chosen_logp
                alive[t, ui, si] = True
            for u, k, (a_t, a_r, a_j) in zip(unit_index, slot_index, chosen.tolist()):
                unit = units[u]
                unit["memories"][k].set_action(a_t, a_r, a_j)
                step_actions[unit["match"]][unit["ids"][k]] = (
                    F.THROTTLE_VALUES[a_t], F.TURN_VALUES[a_r], bool(a_j))
        for m_index, match in enumerate(matches):
            if opponents[m_index] is not None:
                step_actions[m_index].update(opponents[m_index].act(all_obs[m_index]))
            match.act(step_actions[m_index])
        rewards[t] = [unit["tracker"].after() for unit in units]

    stats = []
    for unit in units:
        tracker = unit["tracker"]
        match = matches[unit["match"]]
        own, enemy = match.score[tracker.team], match.score[tracker.enemy]
        stats.append({
            **unit["job"],
            "goals_for": own,
            "goals_against": enemy,
            "result": "win" if own > enemy else "loss" if own < enemy else "draw",
            "reward": {name: tracker.totals[name] for name in COMPONENTS},
            "enemy_deaths": sum(1 for m in match.death_log() if m["team"] == tracker.enemy),
            **tracker.counts,
        })
    out = {
        "obs": obs, "actions": actions, "logp": logp, "rewards": rewards, "alive": alive,
        "stats": stats, "seconds": time.perf_counter() - started, "ticks": T * len(matches),
    }
    if teacher:
        out["teacher_actions"] = teacher_actions
        out["teacher_valid"] = teacher_valid
    if recurrent:
        out["resets"] = resets
        out["h0_t"] = np.array([s[0] for s in stored], dtype=np.int64)
        out["h0_unit"] = np.array([s[1] for s in stored], dtype=np.int64)
        out["h0_slot"] = np.array([s[2] for s in stored], dtype=np.int64)
        out["h0"] = (np.stack([s[3] for s in stored]) if stored
                     else np.zeros((0, actor.hidden_size), dtype=np.float32))
    return out


def worker_main(conn, worker_id):
    """Process loop: receive (weights, mean, std, jobs, config), send collected data."""
    import os
    import traceback
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    while True:
        try:
            message = conn.recv()
        except EOFError:
            break
        if message is None:
            break
        try:
            weights, mean, std, jobs, config = message
            actor = make_actor(weights, mean, std)
            data = collect(actor, jobs, config["reward_weights"], config["max_ticks"],
                           config["seed"], config.get("memory_wipe_prob", 0.0),
                           config.get("hidden_every", 0), config.get("teacher"))
        except Exception:
            data = {"error": traceback.format_exc()}
        try:
            conn.send(data)
        except (BrokenPipeError, OSError):
            break
