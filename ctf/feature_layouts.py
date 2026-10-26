"""Frozen feature layouts of earlier encoder versions, and name-based warm starts.

Files written since feature version 5 carry their column names and action
tables. Older checkpoints and policies do not; their layouts are frozen here
(never edit these when ctf/features.py changes).

Migration from any layout to another (`migrate_mlp`, and `ctf.train.migrate_state`
for checkpoints) changes only the input layer, the normalization and the actor's
output rows:

* input columns are matched by name; a new column gets zero weights and identity
  normalization, so it is ignored until training uses it;
* a previous-action one-hot column for a new action value inherits the weights
  and normalization of the nearest old value of the same head, and the input
  bias is corrected so the network's output on old inputs is unchanged;
* actor output rows map to the nearest old action value per head, and the
  `copies` rows that share one old row get -log(copies) on their bias, so the
  total probability of the original action is preserved (a removed value merges
  into its nearest remaining one, so migrating back undoes a migration).

Numpy only (used by ctf.policy_io).
"""

import numpy as np

HEADS = ("throttle", "turn")
GOAL_HEADS = ("heading", "speed")  # the goal action space (ctf/action_space.py)
NORM_CLIP = 5.0  # normalized features are clipped to +-NORM_CLIP (NumpyActor, trainer)

V4_ACTION_TABLES = {
    "throttle": (-1.0, -0.3, -0.1, -0.05, -0.034, 0.0, 0.015, 0.025, 0.034, 0.037, 0.05, 0.06,
                 0.1, 0.3, 1.0),
    "turn": (-1.0, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0),
}
V4_FEATURE_NAMES = (
    'self.x', 'self.y', 'self.z', 'self.vfwd', 'self.vright', 'self.vz', 'self.speed',
    'self.lethal', 'self.yaw_sin', 'self.yaw_cos', 'self.grounded', 'self.carrying',
    'self.edge_px', 'self.edge_nx', 'self.edge_py', 'self.edge_ny', 'self.player1',
    'self.player2', 'self.player3', 'self.gap', 'self.dspeed', 'self.side_known', 'game.tick',
    'game.lead', 'game.mine', 'game.theirs', 'game.dead_mates', 'own_flag.fwd',
    'own_flag.right', 'own_flag.dist', 'own_flag.at_home', 'own_flag.carried', 'own_flag.vfwd',
    'own_flag.vright', 'enemy_flag.seen', 'enemy_flag.fwd', 'enemy_flag.right',
    'enemy_flag.dist', 'enemy_flag.at_home', 'enemy_flag.mine', 'enemy_flag.mate',
    'own_home.known', 'own_home.fwd', 'own_home.right', 'own_home.dist', 'enemy_home.known',
    'enemy_home.fwd', 'enemy_home.right', 'enemy_home.dist', 'enemy0.present', 'enemy0.age',
    'enemy0.fwd', 'enemy0.right', 'enemy0.dist', 'enemy0.rvfwd', 'enemy0.rvright',
    'enemy0.speed', 'enemy0.speed_adv', 'enemy0.carrying', 'enemy0.dz', 'enemy0.yaw_sin',
    'enemy0.yaw_cos', 'enemy0.grounded', 'enemy1.present', 'enemy1.age', 'enemy1.fwd',
    'enemy1.right', 'enemy1.dist', 'enemy1.rvfwd', 'enemy1.rvright', 'enemy1.speed',
    'enemy1.speed_adv', 'enemy1.carrying', 'enemy1.dz', 'enemy1.yaw_sin', 'enemy1.yaw_cos',
    'enemy1.grounded', 'enemy2.present', 'enemy2.age', 'enemy2.fwd', 'enemy2.right',
    'enemy2.dist', 'enemy2.rvfwd', 'enemy2.rvright', 'enemy2.speed', 'enemy2.speed_adv',
    'enemy2.carrying', 'enemy2.dz', 'enemy2.yaw_sin', 'enemy2.yaw_cos', 'enemy2.grounded',
    'bump0.present', 'bump0.fwd', 'bump0.right', 'bump1.present', 'bump1.fwd', 'bump1.right',
    'bump2.present', 'bump2.fwd', 'bump2.right', 'bump3.present', 'bump3.fwd', 'bump3.right',
    'bump4.present', 'bump4.fwd', 'bump4.right', 'bump5.present', 'bump5.fwd', 'bump5.right',
    'bump6.present', 'bump6.fwd', 'bump6.right', 'bump7.present', 'bump7.fwd', 'bump7.right',
    'bump8.present', 'bump8.fwd', 'bump8.right', 'bump9.present', 'bump9.fwd', 'bump9.right',
    'bump10.present', 'bump10.fwd', 'bump10.right', 'bump11.present', 'bump11.fwd',
    'bump11.right', 'danger0.present', 'danger0.fwd', 'danger0.right', 'danger0.age',
    'danger0.not_wall', 'danger1.present', 'danger1.fwd', 'danger1.right', 'danger1.age',
    'danger1.not_wall', 'danger2.present', 'danger2.fwd', 'danger2.right', 'danger2.age',
    'danger2.not_wall', 'danger3.present', 'danger3.fwd', 'danger3.right', 'danger3.age',
    'danger3.not_wall', 'ray.fwd', 'ray.r45', 'ray.l45', 'ray.r90', 'ray.l90', 'ray.r135',
    'ray.l135', 'ray.back', 'ray.motion', 'ray.motion_blocked', 'ray.known_free',
    'team.mate_a_carrying', 'team.mate_b_carrying', 'team.mate_a_respawn',
    'team.mate_b_respawn', 'team.home_enemy_dist', 'team.home_enemy_count', 'team.flag_closing',
    'team.enemy_flag_waited', 'prev.throttle=-1.0', 'prev.throttle=-0.3', 'prev.throttle=-0.1',
    'prev.throttle=-0.05', 'prev.throttle=-0.034', 'prev.throttle=0.0', 'prev.throttle=0.015',
    'prev.throttle=0.025', 'prev.throttle=0.034', 'prev.throttle=0.037', 'prev.throttle=0.05',
    'prev.throttle=0.06', 'prev.throttle=0.1', 'prev.throttle=0.3', 'prev.throttle=1.0',
    'prev.turn=-1.0', 'prev.turn=-0.6', 'prev.turn=-0.3', 'prev.turn=-0.1', 'prev.turn=0.0',
    'prev.turn=0.1', 'prev.turn=0.3', 'prev.turn=0.6', 'prev.turn=1.0', 'prev.jump',
)

# v3: the v4 layout without the team block, with coarser throttle bins.
V3_ACTION_TABLES = {
    "throttle": (-1.0, -0.3, -0.1, -0.03, 0.0, 0.015, 0.025, 0.034, 0.05, 0.1, 0.3, 1.0),
    "turn": V4_ACTION_TABLES["turn"],
}
V3_FEATURE_NAMES = (
    tuple(n for n in V4_FEATURE_NAMES if not n.startswith(("team.", "prev.")))
    + tuple("prev.throttle=%r" % v for v in V3_ACTION_TABLES["throttle"])
    + tuple("prev.turn=%r" % v for v in V3_ACTION_TABLES["turn"]) + ("prev.jump",)
)

LEGACY_LAYOUTS = {3: (V3_FEATURE_NAMES, V3_ACTION_TABLES), 4: (V4_FEATURE_NAMES, V4_ACTION_TABLES)}
assert len(V4_FEATURE_NAMES) == 191 and len(V3_FEATURE_NAMES) == 180


def heads_of(tables):
    """The action heads of a tables mapping: HEADS, or GOAL_HEADS for the goal space."""
    return GOAL_HEADS if "heading" in tables else HEADS


def normalize_tables(tables):
    """{head: tuple of floats} for its heads, from any mapping of sequences (e.g. JSON lists)."""
    return {head: tuple(float(v) for v in tables[head]) for head in heads_of(tables)}


def _common_heads(old_tables, new_tables):
    heads = heads_of(old_tables)
    if heads_of(new_tables) != heads:
        raise ValueError(f"cannot migrate between action spaces ({'/'.join(heads)} -> "
                         f"{'/'.join(heads_of(new_tables))}); train the new space from scratch")
    return heads


def resolve_layout(names=None, tables=None, version=None, dim=None, current=None):
    """(names, tables) of a checkpoint or policy file.

    Stored names win. Otherwise the layout is the frozen one of `version` (or, when
    the version is unknown, of the width `dim`); `current` = (version, names, tables)
    of the running encoder covers unnamed files of the current version.
    """
    if names is not None:
        if tables is None:
            raise ValueError("feature names stored without action tables")
        return tuple(names), normalize_tables(tables)
    if current is not None and version == current[0] and dim in (None, len(current[1])):
        return tuple(current[1]), normalize_tables(current[2])
    for legacy_version, (legacy_names, legacy_tables) in LEGACY_LAYOUTS.items():
        if version == legacy_version or (version is None and dim == len(legacy_names)):
            if dim is not None and dim != len(legacy_names):
                raise ValueError(f"feature version {version} has {len(legacy_names)} features, "
                                 f"not {dim}")
            return legacy_names, normalize_tables(legacy_tables)
    raise ValueError(f"unknown feature layout (version {version}, {dim} features) without names")


def _action_column(name, tables):
    """(head, value) for a previous-action one-hot column such as 'prev.turn=0.3', else None."""
    if not name.startswith("prev.") or "=" not in name:
        return None
    head, _, value = name[5:].partition("=")
    if head not in tables:
        return None
    return head, float(value)


def _nearest(values, value):
    return min(range(len(values)), key=lambda i: abs(values[i] - value))


def column_sources(old_names, old_tables, new_names, new_tables):
    """For each new column, the index of the old column it takes over (or None)."""
    old_tables, new_tables = normalize_tables(old_tables), normalize_tables(new_tables)
    if len(set(old_names)) != len(old_names) or len(set(new_names)) != len(new_names):
        raise ValueError("feature names must be unique")
    index = {name: i for i, name in enumerate(old_names)}
    by_head = {head: [] for head in _common_heads(old_tables, new_tables)}  # old one-hot columns per head: (value, index)
    for i, name in enumerate(old_names):
        action = _action_column(name, old_tables)
        if action is not None:
            by_head[action[0]].append((action[1], i))
    sources = []
    for name in new_names:
        action = _action_column(name, new_tables)
        if action is not None and by_head[action[0]]:
            values = [v for v, _ in by_head[action[0]]]
            sources.append(by_head[action[0]][_nearest(values, action[1])][1])
        else:
            sources.append(index.get(name))
    return sources


def output_rows(old_tables, new_tables):
    """Old logit row of every new logit row, and the shift added to its bias.

    Logits are laid out as the HEADS in order, then the two jump logits. A new row
    copies the row of the nearest old value; the `copies` rows sharing one old row
    get -log(copies), splitting its probability evenly. An old value with no row of
    its own (removed) adds its mass to the row of the nearest new value, assuming
    the two logits were equal (exact when undoing an earlier split): +log(1 + k).
    """
    old_tables, new_tables = normalize_tables(old_tables), normalize_tables(new_tables)
    rows, shift, offset = [], [], 0
    for head in _common_heads(old_tables, new_tables):
        old, new = old_tables[head], new_tables[head]
        nearest = [_nearest(old, v) for v in new]
        copies = np.bincount(nearest, minlength=len(old))
        merged = np.ones(len(new))
        for i, value in enumerate(old):
            if copies[i] == 0:
                merged[_nearest(new, value)] += 1.0
        rows += [offset + i for i in nearest]
        shift += [float(np.log(merged[j]) - np.log(copies[i])) for j, i in enumerate(nearest)]
        offset += len(old)
    rows += [offset, offset + 1]
    shift += [0.0, 0.0]
    return rows, np.asarray(shift)


def migrate_input(weight, bias, mean, std, sources, old_names, old_tables):
    """Input layer `weight` [n_out, old_dim] (float arrays) -> ([n_out, new_dim], bias).

    `mean`, `std`: the old normalization as the network applies it. An old one-hot
    column copied to k new columns contributes its inactive value k - 1 extra times
    (dropped when k = 0), which the bias absorbs; at most one column of a head is
    active, so outputs on old inputs are unchanged. Dropped non-action columns are
    assumed to sit at their mean (normalized value 0).
    """
    weight = np.asarray(weight, dtype=np.float64)
    if weight.shape[1] != len(old_names):
        raise ValueError(f"input layer has {weight.shape[1]} columns, layout has {len(old_names)}")
    tables = normalize_tables(old_tables)
    new = np.zeros((weight.shape[0], len(sources)))
    for j, i in enumerate(sources):
        if i is not None:
            new[:, j] = weight[:, i]
    bias = np.asarray(bias, dtype=np.float64).copy()
    copies = np.bincount([i for i in sources if i is not None], minlength=len(old_names))
    mean32, std32 = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    inactive = np.clip((np.float32(0.0) - mean32) / std32, -NORM_CLIP, NORM_CLIP).astype(np.float64)
    for i, name in enumerate(old_names):
        if copies[i] != 1 and _action_column(name, tables) is not None:
            bias -= (copies[i] - 1) * weight[:, i] * inactive[i]
    return new, bias


def migrate_norm(mean, var, sources):
    """New (mean, var): copied from each source column, identity (0, 1) for new columns."""
    new_mean, new_var = np.zeros(len(sources)), np.ones(len(sources))
    for j, i in enumerate(sources):
        if i is not None:
            new_mean[j], new_var[j] = mean[i], var[i]
    return new_mean, new_var


def migrate_output(weight, bias, old_tables, new_tables):
    """Actor output layer `weight` [n_logits, hidden] and bias, rows re-mapped per head."""
    rows, shift = output_rows(old_tables, new_tables)
    weight, bias = np.asarray(weight), np.asarray(bias)
    old = normalize_tables(old_tables)
    if weight.shape[0] != sum(len(old[h]) for h in heads_of(old)) + 2:
        raise ValueError("output layer does not match the old action tables")
    return weight[rows].copy(), (bias[rows] + shift).astype(bias.dtype)


def same_layout(names_a, tables_a, names_b, tables_b):
    return (tuple(names_a) == tuple(names_b)
            and normalize_tables(tables_a) == normalize_tables(tables_b))


def lossy_changes(old_names, old_tables, new_names, new_tables):
    """Reasons a migration would discard information (dropped columns or action values)."""
    reasons = []
    new_set = set(new_names)
    for name in old_names:
        if _action_column(name, old_tables) is None and name not in new_set:
            reasons.append(f"feature {name!r} dropped")
    old_t, new_t = normalize_tables(old_tables), normalize_tables(new_tables)
    for head in _common_heads(old_t, new_t):
        nearest = {_nearest(old_t[head], v) for v in new_t[head]}
        for i, value in enumerate(old_t[head]):
            if i not in nearest:
                reasons.append(f"{head} value {value} has no successor")
    return reasons


def migrate_mlp(layers, mean, std, old_names, old_tables, new_names, new_tables,
                allow_lossy=False):
    """Migrate an MLP policy: `layers` [(W [n_in, n_out], b)], normalization mean/std.

    Returns (layers, mean, std) for the new layout, with the input dtypes. Only
    refinements (added features, split action values) are exact; contractions are
    refused unless `allow_lossy`, since merged logits are only approximately right.
    Note: splitting action values preserves each original value's probability mass,
    not the greedy action, so a migrated policy is a warm start, not a replica.
    """
    reasons = lossy_changes(old_names, old_tables, new_names, new_tables)
    if reasons and not allow_lossy:
        raise ValueError("lossy migration refused: " + "; ".join(reasons[:5]))
    sources = column_sources(old_names, old_tables, new_names, new_tables)
    mean, std = np.asarray(mean), np.asarray(std)
    layers = [(np.asarray(w), np.asarray(b)) for w, b in layers]
    w0, b0 = layers[0]
    w_new, b_new = migrate_input(w0.T, b0, mean, std, sources, old_names, old_tables)
    layers[0] = (w_new.T.astype(w0.dtype), b_new.astype(b0.dtype))
    w_last, b_last = layers[-1]
    w_out, b_out = migrate_output(w_last.T, b_last, old_tables, new_tables)
    layers[-1] = (w_out.T.copy(), b_out)
    new_mean, new_var = migrate_norm(mean.astype(np.float64), std.astype(np.float64) ** 2, sources)
    return layers, new_mean.astype(mean.dtype), np.sqrt(new_var).astype(std.dtype)
