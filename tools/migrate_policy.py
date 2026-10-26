"""Migrate an MLP policy file (.npz) to the current feature/action layout.

    python tools/migrate_policy.py OLD.npz NEW.npz

Works from any layout: files with stored feature names (version 5+) and the
frozen v3/v4 layouts of unnamed files (see ctf/feature_layouts.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ctf.policy_io import migrate_policy_arrays, read_policy, save_policy  # noqa: E402

KEEP = ("run", "update", "samples", "notes")


def main(src, dst):
    arch, weights, mean, std, meta = read_policy(src)
    if arch != "mlp":
        raise SystemExit(f"{src}: only MLP policies can be migrated (arch {arch!r})")
    weights, mean, std, new_meta = migrate_policy_arrays(weights, mean, std, meta)
    origin = new_meta.get("migrated_from", {"feature_version": meta.get("feature_version"),
                                            "feature_dim": len(mean)})
    save_policy(dst, weights, mean, std, migrated_from=dict(origin, path=str(src)),
                **{k: v for k, v in meta.items() if k in KEEP})
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
