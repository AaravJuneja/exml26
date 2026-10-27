import numpy as np
import pytest

from ctf.policy_io import read_policy, save_policy
from tools import average_policies as AP


def _policy(path, seed, mean_shift=0.0, space="goal"):
    from ctf import action_space as A
    rng = np.random.default_rng(seed)
    dim = A.get(space).feature_dim
    sizes = [dim, 8, sum(A.get(space).head_sizes)]
    weights = [(rng.normal(size=(a, b)).astype(np.float32), rng.normal(size=b).astype(np.float32))
               for a, b in zip(sizes[:-1], sizes[1:])]
    save_policy(path, weights, np.zeros(dim) + mean_shift, np.ones(dim), space=space, run="t")
    return weights


def test_average_of_copies_is_identity(tmp_path):
    w = _policy(tmp_path / "a.npz", 0)
    _policy(tmp_path / "b.npz", 0)
    AP.main([str(tmp_path / "avg.npz"), str(tmp_path / "a.npz"), str(tmp_path / "b.npz")])
    arch, avg, _, _, meta = read_policy(tmp_path / "avg.npz")
    assert arch == "mlp" and meta["action_space"] == "goal"
    for (w1, b1), (w2, b2) in zip(w, avg):
        np.testing.assert_allclose(w1, w2, rtol=0, atol=1e-6)
        np.testing.assert_allclose(b1, b2, rtol=0, atol=1e-6)


def test_average_is_mean(tmp_path):
    w1 = _policy(tmp_path / "a.npz", 1)
    w2 = _policy(tmp_path / "b.npz", 2)
    AP.main([str(tmp_path / "avg.npz"), str(tmp_path / "a.npz"), str(tmp_path / "b.npz")])
    avg = read_policy(tmp_path / "avg.npz")[1]
    np.testing.assert_allclose(avg[0][0], (w1[0][0] + w2[0][0]) / 2, atol=1e-6)


def test_average_rejects_different_normalisation(tmp_path):
    _policy(tmp_path / "a.npz", 1)
    _policy(tmp_path / "b.npz", 2, mean_shift=1.0)
    with pytest.raises(ValueError, match="normalisation"):
        AP.main([str(tmp_path / "avg.npz"), str(tmp_path / "a.npz"), str(tmp_path / "b.npz")])
