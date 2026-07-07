"""Smoke/correctness tests for the three experiment ports (pytest or direct run)."""

import jax
import jax.numpy as jnp
import numpy as np

from sghmc_jax.figures import figure1, figure2, figure3
from sghmc_jax.mf import data as mf_data
from sghmc_jax.mf import train as mf_train


def test_figure1_densities_normalized():
    res = figure1.run_target("func4", nsample=500, backend="jax")
    for name, y in res.items():
        assert y.shape == figure1.GRID.shape
        np.testing.assert_allclose(y.sum() * figure1.XSTEP, 1.0, rtol=1e-5), name


def test_figure2_traces():
    res = figure2.run()
    assert set(res) == {"hamiltonian_noisy", "hamiltonian_noisy_resample",
                        "friction_noisy", "hamiltonian_exact"}
    for tr in res.values():
        assert tr.shape == (figure2.NSTEP, 2)
    # exact Hamiltonian dynamics stays on the energy shell |z| = 1
    r = np.sqrt((res["hamiltonian_exact"] ** 2).sum(axis=1))
    np.testing.assert_allclose(r, 1.0, atol=5e-3)


def test_figure3_auc_time_iid_is_one():
    """For iid N(0,1) draws rho_j ~ 0 for j >= 1, so aucTime must return ~1."""
    rng = np.random.default_rng(0)
    taus = figure3.auc_time(rng.normal(size=(50_000, 2)))
    assert taus.shape == (2,)
    assert np.all(taus < 4.0), taus


def test_figure3_cov_err_zero_for_exact_target():
    rng = np.random.default_rng(1)
    data = rng.multivariate_normal(np.zeros(2), figure3.COV_S, size=400_000)
    assert figure3.cov_err(data) < 0.01


def test_mf_fold_assignment_snapshot():
    """Regression snapshot of the py2 randint(1,5) emulation from seed 0."""
    fa = mf_data.fold_assignment(20)
    expected = [1 + int(x * 5) for x in _py_random_seq(20)]
    assert list(fa) == expected
    counts = np.bincount(mf_data.fold_assignment(100_000), minlength=6)[1:]
    assert counts.sum() == 100_000
    assert np.all(np.abs(counts - 20_000) < 600), counts


def _py_random_seq(n):
    import random

    rng = random.Random(0)
    return [rng.random() for _ in range(n)]


def test_mf_kernels_sgld_backends_match():
    """MF SGLD: the blackjax mapping is exact, steps must match to float tolerance."""
    steps = {b: mf_train.make_kernel("sgld", b, 8.0, 0.05, 1.0, 1000)
             for b in ("jax", "blackjax")}
    params = {k: jnp.ones((4, 3)) if k.startswith("W") else jnp.ones((4,))
              for k in mf_train.GROUPS}
    moms = jax.tree_util.tree_map(jnp.zeros_like, params)
    g = jax.tree_util.tree_map(lambda t: 0.1 * t, params)
    key = jax.random.key(0)
    outs = {}
    for b, step in steps.items():
        p, _ = step(key, params, moms, g, jnp.asarray(True))
        outs[b] = p
    for a, b in zip(jax.tree_util.tree_leaves(outs["jax"]),
                    jax.tree_util.tree_leaves(outs["blackjax"])):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-7)


def test_mf_sghmc_kernel_noiseless_matches_cpp_update():
    """SGHMC kernel with noise off: p' = exp(-mdecay) p + eta g; theta' += p'."""
    num_train = 1000
    step = mf_train.make_kernel("sghmc", "jax", 0.4, 0.05, 1.0, num_train)
    params = {k: jnp.ones((2, 2)) if k.startswith("W") else jnp.ones((2,))
              for k in mf_train.GROUPS}
    moms = jax.tree_util.tree_map(lambda t: 0.5 * jnp.ones_like(t), params)
    g = jax.tree_util.tree_map(lambda t: 2.0 * jnp.ones_like(t), params)
    p2, m2 = step(jax.random.key(0), params, moms, g, jnp.asarray(False))
    eta = 0.4 / num_train
    exp_m = np.exp(-0.05) * 0.5 + eta * 2.0
    np.testing.assert_allclose(np.asarray(m2["u_bias"]), exp_m, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(p2["u_bias"]), 1.0 + exp_m, rtol=1e-6)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
