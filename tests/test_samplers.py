"""Unit tests for the sampler kernels (runnable directly or via pytest).

Run:  JAX_PLATFORMS=cpu .venv/bin/python tests/test_samplers.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from sghmc_jax.samplers import blackjax_backend, jax_backend


def _exact_grad(_key, x):
    return x  # gradU of U = x^2/2 (standard normal target)


def test_sgld_backends_bit_equivalent():
    """Same key/state/grad must give (near) identical SGLD steps on both backends."""
    step_jax = jax_backend.make_sgld(eta=0.05, v_hat=1.0)
    step_bjx = blackjax_backend.make_sgld(eta=0.05, v_hat=1.0)
    key = jax.random.key(0)
    x = {"a": jnp.arange(3.0), "b": jnp.ones((2, 2))}
    g = jax.tree_util.tree_map(lambda t: 0.3 * t + 1.0, x)
    out_jax = step_jax(key, x, g)
    out_bjx = step_bjx(key, x, g)
    for a, b in zip(jax.tree_util.tree_leaves(out_jax), jax.tree_util.tree_leaves(out_bjx)):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def _run_chain_sghmc(make, eta, alpha, v_hat, n_steps, seed):
    init, step = make(eta, alpha, v_hat)
    key = jax.random.key(seed)
    key, sub = jax.random.split(key)
    x0 = jnp.zeros(())
    p0 = init(sub, x0)

    def body(carry, subkey):
        x, p = carry
        k_grad, k_step = jax.random.split(subkey)
        grad = x + jax.random.normal(k_grad)  # noisy gradient, V-ish = 1
        x, p = step(k_step, x, p, grad)
        return (x, p), x

    _, xs = jax.lax.scan(body, (x0, p0), jax.random.split(key, n_steps))
    return xs


def _sghmc_lyapunov_var(eta, alpha, v_hat, ordering):
    """Stationary Var(x) of the linearized SGHMC chain on N(0,1), by fixed point.

    ordering="pq" is the original's (momentum first, position uses the fresh
    momentum); "qp" is blackjax's (position uses the previous momentum, and
    momentum noise reaches the position only on the next step).
    """
    q = eta**2 * v_hat + 2.0 * eta * (alpha - 0.5 * v_hat * eta)
    if ordering == "pq":
        A = np.array([[1.0 - eta, 1.0 - alpha], [-eta, 1.0 - alpha]])
        B = np.array([[1.0], [1.0]])
    else:
        A = np.array([[1.0, 1.0], [-eta, 1.0 - alpha]])
        B = np.array([[0.0], [1.0]])
    S = np.eye(2)
    for _ in range(200_000):
        S = A @ S @ A.T + q * B @ B.T
    return S[0, 0]


def test_sghmc_backends_match_their_stationary_variance():
    """Each backend's chain on N(0,1) must match its analytic stationary variance.

    The two backends are the same Euler scheme under different splittings:
    the original/jax "pq" ordering has Var(x) ~ 1 + O(eta), while blackjax's
    "qp" ordering inflates the variance by ~ alpha / (alpha - eta) (a
    systematic property of the blackjax diffusion, ~1.25x here). Verifying
    each against its own Lyapunov solution confirms the parameter mapping in
    the wrapper is exact.
    """
    n = 400_000
    for make, ordering in (
        (jax_backend.make_sghmc, "pq"),
        (blackjax_backend.make_sghmc, "qp"),
    ):
        xs = np.asarray(_run_chain_sghmc(make, 0.01, 0.05, 1.0, n, 0)[n // 10:])
        expected = _sghmc_lyapunov_var(0.01, 0.05, 1.0, ordering)
        assert abs(xs.mean()) < 0.08, (ordering, xs.mean())
        assert abs(xs.var() - expected) < 0.09, (ordering, xs.var(), expected)


def test_sghmc_blackjax_marginal_at_eta_equals_alpha():
    """Documented caveat: the blackjax splitting does not contract at eta = alpha.

    The chain's variance must blow up well past the target variance while the
    faithful jax backend stays near 1.
    """
    n = 100_000
    var_jax = float(np.asarray(_run_chain_sghmc(jax_backend.make_sghmc, 0.05, 0.05, 1.0, n, 0)).var())
    var_bjx = float(np.asarray(_run_chain_sghmc(blackjax_backend.make_sghmc, 0.05, 0.05, 1.0, n, 0)).var())
    assert abs(var_jax - 1.0) < 0.1, var_jax
    assert var_bjx > 10.0, var_bjx


def test_sghmc_rejects_too_big_eta():
    for backend in (jax_backend, blackjax_backend):
        try:
            backend.make_sghmc(eta=1.0, alpha=0.01, v_hat=1.0)
        except ValueError:
            continue
        raise AssertionError(f"{backend.__name__} accepted beta_hat > alpha")


def test_hmc_samples_standard_normal():
    u_fn = lambda x: jnp.sum(0.5 * x**2)
    step = jax_backend.hmc_kernel(u_fn, _exact_grad, dt=0.1, nstep=10, m=1.0, mh=True)

    def body(x, key):
        x = step(key, x)
        return x, x

    _, xs = jax.lax.scan(body, jnp.zeros(()), jax.random.split(jax.random.key(1), 20_000))
    xs = np.asarray(xs[2000:])
    assert abs(xs.mean()) < 0.05, xs.mean()
    assert abs(xs.var() - 1.0) < 0.08, xs.var()


def test_physical_sghmc_backends_agree_on_double_well():
    """figure1 parameterization: both backends sample nearly the same histogram.

    The tolerance is loose because the difference is systematic, not
    statistical: blackjax's diffusion advances the position with the previous
    momentum (qp splitting) while the original (jax backend) uses the freshly
    updated momentum (pq splitting), so at dt=0.1 the two stationary
    distributions differ at O(dt) (sample variance 0.78 vs 0.82 here). Both
    discretize the same SDE; the jax backend is the one line-faithful to
    matlab/figure1/sghmc.m.
    """

    def grad_u(key, x):
        return -4.0 * x + 4.0 * x**3 + 2.0 * jax.random.normal(key)

    hists = {}
    for name, mod in (("jax", jax_backend), ("blackjax", blackjax_backend)):
        step = mod.sghmc_physical_kernel(grad_u, dt=0.1, nstep=50, m=1.0, C=3.0, V=4.0)

        def body(x, key):
            x = step(key, x)
            return x, x

        _, xs = jax.lax.scan(body, jnp.zeros(()), jax.random.split(jax.random.key(2), 8000))
        hists[name] = np.histogram(np.asarray(xs), bins=30, range=(-3, 3), density=True)[0]
    # total-variation-ish distance between the two histograms
    assert np.abs(hists["jax"] - hists["blackjax"]).mean() < 0.05


def test_nn_sghmc_noiseless_matches_momentum_sgd():
    step = jax_backend.make_nn_sghmc(eta=0.1, mdecay=0.05, num_train=1000, temp=1.0)
    w = {"w": jnp.array([1.0, -2.0])}
    m = {"w": jnp.array([0.5, 0.5])}
    g = {"w": jnp.array([0.2, -0.1])}
    wd = {"w": 0.01}
    new_w, new_m = step(jax.random.key(0), w, m, g, wd, do_sample=False)
    exp_m = (1 - 0.05) * m["w"] - 0.1 * (g["w"] + 0.01 * w["w"])
    np.testing.assert_allclose(np.asarray(new_m["w"]), np.asarray(exp_m), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(new_w["w"]), np.asarray(w["w"] + exp_m), rtol=1e-6)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
