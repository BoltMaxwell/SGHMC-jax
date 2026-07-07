"""SG-MCMC kernels backed by blackjax's ``sgmcmc.diffusions``.

blackjax implements the SGHMC diffusion of Chen et al. (2014) with parameters
scaled following Ma et al. (2015):

    x <- x + h v
    v <- (1 - alpha_bj h) v + h g + sqrt(h T (2 alpha_bj - h T beta_bj)) n,

where ``g`` is the log-density (ascent) gradient. The original ML-SGHMC
momentum-buffer form (matlab/figure3/sghmc.m)

    p <- (1 - alpha) p - eta grad_U + sqrt(2 eta (alpha - 0.5 V eta)) n
    x <- x + p

maps onto it exactly via  h = sqrt(eta), alpha_bj = alpha / h, beta_bj = V,
p = h v  (substituting gives identical coefficients term by term). The one
irreducible difference is operation order: blackjax advances the position with
the *previous* momentum before refreshing the momentum, while the original
refreshes the momentum first. Both are Euler discretizations of the same SDE
(a qp- vs pq-splitting); chains are not stepwise identical but target the same
stationary distribution, which the tests check statistically.

Stability caveat: because blackjax moves the position before refreshing the
momentum (with the gradient evaluated at the pre-move position), the
linearized one-step map on a Gaussian target has determinant (1 - alpha) + eta
instead of the original ordering's (1 - alpha). The blackjax backend therefore
requires eta < alpha to contract; at eta = alpha (the largest figure3a
setting) it is exactly marginally stable and the chain's variance grows
without bound, while the jax backend remains stable. Even in the stable
regime the qp splitting inflates the stationary variance on a Gaussian target
by ~ alpha / (alpha - eta) (the jax backend's bias is only O(eta)); see
tests/test_samplers.py for the Lyapunov analysis both backends are verified
against.

The figure1 "physical" (m=1, dt, C, V) parameterization maps directly:
h = dt, alpha_bj = C, beta_bj = V, with unit-normal velocity resampling.

Interface mirrors :mod:`sghmc_jax.samplers.jax_backend`; for SGHMC the
momentum state returned by ``init``/``step`` is the blackjax velocity
(v = p / sqrt(eta)) and should be treated as opaque by drivers.
"""

import math

import jax
import jax.numpy as jnp
from blackjax.sgmcmc import diffusions

from .jax_backend import gaussian_like


def _neg(tree):
    return jax.tree_util.tree_map(jnp.negative, tree)


def make_sghmc(eta, alpha, v_hat=0.0):
    """figure3-form SGHMC via the blackjax diffusion. Returns ``(init, step)``."""
    beta_hat = 0.5 * v_hat * eta
    if beta_hat > alpha:
        raise ValueError(f"too big eta: beta_hat={beta_hat} > alpha={alpha}")
    h = math.sqrt(eta)
    one_step = diffusions.sghmc(alpha=alpha / h, beta=v_hat)

    def init(key, x):
        return gaussian_like(key, x)

    def step(key, x, v, grad):
        return one_step(key, x, v, _neg(grad), h)

    return init, step


def make_sgld(eta, v_hat=0.0):
    """figure3-form SGLD via the blackjax overdamped Langevin diffusion.

    The SGFS-style correction sigma^2 = 2 eta (1 - 0.5 V eta) is expressed
    through blackjax's temperature argument: T = 1 - 0.5 V eta.
    Returns ``step(key, x, grad) -> x``.
    """
    beta_hat = 0.5 * v_hat * eta
    if beta_hat > 1.0:
        raise ValueError(f"too big eta: beta_hat={beta_hat} > 1")
    one_step = diffusions.overdamped_langevin()
    temperature = 1.0 - beta_hat

    def step(key, x, grad):
        return one_step(key, x, _neg(grad), eta, temperature)

    return step


def make_nn_sghmc(eta, mdecay, *, num_train, temp=1.0, updater="sghmc"):
    """The bayesnn SGHMCUpdater expressed through blackjax diffusions.

    Mapping for SGHMC: h = sqrt(eta), alpha_bj = mdecay / h, beta_bj = 0,
    temperature = temp / num_train (times 0 when not sampling), log-density
    gradient g = -(grad + wd * w); the momentum state is the blackjax velocity
    v = m / h (opaque to drivers). SGLD (mdecay = 1) uses the overdamped
    Langevin diffusion and is stepwise identical to the original update.
    Returns ``step(key, w, m, g, wd, do_sample) -> (w, m)``.
    """
    base_temp = temp / num_train
    is_sgld = updater == "sgld" or mdecay >= 1.0 - 1e-5

    if is_sgld:
        one_step = diffusions.overdamped_langevin()

        def step(key, w, m, g, wd, do_sample):
            gl = jax.tree_util.tree_map(lambda g_, w_, wd_: -(g_ + wd_ * w_), g, w, wd)
            temperature = base_temp * jnp.where(do_sample, 1.0, 0.0)
            w = one_step(key, w, gl, eta, temperature)
            return w, m

        return step

    h = math.sqrt(eta)
    one_step = diffusions.sghmc(alpha=mdecay / h, beta=0.0)

    def step(key, w, v, g, wd, do_sample):
        gl = jax.tree_util.tree_map(lambda g_, w_, wd_: -(g_ + wd_ * w_), g, w, wd)
        temperature = base_temp * jnp.where(do_sample, 1.0, 0.0)
        return one_step(key, w, v, gl, h, temperature)

    return step


def sghmc_physical_kernel(grad_u, *, dt, nstep, m=1.0, C=3.0, V=4.0):
    """figure1-form SGHMC (momentum resampled per call) via the blackjax diffusion.

    Only unit mass is supported (the blackjax diffusion has no mass parameter);
    all original experiments use m = 1. Returns ``step(key, x) -> new_x``.
    """
    if m != 1.0:
        raise ValueError("blackjax backend supports m=1 only")
    one_step = diffusions.sghmc(alpha=C, beta=V)

    def step(key, x):
        key_p, key_steps = jax.random.split(key)
        v = gaussian_like(key_p, x)

        def inner(carry, subkey):
            x, v = carry
            k_grad, k_diff = jax.random.split(subkey)
            g = -grad_u(k_grad, x)
            x, v = one_step(k_diff, x, v, g, dt)
            return (x, v), None

        (x, _), _ = jax.lax.scan(inner, (x, v), jax.random.split(key_steps, nstep))
        return x

    return step
