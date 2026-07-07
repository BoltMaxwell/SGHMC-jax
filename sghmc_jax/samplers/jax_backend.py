"""Hand-rolled SG-MCMC kernels in pure JAX, faithful to the original ML-SGHMC code.

Each factory cites the original file whose update equations it reproduces:

- ``hmc_kernel``            matlab/figure1/hmc.m       (leapfrog, optional M-H test)
- ``sghmc_physical_kernel`` matlab/figure1/sghmc.m     ((m, dt, C, V) parameterization)
- ``make_sghmc``            matlab/figure3/sghmc.m     (momentum-buffer (eta, alpha, V) form)
- ``make_sgld``             matlab/figure3/sgld.m      (SGFS-style noise correction)
- ``make_nn_sghmc``         bayesnn/nnupdater.py       (SGHMCUpdater with weight decay)

Stochastic gradients are modeled as ``grad_u(key, x)`` so each evaluation draws
fresh noise, exactly as the matlab ``gradU = @(x) ... + randn(1)*2`` closures do.
Gaussian noise on pytrees is drawn as a single flat vector and unraveled, the
same construction as ``blackjax.util.generate_gaussian_noise``, so the blackjax
backend can be made bit-equivalent where the parameterizations coincide.
"""

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def gaussian_like(key, tree):
    """Standard normal noise with the structure of ``tree`` (flat draw + unravel)."""
    flat, unravel = ravel_pytree(tree)
    return unravel(jax.random.normal(key, flat.shape, flat.dtype))


# ---------------------------------------------------------------------------
# figure1: (m, dt, C, V) "physical" parameterization
# ---------------------------------------------------------------------------

def hmc_kernel(u_fn, grad_u, *, dt, nstep, m=1.0, mh=True):
    """HMC with leapfrog and optional Metropolis-Hastings test (figure1/hmc.m).

    ``grad_u(key, x)`` may be a stochastic gradient ("naive SGHMC" when mh=False).
    The kinetic energy is ``p' * m * p / 2`` exactly as in the matlab code
    (which is only the true kinetic energy for m = 1; all experiments use m = 1).
    Returns ``step(key, x) -> new_x``.
    """

    def step(key, x):
        key_p, key_grad, key_mh = jax.random.split(key, 3)
        p = jax.random.normal(key_p, jnp.shape(x)) * jnp.sqrt(m)
        old_energy = jnp.sum(p * m * p) / 2.0 + u_fn(x)
        old_x = x

        def leapfrog(carry, subkey):
            x, p = carry
            k1, k2 = jax.random.split(subkey)
            p = p - grad_u(k1, x) * dt / 2.0
            x = x + p / m * dt
            p = p - grad_u(k2, x) * dt / 2.0
            return (x, p), None

        (x, p), _ = jax.lax.scan(leapfrog, (x, p), jax.random.split(key_grad, nstep))
        p = -p

        if not mh:
            return x
        new_energy = jnp.sum(p * m * p) / 2.0 + u_fn(x)
        accept = jnp.exp(old_energy - new_energy) >= jax.random.uniform(key_mh)
        return jnp.where(accept, x, old_x)

    return step


def sghmc_physical_kernel(grad_u, *, dt, nstep, m=1.0, C=3.0, V=4.0):
    """SGHMC in the (m, dt, C, V) parameterization (figure1/sghmc.m).

    Momentum is resampled at the start of every call; B = 0.5*V*dt is the
    assumed gradient-noise contribution and sqrt(2*(C-B)*dt) the injected noise.
    Returns ``step(key, x) -> new_x``.
    """
    B = 0.5 * V * dt
    D = jnp.sqrt(2.0 * (C - B) * dt)

    def step(key, x):
        key_p, key_steps = jax.random.split(key)
        p = jax.random.normal(key_p, jnp.shape(x)) * jnp.sqrt(m)

        def inner(carry, subkey):
            x, p = carry
            k_grad, k_noise = jax.random.split(subkey)
            noise = jax.random.normal(k_noise, jnp.shape(x))
            p = p - grad_u(k_grad, x) * dt - p * C * dt + noise * D
            x = x + p / m * dt
            return (x, p), None

        (x, _), _ = jax.lax.scan(inner, (x, p), jax.random.split(key_steps, nstep))
        return x

    return step


# ---------------------------------------------------------------------------
# figure3: momentum-buffer (eta, alpha, V) parameterization
# ---------------------------------------------------------------------------

def make_sghmc(eta, alpha, v_hat=0.0):
    """SGHMC momentum-buffer form (figure3/sghmc.m).

    beta_hat = 0.5 * V * eta must satisfy beta_hat < alpha ("too big eta").
    Update: p <- (1-alpha) p - eta grad_u + N(0, 2 eta (alpha - beta_hat));
            x <- x + p.
    Returns ``(init, step)`` with ``init(key, x) -> p0`` (p0 ~ N(0, eta), as in
    the matlab code) and ``step(key, x, p, grad) -> (x, p)`` where ``grad`` is
    the (stochastic) gradient of U evaluated at x.
    """
    beta_hat = 0.5 * v_hat * eta
    if beta_hat > alpha:
        raise ValueError(f"too big eta: beta_hat={beta_hat} > alpha={alpha}")
    sigma = jnp.sqrt(2.0 * eta * (alpha - beta_hat))

    def init(key, x):
        return jax.tree_util.tree_map(
            lambda n: n * jnp.sqrt(eta), gaussian_like(key, x)
        )

    def step(key, x, p, grad):
        noise = gaussian_like(key, x)
        p = jax.tree_util.tree_map(
            lambda p_, g, n: (1.0 - alpha) * p_ - eta * g + sigma * n, p, grad, noise
        )
        x = jax.tree_util.tree_map(lambda x_, p_: x_ + p_, x, p)
        return x, p

    return init, step


def make_sgld(eta, v_hat=0.0):
    """SGLD with SGFS-style gradient-noise correction (figure3/sgld.m).

    beta_hat = 0.5 * V * eta must satisfy beta_hat < 1.
    Update: x <- x - eta grad_u + N(0, 2 eta (1 - beta_hat)).
    Returns ``step(key, x, grad) -> x``.
    """
    beta_hat = 0.5 * v_hat * eta
    if beta_hat > 1.0:
        raise ValueError(f"too big eta: beta_hat={beta_hat} > 1")
    sigma = jnp.sqrt(2.0 * eta * (1.0 - beta_hat))

    def step(key, x, grad):
        noise = gaussian_like(key, x)
        return jax.tree_util.tree_map(
            lambda x_, g, n: x_ - eta * g + sigma * n, x, grad, noise
        )

    return step


# ---------------------------------------------------------------------------
# bayesnn: SGHMC/SGLD updater with weight decay (nnupdater.py SGHMCUpdater)
# ---------------------------------------------------------------------------

def make_nn_sghmc(eta, mdecay, *, num_train, temp=1.0, updater="sghmc"):
    """The bayesnn SGHMCUpdater update, on a pytree of parameters.

    m <- (1 - mdecay) m - eta (g + wd w) [+ N(0, sigma^2) if sampling];
    w <- w + m,
    with sigma = sqrt(2 temp * eta * mdecay / num_train) (SGHMC) or
    sqrt(2 temp * eta / num_train) (SGLD, i.e. mdecay = 1), matching
    NNParam.get_sigma. ``wd`` is a pytree (or scalar-per-leaf) of weight decays
    so the Gibbs hyperupdater can adjust it per parameter group.
    Returns ``step(key, w, m, g, wd, do_sample) -> (w, m)``.
    """
    if updater == "sgld" or mdecay >= 1.0 - 1e-5:
        scale = eta / num_train
    else:
        scale = eta * mdecay / num_train
    sigma = jnp.sqrt(2.0 * temp * scale)

    def step(key, w, m, g, wd, do_sample):
        noise = gaussian_like(key, w)
        m = jax.tree_util.tree_map(
            lambda m_, g_, w_, wd_, n: (1.0 - mdecay) * m_
            - eta * (g_ + wd_ * w_)
            + jnp.where(do_sample, sigma, 0.0) * n,
            m, g, w, wd, noise,
        )
        w = jax.tree_util.tree_map(lambda w_, m_: w_ + m_, w, m)
        return w, m

    return step
