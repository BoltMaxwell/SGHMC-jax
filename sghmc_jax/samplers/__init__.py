"""SG-MCMC sampler kernels with two interchangeable backends.

- ``jax``: hand-rolled kernels, line-for-line faithful to the original
  ML-SGHMC matlab/numpy update equations.
- ``blackjax``: the same samplers expressed through blackjax's
  ``sgmcmc.diffusions``, with the exact parameter mapping documented in
  :mod:`sghmc_jax.samplers.blackjax_backend`.

Shared interface (per backend module):

- ``make_sghmc(eta, alpha, v_hat) -> (init, step)`` with
  ``init(key, x) -> momentum`` and ``step(key, x, momentum, grad) -> (x, momentum)``
- ``make_sgld(eta, v_hat) -> step`` with ``step(key, x, grad) -> x``
- ``sghmc_physical_kernel(grad_u, *, dt, nstep, m, C, V) -> step`` with
  ``step(key, x) -> x`` (momentum resampled per call)

The jax backend additionally provides ``hmc_kernel`` (leapfrog HMC with
optional M-H test) and ``make_nn_sghmc`` (the bayesnn updater form).
"""

from . import blackjax_backend, jax_backend
from .jax_backend import (
    gaussian_like,
    hmc_kernel,
    make_nn_sghmc,
    make_sghmc,
    make_sgld,
    sghmc_physical_kernel,
)

BACKENDS = {
    "jax": jax_backend,
    "blackjax": blackjax_backend,
}

__all__ = [
    "BACKENDS",
    "gaussian_like",
    "hmc_kernel",
    "make_nn_sghmc",
    "make_sghmc",
    "make_sgld",
    "sghmc_physical_kernel",
    "jax_backend",
    "blackjax_backend",
]
