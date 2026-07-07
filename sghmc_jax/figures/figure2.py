"""Figure 2: phase-space traces of Hamiltonian-like dynamics with noisy gradients.

Port of matlab/figure2/figure2.m. Target U = x^2/2, gradient noise
N(0, 0.5^2). Four trajectories from (x, p) = (1, 0), recording (x, p) after
each block of ``niter`` = 50 inner steps, ``nstep`` = 300 blocks:

  1. noisy Hamiltonian dynamics (leapfrog, momentum never resampled)
  2. noisy Hamiltonian dynamics, momentum resampled every block
  3. noisy Hamiltonian dynamics with friction (the SGHMC dynamic, C = 3)
  4. exact Hamiltonian dynamics

Usage: python -m sghmc_jax.figures.figure2 [--plot] [--out figs]
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

M, C, DT = 1.0, 3.0, 0.1
NSTEP, NITER = 300, 50
SIGMA = 0.5  # gradient noise std, V = sigma^2


def _leapfrog_block(grad, resample):
    def block(carry, key):
        x, p = carry
        key_p, key_steps = jax.random.split(key)
        if resample:
            p = jax.random.normal(key_p) * jnp.sqrt(M)

        def inner(carry, subkey):
            x, p = carry
            k1, k2 = jax.random.split(subkey)
            p = p - grad(k1, x) * DT / 2.0
            x = x + p / M * DT
            p = p - grad(k2, x) * DT / 2.0
            return (x, p), None

        (x, p), _ = jax.lax.scan(inner, (x, p), jax.random.split(key_steps, NITER))
        return (x, p), (x, p)

    return block


def _friction_block(grad):
    b_hat = 0.5 * SIGMA**2 * DT
    d = jnp.sqrt(2.0 * (C - b_hat) * DT)

    def block(carry, key):
        def inner(carry, subkey):
            x, p = carry
            k_grad, k_noise = jax.random.split(subkey)
            p = p - grad(k_grad, x) * DT - p * C * DT + jax.random.normal(k_noise) * d
            x = x + p / M * DT
            return (x, p), None

        carry, _ = jax.lax.scan(inner, carry, jax.random.split(key, NITER))
        return carry, carry

    return block


def run(seed=0):
    noisy = lambda key, x: x + SIGMA * jax.random.normal(key)
    exact = lambda key, x: x
    blocks = {
        "hamiltonian_noisy": _leapfrog_block(noisy, resample=False),
        "hamiltonian_noisy_resample": _leapfrog_block(noisy, resample=True),
        "friction_noisy": _friction_block(noisy),
        "hamiltonian_exact": _leapfrog_block(exact, resample=False),
    }
    out = {}
    for i, (name, block) in enumerate(blocks.items()):
        keys = jax.random.split(jax.random.key(seed + i), NSTEP)
        _, (xs, ps) = jax.lax.scan(block, (jnp.asarray(1.0), jnp.asarray(0.0)), keys)
        out[name] = np.stack([np.asarray(xs), np.asarray(ps)], axis=1)
    return out


LABELS = {
    "hamiltonian_noisy": "Noisy Hamiltonian dynamics",
    "hamiltonian_noisy_resample": "Noisy Hamiltonian dynamics(resample r each 50 steps)",
    "friction_noisy": "Noisy Hamiltonian dynamics with friction",
    "hamiltonian_exact": "Hamiltonian dynamics",
}
STYLES = {
    "hamiltonian_noisy": ("v", "r"),
    "hamiltonian_noisy_resample": ("o", "k"),
    "friction_noisy": ("s", "g"),
    "hamiltonian_exact": ("x", "b"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    res = run(args.seed)
    np.savez(os.path.join(args.out, "figure2_trace.npz"), **res)
    for name, tr in res.items():
        r = np.sqrt((tr**2).sum(axis=1))
        print(f"{LABELS[name]:55s} final |z|={r[-1]:8.3f}  max |z|={r.max():8.3f}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4.5))
        for name, tr in res.items():
            marker, color = STYLES[name]
            plt.plot(tr[:, 0], tr[:, 1], marker, color=color, markersize=3,
                     linestyle="none", label=LABELS[name])
        plt.xlabel(r"$\theta$")
        plt.ylabel("r")
        plt.legend(fontsize=7)
        plt.tight_layout()
        path = os.path.join(args.out, "figure2_trace.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"saved {path}")


if __name__ == "__main__":
    main()
