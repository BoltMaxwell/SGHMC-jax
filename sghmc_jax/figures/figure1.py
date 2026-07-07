"""Figure 1: sampled distributions of HMC variants under noisy gradients.

Port of matlab/figure1/{figure1.m,hmccmp.m,hmc.m,sghmc.m}. For each target
(func4: U = -2x^2 + x^4, func2: U = x^2/2) five chains of ``nsample`` draws
are run, each draw being 50 leapfrog/SGHMC steps from the previous draw:

  1. standard HMC (perfect gradient, M-H test)
  2. standard HMC (perfect gradient, no M-H)
  3. naive stochastic gradient HMC (noisy gradient, M-H)
  4. naive stochastic gradient HMC (noisy gradient, no M-H)
  5. SGHMC with friction (C = 3, V = 4)

Gradient noise is N(0, 2^2) per gradient evaluation, exactly as
``gradU = @(x) ... + randn(1)*2``. Histograms are computed on the matlab
grid (bin centers -3:0.1:3) and normalized to densities.

Usage: python -m sghmc_jax.figures.figure1 [--nsample 80000] [--backend jax]
       [--plot] [--out figs]
"""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from sghmc_jax.samplers import BACKENDS, jax_backend

XSTEP = 0.1
GRID = np.arange(-3.0, 3.0 + 1e-9, XSTEP)  # bin centers, as in matlab

TARGETS = {
    "func4": {
        "u": lambda x: -2.0 * x**2 + x**4,
        "grad": lambda x: -4.0 * x + 4.0 * x**3,
    },
    "func2": {
        "u": lambda x: 0.5 * x**2,
        "grad": lambda x: x,
    },
}
NOISE_STD = 2.0  # gradient noise, V = 4
M, DT, NSTEP, C, V = 1.0, 0.1, 50, 3.0, 4.0


def run_chain(step, nsample, seed):
    def body(x, key):
        x = step(key, x)
        return x, x

    keys = jax.random.split(jax.random.key(seed), nsample)
    _, xs = jax.lax.scan(body, jnp.zeros(()), keys)
    return np.asarray(jax.block_until_ready(xs))


def hist_density(samples):
    """matlab hist() on bin centers GRID, normalized like hmccmp.m."""
    edges = np.concatenate([[-np.inf], (GRID[:-1] + GRID[1:]) / 2, [np.inf]])
    y, _ = np.histogram(samples, bins=edges)
    return y / y.sum() / XSTEP


def true_density(u):
    y = np.exp(-u(GRID))
    return y / y.sum() / XSTEP


def run_target(name, nsample, backend, seed=0):
    tgt = TARGETS[name]
    u_fn = tgt["u"]
    exact = lambda key, x: tgt["grad"](x)
    noisy = lambda key, x: tgt["grad"](x) + NOISE_STD * jax.random.normal(key)
    sghmc_physical = BACKENDS[backend].sghmc_physical_kernel

    methods = {
        "hmc_mh": jax_backend.hmc_kernel(u_fn, exact, dt=DT, nstep=NSTEP, m=M, mh=True),
        "hmc_nomh": jax_backend.hmc_kernel(u_fn, exact, dt=DT, nstep=NSTEP, m=M, mh=False),
        "naive_mh": jax_backend.hmc_kernel(u_fn, noisy, dt=DT, nstep=NSTEP, m=M, mh=True),
        "naive_nomh": jax_backend.hmc_kernel(u_fn, noisy, dt=DT, nstep=NSTEP, m=M, mh=False),
        "sghmc": sghmc_physical(noisy, dt=DT, nstep=NSTEP, m=M, C=C, V=V),
    }
    out = {"true": true_density(u_fn)}
    for i, (mname, step) in enumerate(methods.items()):
        out[mname] = hist_density(run_chain(step, nsample, seed + i))
    return out


LABELS = {
    "true": "True Distribution",
    "hmc_mh": "Standard HMC(with MH)",
    "hmc_nomh": "Standard HMC(no MH)",
    "naive_mh": "Naive stochastic gradient HMC(with MH)",
    "naive_nomh": "Naive stochastic gradient HMC(no MH)",
    "sghmc": "SGHMC",
}
STYLES = {
    "true": "b-",
    "hmc_mh": "c-v",
    "hmc_nomh": "r-.",
    "naive_mh": "m-x",
    "naive_nomh": "k-.",
    "sghmc": "g-",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsample", type=int, default=80000)
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="jax")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for name in ("func4", "func2"):
        res = run_target(name, args.nsample, args.backend, args.seed)
        np.savez(os.path.join(args.out, f"figure1_{name}_{args.backend}.npz"),
                 grid=GRID, **res)
        print(f"[{name}] histogram L1 distance to true density:")
        for mname in LABELS:
            if mname == "true":
                continue
            l1 = np.abs(res[mname] - res["true"]).sum() * XSTEP
            print(f"  {LABELS[mname]:45s} {l1:.4f}")
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 4.5))
            for mname in LABELS:
                plt.plot(GRID, res[mname], STYLES[mname], label=LABELS[mname],
                         markersize=4, markevery=3)
            plt.xlabel(r"$\theta$")
            plt.legend(fontsize=7)
            xlim = (-2, 2) if name == "func4" else (-3, 3)
            plt.xlim(xlim)
            plt.tight_layout()
            path = os.path.join(args.out, f"figure1_{name}_{args.backend}.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  saved {path}")


if __name__ == "__main__":
    main()
