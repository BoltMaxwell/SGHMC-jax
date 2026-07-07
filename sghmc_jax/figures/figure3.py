"""Figure 3: SGLD vs SGHMC on a correlated 2D Gaussian (rho = 0.9).

Port of matlab/figure3/{figure3a.m,figure3b.m,sghmc.m,sgld.m,aucTime.m,drawcmp.m}.

figure3b: first 50 samples of each sampler over the target contours
(etaSGLD = etaSGHMC = 0.05, alpha = 0.035).

figure3a: 5 step-size settings per sampler, L steps each (1e7 in the paper
script). SGLD: eta = 0.18 * 0.8^i; SGHMC: dscale = 0.6^i,
eta = 0.05 * dscale^2, alpha = 0.05 * dscale. For each run we record the
autocorrelation time (aucTime.m, the NUTS-paper estimator) and the average
absolute error of the sample covariance (drawcmp.m).

Usage: python -m sghmc_jax.figures.figure3 [--mode {a,b,both}] [--steps 10000000]
       [--backend jax] [--plot] [--out figs]
"""

import argparse
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

from sghmc_jax.samplers import BACKENDS

RHO = 0.9
COV_S = np.array([[1.0, RHO], [RHO, 1.0]])
INV_S = np.linalg.inv(COV_S)
V = 1.0  # gradient noise is N(0, I_2) per evaluation


@functools.partial(jax.jit, static_argnames=("backend", "eta", "n_steps"))
def run_sgld(key, backend, eta, n_steps):
    step = BACKENDS[backend].make_sgld(eta, v_hat=V)
    inv_s = jnp.asarray(INV_S)

    def body(x, subkey):
        k_grad, k_step = jax.random.split(subkey)
        grad = inv_s @ x + jax.random.normal(k_grad, (2,))
        x = step(k_step, x, grad)
        return x, x

    _, xs = jax.lax.scan(body, jnp.zeros(2), jax.random.split(key, n_steps))
    return xs


@functools.partial(jax.jit, static_argnames=("backend", "eta", "alpha", "n_steps"))
def run_sghmc(key, backend, eta, alpha, n_steps):
    init, step = BACKENDS[backend].make_sghmc(eta, alpha, v_hat=V)
    inv_s = jnp.asarray(INV_S)
    key, sub = jax.random.split(key)
    x0 = jnp.zeros(2)
    p0 = init(sub, x0)

    def body(carry, subkey):
        x, p = carry
        k_grad, k_step = jax.random.split(subkey)
        grad = inv_s @ x + jax.random.normal(k_grad, (2,))
        x, p = step(k_step, x, p, grad)
        return (x, p), x

    _, xs = jax.lax.scan(body, (x0, p0), jax.random.split(key, n_steps))
    return xs


def auc_time(data, dvar=1.0):
    """Autocorrelation time per dimension, replicating matlab figure3/aucTime.m.

    Biased autocorrelation (no mean subtraction, normalized by L and dvar).
    Two quirks of the original are reproduced exactly: due to its xcorr
    indexing, the tested value at lag j is rho_j = 0.5 (c_{j-1} + c_{j+1}),
    and the accumulated term is 2 (1 - j/L) itself, NOT weighted by rho:
    tau = 1 + sum_{j >= 1} 2 (1 - j/L), stopping at the first rho_j < 0.05.
    """
    data = np.asarray(data, np.float64)
    L, m = data.shape
    nfft = 1 << (2 * L - 1).bit_length()
    taus = []
    for i in range(m):
        f = np.fft.rfft(data[:, i], nfft)
        acov = np.fft.irfft(f * np.conj(f), nfft)[:L] / L / dvar
        rho = 0.5 * (acov[:-2] + acov[2:])  # rho[j-1] corresponds to lag j
        below = np.nonzero(rho < 0.05)[0]
        jstop = below[0] + 1 if below.size else L  # first j with rho_j < 0.05
        js = np.arange(1, jstop)
        taus.append(1.0 + np.sum(2.0 * (1.0 - js / L)))
    return np.array(taus)


def cov_err(data):
    """Average absolute error of the sample covariance (drawcmp.m)."""
    data = np.asarray(data, np.float64)
    L = data.shape[0]
    cov_e = data.T @ data / L
    mean_e = data.mean(axis=0)
    return np.abs(cov_e - np.outer(mean_e, mean_e) - COV_S).sum() / 4.0


def run_fig3a(n_steps, backend, seed=0, nset=5):
    rows = []
    for i in range(nset):
        eta = 0.18 * 0.8**i
        xs = jax.block_until_ready(run_sgld(jax.random.key(seed + i), backend, eta, n_steps))
        rows.append(("sgld", i, eta, np.nan,
                     float(auc_time(xs).mean()), float(cov_err(xs))))
        del xs
    for i in range(nset):
        dscale = 0.6**i
        eta = 0.05 * dscale * dscale
        alpha = 0.05 * dscale
        xs = jax.block_until_ready(
            run_sghmc(jax.random.key(seed + 100 + i), backend, eta, alpha, n_steps))
        rows.append(("sghmc", i, eta, alpha,
                     float(auc_time(xs).mean()), float(cov_err(xs))))
        del xs
    return rows


def run_fig3b(backend, seed=0):
    sgld_xs = run_sgld(jax.random.key(seed), backend, 0.05, 50)
    sghmc_xs = run_sghmc(jax.random.key(seed + 1), backend, 0.05, 0.035, 50)
    return np.asarray(sgld_xs), np.asarray(sghmc_xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("a", "b", "both"), default="both")
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--nset", type=int, default=5)
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="jax")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="figs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.mode in ("b", "both"):
        sgld_xs, sghmc_xs = run_fig3b(args.backend, args.seed)
        np.savez(os.path.join(args.out, f"figure3b_{args.backend}.npz"),
                 sgld=sgld_xs, sghmc=sghmc_xs)
        print(f"[3b] sgld mean {sgld_xs.mean(0)}, sghmc mean {sghmc_xs.mean(0)}")
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
            zz = np.exp(-0.5 * (xx**2 * INV_S[0, 0] + 2 * xx * yy * INV_S[0, 1]
                                + yy**2 * INV_S[1, 1]))
            zz /= 2 * np.pi * np.sqrt(np.linalg.det(COV_S))
            plt.figure(figsize=(5, 4.5))
            plt.contour(xx, yy, zz)
            plt.scatter(sgld_xs[:, 0], sgld_xs[:, 1], marker="x", c="b", label="SGLD")
            plt.scatter(sghmc_xs[:, 0], sghmc_xs[:, 1], marker="o", facecolors="none",
                        edgecolors="r", label="SGHMC")
            plt.xlabel("x"); plt.ylabel("y"); plt.legend()
            plt.xlim(-2.1, 3); plt.ylim(-2.1, 3)
            plt.tight_layout()
            path = os.path.join(args.out, f"figure3b_{args.backend}.png")
            plt.savefig(path, dpi=150); plt.close()
            print(f"saved {path}")

    if args.mode in ("a", "both"):
        rows = run_fig3a(args.steps, args.backend, args.seed, args.nset)
        arr = np.array([(r[2], np.nan if r[3] is np.nan else r[3], r[4], r[5])
                        for r in rows], np.float64)
        np.savez(os.path.join(args.out, f"figure3a_{args.backend}.npz"),
                 sampler=np.array([r[0] for r in rows]), stats=arr,
                 n_steps=args.steps)
        print(f"[3a] L = {args.steps}")
        print(f"{'sampler':8s} {'eta':>10s} {'alpha':>8s} {'aucTime':>10s} {'covErr':>10s}")
        for sampler, _, eta, alpha, auc, cerr in rows:
            alpha_s = f"{alpha:8.5f}" if not np.isnan(alpha) else "       -"
            print(f"{sampler:8s} {eta:10.6f} {alpha_s} {auc:10.2f} {cerr:10.4f}")
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(5, 4))
            for sampler, style in (("sgld", "b-x"), ("sghmc", "r-o")):
                pts = [(r[4], r[5]) for r in rows if r[0] == sampler]
                pts.sort()
                plt.plot([p[0] for p in pts], [p[1] for p in pts], style,
                         label=sampler.upper())
            plt.xlabel("Autocorrelation Time")
            plt.ylabel("Average Absolute Error of Sample Covariance")
            plt.legend()
            plt.tight_layout()
            path = os.path.join(args.out, f"figure3a_{args.backend}.png")
            plt.savefig(path, dpi=150); plt.close()
            print(f"saved {path}")


if __name__ == "__main__":
    main()
