"""Bayesian matrix factorization on ml-1m with SGHMC/SGLD, port of mf/ (see docs/mf_spec.md).

Model: pred(u, i) = base_score + u_bias[u] + i_bias[i] + <W_user[u], W_item[i]>,
square loss on labels (rating - 1)/4, isotropic Gamma-sampled precisions per
parameter group. A round = one Gibbs hyperparameter update + 2000 minibatch
SGHMC/SGLD steps over the reshuffled train set (persistent wraparound
pointer) + posterior-predictive averaging + RMSE eval (x4, star scale).

Usage: python -m sghmc_jax.mf.train --data_path data [--fold 1] [--sampler sghmc]
       [--backend jax] [--num_round 1000] ...
"""

import argparse
import functools
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.sgmcmc import diffusions

from sghmc_jax.samplers.jax_backend import gaussian_like

from . import data as data_mod

GROUPS = ("W_user", "W_item", "u_bias", "i_bias")


def init_params(key, num_user, num_item, num_factor, init_sigma):
    k1, k2 = jax.random.split(key)
    return {
        "W_user": init_sigma * jax.random.normal(k1, (num_user, num_factor), jnp.float32),
        "W_item": init_sigma * jax.random.normal(k2, (num_item, num_factor), jnp.float32),
        "u_bias": jnp.zeros((num_user,), jnp.float32),
        "i_bias": jnp.zeros((num_item,), jnp.float32),
    }


def predict(params, base_score, u, i):
    return (base_score + params["u_bias"][u] + params["i_bias"][i]
            + jnp.sum(params["W_user"][u] * params["W_item"][i], axis=-1))


def make_kernel(sampler, backend, eta0, mdecay, temp, num_train):
    """One parameter update from the ascent gradient g (= d * mscale / eta).

    jax backend reproduces the C++ update exactly:
    SGHMC: p <- exp(-mdecay) p + eta g + N(0, 2 temp eta mdecay); theta += p
    SGLD:  theta += eta g + N(0, 2 temp eta)
    blackjax backend maps the same update onto blackjax's diffusions
    (SGHMC: h = sqrt(eta), alpha = (1 - exp(-mdecay))/h,
    temperature = temp * mdecay / (1 - exp(-mdecay)); SGLD is exact).
    """
    eta = eta0 / num_train
    if backend == "jax":
        if sampler == "sghmc":
            momentum = float(np.exp(-mdecay))
            sigma = float(np.sqrt(2.0 * temp * eta * mdecay))

            def step(key, params, moms, g, do_noise):
                noise = gaussian_like(key, params)
                moms = jax.tree_util.tree_map(
                    lambda p, g_, n: momentum * p + eta * g_
                    + jnp.where(do_noise, sigma, 0.0) * n,
                    moms, g, noise)
                params = jax.tree_util.tree_map(lambda t, p: t + p, params, moms)
                return params, moms
        else:
            sigma = float(np.sqrt(2.0 * temp * eta))

            def step(key, params, moms, g, do_noise):
                noise = gaussian_like(key, params)
                delta = jax.tree_util.tree_map(
                    lambda g_, n: eta * g_ + jnp.where(do_noise, sigma, 0.0) * n,
                    g, noise)
                params = jax.tree_util.tree_map(lambda t, d: t + d, params, delta)
                return params, moms

        return step

    # blackjax backend
    if sampler == "sghmc":
        h = float(np.sqrt(eta))
        momentum = float(np.exp(-mdecay))
        alpha = (1.0 - momentum) / h
        base_temp = temp * mdecay / (1.0 - momentum)
        one_step = diffusions.sghmc(alpha=alpha, beta=0.0)

        def step(key, params, moms, g, do_noise):
            temperature = base_temp * jnp.where(do_noise, 1.0, 0.0)
            return one_step(key, params, moms, g, h, temperature)
    else:
        one_step = diffusions.overdamped_langevin()

        def step(key, params, moms, g, do_noise):
            temperature = temp * jnp.where(do_noise, 1.0, 0.0)
            return one_step(key, params, g, eta, temperature), moms

    return step


def make_round_fn(step, base_score, num_train, batch_size, nstep, eta0):
    @jax.jit
    def run_round(key, params, moms, lambdas, perm, sptr, uid, iid, label, do_noise):
        lam_out = lambdas["output"]
        mscale = eta0 * lam_out / batch_size
        rscale = -batch_size / (num_train * lam_out)
        eta = eta0 / num_train

        def body(carry, subkey):
            params, moms, sptr = carry
            idx = perm[(sptr + jnp.arange(batch_size)) % num_train]
            u, i, y = uid[idx], iid[idx], label[idx]
            g = y - predict(params, base_score, u, i)
            d = {
                "W_user": jnp.zeros_like(params["W_user"]).at[u].add(
                    g[:, None] * params["W_item"][i]),
                "W_item": jnp.zeros_like(params["W_item"]).at[i].add(
                    g[:, None] * params["W_user"][u]),
                "u_bias": jnp.zeros_like(params["u_bias"]).at[u].add(g),
                "i_bias": jnp.zeros_like(params["i_bias"]).at[i].add(g),
            }
            d = {k: d[k] + lambdas[k] * rscale * params[k] for k in GROUPS}
            # ascent gradient in the kernel's units: g_asc = d * mscale / eta
            g_asc = {k: d[k] * (mscale / eta) for k in GROUPS}
            params, moms = step(subkey, params, moms, g_asc, do_noise)
            sptr = (sptr + batch_size) % num_train
            return (params, moms, sptr), None

        (params, moms, sptr), _ = jax.lax.scan(
            body, (params, moms, sptr), jax.random.split(key, nstep))
        return params, moms, sptr

    return run_round


def make_gibbs_fn(base_score, prior_alpha=1.0, prior_beta=1.0):
    @jax.jit
    def gibbs(key, params, uid, iid, label):
        pred = predict(params, base_score, uid, iid)
        mse_sum = jnp.sum((pred - label) ** 2)
        stats = {k: (params[k].size, jnp.sum(params[k] ** 2)) for k in GROUPS}
        stats["output"] = (label.shape[0], mse_sum)
        lambdas = {}
        for k, kk in zip(sorted(stats), jax.random.split(key, len(stats))):
            n, sqr = stats[k]
            alpha = prior_alpha + 0.5 * n
            beta = prior_beta + 0.5 * sqr
            lambdas[k] = jax.random.gamma(kk, alpha) / beta
        return lambdas

    return gibbs


def make_eval_fn(base_score):
    @jax.jit
    def evaluate(params, predmean, alpha, uid, iid, label):
        pred = predict(params, base_score, uid, iid)
        predmean = predmean * (1.0 - alpha) + alpha * pred
        rmse = jnp.sqrt(jnp.mean((predmean - label) ** 2)) * 4.0
        return predmean, rmse

    return evaluate


def train(args):
    ratings = data_mod.download(args.data_path)
    fold = data_mod.load_fold(ratings, args.fold)
    tr_u, tr_i, tr_y = (jnp.asarray(a) for a in fold["train"])
    te_u, te_i, te_y = (jnp.asarray(a) for a in fold["test"])
    num_train = int(tr_y.shape[0])
    base_score = fold["base_score"] if args.base_score is None else args.base_score

    eta0 = args.eta if args.eta is not None else (0.4 if args.sampler == "sghmc" else 8.0)
    key = jax.random.key(args.seed)
    key, k_init = jax.random.split(key)
    params = init_params(k_init, fold["num_user"], fold["num_item"],
                         args.num_factor, args.init_sigma)
    moms = jax.tree_util.tree_map(jnp.zeros_like, params)
    lambdas = {k: jnp.asarray(1.0, jnp.float32) for k in (*GROUPS, "output")}

    kernel = make_kernel(args.sampler, args.backend, eta0, args.mdecay,
                         args.temp, num_train)
    run_round = make_round_fn(kernel, base_score, num_train, args.batch_size,
                              args.nstep, eta0)
    gibbs = make_gibbs_fn(base_score)
    evaluate = make_eval_fn(base_score)

    predmean_tr = jnp.zeros_like(tr_y)
    predmean_te = jnp.zeros_like(te_y)
    sptr = jnp.asarray(0, jnp.int32)
    sum_wsample = 0.0
    history = []

    for r in range(args.num_round):
        key, k_gibbs, k_perm, k_round = jax.random.split(key, 4)
        if r > args.start_sample:  # rounds 0..start_sample use lambda = 1
            lambdas = gibbs(k_gibbs, params, tr_u, tr_i, tr_y)
        perm = jax.random.permutation(k_perm, num_train)
        do_noise = r >= args.start_sample
        params, moms, sptr = run_round(k_round, params, moms, lambdas, perm,
                                       sptr, tr_u, tr_i, tr_y,
                                       jnp.asarray(do_noise))

        if r > args.num_burn:
            sum_wsample += 1.0
            alpha = 1.0 / sum_wsample
        else:
            alpha = 1.0
        predmean_tr, train_rmse = evaluate(params, predmean_tr, alpha, tr_u, tr_i, tr_y)
        predmean_te, test_rmse = evaluate(params, predmean_te, alpha, te_u, te_i, te_y)
        train_rmse, test_rmse = float(train_rmse), float(test_rmse)
        history.append({"round": r, "train-rmse": train_rmse, "test-rmse": test_rmse,
                        **{f"lambda_{k}": float(v) for k, v in lambdas.items()}})
        print(f"[{r}] train-rmse:{train_rmse:f}\ttest-rmse:{test_rmse:f}", file=sys.stderr)
        sys.stderr.flush()

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        tag = f"{args.sampler}_{args.backend}_fold{args.fold}_seed{args.seed}"
        with open(os.path.join(args.out, f"mf_{tag}.json"), "w") as f:
            json.dump({"args": vars(args), "history": history}, f)
        print(f"saved {args.out}/mf_{tag}.json")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--sampler", choices=("sghmc", "sgld"), default="sghmc")
    ap.add_argument("--backend", choices=("jax", "blackjax"), default="jax")
    ap.add_argument("--num_round", type=int, default=1000)
    ap.add_argument("--nstep", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=4000)
    ap.add_argument("--num_factor", type=int, default=20)
    ap.add_argument("--eta", type=float, default=None,
                    help="base eta (default 0.4 sghmc / 8.0 sgld)")
    ap.add_argument("--mdecay", type=float, default=0.05)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--init_sigma", type=float, default=0.03)
    ap.add_argument("--base_score", type=float, default=None,
                    help="global offset (default: mean label, = 0.645391 on ml-1m)")
    ap.add_argument("--num_burn", type=int, default=50)
    ap.add_argument("--start_sample", type=int, default=2)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
