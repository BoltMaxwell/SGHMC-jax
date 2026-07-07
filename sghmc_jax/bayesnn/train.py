"""Bayesian NN on MNIST with SGHMC, port of bayesnn/ (mnist.py + nnet.py + nnupdater.py).

Model: 784 -> num_hidden (sigmoid) -> 10 (softmax), the original ``mlp2``.
Updaters: sghmc / sgld / sgd, gradients are minibatch means, weight decay is
added in the updater (not the loss). Gibbs Gamma resampling of per-group
weight decay (``gibbs-sep``/``gibbs-joint``): lambda ~ Gamma(1 + n/2,
1 + sum(w^2)/2), wd = lambda / num_train, fired on the 100th qualifying
minibatch (the last minibatch of each round from round 1 on), before that
minibatch's weight update — exactly the original HyperUpdater cadence.
Evaluation: running average of softmax predictions, replacing until round
``num_burn`` then equal-weight averaging (NNEvaluator).

Usage: python -m sghmc_jax.bayesnn.train --data_path data/mnist [--updater sghmc]
       [--backend jax] [--num_round 800] ...
"""

import argparse
import functools
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from sghmc_jax.samplers import BACKENDS

from . import data as data_mod


def init_params(key, input_size, num_hidden, num_class, init_sigma):
    k1, k2 = jax.random.split(key)
    return {
        "w1": init_sigma * jax.random.normal(k1, (input_size, num_hidden), jnp.float32),
        "b1": jnp.zeros((num_hidden,), jnp.float32),
        "w2": init_sigma * jax.random.normal(k2, (num_hidden, num_class), jnp.float32),
        "b2": jnp.zeros((num_class,), jnp.float32),
    }


def forward(params, x):
    h = jax.nn.sigmoid(x @ params["w1"] + params["b1"])
    return h @ params["w2"] + params["b2"]


def loss_fn(params, x, y):
    logp = jax.nn.log_softmax(forward(params, x))
    return -jnp.mean(logp[jnp.arange(y.shape[0]), y])


def gibbs_resample_wd(key, params, num_train, mode, hyper_alpha=1.0, hyper_beta=1.0):
    """HyperUpdater: lambda ~ Gamma(alpha0 + n/2, beta0 + sumsqr/2), wd = lambda/num_train."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    if mode == "gibbs-sep":
        wds = []
        for k, w in zip(keys, leaves):
            alpha = hyper_alpha + 0.5 * w.size
            beta = hyper_beta + 0.5 * jnp.sum(w * w)
            wds.append(jax.random.gamma(k, alpha) / beta / num_train)
        return jax.tree_util.tree_unflatten(treedef, wds)
    if mode == "gibbs-joint":
        alpha = hyper_alpha + 0.5 * sum(w.size for w in leaves)
        beta = hyper_beta + 0.5 * sum(jnp.sum(w * w) for w in leaves)
        wd = jax.random.gamma(keys[0], alpha) / beta / num_train
        return jax.tree_util.tree_unflatten(treedef, [wd] * len(leaves))
    raise ValueError(mode)


def make_round_fn(step, num_train, hyperupdater):
    """One training round: scan over the (fixed-order) minibatches."""

    @jax.jit
    def run_round(key, w, m, wd, xdata, ydata, gibbs_flags, do_sample):
        def body(carry, inp):
            w, m, wd, key = carry
            x, y, gibbs = inp
            key, k_gibbs, k_step = jax.random.split(key, 3)
            if hyperupdater != "none":
                wd = jax.lax.cond(
                    gibbs,
                    lambda: gibbs_resample_wd(k_gibbs, w, num_train, hyperupdater),
                    lambda: wd,
                )
            g = jax.grad(loss_fn)(w, x, y)
            w, m = step(k_step, w, m, g, wd, do_sample)
            return (w, m, wd, key), None

        (w, m, wd, key), _ = jax.lax.scan(body, (w, m, wd, key), (xdata, ydata, gibbs_flags))
        return w, m, wd

    return run_round


@jax.jit
def predict_probs(params, xdata):
    return jax.nn.softmax(jax.vmap(lambda x: forward(params, x))(xdata))


@jax.jit
def eval_avg(o_pred, probs, alpha, ydata):
    o_pred = (1.0 - alpha) * o_pred + alpha * probs
    y_hat = jnp.argmax(o_pred, axis=-1)
    err = jnp.mean((y_hat != ydata).astype(jnp.float32))
    picked = jnp.take_along_axis(o_pred, ydata[..., None], axis=-1)
    nlik = -jnp.mean(jnp.log(picked))
    return o_pred, err, nlik


def train(args):
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = data_mod.load_mnist_batched(
        args.data_path, args.batch_size, args.seed)
    num_train = train_y.size
    print(f"loading end,{num_train} train,{valid_y.size} valid,{test_y.size} test, start update ...")

    key = jax.random.key(args.seed)
    key, k_init = jax.random.split(key)
    w = init_params(k_init, train_x.shape[-1], args.num_hidden, 10, args.init_sigma)
    m = jax.tree_util.tree_map(jnp.zeros_like, w)
    wd = jax.tree_util.tree_map(lambda _: jnp.asarray(args.wd, jnp.float32), w)

    updater = args.updater
    backend = BACKENDS[args.backend]
    if updater in ("sghmc", "sgld"):
        # nncfg.py forces mdecay = 1.0 for sgld (memoryless update)
        mdecay = 1.0 if updater == "sgld" else args.mdecay
        step = backend.make_nn_sghmc(args.eta, mdecay, num_train=num_train,
                                     temp=args.temp, updater=updater)
    elif updater == "sgd":
        base = BACKENDS["jax"].make_nn_sghmc(args.eta, args.mdecay, num_train=num_train,
                                             temp=args.temp)
        step = lambda key, w, m, g, wd, do_sample: base(key, w, m, g, wd, False)
    else:
        raise ValueError(updater)

    run_round = make_round_fn(step, num_train, args.hyperupdater)

    nb = train_x.shape[0]
    train_xd, train_yd = jnp.asarray(train_x), jnp.asarray(train_y)
    valid_xd, valid_yd = jnp.asarray(valid_x), jnp.asarray(valid_y)
    test_xd, test_yd = jnp.asarray(test_x), jnp.asarray(test_y)

    o_preds = {
        "train": jnp.zeros(train_y.shape + (10,), jnp.float32),
        "valid": jnp.zeros(valid_y.shape + (10,), jnp.float32),
        "test": jnp.zeros(test_y.shape + (10,), jnp.float32),
    }
    splits = {"train": (train_xd, train_yd), "valid": (valid_xd, valid_yd),
              "test": (test_xd, test_yd)}
    sum_wsample = 0.0
    history = []

    for r in range(args.num_round):
        gibbs_flags = np.zeros(nb, bool)
        if args.hyperupdater != "none" and r >= args.start_hsample:
            gibbs_flags[-1] = True  # the original fires on the last minibatch
        do_sample = updater != "sgd" and r >= args.start_sample
        key, k_round = jax.random.split(key)
        w, m, wd = run_round(k_round, w, m, wd, train_xd, train_yd,
                             jnp.asarray(gibbs_flags), jnp.asarray(do_sample))

        if r < args.num_burn:
            alpha = 1.0
        else:
            sum_wsample += 1.0
            alpha = 1.0 / sum_wsample
        row = {"round": r}
        line = f"[{r}]"
        for name, (xd, yd) in splits.items():
            probs = predict_probs(w, xd)
            o_preds[name], err, nlik = eval_avg(o_preds[name], probs, alpha, yd)
            row[f"{name}-err"] = float(err)
            row[f"{name}-nlik"] = float(nlik)
            line += f" {name}-err:{float(err):f} {name}-nlik:{float(nlik):f}"
        history.append(row)
        print(line, file=sys.stderr)
        sys.stderr.flush()

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        tag = f"{updater}_{args.backend}_seed{args.seed}"
        with open(os.path.join(args.out, f"bayesnn_{tag}.json"), "w") as f:
            json.dump({"args": vars(args), "history": history}, f)
        print(f"saved {args.out}/bayesnn_{tag}.json")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data/mnist")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--updater", choices=("sghmc", "sgld", "sgd"), default="sghmc")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="jax")
    ap.add_argument("--hyperupdater", choices=("none", "gibbs-sep", "gibbs-joint"),
                    default="gibbs-sep")
    ap.add_argument("--num_round", type=int, default=800)
    ap.add_argument("--num_burn", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--num_hidden", type=int, default=100)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--mdecay", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.00002)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--init_sigma", type=float, default=0.01)
    ap.add_argument("--start_sample", type=int, default=1)
    ap.add_argument("--start_hsample", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    if args.download:
        data_mod.download(args.data_path)
    train(args)


if __name__ == "__main__":
    main()
