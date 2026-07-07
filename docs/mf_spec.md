# Spec: Bayesian matrix factorization with SGHMC/SGLD (extracted from the original C++)

This documents the exact semantics of the original `mf/` implementation
(SVDFeature-based, `code/hmc-bpmf/bpmf_sampler.h` + `bpmf_base.h`), used as the
sole specification for the JAX reimplementation in `sghmc_jax/mf/`.

## Config (run/SGHMC.conf, run/SGLD.conf)

Common: `base_score = 0.645391`, `scale_score = 4` (RMSE rescale), 1000 rounds,
`u_init_sigma = i_init_sigma = 0.03`, `num_item = 3706`, `num_user = 6040`,
`num_factor = 20`, `batch_size = 4000`, `nstep = 2000` inner steps per round,
`start_sample = 2`, `num_burn = 50`, fold 1 of a 5-fold split, LINEAR
activation (no sigmoid, no clipping), `temp = 1`.

SGHMC: `sample_method = 2`, `eta = 0.4`, `mdecay = 0.05`.
SGLD: `sample_method = 1`, `eta = 8.0`.

Hyperprior defaults: all Gamma priors `alpha = beta = 1`. No eta/mdecay decay
schedules (`a = b = 1`, `d_eta = d_mdecay = 0`).

## Model

Parameters: `W_user (6040 x 20)`, `W_item (3706 x 20)`, `u_bias (6040)`,
`i_bias (3706)`. Prediction (labels normalized to [0,1]):

    pred(u, i) = base_score + u_bias[u] + i_bias[i] + <W_user[u], W_item[i]>

`base_score = 0.645391` is the global mean of `(rating - 1)/4` (additive
offset, not sigmoided). Loss is square loss; residual `g = label - pred`.
Init: factors i.i.d. `N(0, 0.03^2)`, biases and momenta zero.

## SGHMC update (per inner step; N = #train, B = 4000)

With `eta = 0.4/N`, `mdecay = 0.05`, `momentum = exp(-mdecay)`,
`sigma = sqrt(2 * temp * eta * mdecay)`, per parameter group theta with prior
precision lambda_theta and momentum v:

    grad_batch = sum_minibatch (label - pred) * x     # x = other factor, or 1 for biases
    v <- exp(-mdecay) v + eta [ lambda_out (N/B) grad_batch - lambda_theta theta ]
         + Normal(0, 2 temp eta mdecay)               # noise only from round >= 2
    theta <- theta + v

Minibatches are B consecutive entries of the shuffled train set with a
persistent wraparound pointer (reshuffled every round).

## SGLD update

Same gradients, no momentum, `eta = 8.0/N`, `sigma = sqrt(2 * temp * eta)`:

    theta <- theta + eta [ lambda_out (N/B) grad_batch - lambda_theta theta ]
             + Normal(0, 2 temp eta)

## Gibbs hyperparameter updates (once per round, before the inner sweep)

Rounds 0-2 use `lambda = 1` for every group; from round 3 on (rate
parameterization):

    lambda_user   ~ Gamma(1 + 0.5 * 6040*20, 1 + 0.5 * sum(W_user^2))
    lambda_item   ~ Gamma(1 + 0.5 * 3706*20, 1 + 0.5 * sum(W_item^2))
    lambda_ubias  ~ Gamma(1 + 0.5 * 6040,    1 + 0.5 * sum(u_bias^2))
    lambda_ibias  ~ Gamma(1 + 0.5 * 3706,    1 + 0.5 * sum(i_bias^2))
    lambda_output ~ Gamma(1 + 0.5 * N,       1 + 0.5 * sum((pred_train - label)^2))

## Training loop and evaluation

A round = Gibbs update + reshuffle + 2000 minibatch steps + eval. 1000 rounds,
round counter r = 0..999. Posterior-predictive averaging: for r > 50 (burn-in),
`predmean <- (1 - 1/k) predmean + (1/k) pred` (equal-weight running average of
current-sample predictions, since the sample weight `cur_sweight` is constant).
Reported metric: `test-rmse = sqrt(mean((predmean - label)^2)) * 4` (star
scale). Each round prints `[r] train-rmse:X test-rmse:Y` to stderr.

## Data pipeline (ml-1m)

`ratings.dat` lines `user::movie::rating::ts`. Contiguous id maps in
first-seen order; labels `(rating - 1)/4`. 5-fold split with Python 2
`random.seed(0)`; each line goes to fold `random.randint(1, 5)`; fold k's
lines are test, the rest train. Fold 1 is the reported experiment. The
original's extra line-shuffle/binary-buffer machinery only permutes row order
and is not semantically relevant (train order is reshuffled every round).

Note on `random.randint`: Python 2's `randint(1, 5)` maps MT19937 `random()`
draws via `1 + int(random() * 5)`, while Python 3 uses `getrandbits`-based
rejection sampling — different sequences from the same seed. The port emulates
the Python 2 mapping (`1 + int(Random(0).random() * 5)`) so fold membership is
bit-identical to the original split.

## RNG caveat

The original uses C `rand()`/`rand_r` (Marsaglia polar for normals,
Marsaglia-Tsang for Gamma) seeded with 10. These streams cannot be reproduced
in JAX; the port matches distributions and seed intent, and equivalence with
the original is verified statistically on the reported RMSE.
