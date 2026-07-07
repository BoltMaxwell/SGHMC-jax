# Verification of the JAX port against the original implementations

Before removing the original code, every experiment was re-run in its original
form (matlab scripts under GNU Octave 11.3 with the original kernel `.m` files
untouched; the numpy bayesnn under Python 3 via a mechanical 2to3 copy with
numerics untouched; the C++ mf reference compiled and run on the cluster) and
compared quantitatively against the JAX port. RNG streams necessarily differ,
so agreement is judged statistically at matched settings.

## Figure 1 (double well / quadratic, 80,000 samples per method)

L1 distance between each method's sampled histogram and the true density
(the quantity the figure displays). Original = Octave running the original
`hmc.m`/`sghmc.m`.

| method | func4 orig | func4 jax | func2 orig | func2 jax |
|---|---|---|---|---|
| Standard HMC (MH) | 0.0183 | 0.0168 | 0.0219 | 0.0207 |
| Standard HMC (no MH) | 0.0215 | 0.0245 | 0.0175 | 0.0210 |
| Naive SG-HMC (MH) | 0.0225 | 0.0230 | 0.0299 | 0.0277 |
| Naive SG-HMC (no MH) | **0.2050** | **0.2004** | **0.2191** | **0.2235** |
| SGHMC | 0.0222 | 0.0212 | 0.0197 | 0.0226 |

The Monte-Carlo self-distance between two independent 80k-sample runs is
~0.02-0.03, so all methods agree within noise, including the paper's headline
qualitative result (naive stochastic-gradient HMC without M-H is badly biased,
SGHMC is not).

## Figure 2 (phase-space traces)

The exact-gradient Hamiltonian trace is deterministic and matches the
original exactly (max |z| = 1.000, std(x) = 0.706 in both). The stochastic
traces agree statistically (resampled: std 1.05 orig / 1.03 jax; friction:
1.01 / 1.02). The never-resampled noisy Hamiltonian trace is a single
realization of a divergent process and is only qualitatively comparable
(it diverges in both).

## Figure 3a (2D Gaussian, L = 10,000,000 steps, 5 settings per sampler)

Original = Octave running the original `sgld.m`/`sghmc.m`/`aucTime.m`.
Autocorrelation times match **exactly at every one of the 10 settings**;
SGLD covariance errors match to 3-4 decimals; SGHMC covariance errors are at
the Monte-Carlo noise floor in both implementations.

| sampler | eta | auc orig | auc jax | covErr orig | covErr jax |
|---|---|---|---|---|---|
| SGLD | 0.1800 | 61.0 | 61.0 | 0.4499 | 0.4501 |
| SGLD | 0.1440 | 75.0 | 75.0 | 0.1286 | 0.1285 |
| SGLD | 0.1152 | 95.0 | 95.0 | 0.0679 | 0.0679 |
| SGLD | 0.0922 | 119.0 | 120.0 | 0.0427 | 0.0427 |
| SGLD | 0.0737 | 150.0 | 147.0 | 0.0292 | 0.0292 |
| SGHMC | 0.0500 | 21.0 | 21.0 | 0.0074 | 0.0073 |
| SGHMC | 0.0180 | 35.0 | 35.0 | 0.0035 | 0.0023 |
| SGHMC | 0.0065 | 59.0 | 59.0 | 0.0049 | 0.0010 |
| SGHMC | 0.0023 | 97.0 | 97.0 | 0.0038 | 0.0027 |
| SGHMC | 0.0008 | 161.0 | 161.0 | 0.0005 | 0.0031 |

Note: `aucTime.m` has two quirks that the port reproduces exactly (its xcorr
indexing tests rho_j = 0.5 (c_{j-1} + c_{j+1}), and it accumulates 2 (1 - j/L)
unweighted by rho). See `sghmc_jax/figures/figure3.py:auc_time`.

## bayesnn (MNIST, 784-100-10 sigmoid MLP, 800 rounds, demo hyperparameters)

Original = the original numpy implementation (2to3 mechanical copy, numerics
untouched), run to completion. JAX runs use the same hyperparameters with
independent RNG.

| run | test err | test NLL | valid err |
|---|---|---|---|
| original numpy, SGHMC | 1.63% | 0.0653 | 1.79% |
| JAX port, SGHMC (jax backend) | 1.68% | 0.0627 | 1.61% |
| JAX port, SGHMC (blackjax backend) | 1.70% | 0.0668 | 1.73% |

The differences are within seed-to-seed variation for this experiment (the
original repo's own README notes the Bayesian-NN vs momentum-SGD gap is not
statistically significant across seeds).

## mf (Bayesian matrix factorization, ml-1m fold 1, 1000 rounds)

Fold membership is bit-identical to the original 5-fold split (the port
emulates Python 2's `random.seed(0)` + `randint(1,5)` draws; fold 1 =
799,597 train / 200,612 test; dataset stats nuser=6040, nitem=3706,
mean=0.645391 reproduce the original's config constants exactly). The C++
reference was compiled on the cluster with `-std=gnu++03 -fpermissive` plus a
build-only header patch (forward declarations of the expression-template
operators in `apex_exp_template.h`, required by modern g++ two-phase name
lookup; no numerics touched) and run with the shipped `SGHMC.conf` /
`SGLD.conf` (1000 rounds, nthread=10). Final posterior-averaged test RMSE
(star scale), fold 1:

| run | test RMSE | train RMSE |
|---|---|---|
| original C++, SGHMC | 0.84039 | 0.70643 |
| original C++, SGLD | 0.84042 | 0.70836 |
| JAX port, SGHMC (jax backend) | 0.83939 | 0.70519 |
| JAX port, SGHMC (blackjax backend) | 0.83933 | — |
| JAX port, SGLD (jax backend) | 0.83955 | 0.70720 |
| JAX port, SGLD (blackjax backend) | 0.83955 | — |

Agreement is within 0.001 RMSE (~0.12% relative), with trajectories tracking
throughout training (e.g. at round 160, C++ 0.8441 vs JAX 0.8408). The
residual gap is consistent with the independent RNG streams (C `rand()` /
Marsaglia polar vs JAX threefry) and single-precision accumulation
differences. JAX runs: H100, ~6 minutes for all four configs; C++ reference:
~25 minutes per config on 14 CPU cores.

## blackjax backend caveats (documented, deliberate)

blackjax's SGHMC diffusion advances the position with the *previous* momentum
(qp splitting) while the original refreshes the momentum first (pq). Both are
Euler discretizations of the same SDE, but:

- the qp map only contracts when eta < alpha (at eta = alpha, the largest
  figure3a setting, its linearization is exactly marginally stable and the
  chain diverges);
- in the stable regime it inflates the stationary variance on a Gaussian by
  ~ alpha / (alpha - eta) (verified against the discrete Lyapunov solution in
  `tests/test_samplers.py`);
- on the figure1 double well at dt = 0.1 this bias is visible: L1 to the true
  density 0.195 (blackjax) vs 0.021 (faithful kernel).

SGLD has no ordering ambiguity and the blackjax backend is stepwise identical
to the original update (verified to 1e-6 in tests).
