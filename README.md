SGHMC-jax
=========

A JAX reimplementation of the experiments from

> Tianqi Chen, Emily B. Fox, Carlos Guestrin.
> "Stochastic Gradient Hamiltonian Monte Carlo". ICML 2014.
> [arXiv:1402.4102](http://arxiv.org/abs/1402.4102)

This repository started as a fork of the original experiment code,
[tqchen/ML-SGHMC](https://github.com/tqchen/ML-SGHMC), and now contains only
the JAX port. Every experiment was verified against the original
implementation (matlab scripts run under Octave with the original kernels,
the numpy Bayesian NN, and the C++ matrix factorization) before the original
code was removed — see [docs/verification.md](docs/verification.md) for the
full comparison. The port was written with the assistance of Claude Code; all
credit for the method and the experimental design belongs to the original
authors:

```bibtex
@inproceedings{pmlr-v32-cheni14,
  title     = {Stochastic Gradient {H}amiltonian {M}onte {C}arlo},
  author    = {Chen, Tianqi and Fox, Emily and Guestrin, Carlos},
  booktitle = {Proceedings of the 31st International Conference on Machine Learning},
  pages     = {1683--1691},
  year      = {2014},
  volume    = {32},
  number    = {2},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v32/cheni14.html}
}
```

Scope of this port
------------------

| original | status |
|---|---|
| `matlab/figure1` — HMC variants on simulated targets | ported (`sghmc_jax/figures/figure1.py`) |
| `matlab/figure2` — phase-space traces | ported (`sghmc_jax/figures/figure2.py`) |
| `matlab/figure3` — SGLD vs SGHMC on a 2D Gaussian | ported (`sghmc_jax/figures/figure3.py`) |
| `bayesnn` — Bayesian MLP on MNIST (SGHMC/SGLD/SGD + Gibbs hyperpriors) | ported (`sghmc_jax/bayesnn/`) |
| `mf` — Bayesian matrix factorization on ml-1m (SGHMC/SGLD) | ported (`sghmc_jax/mf/`, spec in [docs/mf_spec.md](docs/mf_spec.md)) |
| `bayesnn` NAG updater | not ported (not used by any shipped experiment) |
| SVDFeature framework generality (feature files, other model types) | not ported (only the model the experiment uses) |

Samplers: two backends
----------------------

Every sampler is available through two interchangeable backends
(`sghmc_jax/samplers/`):

- **`jax`** — hand-rolled kernels, line-for-line faithful to the original
  update equations (each factory's docstring cites the original file).
- **`blackjax`** — the same samplers expressed through
  [blackjax](https://github.com/blackjax-devs/blackjax)'s
  `sgmcmc.diffusions`, with exact parameter mappings documented in
  `sghmc_jax/samplers/blackjax_backend.py`.

The SGLD mapping is stepwise identical to the original update. The SGHMC
mapping is coefficient-exact, but blackjax's diffusion advances the position
with the *previous* momentum (a qp- instead of pq-splitting of the same Euler
scheme). This has real consequences that this repo documents and tests
(`tests/test_samplers.py`, verified against the discrete Lyapunov solution):

- it only contracts on a Gaussian target when `eta < alpha`; at `eta = alpha`
  (the largest figure-3a setting) the chain's variance grows without bound;
- in the stable regime it inflates the stationary variance by roughly
  `alpha / (alpha - eta)`;
- on the figure-1 double well at `dt = 0.1` the resulting density bias is
  visible (L1 error to the true density 0.195 vs 0.021 for the faithful
  kernel).

Layout
------

```
sghmc_jax/
├── samplers/
│   ├── jax_backend.py       # faithful kernels (HMC, SGHMC x2 forms, SGLD, bayesnn updater)
│   └── blackjax_backend.py  # the same via blackjax.sgmcmc.diffusions
├── figures/                 # matlab/ ports: figure1.py, figure2.py, figure3.py
├── bayesnn/                 # MNIST Bayesian NN: data.py, train.py
└── mf/                      # ml-1m matrix factorization: data.py, train.py
docs/
├── mf_spec.md               # exact semantics extracted from the original C++
└── verification.md          # port-vs-original comparisons (the removal gate)
tests/                       # sampler + experiment tests (CPU, seconds)
benchmarks/                  # jax-vs-blackjax throughput
figs/                        # generated figures (committed)
```

Install
-------

```bash
pip install -e '.[plot]'
```

Simulated experiments (figures 1-3)
-----------------------------------

All three run in minutes on a laptop CPU:

```bash
python -m sghmc_jax.figures.figure1 --plot            # 80k samples x 5 methods x 2 targets
python -m sghmc_jax.figures.figure2 --plot            # phase-space traces
python -m sghmc_jax.figures.figure3 --plot            # L = 1e7 sweep + 50-step scatter
```

Each also writes an `.npz` with the raw histograms/statistics next to the
figure. `--backend blackjax` switches the SGHMC/SGLD kernels.

Bayesian neural network on MNIST
--------------------------------

```bash
python -m sghmc_jax.bayesnn.train --download --data_path data/mnist   # first run
python -m sghmc_jax.bayesnn.train --data_path data/mnist \
    --updater sghmc --backend jax --num_round 800 --out runs
```

Defaults reproduce `bayesnn/demo/mnist-sghmc.py`: 784-100-10 sigmoid MLP,
batch 500, eta 0.1, mdecay 0.01, per-group Gibbs Gamma hyperpriors
(`--hyperupdater gibbs-sep`), burn-in 50 rounds, running posterior-predictive
averaging. `--updater sgld` forces mdecay = 1 exactly as the original.
~4 minutes on an M-series CPU for the full 800 rounds.

Bayesian matrix factorization on MovieLens ml-1m
------------------------------------------------

```bash
python -m sghmc_jax.mf.train --data_path data --fold 1 \
    --sampler sghmc --backend jax --num_round 1000 --out runs
```

Downloads ml-1m on first use. Reproduces the original `SGHMC.conf` /
`SGLD.conf` experiment: 20 latent factors, batch 4000, 2000 minibatch steps
per round, 1000 rounds, five isotropic Gamma-sampled precisions resampled
once per round, equal-weight posterior-predictive averaging after 50 burn-in
rounds, RMSE reported on the star scale. The 5-fold split is bit-identical
to the original `mknfold.py` (Python 2 RNG emulated). ~1 h on CPU, ~6 min
on an H100.

Results
-------

Full port-vs-original tables are in [docs/verification.md](docs/verification.md).
Headline numbers (originals re-run, not quoted from the paper):

| experiment | original | JAX (jax backend) | JAX (blackjax backend) |
|---|---|---|---|
| figure 3a autocorrelation times (10 settings, L=1e7) | — | identical to the original at every setting | see splitting caveat |
| MNIST SGHMC, 800 rounds: test error | 1.63% | 1.68% (seeds 0-2: 1.67-1.74%) | 1.70% |
| ml-1m fold 1 SGHMC, 1000 rounds: test RMSE | 0.8404 | 0.8394 | 0.8393 |
| ml-1m fold 1 SGLD, 1000 rounds: test RMSE | 0.8404 | 0.8396 | 0.8396 |

Speed: migration gains
----------------------

Wall-clock for the full experiments, original implementation vs this port
(same configurations; local CPU is an Apple M-series, GPU is an H100):

| experiment | original | JAX (CPU) | JAX (H100) | GPU speedup |
|---|---|---|---|---|
| bayesnn MNIST, 800 rounds | numpy, 194 s | 105 s | 21.9 s | 8.9x |
| mf ml-1m, 1000 rounds | C++ (14 cores), ~25 min | ~60 min | 88.5 s | ~17x |
| figure 3a sweep, L = 1e7 x 10 | matlab/Octave, ~45 min | 4.1 min | — | 11x (CPU) |

The bayesnn original is vectorized numpy, so the CPU-to-CPU gain is modest
(~1.9x); the wins come from running the identical code on GPU, which the
original implementations cannot do.

Tests
-----

```bash
JAX_PLATFORMS=cpu python tests/test_samplers.py
JAX_PLATFORMS=cpu python tests/test_experiments.py
```

(or `pytest tests/`). The sampler tests include the bit-equivalence check
between backends for SGLD and the Lyapunov-verified stationary-variance
checks for both SGHMC splittings.

License
-------

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The upstream
code, [tqchen/ML-SGHMC](https://github.com/tqchen/ML-SGHMC), carries no license
of its own; this port is an independent reimplementation.
