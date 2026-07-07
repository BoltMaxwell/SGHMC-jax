"""Throughput benchmark: jax vs blackjax sampler kernels (synchronized timing).

Runs the figure3-style chains (2D correlated Gaussian) and a bayesnn-sized
pytree update, reporting median wall time per 1e6 steps after warmup, with
block_until_ready around every measured region.

Usage: JAX_PLATFORMS=cpu python benchmarks/bench_samplers.py [--steps 1000000]
"""

import argparse
import statistics
import time

import jax
import jax.numpy as jnp

from sghmc_jax.figures.figure3 import run_sghmc, run_sgld


def timed(fn, *args, repeats=5):
    fn(*args).block_until_ready()  # compile + warmup
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    key = jax.random.key(0)
    print(f"device: {jax.devices()[0].device_kind}, steps: {args.steps}")
    print(f"{'kernel':24s} {'backend':10s} {'median s':>10s} {'Msteps/s':>10s}")
    for backend in ("jax", "blackjax"):
        t = timed(lambda: run_sgld(key, backend, 0.05, args.steps), repeats=args.repeats)
        print(f"{'sgld (2D gaussian)':24s} {backend:10s} {t:10.3f} {args.steps / t / 1e6:10.2f}")
    for backend in ("jax", "blackjax"):
        t = timed(lambda: run_sghmc(key, backend, 0.01, 0.05, args.steps),
                  repeats=args.repeats)
        print(f"{'sghmc (2D gaussian)':24s} {backend:10s} {t:10.3f} {args.steps / t / 1e6:10.2f}")


if __name__ == "__main__":
    main()
