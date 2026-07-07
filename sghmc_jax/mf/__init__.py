"""JAX port of mf/: Bayesian matrix factorization on MovieLens ml-1m with SGHMC/SGLD.

The exact semantics of the original C++ (SVDFeature-based) implementation are
documented in docs/mf_spec.md, which this port follows line by line.
"""
