import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import itertools as it
from collections import Counter, defaultdict, deque
from scipy.stats import bernoulli, binom
from functools import lru_cache

# ---------- Generative process ---------- #

def sample_table(n, p, q0, q1):
    """Sample a contingency table with the given parameters."""
    n1 = binom(n, p).rvs()
    n0 = n - n1
    k0 = binom(n0, q0).rvs()
    k1 = binom(n1, q1).rvs()
    return np.array([[n0 - k0, k0], [n1-k1, k1]])

def expected_cell_rate(p, q0, q1):
    return np.array([
        [(1-p) * (1 - q0), (1-p) * q0],
        [p * (1 - q1), p * q1]
    ])

def sample_sequence(n, p, q0, q1):
    x = bernoulli(p).rvs(n)
    y0 = bernoulli(q0).rvs(n)
    y1 = bernoulli(q1).rvs(n)
    y = np.where(x, y1, y0)
    return x, y


# ---------- Reparameterization ---------- #

def phi(p, q, q0, q1):
    """Expected correlation coefficient."""
    return (q1 - q0) * p * (1-p) * (p * (1-p) * q * (1-q)) ** -0.5

def get_q1(p, q, q0):
    """Choose q1 such that the marginal q is maintained."""
    return(q - (1-p) * q0) / p

from functools import update_wrapper

def factory(func):
    return update_wrapper(func(), func)

@factory
def get_q0():
    from sympy.solvers import solveset
    from sympy import symbols
    from sympy.utilities.lambdify import lambdify
    p, q, q0, q1, φ = symbols('p q q0 q1 φ', real=True)
    q1 = get_q1(p, q, q0)
    return lambdify([p, q, φ], 
                    solveset(phi(p, q, q0, q1) - φ, q0))


@lru_cache(None)
def get_q0_q1(p, q, φ):
    q0 = get_q0(p, q, φ).pop()
    q1 = get_q1(p, q, q0)
    if not (0 <= q0 <= 1) and (0 <= q1 <= 1):
        return np.nan, np.nan
    return q0, q1


# ---------- Inference ---------- #

GRID_SIZE = 101
GRID = np.linspace(0, 1, GRID_SIZE)

def normalize(x):
    x = x.astype(float)
    x /= x.sum()
    return x

def is_equal(x, y):
    return abs(x - y) < 1 / GRID_SIZE

from toolz import memoize
@memoize
def h1_prior(p, q):
    Q = p * GRID + ((1-p)* GRID)[:, None]
    return normalize(is_equal(Q, q))

@memoize
def h0_prior(q):
    return np.diag(normalize(is_equal(GRID, q)))

def likelihood_grid(tbl):
    """Binomial likelihood of the counts in tbl for a 2D GRID of q0 and q1 values."""
    n0, n1 = tbl.sum(axis=1)  # counts for x
    k0, k1 = tbl[:, 1]  # counts for x given y=1
    pk0 = binom(n0, p=GRID).pmf(k0)
    pk1 = binom(n1, p=GRID).pmf(k1)
    return np.outer(pk0, pk1)

def log_likelihood_ratio(p, q, tbl):
    L = likelihood_grid(tbl)
    h0 = (h0_prior(q) * L).sum()
    h1 = (h1_prior(p, q) * L).sum()
    return np.log(h1) - np.log(h0)


# ---------- Biased memory ---------- #

def forget(tbl, mem_rate, bias=None, ecr=None):
    """Stochastically drop (1-mem_rate) proportion of entries in tbl.
    
    If `bias` (a 2x2 array) is given, we make memory rate for each cell 
    proportion to the corresponding element of `bias`. Expected proportions
    of each cell must be provided"""
    if bias is None:
        cell_rates = np.ones(4) * mem_rate
    else:
        bias = np.array(bias, dtype=float)
        beta = mem_rate / np.dot(bias, ecr.flat) 
        cell_rates = beta * bias
        cell_rates = cell_rates.clip(0, 1)  # floating point error
    new_tbl = 0.1 * np.ones((2, 2)) + binom(tbl, cell_rates.reshape((2,2))).rvs()
    true_mem_rate = new_tbl.sum() / tbl.sum()
    return np.ceil(new_tbl / true_mem_rate).astype(int)


