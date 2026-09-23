"""Statistics for the benchmark (spec §11.4): bootstrap CIs, paired permutation tests, Holm
correction, paired effect sizes, Cohen's kappa. Pure numpy, seeded, deterministic."""

from collections.abc import Hashable, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

Floats = Sequence[float] | NDArray[np.float64]


class CI(BaseModel):
    mean: float
    lo: float
    hi: float
    n: int


def bootstrap_ci(values: Floats, n_boot: int = 2000, alpha: float = 0.05, seed: int = 13) -> CI:
    """Percentile bootstrap CI of the mean. Empty input -> NaNs."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return CI(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(mean=float(x.mean()), lo=float(lo), hi=float(hi), n=int(x.size))


def paired_permutation_test(a: Floats, b: Floats, n_perm: int = 10000, seed: int = 13) -> float:
    """Two-sided sign-flip permutation test on paired differences (H0: mean diff = 0).
    Returns a p-value with the +1 correction so it is never exactly 0."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if d.size == 0:
        return 1.0
    observed = abs(d.mean())
    if observed == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= observed - 1e-12)) / (n_perm + 1))


def holm(pvalues: Floats) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def paired_effect_size(a: Floats, b: Floats) -> float:
    """Cohen's d_z for paired samples: mean(diff) / sd(diff). 0 when there is no variation."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if d.size < 2:
        return 0.0
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def cohens_kappa(rater1: Sequence[Hashable], rater2: Sequence[Hashable]) -> float:
    if len(rater1) != len(rater2) or not rater1:
        raise ValueError("raters must label the same non-empty set of items")
    cats = sorted(set(rater1) | set(rater2), key=str)
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(rater1, rater2, strict=True):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = float((m.sum(axis=0) * m.sum(axis=1)).sum() / n**2)
    return 1.0 if pe == 1.0 else float((po - pe) / (1 - pe))
