"""Statistics for the benchmark (spec §11.4): bootstrap CIs, paired permutation tests, Holm
correction, paired effect sizes, Cohen's kappa (plain and linear-weighted). Pure numpy,
seeded, deterministic.

Used by: bench/run.py (report stage), bench/validate.py (Cohen's kappa for judge agreement).
"""

# Imports: numpy for the resampling math, pydantic for the small CI result object.
from collections.abc import Hashable, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

# Accept either a plain list of floats or a numpy array.
Floats = Sequence[float] | NDArray[np.float64]


# A mean with its lower and upper confidence bounds and the sample size.
class CI(BaseModel):
    mean: float
    lo: float
    hi: float
    n: int


# Resample the questions many times to get a range for the mean, not one lucky number.
def bootstrap_ci(values: Floats, n_boot: int = 2000, alpha: float = 0.05, seed: int = 13) -> CI:
    """Percentile bootstrap CI of the mean. Empty input -> NaNs."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return CI(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    # Seeded RNG so the same input always gives the same interval.
    rng = np.random.default_rng(seed)
    # Each row is one resample of question indices drawn with replacement.
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(mean=float(x.mean()), lo=float(lo), hi=float(hi), n=int(x.size))


# Is system A really different from B on the same questions? Returns a p-value.
def paired_permutation_test(a: Floats, b: Floats, n_perm: int = 10000, seed: int = 13) -> float:
    """Two-sided sign-flip permutation test on paired differences (H0: mean diff = 0).
    Returns a p-value with the +1 correction so it is never exactly 0."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if d.size == 0:
        return 1.0
    observed = abs(d.mean())
    if observed == 0:
        return 1.0
    # Randomly flip the sign of each paired difference to simulate "no real difference".
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= observed - 1e-12)) / (n_perm + 1))


# Adjust p-values for testing many system pairs at once, so we do not over-claim wins.
def holm(pvalues: Floats) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    # Walk from smallest p-value up, scaling by remaining tests and keeping values non-decreasing.
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


# Effect size of a paired comparison: how big the gap is relative to its spread.
def paired_effect_size(a: Floats, b: Floats) -> float:
    """Cohen's d_z for paired samples: mean(diff) / sd(diff). 0 when there is no variation."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if d.size < 2:
        return 0.0
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


# Agreement between two raters beyond chance, e.g. LLM judge versus a human label.
def cohens_kappa(rater1: Sequence[Hashable], rater2: Sequence[Hashable]) -> float:
    if len(rater1) != len(rater2) or not rater1:
        raise ValueError("raters must label the same non-empty set of items")
    # Build a confusion matrix of rater1 labels against rater2 labels.
    cats = sorted(set(rater1) | set(rater2), key=str)
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(rater1, rater2, strict=True):
        m[idx[x], idx[y]] += 1
    # Observed agreement is the diagonal; expected agreement comes from each rater's label mix.
    n = m.sum()
    po = np.trace(m) / n
    pe = float((m.sum(axis=0) * m.sum(axis=1)).sum() / n**2)
    return 1.0 if pe == 1.0 else float((po - pe) / (1 - pe))


# Kappa for ordered scores, where being off by one step is only a partial disagreement.
def weighted_kappa(rater1: Sequence[float], rater2: Sequence[float]) -> float:
    """Linear-weighted Cohen's kappa for ordinal scores (e.g. 0 / 0.5 / 1): a one-step
    disagreement costs half of a two-step one. Weights are |a - b| / (max - min)."""
    if len(rater1) != len(rater2) or not rater1:
        raise ValueError("raters must label the same non-empty set of items")
    a = np.asarray(rater1, dtype=np.float64)
    b = np.asarray(rater2, dtype=np.float64)
    cats = np.unique(np.concatenate([a, b]))
    span = float(cats.max() - cats.min())
    if span == 0:
        return 1.0
    # Observed average distance between the two raters, scaled to 0..1.
    observed = float(np.abs(a - b).mean()) / span
    # Expected distance if the raters labeled independently with their own label frequencies.
    pa = np.array([(a == c).mean() for c in cats])
    pb = np.array([(b == c).mean() for c in cats])
    expected = float((np.outer(pa, pb) * np.abs(cats[:, None] - cats[None, :])).sum()) / span
    return 1.0 if expected == 0 else 1.0 - observed / expected
