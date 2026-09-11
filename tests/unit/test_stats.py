"""Bootstrap CIs, permutation tests, Holm correction, effect sizes and Cohen's kappa.

Covers bench/stats.py (partly property-based).
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fixgraph.bench.stats import (
    bootstrap_ci,
    cohens_kappa,
    holm,
    paired_effect_size,
    paired_permutation_test,
)


def test_bootstrap_ci_contains_mean_and_is_deterministic() -> None:
    x = [0.0, 0.5, 1.0, 1.0, 0.0, 1.0, 0.5, 1.0]
    ci = bootstrap_ci(x)
    assert ci.lo <= ci.mean <= ci.hi
    assert ci == bootstrap_ci(x)
    assert math.isclose(ci.mean, float(np.mean(x)))
    assert bootstrap_ci([1.0] * 5).lo == 1.0
    assert bootstrap_ci([]).n == 0


def test_permutation_detects_real_difference_and_not_noise() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.3, 60)
    b = a - 0.5 + rng.normal(0, 0.1, 60)
    assert paired_permutation_test(a, b) < 0.001
    c = a + rng.normal(0, 0.3, 60)
    assert paired_permutation_test(a, c) > 0.05
    assert paired_permutation_test([1, 1], [1, 1]) == 1.0


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(0, 1), min_size=2, max_size=30))
def test_permutation_symmetric(xs: list[float]) -> None:
    ys = [1 - x for x in xs]
    assert math.isclose(paired_permutation_test(xs, ys), paired_permutation_test(ys, xs))


def test_holm_matches_hand_computation() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]
    assert holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_effect_size() -> None:
    assert paired_effect_size([2, 3, 4], [1, 2, 3]) == 0.0  # constant diff -> sd 0
    assert paired_effect_size([2, 3, 5], [1, 2, 3]) > 0


def test_cohens_kappa() -> None:
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    # po = 35/50 = 0.7; marginals 35/15 and 30/20 -> pe = (35*30 + 15*20)/2500 = 0.54
    r1 = ["y"] * 25 + ["y"] * 10 + ["n"] * 5 + ["n"] * 10
    r2 = ["y"] * 25 + ["n"] * 10 + ["y"] * 5 + ["n"] * 10
    assert cohens_kappa(r1, r2) == pytest.approx((0.7 - 0.54) / 0.46)
    with pytest.raises(ValueError):
        cohens_kappa([1], [1, 2])
