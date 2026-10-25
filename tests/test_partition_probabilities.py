"""Balanced-partition combinatorics (paper Lemma 2) — exhaustive."""

from fractions import Fraction
from math import comb, sqrt

import pytest

from quantum_counterfeit import model

pytestmark = pytest.mark.reference


@pytest.mark.parametrize("N", [2, 4, 5, 6, 7, 8])
def test_splits_are_exactly_the_legal_equal_pan_splits(N):
    for s in model.even_selections(N):
        splits = model.balanced_splits(s)
        w = sum(s)
        assert len(splits) == comb(w, w // 2) == len(set(splits))
        I = set(model.support(s))
        for q in splits:
            assert model.is_legal(q)
            assert {i for i, v in enumerate(q) if v} == I


@pytest.mark.parametrize("N,k", [(4, 1), (5, 2), (6, 2), (7, 2), (7, 3), (8, 3)])
def test_enumerated_probability_equals_formula(N, k):
    for x in model.all_secrets(N, k):
        for s in model.even_selections(N):
            w, m = sum(s), model.m_of(s, x)
            p_enum = model.balanced_probability_enum(s, x)
            p_form = model.balanced_probability_formula(w, m)
            assert p_enum == p_form
            if m % 2:
                assert p_enum == 0  # odd m: never balanced
            else:
                assert p_enum > 0  # even m: at least one balanced split
            if m == 0:
                assert p_enum == 1


@pytest.mark.parametrize("N,k", [(4, 1), (6, 2), (7, 3), (8, 2), (8, 3)])
def test_p_min_equals_exhaustive_minimum_and_ignores_secret(N, k):
    exhaustive = min(
        model.balanced_probability_enum(s, x)
        for x in model.all_secrets(N, k)
        for s in model.even_selections(N)
        if model.balanced_probability_enum(s, x) > 0
    )
    assert model.p_min(N, k) == exhaustive


@pytest.mark.parametrize("N,k", [(6, 2), (8, 3), (10, 4)])
def test_multiplicities_are_secret_independent(N, k):
    for x in model.all_secrets(N, k)[:5]:
        counts = {}
        for s in model.even_selections(N):
            key = (sum(s), model.m_of(s, x))
            counts[key] = counts.get(key, 0) + 1
        assert counts == {(w, m): c for w, m, c in model.admissible_wm(N, k)}


def test_N_independent_lower_bound():
    """p(w, m) >= C(m, m/2) / 2^m >= 1/sqrt(2m) for every even m >= 2 and w >= m,
    so p_min(N, k) >= 1/sqrt(2k) independent of N (the paper's Omega(1/sqrt k))."""
    for w in range(2, 41, 2):
        for m in range(2, w + 1, 2):
            p = model.balanced_probability_formula(w, m)
            lb = Fraction(comb(m, m // 2), 2**m)
            assert p >= lb
            assert float(lb) >= 1 / sqrt(2 * m) - 1e-15
    for N in range(5, 30):
        for k in range(1, (N + 1) // 2):
            if 2 * k < N:
                assert float(model.p_min(N, k)) >= 1 / sqrt(2 * k) - 1e-15
