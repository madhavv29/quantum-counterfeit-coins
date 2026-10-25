"""Exhaustive tests of the classical reference model (Phase A)."""

from itertools import product
from math import comb

import pytest

from quantum_counterfeit import model
from quantum_counterfeit.classical import all_distinguishable, or_shortcut, parity_shortcut

pytestmark = pytest.mark.reference
SIZES = [(4, 1), (5, 1), (5, 2), (6, 1), (6, 2), (7, 1), (7, 2), (7, 3), (8, 3)]


@pytest.mark.parametrize("N,k", [(4, 2), (6, 3), (5, 3), (4, 0), (1, 0)])
def test_k_below_half_is_enforced(N, k):
    with pytest.raises(ValueError):
        model.validate_params(N, k)


@pytest.mark.parametrize("N,k", SIZES)
def test_secrets_have_weight_k(N, k):
    xs = model.all_secrets(N, k)
    assert len(xs) == comb(N, k) == len(set(xs))
    assert all(sum(x) == k and set(x) <= {0, 1} for x in xs)


@pytest.mark.parametrize("N", range(2, 9))
def test_legal_queries_have_equal_pans(N):
    qs = list(model.legal_queries(N))
    assert len(qs) == model.num_legal_queries(N)
    for q in qs:
        assert sum(q) == 0
        left, right = model.pan_sizes(q)
        assert left == right
    # completeness: every q in {-1,0,1}^N with sum 0 is listed exactly once
    assert set(qs) == {q for q in product((-1, 0, 1), repeat=N) if sum(q) == 0}


@pytest.mark.parametrize("N,k", SIZES)
def test_chi_matches_weight_difference_and_complement(N, k):
    for x in model.all_secrets(N, k):
        xc = model.complement(x)
        for q in model.legal_queries(N):
            d = sum(a * b for a, b in zip(x, q))
            assert model.chi(x, q) == (0 if d == 0 else 1)
            assert model.chi(x, q) == model.chi(xc, q)  # B-oracle cannot see complement


def test_illegal_query_rejected():
    with pytest.raises(ValueError):
        model.chi((1, 0, 0, 0), (1, 1, -1, 0))


def test_required_counterexample_N6_k2():
    q = (+1, +1, -1, -1, 0, 0)
    x_same_pan = (1, 1, 0, 0, 0, 0)
    x_split = (1, 0, 1, 0, 0, 0)
    # both selections contain an EVEN number (2) of counterfeits ...
    assert parity_shortcut(x_same_pan, q) == parity_shortcut(x_split, q) == 0
    # ... yet one tilts and one balances
    assert model.chi(x_same_pan, q) == 1
    assert model.chi(x_split, q) == 0


@pytest.mark.parametrize("N,k", [(5, 2), (6, 2), (7, 3)])
def test_parity_and_or_shortcuts_are_wrong_for_k_gt_1(N, k):
    bad_par = bad_or = 0
    for x in model.all_secrets(N, k):
        for q in model.legal_queries(N):
            bad_par += parity_shortcut(x, q) != model.chi(x, q)
            bad_or += or_shortcut(x, q) != model.chi(x, q)
    assert bad_par > 0 and bad_or > 0


@pytest.mark.parametrize("N", [4, 5, 6, 7])
def test_k1_balance_equals_parity_of_selection(N):
    """For k = 1 the B-oracle IS the inner-product oracle (paper §1)."""
    for x in model.all_secrets(N, 1):
        for q in model.legal_queries(N):
            assert model.chi(x, q) == parity_shortcut(x, q)


@pytest.mark.parametrize("N,k", SIZES)
def test_all_secrets_distinguishable(N, k):
    assert all_distinguishable(N, k)


@pytest.mark.parametrize("N", [4, 5, 6])
def test_encoding_reference(N):
    for sel in product((0, 1), repeat=N):
        for side in product((0, 1), repeat=N):
            q = model.query_from_encoding(sel, side)
            assert all((qi != 0) == bool(si) for qi, si in zip(q, sel))
            assert model.encoding_is_legal(sel, side) == (sum(q) == 0)


@pytest.mark.parametrize("N", [5, 6, 7])
def test_ideal_parity_decoding_lemma1(N):
    """H^N applied to the ideal parity state gives (|x> + |x̄>)/sqrt 2 (Lemma 1)."""
    import numpy as np

    for k in range(1, (N + 1) // 2):
        for x in model.all_secrets(N, k):
            psi = model.ideal_parity_state(x)
            out = {}
            for z in product((0, 1), repeat=N):
                out[z] = sum(a * (-1) ** sum(si & zi for si, zi in zip(s, z)) for s, a in psi.items()) / 2 ** (N / 2)
            for z, a in out.items():
                expect = 1 / np.sqrt(2) if z in (x, model.complement(x)) else 0.0
                assert abs(a - expect) < 1e-12
            assert model.decode_with_weight(model.complement(x), k) == x
