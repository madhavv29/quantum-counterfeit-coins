"""Coherent amplification and the balance-to-parity phase (Phase E).

Branch values are MEASURED from one coherent simulation over all s, then
compared with the exact 2x2 prediction — not asserted from theory alone."""

import numpy as np
import pytest

from quantum_counterfeit import model
from quantum_counterfeit.amplitude import (
    choose_L,
    grover_worst_case,
    simulate_two_level,
    two_level_phase_value,
    width,
    ylc_schedule,
    ylc_success,
)
from quantum_counterfeit.b_oracle import BalanceOracle
from quantum_counterfeit.solver import branch_diagnostics


@pytest.mark.parametrize("L", [1, 3, 5, 7, 9, 13])
@pytest.mark.parametrize("delta", [0.5, 0.2, 0.1, 0.03])
def test_ylc_formula_matches_exact_two_level_evolution(L, delta):
    s = ylc_schedule(L, delta)
    for lam in np.linspace(0, 1, 201):
        assert abs(abs(simulate_two_level(lam, s)) ** 2 - ylc_success(lam, L, delta)) < 1e-9


@pytest.mark.parametrize("delta", [0.2, 0.1, 0.05])
@pytest.mark.parametrize("pm", [0.6, 0.47, 0.25, 0.1])
def test_guarantee_above_p_min_and_zero_branch_preserved(pm, delta):
    L = choose_L(pm, delta)
    s = ylc_schedule(L, delta)
    assert width(L, delta) <= pm + 1e-15
    if L > 1:
        assert width(L - 2, delta) > pm  # minimality
    for lam in np.linspace(pm, 1, 300):
        assert abs(simulate_two_level(lam, s)) ** 2 >= 1 - delta**2 - 1e-12
    assert abs(simulate_two_level(0.0, s)) < 1e-15  # p = 0 stays exactly unamplified
    assert two_level_phase_value(0.0, s) == pytest.approx(-1.0, abs=1e-15)


@pytest.mark.parametrize("N,k", [(8, 2), (10, 4), (12, 5)])
def test_fixed_iteration_grover_cannot_amplify_all_branches(N, k):
    """Why QSearch cannot be replaced by plain Grover with a fixed count."""
    ps = [float(model.balanced_probability_formula(w, m)) for w, m, _ in model.admissible_wm(N, k)]
    ps = [p for p in ps if p > 0]
    for j, worst in grover_worst_case(ps, 10)[1:]:
        assert worst < 0.99


@pytest.mark.coherent_multi
@pytest.mark.parametrize("N,k,stride", [(6, 2, 1), (8, 3, 11)])
def test_branchwise_phase_and_leakage_measured(N, k, stride):
    delta = 0.1
    for x in model.all_secrets(N, k)[::stride]:
        recs, sc = branch_diagnostics(BalanceOracle(x), x, delta=delta)
        sched = sc.schedule
        assert sc.counted_queries == 2 * sched.L - 1
        for r in recs:
            assert abs(r.value.imag) < 1e-9
            if r.m % 2:  # p = 0: exact phase -1, nothing leaks
                assert r.p == 0
                assert abs(r.value + 1) < 1e-9 and r.leakage < 1e-9
            else:  # p >= p_min: phase ~ +1 within the fixed-point guarantee
                pred = two_level_phase_value(r.p, sched)
                assert abs(r.value.real - pred) < 1e-9
                assert r.value.real >= 1 - 2 * delta**2 - 1e-9
                assert abs(r.value.real**2 + r.leakage**2 - 1) < 1e-9


@pytest.mark.coherent_multi
def test_p_zero_vs_positive_distinguished_by_sign():
    x = (0, 1, 0, 1, 1, 0, 0, 0)
    recs, _ = branch_diagnostics(BalanceOracle(x), x, delta=0.2)
    for r in recs:
        assert np.sign(r.value.real) == (1 if r.p > 0 else -1)
