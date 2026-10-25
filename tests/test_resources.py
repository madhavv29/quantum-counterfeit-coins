"""Resource accounting and classical baselines (Phase G)."""

import pytest

from quantum_counterfeit import model
from quantum_counterfeit.amplitude import ylc_schedule
from quantum_counterfeit.b_oracle import BalanceOracle, NullOracle, count_oracle_calls
from quantum_counterfeit.classical import GreedySolver, info_lower_bound, minimax_queries
from quantum_counterfeit.resources import compiled_counts, logical_inventory
from quantum_counterfeit.solver import build_find_star, build_one_coin


@pytest.mark.parametrize("L", [1, 3, 5, 7])
def test_call_graph_count_equals_schedule_count(L):
    o = BalanceOracle((0, 1, 0, 0, 1, 0))
    sc = build_find_star(o, L=L, delta=0.1)
    assert sc.counted_queries == 2 * L - 1 == sc.scheduled_queries
    assert 2 * ylc_schedule(L, 0.1).oracle_calls + 1 == 2 * L - 1


def test_null_oracle_has_same_call_graph_and_no_gates():
    real = build_find_star(BalanceOracle((0, 1, 0, 0, 1, 0)))
    null = build_find_star(NullOracle(6, 2))
    assert count_oracle_calls(real.circuit) == count_oracle_calls(null.circuit)
    assert logical_inventory(null.circuit)["total"] < logical_inventory(real.circuit)["total"]


def test_compiled_counts_small_circuit():
    sc = build_one_coin(BalanceOracle((0, 1, 0, 0)))
    c = compiled_counts(sc.circuit)
    assert c["cx"] > 0 and c["depth"] > 0


@pytest.mark.parametrize("N,k", [(4, 1), (5, 1), (6, 2), (7, 2), (8, 3)])
def test_greedy_uses_only_oracle_and_is_correct(N, k):
    g = GreedySolver(N, k)
    worst = 0
    for x in model.all_secrets(N, k):
        o = BalanceOracle(x)
        found, used = g.solve(o)
        assert found == x
        assert used == o.classical_calls
        worst = max(worst, used)
    assert worst >= info_lower_bound(N, k)  # the bound is a worst-case bound


@pytest.mark.parametrize("N,k,expected", [(4, 1, 2), (6, 1, 3), (6, 2, 4), (8, 2, 5)])
def test_minimax_small_instances(N, k, expected):
    assert minimax_queries(N, k, time_limit=60) == expected


# ------------------------------------------------- schedule analytics (ablation)
def test_fast_analytics_match_exact_versions():
    from quantum_counterfeit.experiments import overlap_fast, p_min_float
    from quantum_counterfeit.solver import predicted_ideal_overlap

    for N, k in [(6, 2), (8, 3), (10, 4), (13, 6), (20, 3)]:
        assert abs(p_min_float(N, k) - float(model.p_min(N, k))) < 1e-12
        for L, d in [(1, 0.3), (3, 0.1), (5, 0.05), (7, 0.2)]:
            assert abs(overlap_fast(N, k, L, d) - predicted_ideal_overlap(N, k, ylc_schedule(L, d))) < 1e-10


def test_tuned_rule_never_worse_and_meets_target_on_every_secret():
    from quantum_counterfeit.experiments import rule_exact_pmin, rule_generic, rule_tuned
    from quantum_counterfeit.solver import run

    T = 0.99
    for N, k in [(6, 2), (8, 3), (64, 8)]:
        Lt, _ = rule_tuned(N, k, T)
        # p_min(N,k) >= generic bound, so the exact rule never needs a longer sequence
        assert Lt <= rule_exact_pmin(N, k, T)[0] <= rule_generic(N, k, T)[0]
    L, d = rule_tuned(6, 2, T)
    for x in model.all_secrets(6, 2):
        r = run(build_find_star(BalanceOracle(x), delta=d, L=L))
        assert r.success_probability(x) >= T
