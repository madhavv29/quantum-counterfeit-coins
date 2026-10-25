"""End-to-end solvers (Phases C and F) and anti-cheating checks."""

import inspect
from math import comb

import numpy as np
import pytest
from qiskit import QuantumCircuit

import quantum_counterfeit.amplitude as amplitude_mod
import quantum_counterfeit.parity_approx as parity_mod
import quantum_counterfeit.solver as solver_mod
import quantum_counterfeit.state_prep as prep_mod
from quantum_counterfeit import model
from quantum_counterfeit.b_oracle import BalanceOracle, phase_oracle
from quantum_counterfeit.solver import (
    build_find_star,
    build_one_coin,
    predicted_ideal_overlap,
    run,
    solve,
)
from quantum_counterfeit.sparse_sim import SparseState
from quantum_counterfeit.state_prep import alternating_partition_circuit, w_ghz_circuit


# ------------------------------------------------------------------ k = 1
@pytest.mark.k1
@pytest.mark.parametrize("N", [3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("w_variant", ["paper", "ghz"])
def test_one_coin_exact_with_one_query(N, w_variant):
    for x in model.all_secrets(N, 1):
        sc = build_one_coin(BalanceOracle(x), w_variant)
        assert sc.counted_queries == 1
        r = run(sc)
        assert r.success_probability(x) == pytest.approx(1.0, abs=1e-12)
        assert r.clean_probability == pytest.approx(1.0, abs=1e-12)


@pytest.mark.k1
@pytest.mark.parametrize("N", [4, 5, 6])
def test_one_coin_amplitudes_after_hadamards(N):
    """Before the final GHZ un-preparation, H^N yields (|x> + |x̄>)/sqrt 2."""
    for x in model.all_secrets(N, 1):
        o = BalanceOracle(x)
        lay = o.layout
        qc = QuantumCircuit(lay.num_qubits)
        qc.compose(w_ghz_circuit(N), qubits=list(range(N)), inplace=True)
        alt = alternating_partition_circuit(N)
        qc.compose(alt, qubits=list(range(2 * N)), inplace=True)
        qc.compose(phase_oracle(lay, o.bit_gate()), inplace=True)
        qc.compose(alt.inverse(), qubits=list(range(2 * N)), inplace=True)
        for i in range(N):
            qc.h(i)
        st = SparseState.basis(lay.num_qubits).evolve(qc).as_dict()
        xi = sum(b << i for i, b in enumerate(x))
        xc = sum((1 - b) << i for i, b in enumerate(x))
        assert set(st) == {xi, xc}
        phase = st[xi] / abs(st[xi])
        for key in (xi, xc):
            assert abs(st[key] / phase - 1 / np.sqrt(2)) < 1e-12


# ------------------------------------------------------- coherent k > 1
def _check_instance(N, k, x, delta=0.1, w_variant="paper"):
    sc = build_find_star(BalanceOracle(x), delta=delta, w_variant=w_variant)
    r = run(sc)
    bound = predicted_ideal_overlap(N, k, sc.schedule) ** 2
    p = r.success_probability(x)
    assert sc.counted_queries == sc.scheduled_queries == 2 * sc.schedule.L - 1
    assert p >= bound - 1e-9
    assert p >= 0.99
    for ans in r.answer_probs:
        assert ans is None or (len(ans) == N and sum(ans) == k)  # full k-set
    return p


@pytest.mark.coherent_multi
@pytest.mark.parametrize("w_variant", ["paper", "ghz"])
def test_find_star_exhaustive_6_2(w_variant):
    ps = [_check_instance(6, 2, x, w_variant=w_variant) for x in model.all_secrets(6, 2)]
    assert len(ps) == comb(6, 2)


@pytest.mark.coherent_multi
def test_find_star_exhaustive_8_2():
    ps = [_check_instance(8, 2, x) for x in model.all_secrets(8, 2)]
    assert len(ps) == 28


@pytest.mark.coherent_multi
@pytest.mark.slow
def test_find_star_exhaustive_8_3():
    ps = [_check_instance(8, 3, x) for x in model.all_secrets(8, 3)]
    assert len(ps) == 56


@pytest.mark.coherent_multi
@pytest.mark.slow
def test_find_star_exhaustive_10_4():
    ps = [_check_instance(10, 4, x) for x in model.all_secrets(10, 4)]
    assert len(ps) == 210


@pytest.mark.coherent_multi
def test_find_star_sampled_shots_agree_with_exact():
    x = (0, 0, 1, 0, 1, 0, 0, 1)
    r = run(build_find_star(BalanceOracle(x), delta=0.1))
    shots = r.sample(4000, seed=7)
    emp = sum(1 for a in shots if a == x) / len(shots)
    p = r.success_probability(x)
    assert abs(emp - p) < 4 * np.sqrt(p * (1 - p) / len(shots)) + 1e-3


@pytest.mark.coherent_multi
def test_solve_returns_full_counterfeit_set():
    for x in [(1, 0, 0, 1, 0, 0), (0, 1, 0, 0, 1, 1, 0, 0)]:
        assert solve(BalanceOracle(x), seed=3) == x


# ------------------------------------------------------------ anti-cheating
class GuardedOracle:
    """Exposes ONLY the public interface; any other attribute access fails."""

    def __init__(self, inner: BalanceOracle):
        object.__setattr__(self, "_inner", inner)

    def __getattribute__(self, name):
        if name in {"N", "k", "layout", "bit_gate"}:
            return getattr(object.__getattribute__(self, "_inner"), name)
        raise AttributeError(f"solver tried to access forbidden attribute {name!r}")


@pytest.mark.coherent_multi
def test_solver_uses_only_the_public_oracle_interface():
    x = (0, 1, 1, 0, 0, 0)
    sc = build_find_star(GuardedOracle(BalanceOracle(x)))
    assert run(sc).success_probability(x) > 0.99


@pytest.mark.coherent_multi
def test_solver_follows_the_gate_not_decoy_attributes():
    x_true, x_decoy = (0, 0, 0, 1, 1, 0), (1, 1, 0, 0, 0, 0)
    o = BalanceOracle(x_true)
    o.x = x_decoy  # planted misleading attributes
    o.secret = x_decoy
    r = run(build_find_star(o))
    assert r.success_probability(x_true) > 0.99
    assert r.success_probability(x_decoy) < 1e-3


def test_solver_source_never_touches_secret_or_gate_internals():
    for mod in (solver_mod, parity_mod, prep_mod, amplitude_mod):
        src = inspect.getsource(mod)
        assert "_BalanceOracle__x" not in src
        assert ".definition" not in src
        assert "classical_query" not in src


@pytest.mark.coherent_multi
def test_success_is_not_from_state_initialisation():
    """The circuit contains no initialize/unitary/statevector-preparation ops."""
    sc = build_find_star(BalanceOracle((1, 0, 1, 0, 0, 0)))

    def walk(qc):
        for inst in qc.data:
            op = inst.operation
            assert op.name not in {"initialize", "unitary", "state_preparation", "isometry"}
            if op.name in {
                "W", "W_dg", "ParityBlock", "U_amp", "U_amp_dg", "A_part", "A_part_dg", "S_t", "S_0",
                "B_phase", "S_t_dg", "S_0_dg", "B_phase_dg"}:
                walk(op.definition)

    walk(sc.circuit)


# --------------------------------------------------------------- Aer checks
@pytest.mark.aer
@pytest.mark.k1
def test_one_coin_aer_matches_sparse():
    from quantum_counterfeit.aer_backend import aer_statevector

    sc = build_one_coin(BalanceOracle((0, 0, 1, 0)), "paper")
    vec = aer_statevector(sc.circuit)
    sp = SparseState.basis(sc.num_qubits).evolve(sc.circuit).to_dense()
    assert np.allclose(vec, sp, atol=1e-10)


@pytest.mark.aer
@pytest.mark.coherent_multi
@pytest.mark.slow
def test_find_star_6_2_aer_statevector_and_shots_match_sparse():
    from quantum_counterfeit.aer_backend import aer_counts, aer_statevector

    x = (0, 1, 0, 0, 1, 0)
    sc = build_find_star(BalanceOracle(x), w_variant="ghz")
    assert sc.num_qubits <= 24
    vec = aer_statevector(sc.circuit)
    sp = SparseState.basis(sc.num_qubits).evolve(sc.circuit).to_dense()
    assert np.max(np.abs(vec - sp)) < 1e-9
    shots = 4000
    counts = aer_counts(sc.circuit, sc.S_qubits, shots, seed=11)
    hits = 0
    for bitstr, c in counts.items():
        bits = tuple(int(b) for b in reversed(bitstr))
        if solver_mod.decode(bits, 2, "ghz") == x:
            hits += c
    p = run(sc).success_probability(x)
    assert abs(hits / shots - p) < 4 * np.sqrt(max(p * (1 - p), 1e-4) / shots) + 2e-3


@pytest.mark.aer
def test_aer_rewrite_of_controlled_ry_preserves_the_unitary():
    from qiskit.circuit.library import RYGate
    from qiskit.quantum_info import Operator

    from quantum_counterfeit.aer_backend import aer_native
    from quantum_counterfeit.state_prep import partition_prep_circuit

    A = partition_prep_circuit(3)
    assert Operator(A) == Operator(aer_native(A))  # exact, including global phase
    qc = QuantumCircuit(5, global_phase=0.4)
    qc.append(RYGate(0.77).control(3, ctrl_state=5, annotated=False), [0, 1, 2, 3])
    qc.h(4)
    sub = QuantumCircuit(5, global_phase=0.3)
    sub.x(0)
    qc.append(sub.to_gate(label="sub"), range(5))
    assert Operator(qc) == Operator(aer_native(qc))
