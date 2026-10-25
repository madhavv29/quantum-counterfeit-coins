"""Circuit-level B-oracle (Phase B): exhaustive agreement with chi, clean
scratch, invalid-encoding convention, unitarity, phase kick-back."""

from itertools import product

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from quantum_counterfeit import model
from quantum_counterfeit.b_oracle import (
    BalanceOracle,
    NullOracle,
    OracleLayout,
    count_oracle_calls,
    phase_oracle,
)
from quantum_counterfeit.sparse_sim import SparseState, permute_basis

CASES = [(4, 1), (5, 1), (5, 2), (6, 1), (6, 2), (7, 3)]


def _pack(lay: OracleLayout, sel, side, out=0, E=0, D=0) -> int:
    N = lay.N
    v = 0
    for i in range(N):
        v |= sel[i] << i
        v |= side[i] << (N + i)
    v |= out << (2 * N)
    v |= E << (2 * N + 1)
    v |= D << (2 * N + 1 + lay.bE)
    return v


def _bit_circuit(o: BalanceOracle) -> QuantumCircuit:
    qc = QuantumCircuit(o.layout.num_qubits)
    qc.append(o.bit_gate(), qc.qubits)
    return qc


@pytest.mark.parametrize("N,k", CASES)
def test_bit_oracle_matches_reference_on_every_encoding(N, k):
    secrets = model.all_secrets(N, k)
    for x in secrets[:: max(1, len(secrets) // 6)]:
        o = BalanceOracle(x)
        lay = o.layout
        encs = [(sel, side, out) for sel in product((0, 1), repeat=N)
                for side in product((0, 1), repeat=N) for out in (0, 1)]
        inputs = np.array([_pack(lay, s, t, b) for s, t, b in encs], dtype=np.int64)
        outputs = permute_basis(_bit_circuit(o), inputs)
        for (sel, side, out), v in zip(encs, outputs):
            expect = _pack(lay, sel, side, out ^ model.oracle_bit_reference(x, sel, side))
            assert int(v) == expect  # correct answer, sel/side untouched, scratch clean


@pytest.mark.parametrize("N,k", [(4, 1), (5, 2)])
def test_bit_oracle_is_bijective_and_self_inverse_on_full_space(N, k):
    """Unitarity on the COMPLETE basis, including dirty scratch inputs."""
    o = BalanceOracle(model.all_secrets(N, k)[1])
    n = o.layout.num_qubits
    allin = np.arange(2**n, dtype=np.int64)
    once = permute_basis(_bit_circuit(o), allin)
    assert np.unique(once).size == 2**n
    twice = permute_basis(_bit_circuit(o), once)
    assert np.array_equal(twice, allin)


def test_bit_oracle_operator_unitary_N3():
    o = BalanceOracle((0, 1, 0))
    U = Operator(_bit_circuit(o)).data
    assert np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-12)
    assert np.allclose(U @ U, np.eye(U.shape[0]), atol=1e-12)  # self-inverse


@pytest.mark.parametrize("N,k", [(4, 1), (5, 2), (6, 2)])
def test_phase_oracle_signs_and_invalid_identity(N, k):
    x = model.all_secrets(N, k)[-1]
    o = BalanceOracle(x)
    lay = o.layout
    ph = phase_oracle(lay, o.bit_gate())
    encs = [(sel, side) for sel in product((0, 1), repeat=N) for side in product((0, 1), repeat=N)]
    amp = 1 / np.sqrt(len(encs))
    st = SparseState.from_dict(lay.num_qubits, {_pack(lay, s, t): amp for s, t in encs})
    res = st.evolve(ph).as_dict(tol=0.0)
    n_illegal = 0
    for sel, side in encs:
        key = _pack(lay, sel, side)
        q = model.query_from_encoding(sel, side)
        if model.is_legal(q):
            sign = (-1) ** model.chi(x, q)
        else:
            sign = 1  # documented convention: identity on unequal-pan encodings
            n_illegal += 1
        assert abs(res[key] - sign * amp) < 1e-12
    assert n_illegal > 0
    assert len(res) == len(encs)  # nothing leaked into out/scratch


def test_phase_oracle_counterexample_at_circuit_level():
    lay = OracleLayout.for_params(6, 2)
    sel, side = (1, 1, 1, 1, 0, 0), (1, 1, 0, 0, 0, 0)  # q = (+1,+1,-1,-1,0,0)
    for x, expect in [((1, 1, 0, 0, 0, 0), 1), ((1, 0, 1, 0, 0, 0), 0)]:
        o = BalanceOracle(x)
        out = permute_basis(_bit_circuit(o), np.array([_pack(lay, sel, side)]))[0]
        assert (int(out) >> 12) & 1 == expect


@pytest.mark.aer
def test_phase_oracle_matches_qiskit_statevector_and_aer():
    from quantum_counterfeit.aer_backend import aer_statevector

    o = BalanceOracle((0, 0, 1, 0))
    lay = o.layout
    qc = QuantumCircuit(lay.num_qubits)
    for i in range(2 * lay.N):
        qc.h(i)  # superposition over all (sel, side) encodings
    qc.compose(phase_oracle(lay, o.bit_gate()), inplace=True)
    ref = Statevector(qc).data
    assert np.allclose(aer_statevector(qc), ref, atol=1e-10)
    st = SparseState.basis(lay.num_qubits).evolve(qc)
    assert np.allclose(st.to_dense(), ref, atol=1e-12)


def test_query_counting_includes_inverse_and_nesting():
    o = BalanceOracle((1, 0, 0, 0, 0))
    g = o.bit_gate()
    inner = QuantumCircuit(o.layout.num_qubits)
    inner.append(g, inner.qubits)
    inner.append(g.inverse(), inner.qubits)
    outer = QuantumCircuit(o.layout.num_qubits)
    outer.append(inner.to_gate(label="inner"), outer.qubits)
    outer.append(inner.to_gate(label="inner").inverse(), outer.qubits)
    outer.compose(phase_oracle(o.layout, g), inplace=True)
    assert count_oracle_calls(outer) == 5
    n = NullOracle(5, 1)
    assert n.layout == o.layout and n.bit_gate().name == g.name


def test_secret_not_exposed_by_repr():
    o = BalanceOracle((0, 1, 0, 0, 0))
    assert "1" not in repr(o).replace("k=1", "")
    assert not hasattr(o, "x") and not hasattr(o, "secret")


def test_classical_queries_are_counted_and_illegal_rejected():
    o = BalanceOracle((0, 1, 0, 0, 0))
    assert o.classical_query((1, -1, 0, 0, 0)) == 1
    assert o.classical_query((1, 0, -1, 0, 0)) == 0
    with pytest.raises(ValueError):
        o.classical_query((1, 1, -1, 0, 0))
    assert o.classical_calls == 3
