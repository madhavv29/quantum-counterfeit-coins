"""W (Appendix A) and the coherent s-controlled partition preparation A (Phase D)."""

from itertools import product
from math import comb

import numpy as np
import pytest
from qiskit import QuantumCircuit

from quantum_counterfeit import model
from quantum_counterfeit.sparse_sim import SparseState
from quantum_counterfeit.state_prep import (
    alternating_partition_circuit,
    in_S_lh,
    partition_prep_circuit,
    partition_register_sizes,
    partition_state_reference,
    rotation_table,
    w_ghz_circuit,
    w_paper_circuit,
)


def _idx(bits) -> int:
    return sum(b << i for i, b in enumerate(bits))


def _psi(x):
    return {_idx(s): a for s, a in model.ideal_parity_state(x).items()}


@pytest.mark.parametrize("N", [3, 4, 5, 6, 7])
def test_w_paper_maps_basis_to_psi(N):
    W = w_paper_circuit(N)
    for x in product((0, 1), repeat=N):
        if not in_S_lh(x):
            continue
        st = SparseState.basis(W.num_qubits, _idx(x)).evolve(W).as_dict()
        target = _psi(x)  # R ancilla and counter must be back to 0
        assert set(st) == set(target)
        for key, a in target.items():
            assert abs(st[key] - a) < 1e-12


@pytest.mark.parametrize("N", [4, 5, 6, 7])
def test_w_paper_inverse_decodes_light_representative(N):
    W = w_paper_circuit(N)
    for k in range(0, (N + 1) // 2):
        if 2 * k >= N:
            continue
        for x in (model.all_secrets(N, k) if k else [tuple([0] * N)]):
            psi = SparseState.from_dict(W.num_qubits, _psi(x))
            out = psi.evolve(W.inverse()).as_dict()
            assert len(out) == 1 and abs(out[_idx(x)] - 1) < 1e-12


@pytest.mark.parametrize("N", [3, 4, 5, 6])
def test_w_ghz_variant(N):
    W = w_ghz_circuit(N)
    for d in product((0, 1), repeat=N - 1):
        x = (0,) + d
        st = SparseState.basis(N, _idx(x)).evolve(W).as_dict()
        target = _psi(x)
        assert all(abs(st[k] - v) < 1e-12 for k, v in target.items()) and len(st) == len(target)


@pytest.mark.parametrize("N", [4, 5, 6, 7, 8])
def test_partition_prep_exact_for_every_even_selection(N):
    A = partition_prep_circuit(N)
    br, bn = partition_register_sizes(N)
    assert A.num_qubits == 2 * N + br + bn
    for s in model.even_selections(N):
        st = SparseState.basis(A.num_qubits, _idx(s)).evolve(A).as_dict()
        ref = partition_state_reference(s)
        # overlap with the uniform balanced-split state must be 1: same support,
        # same amplitudes, S untouched, counters r and n back to 0.
        expect = {_idx(s) | (_idx(t) << N): a for t, a in ref.items()}
        assert set(st) == set(expect)
        ov = sum(np.conj(expect[k]) * st[k] for k in expect)
        assert abs(ov - 1) < 1e-12
        w = sum(s)
        assert len(st) == comb(w, w // 2)


@pytest.mark.parametrize("N", [5, 6])
def test_partition_prep_all_support_is_clean_for_odd_weights_too(N):
    A = partition_prep_circuit(N, support="all")
    for s in product((0, 1), repeat=N):
        st = SparseState.basis(A.num_qubits, _idx(s)).evolve(A).as_dict()
        w = sum(s)
        assert len(st) == comb(w, w // 2)
        for key in st:
            assert key >> (2 * N) == 0  # counters clean
            t = (key >> N) & ((1 << N) - 1)
            assert t & ~_idx(s) == 0 and bin(t).count("1") == w // 2


@pytest.mark.parametrize("N", [4, 6, 8])
def test_partition_prep_coherent_over_superposed_selections(N):
    """ONE simulation with s in superposition (not a loop over s)."""
    A = partition_prep_circuit(N)
    qc = QuantumCircuit(A.num_qubits)
    qc.compose(w_ghz_circuit(N), qubits=list(range(N)), inplace=True)  # sum over even s
    qc.compose(A, inplace=True)
    st = SparseState.basis(A.num_qubits).evolve(qc).as_dict()
    c = 2 ** (-(N - 1) / 2)
    expect = {}
    for s in model.even_selections(N):
        for t, a in partition_state_reference(s).items():
            expect[_idx(s) | (_idx(t) << N)] = c * a
    assert set(st) == set(expect)
    assert max(abs(st[k] - v) for k, v in expect.items()) < 1e-12
    # round trip: A^dag returns every branch to |s>|0>
    back = SparseState.basis(A.num_qubits).evolve(qc).evolve(A.inverse()).as_dict()
    assert set(back) == {_idx(s) for s in model.even_selections(N)}


@pytest.mark.parametrize("N", [4, 5, 6, 7])
def test_alternating_split_is_legal(N):
    alt = alternating_partition_circuit(N)
    for s in model.even_selections(N):
        st = SparseState.basis(2 * N, _idx(s)).evolve(alt).as_dict()
        (key,) = st
        side = [(key >> (N + i)) & 1 for i in range(N)]
        assert model.encoding_is_legal(s, side)


def test_rotation_table_is_polynomial():
    for N in [4, 6, 8, 10, 12]:
        tot = sum(len(t) for t in rotation_table(N))
        assert tot <= N**3
