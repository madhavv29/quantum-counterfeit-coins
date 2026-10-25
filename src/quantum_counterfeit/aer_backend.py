"""Qiskit Aer execution (independent cross-check of the sparse simulator).

Dense state vectors cost 16 * 2^n bytes, so this is used only up to 24 qubits
(256 MiB). Larger circuits are run with ``sparse_sim``.

Aer executes MCX and multi-controlled phase gates natively but decomposes
every multi-controlled RY into hundreds of gates. Before handing a circuit to
Aer we therefore (1) unroll the package's own composite instructions and
(2) rewrite each controlled RY with the exact identity

    C^n-RY(theta) = MCX . RY(-theta/2) . MCX . RY(theta/2)

(controls off: RY(-t/2)RY(t/2) = I; controls on: X RY(-t/2) X RY(t/2) = RY(t)).
``tests/test_solver.py`` checks this rewrite preserves the unitary.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import MCXGate, RYGate
from qiskit_aer import AerSimulator

from .b_oracle import _is_user_composite

MAX_DENSE_QUBITS = 24


def aer_native(qc: QuantumCircuit) -> QuantumCircuit:
    out = QuantumCircuit(qc.num_qubits)
    _emit(qc, list(out.qubits), out)
    return out


def _emit(qc: QuantumCircuit, qmap, out: QuantumCircuit) -> None:
    index = {q: i for i, q in enumerate(qc.qubits)}
    out.global_phase += qc.global_phase  # every (sub)circuit's phase counted once
    for inst in qc.data:
        op = inst.operation
        qs = [qmap[index[q]] for q in inst.qubits]
        if isinstance(op, ControlledGate) and op.base_gate.name == "ry" and op.num_ctrl_qubits >= 1:
            theta = float(op.base_gate.params[0])
            n = op.num_ctrl_qubits
            ctrl, tgt = qs[:n], qs[n]
            mcx = MCXGate(n, ctrl_state=op.ctrl_state)
            out.append(RYGate(theta / 2), [tgt])
            out.append(mcx, ctrl + [tgt])
            out.append(RYGate(-theta / 2), [tgt])
            out.append(mcx, ctrl + [tgt])
        elif _is_user_composite(op):
            _emit(op.definition, qs, out)
        else:
            out.append(op, qs)


def aer_statevector(qc: QuantumCircuit) -> np.ndarray:
    if qc.num_qubits > MAX_DENSE_QUBITS:
        raise ValueError(f"{qc.num_qubits} qubits exceeds the dense limit {MAX_DENSE_QUBITS}")
    sim = AerSimulator(method="statevector")
    c = aer_native(qc)
    c.save_statevector()
    t = transpile(c, sim, optimization_level=0)
    res = sim.run(t).result()
    return np.asarray(res.get_statevector(), dtype=complex)


def aer_counts(qc: QuantumCircuit, measured: list[int], shots: int, seed: int) -> dict[str, int]:
    """Shot-based sampling of ``measured`` qubits with Aer (seeded).

    Returned bitstrings follow Qiskit's convention: the right-most character is
    measured[0]."""
    if qc.num_qubits > MAX_DENSE_QUBITS:
        raise ValueError("too many qubits for dense Aer sampling")
    c = QuantumCircuit(qc.num_qubits, len(measured))
    c.compose(aer_native(qc), inplace=True)
    c.measure(measured, list(range(len(measured))))
    sim = AerSimulator(method="statevector", seed_simulator=seed)
    t = transpile(c, sim, optimization_level=0, seed_transpiler=seed)
    return dict(sim.run(t, shots=shots).result().get_counts())
