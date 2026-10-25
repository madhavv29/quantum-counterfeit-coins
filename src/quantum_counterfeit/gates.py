"""Low-level reversible building blocks shared by every circuit in the package.

All multi-controlled operations take controls as a list of ``(qubit, value)``
pairs, so "control on 0" never needs explicit X-conjugation in the caller.
Only standard Qiskit gates are emitted (MCX, controlled RY / Phase / X), so the
circuits run unchanged on Qiskit Aer and on the package's sparse simulator.
"""

from __future__ import annotations

from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import MCXGate, PhaseGate, RYGate, XGate

Control = tuple[Qubit, int]


def _ctrl_state(controls: Sequence[Control]) -> int:
    # Qiskit convention: bit j of ctrl_state refers to the j-th control qubit.
    return sum((int(v) & 1) << j for j, (_, v) in enumerate(controls))


def mc_gate(qc: QuantumCircuit, base: Gate, controls: Sequence[Control], target: Qubit) -> None:
    """Append ``base`` on ``target`` controlled on every (qubit == value)."""
    if not controls:
        qc.append(base, [target])
        return
    if base.name == "x":
        gate = MCXGate(len(controls), ctrl_state=_ctrl_state(controls))
    else:
        # annotated=False keeps an explicit ControlledGate (stable across Qiskit 2.x/3.0)
        gate = base.control(len(controls), ctrl_state=_ctrl_state(controls), annotated=False)
    qc.append(gate, [q for q, _ in controls] + [target])


def mcx(qc: QuantumCircuit, controls: Sequence[Control], target: Qubit) -> None:
    mc_gate(qc, XGate(), controls, target)


def mcry(qc: QuantumCircuit, theta: float, controls: Sequence[Control], target: Qubit) -> None:
    mc_gate(qc, RYGate(theta), controls, target)


def mcphase_on_pattern(qc: QuantumCircuit, phi: float, pattern: Sequence[Control]) -> None:
    """Multiply the amplitude of every basis state matching ``pattern`` by e^{i phi}.

    Implemented as a phase gate on the last pattern qubit (conjugated by X if
    its required value is 0) controlled on the remaining pattern qubits."""
    if not pattern:
        qc.global_phase += phi
        return
    *ctrls, (tq, tv) = list(pattern)
    if tv == 0:
        qc.x(tq)
    mc_gate(qc, PhaseGate(phi), ctrls, tq)
    if tv == 0:
        qc.x(tq)


def increment(qc: QuantumCircuit, reg: Sequence[Qubit], controls: Sequence[Control] = ()) -> None:
    """reg <- reg + 1 (mod 2^len(reg)), conditioned on ``controls``.

    Bit j flips iff all lower bits are 1; processing the top bit first keeps the
    lower bits unchanged while they are being read. len(reg) multi-controlled X."""
    reg = list(reg)
    for j in reversed(range(len(reg))):
        mcx(qc, list(controls) + [(reg[i], 1) for i in range(j)], reg[j])


def decrement(qc: QuantumCircuit, reg: Sequence[Qubit], controls: Sequence[Control] = ()) -> None:
    """reg <- reg - 1 (mod 2^len(reg)), conditioned on ``controls``."""
    reg = list(reg)
    for j in reversed(range(len(reg))):
        mcx(qc, list(controls) + [(reg[i], 0) for i in range(j)], reg[j])


def equals_pattern(reg: Sequence[Qubit], value: int) -> list[Control]:
    """Controls that fire iff the little-endian register holds ``value``."""
    if value < 0 or value >= 2 ** len(reg):
        raise ValueError(f"value {value} does not fit in {len(reg)} bits")
    return [(q, (value >> j) & 1) for j, q in enumerate(reg)]


def bits_for(max_value: int) -> int:
    """Smallest register width that can hold 0..max_value (at least 1)."""
    return max(1, int(max_value).bit_length())
