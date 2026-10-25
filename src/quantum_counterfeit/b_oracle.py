"""Reversible circuit implementation of the balance oracle (B-oracle).

Interface (``OracleLayout``), qubit order inside the gate::

    sel[0..N-1] | side[0..N-1] | out | E[0..bE-1] | D[0..bD-1]

* ``(sel, side)`` encode the query: ``q_i = sel_i * (+1 if side_i else -1)``.
* ``out`` receives ``out XOR [legal(q)] * chi(x; q)``  (bit oracle).
* ``E`` and ``D`` are the oracle's private scratch: they must enter in |0> and
  are returned in |0> (verified exhaustively in the tests).

Invalid (unequal-pan) encodings: the oracle acts as the identity on them
(``out`` is not flipped). This is a documented, unitary convention; the solver
never populates such states (also verified).

Internals for a fixed hidden ``x`` (the only place the secret is used)::

    E  += sum_{i}          sel_i (2 side_i - 1)   (mod 2^bE, 2^bE > N)  -> pan-size difference
    D  += sum_{i in supp x} sel_i (2 side_i - 1)   (mod 2^bD, 2^bD > k)  -> counterfeit difference
    out ^= [E == 0]                         \\  together: out ^= [E==0] AND [D != 0]
    out ^= [E == 0] AND [D == 0]            /
    uncompute D, uncompute E

One *logical B-query* = one application of this gate (it is self-inverse).
The phase oracle ``O_B|q> = (-1)^chi |q>`` is one bit-oracle call with ``out``
prepared in |->; see ``phase_oracle``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from . import model
from .gates import bits_for, decrement, increment, mcx

ORACLE_NAME = "B_oracle"
ORACLE_NAMES = frozenset({ORACLE_NAME, ORACLE_NAME + "_dg"})


@dataclass(frozen=True)
class OracleLayout:
    N: int
    bE: int
    bD: int

    @property
    def num_qubits(self) -> int:
        return 2 * self.N + 1 + self.bE + self.bD

    @property
    def num_scratch(self) -> int:
        return self.bE + self.bD

    @staticmethod
    def for_params(N: int, k: int) -> "OracleLayout":
        # 2^bE > N guarantees E == 0 (mod 2^bE) iff the pans have equal size;
        # 2^bD > k guarantees D == 0 (mod 2^bD) iff equal counterfeit counts.
        return OracleLayout(N=N, bE=bits_for(N), bD=bits_for(k))


def _count_into(qc: QuantumCircuit, sel, side, reg, coins: Sequence[int]) -> None:
    for i in coins:
        increment(qc, reg, [(sel[i], 1), (side[i], 1)])
        decrement(qc, reg, [(sel[i], 1), (side[i], 0)])


def _bit_oracle_circuit(x: Sequence[int], layout: OracleLayout, name: str = ORACLE_NAME) -> QuantumCircuit:
    N = layout.N
    qc = QuantumCircuit(layout.num_qubits, name=name)
    q = qc.qubits
    sel, side, out = q[:N], q[N : 2 * N], q[2 * N]
    E = q[2 * N + 1 : 2 * N + 1 + layout.bE]
    D = q[2 * N + 1 + layout.bE :]
    supp = [i for i, b in enumerate(x) if b]

    comp = QuantumCircuit(qc.qubits)
    _count_into(comp, sel, side, E, range(N))
    _count_into(comp, sel, side, D, supp)

    qc.compose(comp, inplace=True)
    mcx(qc, [(e, 0) for e in E], out)
    mcx(qc, [(e, 0) for e in E] + [(d, 0) for d in D], out)
    qc.compose(comp.inverse(), inplace=True)
    return qc


class BalanceOracle:
    """A black-box instance of the counterfeit-coin problem.

    The secret is stored in a name-mangled attribute and is used only to build
    the oracle's internal circuit and to answer classical queries. Solvers must
    use :meth:`bit_gate` (quantum) or :meth:`classical_query` (classical).
    """

    def __init__(self, x: Sequence[int], k: int | None = None):
        x = tuple(int(b) for b in x)
        if any(b not in (0, 1) for b in x):
            raise ValueError("secret must be a 0/1 vector")
        kk = sum(x)
        if k is not None and k != kk:
            raise ValueError(f"secret has weight {kk}, expected k={k}")
        model.validate_params(len(x), kk)
        self.__x = x
        self.N = len(x)
        self.k = kk  # k is public in the problem statement
        self.layout = OracleLayout.for_params(self.N, self.k)
        self._classical_calls = 0
        self._gate: Gate | None = None

    # ------------------------------------------------------------- classical
    def classical_query(self, q: Sequence[int]) -> int:
        """One classical weighing. Raises on illegal queries (they are not
        physical weighings)."""
        self._classical_calls += 1
        return model.chi(self.__x, q)

    @property
    def classical_calls(self) -> int:
        return self._classical_calls

    def reset_counter(self) -> None:
        self._classical_calls = 0

    # --------------------------------------------------------------- quantum
    def bit_gate(self) -> Gate:
        """The bit oracle as a single opaque-to-the-solver instruction."""
        if self._gate is None:
            self._gate = _bit_oracle_circuit(self.__x, self.layout).to_gate(label=ORACLE_NAME)
        return self._gate

    def __repr__(self) -> str:  # never print the secret
        return f"BalanceOracle(N={self.N}, k={self.k})"


class NullOracle:
    """Same interface and name as a real oracle but an identity body.

    Used only for resource accounting (gates *outside* the oracle)."""

    def __init__(self, N: int, k: int):
        self.N, self.k = N, k
        self.layout = OracleLayout.for_params(N, k)

    def bit_gate(self) -> Gate:
        return QuantumCircuit(self.layout.num_qubits, name=ORACLE_NAME).to_gate(label=ORACLE_NAME)


def phase_oracle(layout: OracleLayout, bit_gate: Gate) -> QuantumCircuit:
    """O_B on (sel, side): (-1)^{legal * chi}, using ``out`` (|0> in, |0> out)
    as a |-> kick-back ancilla. Exactly one bit-oracle call."""
    qc = QuantumCircuit(layout.num_qubits, name="B_phase")
    out = qc.qubits[2 * layout.N]
    qc.x(out)
    qc.h(out)
    qc.append(bit_gate, qc.qubits)
    qc.h(out)
    qc.x(out)
    return qc


def count_oracle_calls(qc: QuantumCircuit) -> int:
    """Number of logical B-oracle invocations anywhere in the circuit's call
    graph (forward or inverse direction, at any nesting depth)."""
    n = 0
    for inst in qc.data:
        op = inst.operation
        if op.name in ORACLE_NAMES:
            n += 1
        elif _is_user_composite(op):
            n += count_oracle_calls(op.definition)
    return n


_STD = None


def _is_user_composite(op) -> bool:
    """True for package-built composite instructions (not library gates)."""
    global _STD
    if _STD is None:
        from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping

        _STD = set(get_standard_gate_name_mapping())
    from qiskit.circuit import ControlledGate

    if isinstance(op, ControlledGate) or op.name in _STD or op.name.startswith("mcx"):
        return False
    return getattr(op, "definition", None) is not None
