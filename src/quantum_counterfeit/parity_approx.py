"""Balance-to-parity conversion: an approximate IP (parity) phase oracle built
coherently from B-oracle calls.

Qubit order of every circuit in this module (the *parity block*)::

    S[N] | side[N] | r[br] | n[bn] | out | E[bE] | D[bD]

``S`` is the selection register (control only). ``side, r, n`` is the
workspace of the partition preparation A. ``out, E, D`` belong to the oracle.

Construction (paper Steps 2.1-2.4, with QSearch replaced by YLC fixed-point
amplification — see ``amplitude.py`` for why):

    U   = G(a_l, b_l) ... G(a_1, b_1) A,     G(a, b) = A S_0(a) A^dag S_t(b)
    P~  = U^dag  O_B  U

* A is block-diagonal in s, so A S_0 A^dag is the reflection about A_s|0> in
  every branch simultaneously — no explicit control on s is needed and no
  branch is ever measured.
* S_t(b): phase e^{ib} on *balanced* queries = bit oracle, phase on out = 0,
  bit oracle  (2 B-queries).
* O_B: phase -1 on *tilted* queries (1 B-query, |-> kick-back).
* For one branch s: <s,0|P~|s,0> = P_good(s) - P_bad(s) = 2 P_L(p(s,x)) - 1.
  Odd m: p = 0, the value is exactly -1 and nothing leaks. Even m: >= 1 - 2 delta^2.
  The global -1 of each YLC iterate is identical in every branch and is
  dropped (it cancels between U and U^dag anyway).

B-queries: 2l in U, 2l in U^dag, 1 in O_B  ->  2(L-1) + 1 = 2L - 1.
"""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from .amplitude import FixedPointSchedule
from .b_oracle import OracleLayout, phase_oracle
from .gates import mcphase_on_pattern
from .state_prep import partition_prep_circuit, partition_register_sizes


@dataclass(frozen=True)
class BlockLayout:
    N: int
    br: int
    bn: int
    oracle: OracleLayout

    @property
    def S(self) -> range:
        return range(0, self.N)

    @property
    def side(self) -> range:
        return range(self.N, 2 * self.N)

    @property
    def r(self) -> range:
        return range(2 * self.N, 2 * self.N + self.br)

    @property
    def n(self) -> range:
        return range(2 * self.N + self.br, 2 * self.N + self.br + self.bn)

    @property
    def out(self) -> int:
        return 2 * self.N + self.br + self.bn

    @property
    def E(self) -> range:
        o = self.out + 1
        return range(o, o + self.oracle.bE)

    @property
    def D(self) -> range:
        o = self.out + 1 + self.oracle.bE
        return range(o, o + self.oracle.bD)

    @property
    def num_qubits(self) -> int:
        return 2 * self.N + self.br + self.bn + 1 + self.oracle.bE + self.oracle.bD

    @property
    def workspace(self) -> list[int]:
        """Registers A acts on besides S: the S_0 reflection is about their |0>."""
        return list(self.side) + list(self.r) + list(self.n)

    @property
    def oracle_qubits(self) -> list[int]:
        return list(self.S) + list(self.side) + [self.out] + list(self.E) + list(self.D)

    @property
    def a_qubits(self) -> list[int]:
        return list(self.S) + list(self.side) + list(self.r) + list(self.n)

    @staticmethod
    def for_params(N: int, k: int) -> "BlockLayout":
        br, bn = partition_register_sizes(N)
        return BlockLayout(N=N, br=br, bn=bn, oracle=OracleLayout.for_params(N, k))


def _st_circuit(lay: BlockLayout, bit_gate: Gate, beta: float) -> QuantumCircuit:
    """S_t(beta): phase e^{i beta} on balanced (good) queries. 2 B-queries."""
    qc = QuantumCircuit(lay.num_qubits, name="S_t")
    oq = [qc.qubits[i] for i in lay.oracle_qubits]
    out = qc.qubits[lay.out]
    qc.append(bit_gate, oq)
    qc.x(out)
    qc.p(beta, out)
    qc.x(out)
    qc.append(bit_gate, oq)
    return qc


def _s0_circuit(lay: BlockLayout, alpha: float) -> QuantumCircuit:
    """S_0(alpha): phase e^{-i alpha} on |0> of the A-workspace. No queries."""
    qc = QuantumCircuit(lay.num_qubits, name="S_0")
    mcphase_on_pattern(qc, -alpha, [(qc.qubits[i], 0) for i in lay.workspace])
    return qc


def amplification_circuit(lay: BlockLayout, bit_gate: Gate, sched: FixedPointSchedule,
                          a_gate: Gate | None = None) -> QuantumCircuit:
    """U = prod_j [A S_0(alpha_j) A^dag S_t(beta_j)] A."""
    qc = QuantumCircuit(lay.num_qubits, name="U_amp")
    if a_gate is None:
        a_gate = partition_prep_circuit(lay.N).to_gate(label="A_part")
    aq = [qc.qubits[i] for i in lay.a_qubits]
    qc.append(a_gate, aq)
    for a, b in zip(sched.alphas, sched.betas):
        qc.append(_st_circuit(lay, bit_gate, b).to_gate(label="S_t"), qc.qubits)
        qc.append(a_gate.inverse(), aq)
        qc.append(_s0_circuit(lay, a).to_gate(label="S_0"), qc.qubits)
        qc.append(a_gate, aq)
    return qc


def parity_block(lay: BlockLayout, bit_gate: Gate, sched: FixedPointSchedule,
                 a_gate: Gate | None = None) -> QuantumCircuit:
    """P~ = U^dag O_B U (approximate (-1)^{s.x} on S, workspace returned ~|0>)."""
    U = amplification_circuit(lay, bit_gate, sched, a_gate)
    qc = QuantumCircuit(lay.num_qubits, name="ParityBlock")
    qc.append(U.to_gate(label="U_amp"), qc.qubits)
    ph = phase_oracle(lay.oracle, bit_gate)
    qc.append(ph.to_gate(label="B_phase"), [qc.qubits[i] for i in lay.oracle_qubits])
    qc.append(U.to_gate(label="U_amp").inverse(), qc.qubits)
    return qc


def expected_queries(sched: FixedPointSchedule) -> int:
    return 2 * sched.oracle_calls + 1
