"""Solvers: the one-query k = 1 baseline and the coherent multi-counterfeit
Find*(k) circuit (bounded error, paper-inspired coherent balance-to-parity
implementation).

The solver receives an oracle object and uses **only** its ``bit_gate()``
instruction (plus the public N, k and register layout). It never reads the
secret, never looks inside the gate, and never initialises a state from
knowledge of x. Recovery happens by simulating the circuit and measuring the
selection register.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Literal, Protocol, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

from . import model
from .amplitude import FixedPointSchedule, choose_L, ylc_schedule
from .b_oracle import OracleLayout, count_oracle_calls, phase_oracle
from .gates import bits_for
from .parity_approx import BlockLayout, expected_queries, parity_block
from .sparse_sim import SparseState, simulate
from .state_prep import alternating_partition_circuit, w_ghz_circuit, w_paper_circuit

WVariant = Literal["paper", "ghz"]


class OracleLike(Protocol):
    N: int
    k: int
    layout: OracleLayout

    def bit_gate(self) -> Gate: ...


# ============================================================ circuit objects
@dataclass
class SolverCircuit:
    circuit: QuantumCircuit
    N: int
    k: int
    kind: str  # "one_coin" or "find_star"
    w_variant: WVariant
    S_qubits: list[int]
    clean_qubits: list[int]  # everything that should end in |0> ideally
    schedule: FixedPointSchedule | None = None
    p_min: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def num_qubits(self) -> int:
        return self.circuit.num_qubits

    @property
    def counted_queries(self) -> int:
        """Logical B-queries counted from the circuit's call graph."""
        return count_oracle_calls(self.circuit)

    @property
    def scheduled_queries(self) -> int:
        """Logical B-queries predicted from the schedule (independent count)."""
        return 1 if self.schedule is None else expected_queries(self.schedule)


def _attach_w(N: int, body_qubits: int, w_variant: WVariant):
    """Return (total qubits, S indices, W-register indices) for the W choice."""
    S = list(range(N))
    if w_variant == "paper":
        wanc = body_qubits
        wcnt = list(range(body_qubits + 1, body_qubits + 1 + bits_for(N)))
        return body_qubits + 1 + bits_for(N), S, [wanc] + wcnt
    if w_variant == "ghz":
        return body_qubits, S, []
    raise ValueError(f"unknown W variant {w_variant!r}")


def _w_gate(N: int, w_variant: WVariant) -> Gate:
    if w_variant == "paper":
        return w_paper_circuit(N).to_gate(label="W")
    return w_ghz_circuit(N).to_gate(label="W_ghz")


def build_one_coin(oracle: OracleLike, w_variant: WVariant = "paper") -> SolverCircuit:
    """k = 1: W, fixed alternating split of I(s), ONE phase-oracle call, undo, W^dag.

    For k = 1 every equal split of I(s) tilts iff m = s.x = 1, so the B-oracle
    is exactly the parity oracle on even s (paper §1; Terhal-Smolin)."""
    N, lay = oracle.N, oracle.layout
    if oracle.k != 1:
        raise ValueError("the one-coin circuit is only valid for k = 1")
    body = lay.num_qubits  # S | side | out | E | D  (oracle order)
    total, S, wregs = _attach_w(N, body, w_variant)
    qc = QuantumCircuit(total, name=f"one_coin_N{N}")
    W = _w_gate(N, w_variant)
    wq = S + wregs
    alt = alternating_partition_circuit(N).to_gate(label="Alt_part")
    qc.append(W, wq)
    qc.append(alt, list(range(2 * N)))
    qc.append(phase_oracle(lay, oracle.bit_gate()).to_gate(label="B_phase"), list(range(body)))
    qc.append(alt.inverse(), list(range(2 * N)))
    qc.append(W.inverse(), wq)
    clean = [q for q in range(total) if q not in S]
    return SolverCircuit(qc, N, 1, "one_coin", w_variant, S, clean)


def build_find_star(
    oracle: OracleLike,
    delta: float = 0.1,
    L: int | None = None,
    w_variant: WVariant = "paper",
    include_w_dagger: bool = True,
) -> SolverCircuit:
    """The coherent bounded-error multi-counterfeit circuit.

    The amplification schedule depends only on (N, k, delta): L is the
    smallest odd length whose YLC width is <= p_min(N, k)."""
    N, k = oracle.N, oracle.k
    model.validate_params(N, k)
    pm = float(model.p_min(N, k))
    if L is None:
        L = choose_L(pm, delta)
    sched = ylc_schedule(L, delta)
    blk = BlockLayout.for_params(N, k)
    if blk.oracle != oracle.layout:
        raise ValueError("oracle layout does not match (N, k)")
    total, S, wregs = _attach_w(N, blk.num_qubits, w_variant)
    qc = QuantumCircuit(total, name=f"find_star_N{N}_k{k}_L{L}")
    W = _w_gate(N, w_variant)
    wq = S + wregs
    qc.append(W, wq)
    qc.append(parity_block(blk, oracle.bit_gate(), sched).to_gate(label="ParityBlock"),
              list(range(blk.num_qubits)))
    if include_w_dagger:
        qc.append(W.inverse(), wq)
    clean = [q for q in range(total) if q not in S]
    return SolverCircuit(qc, N, k, "find_star", w_variant, S, clean, sched, pm,
                         meta={"block": blk, "W_qubits": wq})


# ================================================================ execution
def decode(bits: Sequence[int], k: int, w_variant: WVariant) -> tuple[int, ...] | None:
    """Classical post-processing of the measured selection register."""
    b = tuple(int(v) for v in bits)
    if w_variant == "paper":  # W^dag returns the light representative itself
        return b if sum(b) == k else None
    return model.decode_with_weight(b, k)


def int_to_bits(v: int, N: int) -> tuple[int, ...]:
    return tuple((v >> i) & 1 for i in range(N))


@dataclass
class RunResult:
    N: int
    k: int
    outcome_probs: dict[tuple[int, ...], float]  # exact distribution of S
    answer_probs: dict[tuple[int, ...] | None, float]  # after decoding
    clean_probability: float  # prob. all non-S qubits end in |0>
    nnz_final: int
    counted_queries: int

    def success_probability(self, x: Sequence[int]) -> float:
        return self.answer_probs.get(tuple(x), 0.0)

    def sample(self, shots: int, seed: int) -> list[tuple[int, ...] | None]:
        rng = np.random.default_rng(seed)
        keys = list(self.answer_probs)
        p = np.array([self.answer_probs[k] for k in keys])
        p = p / p.sum()
        idx = rng.choice(len(keys), size=shots, p=p)
        return [keys[i] for i in idx]


def run(sc: SolverCircuit, backend: Literal["sparse", "aer"] = "sparse") -> RunResult:
    """Exact output distribution of the measured selection register."""
    if backend == "sparse":
        st = simulate(sc.circuit)
    elif backend == "aer":
        from .aer_backend import aer_statevector

        vec = aer_statevector(sc.circuit)
        nz = np.nonzero(np.abs(vec) > 1e-14)[0]
        st = SparseState(sc.num_qubits, nz.astype(np.int64), vec[nz])
    else:
        raise ValueError(backend)
    marg = st.marginal(sc.S_qubits)
    outcome = {int_to_bits(v, sc.N): p for v, p in marg.items()}
    answers: dict = {}
    for bits, p in outcome.items():
        a = decode(bits, sc.k, sc.w_variant)
        answers[a] = answers.get(a, 0.0) + p
    clean = st.marginal(sc.clean_qubits).get(0, 0.0)
    return RunResult(sc.N, sc.k, outcome, answers, clean, st.nnz(), sc.counted_queries)


def solve(oracle: OracleLike, delta: float = 0.1, seed: int = 0,
          w_variant: WVariant = "paper") -> tuple[int, ...] | None:
    """Run the appropriate solver once and return the measured answer:
    the indicator vector of the k counterfeit coins (or None on a decoding
    failure). One sampled measurement of the exact output distribution."""
    if oracle.k == 1:
        sc = build_one_coin(oracle, w_variant)
    else:
        sc = build_find_star(oracle, delta=delta, w_variant=w_variant)
    return run(sc).sample(1, seed)[0]


# ======================================================== branch diagnostics
@dataclass
class BranchRecord:
    s: tuple[int, ...]
    w: int
    m: int
    p: float  # exact balanced-partition probability p(w, m)
    value: complex  # measured <s, 0_ws | state> / c_s  (ideal: (-1)^m)
    leakage: float  # norm of the dirty-workspace part of branch s, / c_s


def branch_diagnostics(oracle: OracleLike, x_for_labels: Sequence[int], delta: float = 0.1,
                       L: int | None = None) -> tuple[list[BranchRecord], SolverCircuit]:
    """Simulate W + parity block (no W^dag) coherently over ALL s at once and
    read off, for every even s, the coefficient left on |s>|0...0> and the
    leaked norm. ``x_for_labels`` is used only to label branches by m for the
    report — it does not enter the circuit."""
    sc = build_find_star(oracle, delta=delta, L=L, w_variant="paper", include_w_dagger=False)
    st = simulate(sc.circuit)
    N = sc.N
    c = 2 ** (-(N - 1) / 2)
    d = st.as_dict(tol=0.0)
    smask = (1 << N) - 1
    branch_norm2: dict[int, float] = {}
    for key, a in d.items():
        sv = key & smask
        branch_norm2[sv] = branch_norm2.get(sv, 0.0) + abs(a) ** 2
    recs = []
    for s in model.even_selections(N):
        sv = sum(b << i for i, b in enumerate(s))
        a0 = d.get(sv, 0.0)  # all other qubits zero
        tot = branch_norm2.get(sv, 0.0)
        leak = np.sqrt(max(0.0, tot - abs(a0) ** 2)) / c
        w, m = sum(s), model.m_of(s, x_for_labels)
        recs.append(BranchRecord(s, w, m, float(model.balanced_probability_formula(w, m)), a0 / c, float(leak)))
    return recs, sc


def predicted_ideal_overlap(N: int, k: int, sched: FixedPointSchedule) -> float:
    """|<ideal, 0|final>| = E_s[(-1)^m v_s], from the exact per-branch values
    v_s = 2 P_L(p(w, m)) - 1 (2x2 evolution) and the exact (w, m) multiplicities.
    P(success) >= this squared. Depends only on (N, k, schedule)."""
    from .amplitude import two_level_phase_value

    tot = 0.0
    for w, m, mult in model.admissible_wm(N, k):
        p = float(model.balanced_probability_formula(w, m))
        v = two_level_phase_value(p, sched) if p > 0 else -1.0
        tot += mult * ((-1) ** m) * v
    return tot / 2 ** (N - 1)
