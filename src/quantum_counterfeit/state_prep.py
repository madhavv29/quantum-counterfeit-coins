"""State preparation: the transformation W (paper Appendix A) and the coherent,
s-controlled uniform balanced-partition preparation A (paper Step 2.1).

None of these circuits touches the oracle or the secret.

Partition preparation ``A`` (the main new construction)
------------------------------------------------------
Registers: ``S`` (N, control, never modified) | ``side`` (N) | ``r`` | ``n``.

    A |s>|0>|0>|0> = |s> (1/sqrt(C(w, w/2))) sum_{T ⊆ I(s), |T| = w/2} |T>|0>|0>

``side_i = 1`` puts selected coin i on the left pan. The construction is
reversible *selection sampling* (Knuth's Algorithm S) run coherently:

1. ``r <- popcount(S)`` (controlled increments), ``n <- r >> 1`` (CNOT copy).
2. For i = 0..N-1, if ``S_i = 1``: rotate ``side_i`` so that it is 1 with
   amplitude sqrt(n / r)  (r = selected coins not yet processed, including i;
   n = left-pan places still to fill); then ``n -= side_i``, ``r -= 1``.
3. Every branch ends with r = 0 and n = 0, so **the counters clean themselves**
   — no separate uncomputation is needed.

The rotation in step 2 is a multiplexer over the (r, n) values that are
actually reachable at coin i (computed exactly by ``rotation_table``), each a
multi-controlled RY with controls (S_i = 1, r = rv, n = nv). The amplitude
product along any path is sqrt(1 / C(w, w/2)), all real and positive.

Cost: O(N) controlled increments plus a multiplexer whose size is the number
of reachable (r, n) pairs — O(N^3) multi-controlled rotations in total, each
with O(log N) controls. Polynomial, not a dense unitary.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations, product
from math import asin, comb, sqrt
from typing import Literal, Sequence

from qiskit import QuantumCircuit

from .gates import bits_for, decrement, equals_pattern, increment, mcry, mcx


# ===================================================================== W
def w_paper_circuit(N: int) -> QuantumCircuit:
    """Algorithm A_W of Appendix A.

    Qubits: S[0..N-1] | R (1 ancilla) | cnt[0..bits_for(N)-1] (scratch).
    For every x in S_lh:  W |x>|0>|0> = |psi_x>|0>|0>, where
    psi_x = 2^{-(N-1)/2} sum_{s even} (-1)^{s.x} |s>, and
    S_lh = {wt(x) < N/2} ∪ {wt(x) = N/2 and x_0 = 0}  (one representative of
    each complementary pair; for even N the paper leaves the tie-break open,
    we fix it with x_0 = 0)."""
    bc = bits_for(N)
    qc = QuantumCircuit(N + 1 + bc, name="W")
    S = qc.qubits[:N]
    R = qc.qubits[N]
    cnt = qc.qubits[N + 1 :]
    qc.h(R)  # step 1.1
    for s in S:  # step 1.2: if R = 1 flip all of S
        qc.cx(R, s)
    pop = QuantumCircuit(qc.qubits)  # step 1.3: flip R if S not in S_lh
    for s in S:
        increment(pop, cnt, [(s, 1)])
    qc.compose(pop, inplace=True)
    for v in range(N + 1):
        if 2 * v > N:
            mcx(qc, equals_pattern(cnt, v), R)
        elif 2 * v == N:
            mcx(qc, equals_pattern(cnt, v) + [(S[0], 1)], R)
    qc.compose(pop.inverse(), inplace=True)
    for s in S:  # step 2: Hadamard transform
        qc.h(s)
    return qc


def w_ghz_circuit(N: int) -> QuantumCircuit:
    """Cheaper equivalent: W'|0,d> = psi_{(0,d)} (labels each complementary pair
    by its x_0 = 0 representative instead of the light one). Qubits: S only.
    Decoding then needs the classical complement step ``decode_with_weight``."""
    qc = QuantumCircuit(N, name="W_ghz")
    qc.h(0)
    for i in range(1, N):
        qc.cx(0, i)
    for i in range(N):
        qc.h(i)
    return qc


def in_S_lh(x: Sequence[int]) -> bool:
    N, w = len(x), sum(x)
    return 2 * w < N or (2 * w == N and x[0] == 0)


# ===================================================== balanced partitions
@lru_cache(maxsize=None)
def rotation_table(N: int, support: str = "even") -> tuple[tuple[tuple[int, int], ...], ...]:
    """For each coin i, the sorted (r, n) counter values with n >= 1 reachable
    when coin i is selected, over all selections s of the given parity class
    (``"even"``: even weight only — what the solver uses; ``"all"``)."""
    table: list[set[tuple[int, int]]] = [set() for _ in range(N)]
    for s in product((0, 1), repeat=N):
        w = sum(s)
        if support == "even" and w % 2:
            continue
        states = {(w, w // 2)}
        for i in range(N):
            if not s[i]:
                continue
            nxt = set()
            for r, n in states:
                if n >= 1:
                    table[i].add((r, n))
                    nxt.add((r - 1, n - 1))  # coin i -> left pan
                if n < r:
                    nxt.add((r - 1, n))  # coin i -> right pan
            states = nxt
    return tuple(tuple(sorted(t)) for t in table)


def partition_register_sizes(N: int) -> tuple[int, int]:
    """(bits of r, bits of n)."""
    return bits_for(N), bits_for(N // 2)


def partition_prep_circuit(N: int, support: Literal["even", "all"] = "even") -> QuantumCircuit:
    """The controlled partition preparation A. Qubits: S | side | r | n."""
    br, bn = partition_register_sizes(N)
    qc = QuantumCircuit(2 * N + br + bn, name="A_part")
    S = qc.qubits[:N]
    side = qc.qubits[N : 2 * N]
    r = qc.qubits[2 * N : 2 * N + br]
    n = qc.qubits[2 * N + br :]
    for s in S:
        increment(qc, r, [(s, 1)])
    for j in range(bn):
        if j + 1 < br:
            qc.cx(r[j + 1], n[j])
    table = rotation_table(N, support)
    for i in range(N):
        for rv, nv in table[i]:
            theta = 2 * asin(sqrt(nv / rv))
            mcry(qc, theta, [(S[i], 1)] + equals_pattern(r, rv) + equals_pattern(n, nv), side[i])
        decrement(qc, n, [(S[i], 1), (side[i], 1)])
        decrement(qc, r, [(S[i], 1)])
    return qc


def partition_workspace_size(N: int) -> int:
    br, bn = partition_register_sizes(N)
    return N + br + bn


def partition_state_reference(s: Sequence[int]) -> dict[tuple[int, ...], float]:
    """Mathematical target A_s|0> restricted to the side register (reference only):
    uniform amplitude on every side string T ⊆ I(s) with |T| = w/2."""
    I = [i for i, b in enumerate(s) if b]
    w = len(I)
    amp = 1 / sqrt(comb(w, w // 2))
    out = {}
    for L in combinations(I, w // 2):
        t = [0] * len(s)
        for i in L:
            t[i] = 1
        out[tuple(t)] = amp
    return out


# ========================================== k = 1 baseline: fixed partition
def alternating_partition_circuit(N: int) -> QuantumCircuit:
    """side_i <- s_0 XOR ... XOR s_{i-1}  (qubits: S | side).

    Selected coins alternate right/left, so any even-weight s gives a legal
    equal-pan query. Any legal split works for k = 1 (paper §2: 'an arbitrary
    one half')."""
    qc = QuantumCircuit(2 * N, name="Alt_part")
    S, side = qc.qubits[:N], qc.qubits[N:]
    for i in range(1, N):
        qc.cx(side[i - 1], side[i])
        qc.cx(S[i - 1], side[i])
    return qc
