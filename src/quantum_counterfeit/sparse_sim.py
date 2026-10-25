"""Exact sparse state-vector simulator for the circuits built by this package.

Why it exists: the full coherent solver for N = 8 needs 35 qubits — 512 GiB as
a dense complex128 vector — yet the state it actually visits is small: the
amplitudes live on |s>|partition>|scratch> basis states reachable from |0>
(≈10^4 nonzeros at N = 8, ≈2·10^6 at N = 12). This simulator stores only
nonzero amplitudes and applies each gate exactly (no truncation beyond
dropping amplitudes with |a| < 1e-14).

It is *not* an approximation of the circuit: it executes the same Qiskit
``QuantumCircuit`` gate by gate (recursing into composite gates such as the
opaque ``B_oracle`` instruction). It is cross-checked against Qiskit Aer and
``qiskit.quantum_info.Statevector`` in ``tests/test_sparse_sim.py``.

Supported primitives: any single-qubit gate with a matrix (X, H, RY, P, Z, ...)
and any ``ControlledGate`` whose base is a single-qubit gate, with arbitrary
``ctrl_state``. Everything else must have a ``definition`` to recurse into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate, Instruction

_SKIP = {"barrier", "delay"}
_TOL = 1e-14


@dataclass(frozen=True)
class _Op:
    kind: str  # "x", "diag", "u"
    target: int
    cmask: int
    cval: int
    mat: np.ndarray | None


def _is_x(m: np.ndarray) -> bool:
    return np.allclose(m, [[0, 1], [1, 0]], atol=1e-14)


def _is_diag(m: np.ndarray) -> bool:
    return abs(m[0, 1]) < 1e-15 and abs(m[1, 0]) < 1e-15


def _flatten(op: Instruction, qargs: Sequence[int], out: list, phase: list) -> None:
    name = op.name
    if name in _SKIP:
        return
    if name == "measure" or name == "reset":
        raise ValueError("sparse simulator executes unitary circuits only (no measure/reset)")
    if isinstance(op, ControlledGate) and op.base_gate.num_qubits == 1:
        nc = op.num_ctrl_qubits
        ctrl = list(qargs[:nc])
        tgt = qargs[nc]
        cs = op.ctrl_state
        cmask = sum(1 << c for c in ctrl)
        cval = sum(((cs >> j) & 1) << c for j, c in enumerate(ctrl))
        _push(np.asarray(op.base_gate.to_matrix(), dtype=complex), tgt, cmask, cval, out)
        return
    if op.num_qubits == 1:
        try:
            mat = np.asarray(op.to_matrix(), dtype=complex)
        except Exception:  # composite 1-qubit instruction without a matrix
            mat = None
        if mat is not None:
            _push(mat, qargs[0], 0, 0, out)
            return
    defn = getattr(op, "definition", None)
    if defn is None:
        raise ValueError(f"cannot simulate instruction {name!r}: no matrix and no definition")
    _flatten_circuit(defn, list(qargs), out, phase)


def _push(mat: np.ndarray, tgt: int, cmask: int, cval: int, out: list) -> None:
    if _is_x(mat):
        out.append(_Op("x", tgt, cmask, cval, None))
    elif _is_diag(mat):
        out.append(_Op("diag", tgt, cmask, cval, np.array([mat[0, 0], mat[1, 1]])))
    else:
        out.append(_Op("u", tgt, cmask, cval, mat))


def _flatten_circuit(qc: QuantumCircuit, qmap: Sequence[int], out: list, phase: list) -> None:
    if qc.global_phase:
        phase[0] += float(qc.global_phase)
    index = {q: i for i, q in enumerate(qc.qubits)}
    for inst in qc.data:
        _flatten(inst.operation, [qmap[index[q]] for q in inst.qubits], out, phase)


def flatten(qc: QuantumCircuit) -> tuple[list[_Op], float]:
    """Compile a circuit to primitive (multi-)controlled single-qubit operations."""
    out: list[_Op] = []
    phase = [0.0]
    _flatten_circuit(qc, list(range(qc.num_qubits)), out, phase)
    return out, phase[0]


class SparseState:
    """A state vector stored as parallel arrays of basis indices and amplitudes."""

    def __init__(self, num_qubits: int, keys: np.ndarray, amps: np.ndarray):
        if num_qubits > 63:
            raise ValueError("at most 63 qubits are supported")
        self.num_qubits = num_qubits
        self.keys = np.asarray(keys, dtype=np.int64)
        self.amps = np.asarray(amps, dtype=complex)

    # ---------------------------------------------------------------- creation
    @classmethod
    def basis(cls, num_qubits: int, index: int = 0) -> "SparseState":
        return cls(num_qubits, np.array([index], dtype=np.int64), np.array([1.0 + 0j]))

    @classmethod
    def from_dict(cls, num_qubits: int, d: dict[int, complex]) -> "SparseState":
        ks = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
        vs = np.fromiter(d.values(), dtype=complex, count=len(d))
        return cls(num_qubits, ks, vs)

    def copy(self) -> "SparseState":
        return SparseState(self.num_qubits, self.keys.copy(), self.amps.copy())

    # --------------------------------------------------------------- evolution
    def evolve(self, qc: QuantumCircuit | Iterable[_Op], global_phase: float = 0.0) -> "SparseState":
        if isinstance(qc, QuantumCircuit):
            if qc.num_qubits != self.num_qubits:
                raise ValueError("circuit width does not match state")
            ops, global_phase = flatten(qc)
        else:
            ops = list(qc)
        st = self.copy()
        for op in ops:
            st._apply(op)
        if global_phase:
            st.amps *= np.exp(1j * global_phase)
        return st

    def _apply(self, op: _Op) -> None:
        keys, amps = self.keys, self.amps
        tbit = np.int64(1) << np.int64(op.target)
        if op.cmask:
            cond = (keys & np.int64(op.cmask)) == np.int64(op.cval)
        else:
            cond = np.ones(keys.shape, dtype=bool)
        if op.kind == "x":
            keys[cond] ^= tbit
            return
        if op.kind == "diag":
            d0, d1 = op.mat
            bit = (keys & tbit) != 0
            if d1 != 1:
                amps[cond & bit] *= d1
            if d0 != 1:
                amps[cond & ~bit] *= d0
            return
        # general single-qubit unitary on the controlled subspace
        idx = np.nonzero(cond)[0]
        if idx.size == 0:
            return
        k_aff = keys[idx]
        a_aff = amps[idx]
        base = k_aff & ~tbit
        b = (k_aff & tbit) != 0
        ub, inv = np.unique(base, return_inverse=True)
        a0 = np.zeros(ub.size, dtype=complex)
        a1 = np.zeros(ub.size, dtype=complex)
        a0[inv[~b]] = a_aff[~b]
        a1[inv[b]] = a_aff[b]
        U = op.mat
        n0 = U[0, 0] * a0 + U[0, 1] * a1
        n1 = U[1, 0] * a0 + U[1, 1] * a1
        nk = np.concatenate([ub, ub | tbit])
        na = np.concatenate([n0, n1])
        keep = np.abs(na) > _TOL
        rest = ~cond
        self.keys = np.concatenate([keys[rest], nk[keep]])
        self.amps = np.concatenate([amps[rest], na[keep]])

    # ------------------------------------------------------------- inspection
    def norm(self) -> float:
        return float(np.sqrt(np.sum(np.abs(self.amps) ** 2)))

    def nnz(self) -> int:
        return int(self.keys.size)

    def as_dict(self, tol: float = 1e-12) -> dict[int, complex]:
        return {int(k): complex(a) for k, a in zip(self.keys, self.amps) if abs(a) > tol}

    def to_dense(self) -> np.ndarray:
        if self.num_qubits > 26:
            raise ValueError("refusing to densify more than 26 qubits")
        v = np.zeros(2 ** self.num_qubits, dtype=complex)
        np.add.at(v, self.keys, self.amps)
        return v

    def marginal(self, qubits: Sequence[int]) -> dict[int, float]:
        """Probability distribution of the integer formed by ``qubits``
        (qubits[0] is the least-significant bit)."""
        sub = np.zeros(self.keys.shape, dtype=np.int64)
        for j, q in enumerate(qubits):
            sub |= ((self.keys >> np.int64(q)) & np.int64(1)) << np.int64(j)
        p = np.abs(self.amps) ** 2
        u, inv = np.unique(sub, return_inverse=True)
        tot = np.zeros(u.size)
        np.add.at(tot, inv, p)
        return {int(a): float(b) for a, b in zip(u, tot) if b > 1e-15}

    def project(self, qubits: Sequence[int], value: int) -> "SparseState":
        """Unnormalised projection onto ``qubits == value`` (bit j <- qubits[j])."""
        mask = 0
        val = 0
        for j, q in enumerate(qubits):
            mask |= 1 << q
            val |= ((value >> j) & 1) << q
        keep = (self.keys & np.int64(mask)) == np.int64(val)
        return SparseState(self.num_qubits, self.keys[keep], self.amps[keep])


def permute_basis(qc: QuantumCircuit, inputs: np.ndarray) -> np.ndarray:
    """For a circuit made only of (multi-controlled) X gates, return the image
    of every input basis index. Raises if the circuit is not a permutation."""
    ops, phase = flatten(qc)
    if phase or any(op.kind != "x" for op in ops):
        raise ValueError("circuit is not a pure classical-reversible permutation")
    keys = np.asarray(inputs, dtype=np.int64).copy()
    for op in ops:
        cond = (keys & np.int64(op.cmask)) == np.int64(op.cval) if op.cmask else slice(None)
        keys[cond] ^= np.int64(1) << np.int64(op.target)
    return keys


def simulate(qc: QuantumCircuit, initial: SparseState | None = None) -> SparseState:
    st = initial if initial is not None else SparseState.basis(qc.num_qubits, 0)
    return st.evolve(qc)
