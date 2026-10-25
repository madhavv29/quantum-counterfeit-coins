"""Resource accounting kept in separate units:

* logical B-oracle queries (call-graph count; ``b_oracle.count_oracle_calls``);
* logical gate inventory — primitive (multi-)controlled single-qubit gates as
  built, bucketed by number of controls (no synthesis assumptions);
* compiled cost — Qiskit ``transpile`` to a fixed basis (default {cx, u}),
  fixed optimisation level and seed: CX count, 1-qubit count, depth;
* the same compiled cost for the circuit with every oracle replaced by an
  empty instruction of the same name (``NullOracle``), so the oracle's share
  of the gate cost is measured, never assumed free.

No T-count is reported: the circuits contain arbitrary-angle rotations whose
Clifford+T cost depends on an approximation tolerance we do not fix.
"""

from __future__ import annotations

import time
from collections import Counter

from qiskit import QuantumCircuit, transpile

from .sparse_sim import flatten

DEFAULT_BASIS = ("cx", "u")


def logical_inventory(qc: QuantumCircuit) -> dict[str, int]:
    ops, _ = flatten(qc)
    c: Counter = Counter()
    for op in ops:
        nc = op.cmask.bit_count()
        c[f"{op.kind}_c{nc}"] += 1
        c["total"] += 1
        if nc >= 2:
            c["multi_controlled"] += 1
        c["max_controls"] = max(c["max_controls"], nc)
    return dict(c)


def compiled_counts(qc: QuantumCircuit, basis=DEFAULT_BASIS, optimization_level: int = 1,
                    seed: int = 1234) -> dict[str, float]:
    t0 = time.perf_counter()
    t = transpile(qc, basis_gates=list(basis), optimization_level=optimization_level,
                  seed_transpiler=seed)
    ops = t.count_ops()
    return {
        "cx": int(ops.get("cx", 0)),
        "one_qubit": int(sum(v for k, v in ops.items() if k != "cx")),
        "depth": int(t.depth()),
        "size": int(t.size()),
        "transpile_seconds": round(time.perf_counter() - t0, 3),
    }
