"""Deterministic experiment orchestration. Every function returns a pandas
DataFrame of raw rows; ``experiments/run_all.py`` writes them to CSV/JSON and
``experiments/make_plots.py`` draws the figures from those files only.

Every column is labelled by provenance:
* ``exact_*``     — exact probability from simulating the full circuit;
* ``shots_*``     — finite-shot estimate (seeded sampling);
* ``bound_*``     — rigorous analytic lower bound computed from (N, k, schedule);
* ``schedule_*``  — resource count derived from the implemented schedule rule
                    (no circuit simulated at that size);
* ``theory_*``    — published asymptotic statements, shown as references only.
"""

from __future__ import annotations

import time
from math import ceil, comb, lgamma, log, log2, sqrt

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from . import model
from .amplitude import choose_L, width, ylc_schedule
from .b_oracle import BalanceOracle, NullOracle, count_oracle_calls
from .classical import GreedySolver, info_lower_bound, minimax_queries
from .parity_approx import BlockLayout, amplification_circuit
from .resources import compiled_counts, logical_inventory
from .solver import (
    branch_diagnostics,
    build_find_star,
    build_one_coin,
    predicted_ideal_overlap,
    run,
)
from .sparse_sim import SparseState, permute_basis
from .state_prep import partition_prep_circuit, w_ghz_circuit


# ================================================================ analytics
def _lc(n, r):
    return lgamma(n + 1) - lgamma(r + 1) - lgamma(n - r + 1)


def p_min_float(N: int, k: int) -> float:
    """p_min(N, k) in floating point for large N (p(w, m) decreases in w, so
    only the largest admissible w per even m matters)."""
    model.validate_params(N, k)
    best = 1.0
    for m in range(2, k + 1, 2):
        w = min(N, N - k + m)
        w -= w % 2
        lp = _lc(m, m // 2) + _lc(w - m, (w - m) // 2) - _lc(w, w // 2)
        best = min(best, float(np.exp(lp)))
    return best


def ylc_success_vec(lam: np.ndarray, L: int, delta: float) -> np.ndarray:
    """Vectorised YLC P_L(lambda) (identical formula to amplitude.ylc_success)."""
    T1L = np.cosh(np.arccosh(1 / delta) / L)
    z = T1L * np.sqrt(np.clip(1 - lam, 0, 1))
    TL = np.where(z <= 1, np.cos(L * np.arccos(np.clip(z, -1, 1))), np.cosh(L * np.arccosh(np.maximum(z, 1))))
    return 1 - delta**2 * TL**2


def overlap_fast(N: int, k: int, L: int, delta: float) -> float:
    """Same quantity as solver.predicted_ideal_overlap, vectorised in log space
    for large N. Exact up to floating point."""
    ws, ms = np.meshgrid(np.arange(0, N + 1, 2), np.arange(0, k + 1), indexing="ij")
    ok = (ws - ms >= 0) & (ws - ms <= N - k) & (ms <= ws)
    ws, ms = ws[ok], ms[ok]
    from scipy.special import gammaln

    lc = lambda n, r: gammaln(n + 1) - gammaln(r + 1) - gammaln(n - r + 1)
    lw = lc(k, ms) + lc(N - k, ws - ms) - (N - 1) * log(2)
    wt = np.exp(lw)
    even = ms % 2 == 0
    lp = lc(ms, ms // 2) + lc(ws - ms, (ws - ms) // 2) - lc(ws, ws // 2)
    p = np.where(even, np.exp(lp), 0.0)
    v = np.where(even, 2 * ylc_success_vec(p, L, delta) - 1, 1.0)  # (-1)^m v_s
    if L == 1:
        v = np.where(even, 2 * p - 1, 1.0)
    return float(np.sum(wt * v))


def info_bound_float(N: int, k: int) -> float:
    return _lc(N, k) / log(2)


def per_branch_delta(target: float) -> float:
    """delta such that per-branch P_good >= 1 - delta^2 implies the rigorous
    overall bound (1 - 2 delta^2)^2 >= target."""
    return sqrt((1 - sqrt(target)) / 2)


def generic_p_lb(k: int) -> float:
    """N-independent analytic bound p >= C(k', k'/2) / 2^k', k' = 2 floor(k/2)."""
    kp = 2 * (k // 2)
    return 1.0 if kp == 0 else comb(kp, kp // 2) / 2**kp


def rule_generic(N: int, k: int, target: float) -> tuple[int, float]:
    d = per_branch_delta(target)
    return choose_L(generic_p_lb(k), d), d


def rule_exact_pmin(N: int, k: int, target: float) -> tuple[int, float]:
    d = per_branch_delta(target)
    return choose_L(p_min_float(N, k), d), d


def rule_tuned(N: int, k: int, target: float, L_max: int = 401) -> tuple[int, float]:
    """Smallest L (then the delta maximising the rigorous bound) such that
    overlap(N, k, L, delta)^2 >= target. Uses only (N, k)."""
    deltas = np.concatenate([np.logspace(-3, 0, 61)])
    for L in range(1, L_max + 1, 2):
        best = max(((overlap_fast(N, k, L, d), d) for d in deltas), key=lambda t: t[0])
        if best[0] ** 2 >= target:
            return L, float(best[1])
    raise ValueError("target not reachable")


# ============================================================ experiments
def exp_oracle_truth_table(cases) -> pd.DataFrame:
    """Circuit bit oracle vs chi on EVERY (sel, side, out) encoding and EVERY secret."""
    rows = []
    for N, k in cases:
        t0 = time.perf_counter()
        encs = np.arange(2 ** (2 * N + 1), dtype=np.int64)
        mism = legal_checked = illegal_checked = 0
        for x in model.all_secrets(N, k):
            o = BalanceOracle(x)
            qc = QuantumCircuit(o.layout.num_qubits)
            qc.append(o.bit_gate(), qc.qubits)
            out = permute_basis(qc, encs)
            sel = [(encs >> i) & 1 for i in range(N)]
            side = [(encs >> (N + i)) & 1 for i in range(N)]
            E = sum(s * (2 * t - 1) for s, t in zip(sel, side))
            D = sum(sel[i] * (2 * side[i] - 1) for i in range(N) if x[i])
            ref_bit = ((E == 0) & (D != 0)).astype(np.int64)
            expect = encs ^ (ref_bit << (2 * N))
            mism += int(np.sum(out != expect))
            legal_checked += int(np.sum(E == 0))
            illegal_checked += int(np.sum(E != 0))
        rows.append(dict(N=N, k=k, secrets=comb(N, k), encodings_per_secret=len(encs),
                         legal_inputs_checked=legal_checked, illegal_inputs_checked=illegal_checked,
                         mismatches=mism, seconds=round(time.perf_counter() - t0, 2)))
    return pd.DataFrame(rows)


def exp_partition_table(cases) -> pd.DataFrame:
    rows = []
    for N, k in cases:
        x = model.all_secrets(N, k)[0]
        for w, m, mult in model.admissible_wm(N, k):
            p = model.balanced_probability_formula(w, m)
            enum_ok = None
            if N <= 8:  # verify by enumeration on every s with this (w, m)
                ss = [s for s in model.even_selections(N) if sum(s) == w and model.m_of(s, x) == m]
                enum_ok = all(model.balanced_probability_enum(s, x) == p for s in ss)
            rows.append(dict(N=N, k=k, w=w, m=m, multiplicity=mult, exact_p=float(p),
                             p_fraction=str(p), verified_by_enumeration=enum_ok,
                             is_p_min=(p == model.p_min(N, k))))
    return pd.DataFrame(rows)


def _good_prob_per_branch(st: SparseState, N: int, x) -> dict[int, tuple[float, float]]:
    keys = st.keys
    probs = np.abs(st.amps) ** 2
    s_val = keys & ((1 << N) - 1)
    side = (keys >> N) & ((1 << N) - 1)
    out = {}
    for sv in np.unique(s_val):
        idx = s_val == sv
        tot = probs[idx].sum()
        good = 0.0
        for kk, pp in zip(side[idx], probs[idx]):
            s = tuple((int(sv) >> i) & 1 for i in range(N))
            t = tuple((int(kk) >> i) & 1 for i in range(N))
            q = model.query_from_encoding(s, t)
            if model.is_legal(q) and model.chi(x, q) == 0:
                good += pp
        out[int(sv)] = (good, tot)
    return out


def exp_amplification(N: int, k: int, x, delta: float) -> pd.DataFrame:
    """Per-branch probability of a BALANCED weighing, measured directly from
    the coherent state before (W, A) and after (W, U) amplification, plus the
    branch coefficient and leakage measured after the full parity block."""
    o = BalanceOracle(x)
    pm = float(model.p_min(N, k))
    L = choose_L(pm, delta)
    sched = ylc_schedule(L, delta)
    lay = BlockLayout.for_params(N, k)
    base = QuantumCircuit(lay.num_qubits)
    base.compose(w_ghz_circuit(N), qubits=list(lay.S), inplace=True)
    before = base.copy()
    before.compose(partition_prep_circuit(N), qubits=lay.a_qubits, inplace=True)
    after = base.copy()
    after.compose(amplification_circuit(lay, o.bit_gate(), sched), inplace=True)
    b = _good_prob_per_branch(SparseState.basis(lay.num_qubits).evolve(before), N, x)
    a = _good_prob_per_branch(SparseState.basis(lay.num_qubits).evolve(after), N, x)
    recs, _ = branch_diagnostics(o, x, delta=delta, L=L)
    rows = []
    for r in recs:
        sv = sum(bit << i for i, bit in enumerate(r.s))
        gb, tb = b[sv]
        ga, ta = a[sv]
        rows.append(dict(N=N, k=k, x="".join(map(str, x)), L=L, delta=delta,
                         s="".join(map(str, r.s)), w=r.w, m=r.m, exact_p=r.p,
                         exact_good_before=gb / tb, exact_good_after=ga / ta,
                         exact_branch_value=r.value.real, exact_branch_value_imag=r.value.imag,
                         exact_leakage=r.leakage, ideal_value=(-1) ** r.m))
    return pd.DataFrame(rows)


def _wilson(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    den = 1 + z * z / n
    c = (p_hat + z * z / (2 * n)) / den
    h = z * sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def exp_solver(N: int, k: int, delta: float = 0.1, L: int | None = None, w_variant="paper",
               shots: int = 1000, seed: int = 2026, secrets=None, label: str = "") -> pd.DataFrame:
    rows = []
    xs = secrets if secrets is not None else model.all_secrets(N, k)
    for i, x in enumerate(xs):
        o = BalanceOracle(x)
        t0 = time.perf_counter()
        sc = build_one_coin(o, w_variant) if (k == 1 and L is None) else build_find_star(o, delta=delta, L=L, w_variant=w_variant)
        r = run(sc)
        sec = time.perf_counter() - t0
        p = r.success_probability(x)
        samp = r.sample(shots, seed + i)
        hits = sum(1 for a in samp if a == tuple(x))
        lo, hi = _wilson(hits / shots, shots)
        bound = 1.0 if sc.schedule is None else predicted_ideal_overlap(N, k, sc.schedule) ** 2
        rows.append(dict(label=label, N=N, k=k, x="".join(map(str, x)), kind=sc.kind, w_variant=w_variant,
                         L=(sc.schedule.L if sc.schedule else 1), delta=(sc.schedule.delta if sc.schedule else None),
                         qubits=sc.num_qubits, counted_queries=sc.counted_queries,
                         scheduled_queries=sc.scheduled_queries, exact_success=p,
                         exact_clean_workspace=r.clean_probability, bound_success=bound,
                         shots=shots, shots_success=hits / shots, shots_wilson_lo=lo, shots_wilson_hi=hi,
                         shot_seed=seed + i, final_nnz=r.nnz_final, sim_seconds=round(sec, 3)))
    return pd.DataFrame(rows)


def exp_resources(N: int, k: int, delta: float = 0.1, L: int | None = None, w_variant="paper",
                  compile_: bool = True) -> dict:
    x = model.all_secrets(N, k)[0]
    real = (build_one_coin(BalanceOracle(x), w_variant) if (k == 1 and L is None)
            else build_find_star(BalanceOracle(x), delta=delta, L=L, w_variant=w_variant))
    null = (build_one_coin(NullOracle(N, k), w_variant) if (k == 1 and L is None)
            else build_find_star(NullOracle(N, k), delta=delta, L=L, w_variant=w_variant))
    inv = logical_inventory(real.circuit)
    inv_null = logical_inventory(null.circuit)
    row = dict(N=N, k=k, w_variant=w_variant, L=(real.schedule.L if real.schedule else 1),
               delta=(real.schedule.delta if real.schedule else None), qubits=real.num_qubits,
               counted_queries=count_oracle_calls(real.circuit),
               logical_ops_total=inv["total"], logical_ops_outside_oracle=inv_null["total"],
               logical_multi_controlled=inv.get("multi_controlled", 0), max_controls=inv["max_controls"])
    if compile_:
        c = compiled_counts(real.circuit)
        cn = compiled_counts(null.circuit)
        o = BalanceOracle(x)
        oc = QuantumCircuit(o.layout.num_qubits)
        oc.append(o.bit_gate(), oc.qubits)
        co = compiled_counts(oc)
        row.update(compiled_cx=c["cx"], compiled_1q=c["one_qubit"], compiled_depth=c["depth"],
                   compiled_cx_outside_oracle=cn["cx"], compiled_depth_outside_oracle=cn["depth"],
                   oracle_call_cx=co["cx"], oracle_call_depth=co["depth"],
                   transpile_seconds=c["transpile_seconds"] + cn["transpile_seconds"])
    return row


def exp_classical(cases, minimax_time_limit: float = 120.0) -> pd.DataFrame:
    rows = []
    for N, k in cases:
        g = GreedySolver(N, k)
        used = []
        for x in model.all_secrets(N, k):
            o = BalanceOracle(x)
            found, u = g.solve(o)
            assert found == x and u == o.classical_calls
            used.append(u)
        t0 = time.perf_counter()
        mm = minimax_queries(N, k, time_limit=minimax_time_limit)
        rows.append(dict(N=N, k=k, info_lower_bound=info_lower_bound(N, k),
                         minimax_optimal_worst_case=mm, minimax_seconds=round(time.perf_counter() - t0, 2),
                         greedy_worst_case=max(used), greedy_mean=float(np.mean(used)),
                         legal_queries=model.num_legal_queries(N, include_empty=False)))
    return pd.DataFrame(rows)


def exp_asymptotic_schedule(N_values, k_values, delta: float = 0.1) -> pd.DataFrame:
    """Queries of the IMPLEMENTED schedule rule at sizes we do not simulate."""
    rows = []
    for N in N_values:
        for k in k_values:
            if not (1 <= k and 2 * k < N):
                continue
            pm = p_min_float(N, k)
            L = choose_L(pm, delta)
            rows.append(dict(N=N, k=k, delta=delta, p_min=pm, schedule_L=L,
                             schedule_queries=2 * L - 1, bound_success=overlap_fast(N, k, L, delta) ** 2,
                             classical_info_bound=ceil(info_bound_float(N, k) - 1e-9),
                             theory_k_quarter=k ** 0.25,
                             theory_k_log_N_over_k=k * log2(N / k)))
    return pd.DataFrame(rows)


def exp_schedule_rules(N_values, k_values, target: float) -> pd.DataFrame:
    rows = []
    for N in N_values:
        for k in k_values:
            if not (1 <= k and 2 * k < N):
                continue
            for name, rule in [("generic_bound", rule_generic), ("exact_p_min", rule_exact_pmin),
                               ("tuned_overlap", rule_tuned)]:
                L, d = rule(N, k, target)
                rows.append(dict(N=N, k=k, rule=name, target=target, L=L, delta=d,
                                 schedule_queries=2 * L - 1, bound_success=overlap_fast(N, k, L, d) ** 2))
    return pd.DataFrame(rows)
