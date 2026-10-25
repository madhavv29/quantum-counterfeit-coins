"""Render Markdown summary tables from experiments/results/*.csv.

    python experiments/summarize.py   # writes experiments/results/summary_tables.md

docs/results.md and the README quote these tables verbatim; regenerate them
here rather than editing numbers by hand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent / "results"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != v:
            return "—"
        if v == int(v) and abs(v) < 1e9:
            return str(int(v))
        return f"{v:.6f}" if abs(v) < 10 else f"{v:,.1f}"
    return str(v)


def solver_table() -> pd.DataFrame:
    d = pd.read_csv(RES / "solver_runs.csv", dtype={"x": str, "s": str})
    g = d.groupby(["label", "N", "k"], sort=False).agg(
        secrets=("x", "count"), L=("L", "first"), B_queries=("counted_queries", "max"),
        schedule_queries=("scheduled_queries", "max"), qubits=("qubits", "first"),
        exact_success_min=("exact_success", "min"), exact_success_max=("exact_success", "max"),
        bound=("bound_success", "min"), shots_total=("shots", "sum"),
        shots_success=("shots_success", "mean"), sim_s_per_secret=("sim_seconds", "mean"),
        final_nonzeros=("final_nnz", "max")).reset_index()
    g["shots_success"] = g["shots_success"].round(4)
    g["sim_s_per_secret"] = g["sim_s_per_secret"].round(2)
    return g


def equal_budget_table() -> pd.DataFrame:
    """Quantum success vs the best possible classical success with the SAME
    number of oracle calls. Classical cap: a q-weighing strategy is a binary
    decision tree with <= 2^q leaves, hence correct on <= 2^q of the C(N,k)
    secrets; by Yao's principle no randomized strategy exceeds 2^q / C(N,k)
    on its worst secret. Rigorous; no classical simulation involved."""
    from math import comb

    s = pd.read_csv(RES / "solver_runs.csv", dtype={"x": str})
    rows = []
    for (lab, N, k), g in s.groupby(["label", "N", "k"], sort=False):
        if lab == "find_star_ghz":
            continue
        q = int(g.counted_queries.max())
        rows.append(dict(N=N, k=k, schedule="delta=0.1" if k > 1 else "k=1 circuit",
                         secrets_simulated=len(g), B_queries=q,
                         quantum_success_min=g.exact_success.min(),
                         quantum_bound_all_secrets=g.bound_success.min(),
                         classical_success_cap=min(1.0, 2**q / comb(N, k))))
    f = RES / "ablation_runs.csv"
    if f.exists():
        a = pd.read_csv(f, dtype={"x": str})
        for (N, k), g in a[a.label == "tuned_overlap"].groupby(["N", "k"], sort=False):
            q = int(g.counted_queries.max())
            rows.append(dict(N=N, k=k, schedule="tuned (target 0.99)", secrets_simulated=len(g), B_queries=q,
                             quantum_success_min=g.exact_success.min(),
                             quantum_bound_all_secrets=g.bound_success.min(),
                             classical_success_cap=min(1.0, 2**q / comb(N, k))))
    df = pd.DataFrame(rows).sort_values(["k", "N", "schedule"]).reset_index(drop=True)
    df["separation"] = df.quantum_bound_all_secrets > df.classical_success_cap
    df.to_csv(RES / "equal_budget_comparison.csv", index=False)
    return df


def main():
    out = []
    out.append("## Solver runs (every row aggregates the listed secrets)\n\n" + md(solver_table()))
    f = RES / "aer_crosscheck.csv"
    if f.exists():
        a = pd.read_csv(f, dtype={"x": str})
        g = a.groupby(["kind", "N", "k", "w_variant"], sort=False).agg(
            secrets=("x", "count"), qubits=("qubits", "first"),
            max_abs_amplitude_diff=("max_abs_diff_aer_vs_sparse", "max"),
            exact_success_min=("exact_success", "min"), aer_shot_success_mean=("aer_shot_success", "mean"),
            shots_per_secret=("aer_shots", "first"), aer_seconds=("aer_statevector_seconds", "mean")).reset_index()
        g["max_abs_amplitude_diff"] = g["max_abs_amplitude_diff"].map(lambda v: f"{v:.1e}")
        out.append("## Aer cross-check\n\n" + md(g))
    f = RES / "resources.csv"
    if f.exists():
        r = pd.read_csv(f, dtype={"x": str})
        cols = [c for c in ["N", "k", "w_variant", "L", "qubits", "counted_queries", "logical_ops_total",
                            "logical_ops_outside_oracle", "max_controls", "compiled_cx",
                            "compiled_cx_outside_oracle", "oracle_call_cx", "compiled_1q", "compiled_depth"]
                if c in r]
        out.append("## Resources\n\n" + md(r[cols]))
    f = RES / "delta_sweep_runs.csv"
    if f.exists():
        dr = pd.read_csv(f, dtype={"x": str})
        rr = pd.read_csv(RES / "delta_sweep_resources.csv", dtype={"x": str, "s": str})
        g = dr.groupby("delta").agg(L=("L", "first"), B_queries=("counted_queries", "first"),
                                    secrets=("x", "count"), exact_success_min=("exact_success", "min"),
                                    bound=("bound_success", "min")).reset_index()
        if "compiled_cx" in rr:
            g = g.merge(rr[["delta", "compiled_cx", "compiled_depth"]], on="delta")
        out.append("## Delta sweep\n\n" + md(g.sort_values("delta", ascending=False)))
    f = RES / "classical.csv"
    if f.exists():
        c = pd.read_csv(f, dtype={"x": str})
        s = pd.read_csv(RES / "solver_runs.csv", dtype={"x": str, "s": str})
        s = s[s.label.isin(["one_coin", "find_star"])].groupby(["N", "k"]).agg(
            quantum_B_queries=("counted_queries", "max"), quantum_success_min=("exact_success", "min")).reset_index()
        c = c.merge(s, on=["N", "k"], how="left")
        out.append("## Classical vs quantum at simulable sizes\n\n" + md(c))
        out.append("## Success at equal query budget\n\n" + md(equal_budget_table()))
    f = RES / "asymptotic_schedule.csv"
    if f.exists():
        a = pd.read_csv(f, dtype={"x": str})
        a = a[a.k.isin([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2047])]
        out.append("## Schedule-derived query counts (not simulated)\n\n" +
                   md(a[["N", "k", "p_min", "schedule_L", "schedule_queries", "bound_success",
                         "classical_info_bound"]].round({"p_min": 4, "bound_success": 5})))
    f = RES / "ablation_runs.csv"
    if f.exists():
        ar = pd.read_csv(f, dtype={"x": str})
        g = ar.groupby(["N", "k", "label"], sort=False).agg(
            L=("L", "first"), delta=("delta", "first"), B_queries=("counted_queries", "first"),
            secrets=("x", "count"), exact_success_min=("exact_success", "min"),
            bound=("bound_success", "min")).reset_index().rename(columns={"label": "rule"})
        res = pd.read_csv(RES / "ablation_resources.csv", dtype={"x": str, "s": str})
        if "compiled_cx" in res:
            g = g.merge(res[["N", "k", "rule", "compiled_cx", "compiled_depth"]], on=["N", "k", "rule"])
        g["delta"] = g["delta"].round(4)
        out.append("## Ablation: schedule rules at a fixed rigorous target 0.99 (simulated)\n\n" + md(g))
        rules = pd.read_csv(RES / "ablation_schedule_rules.csv", dtype={"x": str, "s": str})
        p = rules.pivot_table(index=["N", "k"], columns="rule", values="schedule_queries").reset_index()
        out.append("## Ablation: schedule-derived B-queries at target 0.99 (not simulated)\n\n" + md(p))
    (RES / "summary_tables.md").write_text("\n\n".join(out) + "\n")
    print("\n\n".join(out))


if __name__ == "__main__":
    main()
