"""Run every experiment and write machine-readable results.

    python experiments/run_all.py                 # full run (~30-60 min on 2 cores)
    python experiments/run_all.py --quick         # smoke run, small cases only
    python experiments/run_all.py --only solver   # one section

Outputs go to experiments/results/ (CSV per table + run_metadata.json).
Nothing in results/ is edited by hand; plots are drawn from these files only.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))

from quantum_counterfeit import model  # noqa: E402
from quantum_counterfeit import experiments as ex  # noqa: E402

RESULTS = ROOT / "results"


def _save(df: pd.DataFrame, name: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    print(f"  wrote {name}.csv ({len(df)} rows)", flush=True)


def section(name):
    def deco(fn):
        fn.section = name
        return fn
    return deco


@section("oracle")
def run_oracle(cfg, quick):
    cases = [tuple(c) for c in cfg["oracle_truth_table_cases"]]
    _save(ex.exp_oracle_truth_table(cases[:3] if quick else cases), "oracle_truth_table")


@section("partitions")
def run_partitions(cfg, quick):
    cases = [tuple(c) for c in cfg["partition_cases"]]
    _save(ex.exp_partition_table(cases[:1] if quick else cases), "partition_probabilities")


@section("amplification")
def run_amplification(cfg, quick):
    frames = []
    for inst in cfg["amplification_instances"][: 1 if quick else None]:
        frames.append(ex.exp_amplification(inst["N"], inst["k"], tuple(inst["x"]), cfg["delta"]))
    _save(pd.concat(frames), "amplification_branches")


@section("solver")
def run_solver(cfg, quick):
    d, shots, seed = cfg["delta"], cfg["shots"], cfg["shot_seed"]
    frames = []
    for N in cfg["one_coin_sizes"][: 2 if quick else None]:
        frames.append(ex.exp_solver(N, 1, shots=shots, seed=seed, label="one_coin"))
    for N, k in [tuple(c) for c in cfg["solver_cases"]][: 1 if quick else None]:
        frames.append(ex.exp_solver(N, k, delta=d, shots=shots, seed=seed, label="find_star"))
    if not quick:
        frames.append(ex.exp_solver(6, 2, delta=d, w_variant="ghz", shots=shots, seed=seed, label="find_star_ghz"))
        for spec in cfg["solver_sample_cases"]:
            rng = random.Random(spec["seed"])
            xs = rng.sample(model.all_secrets(spec["N"], spec["k"]), spec["n_secrets"])
            frames.append(ex.exp_solver(spec["N"], spec["k"], delta=d, shots=shots, seed=seed, secrets=xs,
                                        label="find_star_sampled_secrets"))
    _save(pd.concat(frames, ignore_index=True), "solver_runs")


@section("aer")
def run_aer(cfg, quick):
    """Independent execution on Qiskit Aer: state vectors and seeded shots."""
    from quantum_counterfeit.aer_backend import aer_counts, aer_statevector
    from quantum_counterfeit.b_oracle import BalanceOracle
    from quantum_counterfeit.solver import build_find_star, build_one_coin, decode, run
    from quantum_counterfeit.sparse_sim import SparseState

    rows = []
    jobs = [("one_coin", 4, 1, "paper"), ("one_coin", 6, 1, "ghz"), ("find_star", 6, 2, "ghz")]
    for kind, N, k, wv in jobs[: 1 if quick else None]:
        secrets = model.all_secrets(N, k)
        if kind == "find_star":
            secrets = secrets[:: 3]
        for x in secrets:
            o = BalanceOracle(x)
            sc = build_one_coin(o, wv) if kind == "one_coin" else build_find_star(o, delta=cfg["delta"], w_variant=wv)
            t0 = time.perf_counter()
            vec = aer_statevector(sc.circuit)
            t_aer = time.perf_counter() - t0
            sp = SparseState.basis(sc.num_qubits).evolve(sc.circuit).to_dense()
            counts = aer_counts(sc.circuit, sc.S_qubits, cfg["shots"], cfg["aer_seed"])
            hits = sum(c for b, c in counts.items() if decode(tuple(int(v) for v in reversed(b)), k, wv) == tuple(x))
            rows.append(dict(kind=kind, N=N, k=k, w_variant=wv, x="".join(map(str, x)), qubits=sc.num_qubits,
                             max_abs_diff_aer_vs_sparse=float(np.max(np.abs(vec - sp))),
                             exact_success=run(sc).success_probability(x), aer_shots=cfg["shots"],
                             aer_shot_success=hits / cfg["shots"], aer_seed=cfg["aer_seed"],
                             aer_statevector_seconds=round(t_aer, 2)))
    _save(pd.DataFrame(rows), "aer_crosscheck")


@section("resources")
def run_resources(cfg, quick):
    rows = []
    for N, k in [tuple(c) for c in cfg["resource_cases"]][: 3 if quick else None]:
        t0 = time.perf_counter()
        rows.append(ex.exp_resources(N, k, delta=cfg["delta"]))
        print(f"    resources ({N},{k}) {time.perf_counter() - t0:.1f}s", flush=True)
    if not quick:
        rows.append(ex.exp_resources(6, 2, delta=cfg["delta"], w_variant="ghz"))
    _save(pd.DataFrame(rows), "resources")


@section("delta")
def run_delta(cfg, quick):
    sw = cfg["delta_sweep"]
    N, k = sw["N"], sw["k"]
    xs = model.all_secrets(N, k)[:: sw["secrets_stride"]]
    runs, res = [], []
    for d in sw["deltas"][: 3 if quick else None]:
        runs.append(ex.exp_solver(N, k, delta=d, shots=cfg["shots"], seed=cfg["shot_seed"], secrets=xs,
                                  label=f"delta_{d}"))
        res.append(ex.exp_resources(N, k, delta=d, compile_=not quick))
    _save(pd.concat(runs, ignore_index=True), "delta_sweep_runs")
    _save(pd.DataFrame(res), "delta_sweep_resources")


@section("classical")
def run_classical(cfg, quick):
    cases = [tuple(c) for c in cfg["classical_cases"]]
    _save(ex.exp_classical(cases[:4] if quick else cases, cfg["minimax_time_limit_s"]), "classical")


@section("asymptotic")
def run_asymptotic(cfg, quick):
    a = cfg["asymptotic"]
    _save(ex.exp_asymptotic_schedule(a["N_values"][: 2 if quick else None], a["k_values"], cfg["delta"]),
          "asymptotic_schedule")


@section("ablation")
def run_ablation(cfg, quick):
    ab = cfg["ablation"]
    T = ab["target"]
    _save(ex.exp_schedule_rules(ab["schedule_N_values"], ab["schedule_k_values"][: 6 if quick else None], T),
          "ablation_schedule_rules")
    runs, res = [], []
    for N, k in [tuple(c) for c in ab["simulated_cases"]][: 1 if quick else None]:
        xs = model.all_secrets(N, k)
        if len(xs) > 60:
            xs = xs[:: 5]
        for name, rule in [("generic_bound", ex.rule_generic), ("exact_p_min", ex.rule_exact_pmin),
                           ("tuned_overlap", ex.rule_tuned)]:
            L, d = rule(N, k, T)
            df = ex.exp_solver(N, k, delta=d, L=L, shots=cfg["shots"], seed=cfg["shot_seed"], secrets=xs,
                               label=name)
            runs.append(df)
            r = ex.exp_resources(N, k, delta=d, L=L, compile_=not quick)
            r["rule"] = name
            res.append(r)
    _save(pd.concat(runs, ignore_index=True), "ablation_runs")
    _save(pd.DataFrame(res), "ablation_resources")


SECTIONS = [run_oracle, run_partitions, run_amplification, run_solver, run_aer, run_resources,
            run_delta, run_classical, run_asymptotic, run_ablation]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    timings = {}
    for fn in SECTIONS:
        if args.only and fn.section not in args.only:
            continue
        print(f"[{fn.section}]", flush=True)
        t0 = time.perf_counter()
        fn(cfg, args.quick)
        timings[fn.section] = round(time.perf_counter() - t0, 1)
    import qiskit
    import qiskit_aer

    meta_path = RESULTS / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "config": cfg,
        "quick": args.quick,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "section_seconds": {**meta.get("section_seconds", {}), **timings},
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dense_statevector_limit_qubits": 24,
        "memory_note": "Dense state vectors (Aer cross-check) are limited to <=24 qubits = 256 MiB at complex128. "
                       "Full solver circuits (27-44 qubits) run on the exact sparse simulator, whose memory is "
                       "proportional to the number of nonzero amplitudes.",
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    print("timings:", timings)


if __name__ == "__main__":
    main()
