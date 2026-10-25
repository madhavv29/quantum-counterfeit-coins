"""Draw every figure from experiments/results/*.csv (never from live runs).

    python experiments/make_plots.py

Palette: validated categorical slots 1-3 (blue, orange, aqua; all-pairs CVD
safe); neutral gray dashed lines are reserved for theoretical references and
thresholds. Every figure has a caption in docs/results.md stating what is
measured, what is exact, and what is theory.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT / "plots"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF, REF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#8a8984"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9.5,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 8.5, "font.size": 9.5, "lines.linewidth": 2, "lines.markersize": 6,
    })


def label(N, k):
    return f"({N},{k})"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ figure 1
def fig_correctness():
    df = pd.read_csv(RES / "solver_runs.csv", dtype={"x": str, "s": str})
    df = df[df.label.isin(["one_coin", "find_star", "find_star_sampled_secrets"])]
    groups = list(dict.fromkeys(zip(df.N, df.k)))
    groups.sort(key=lambda t: (t[1] > 1, t[1], t[0]))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rng = np.random.default_rng(0)
    for i, (N, k) in enumerate(groups):
        g = df[(df.N == N) & (df.k == k)]
        jitter = rng.uniform(-0.18, 0.18, len(g))
        ax.scatter(i + jitter, g.exact_success, s=10, color=BLUE, alpha=0.55, lw=0,
                   label="exact success, each secret (simulated circuit)" if i == 0 else None)
        b = g.bound_success.min()
        ax.plot([i - 0.3, i + 0.3], [b, b], color=REF, lw=1.2, ls="--",
                label="rigorous lower bound from (N,k,schedule)" if i == 0 else None)
        n = int(g.shots.sum())
        h = float((g.shots_success * g.shots).sum())
        p = h / n
        z = 1.96
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        ax.errorbar(i + 0.38, p, yerr=[[max(0.0, p - max(0, c - hw))], [max(0.0, min(1, c + hw) - p)]], fmt="D", ms=4.5,
                    color=ORANGE, ecolor=ORANGE, elinewidth=1.2, capsize=2,
                    label="pooled seeded shots, 95% Wilson interval" if i == 0 else None)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([label(*g) + ("\nk=1" if g[1] == 1 else "") for g in groups])
    ax.set_xlabel("instance (N, k)")
    ax.set_ylabel("P(output = full counterfeit set)")
    ax.set_ylim(0.982, 1.002)
    ax.set_title("Full counterfeit-set recovery, every secret (k = 1: one query; k > 1: coherent Find*)")
    ax.legend(loc="lower left")
    save(fig, "fig1_correctness")


# ------------------------------------------------------------------ figure 2
def fig_amplification():
    df = pd.read_csv(RES / "amplification_branches.csv", dtype={"x": str, "s": str})
    insts = list(dict.fromkeys(zip(df.N, df.k, df.x)))
    fig, axes = plt.subplots(1, len(insts), figsize=(4.6 * len(insts), 4.2), squeeze=False)
    for ax, (N, k, x) in zip(axes[0], insts):
        g = df[(df.N == N) & (df.k == k) & (df.x == x)]
        d = g.delta.iloc[0]
        L = g.L.iloc[0]
        ev, od = g[g.m % 2 == 0], g[g.m % 2 == 1]
        ax.plot([0, 1], [0, 1], color=REF, lw=1, ls="--", label="no amplification (after = before)")
        ax.axhline(1 - d**2, color=REF, lw=1, ls=":", label=f"guarantee 1 - δ² = {1 - d**2:.2f}")
        ax.scatter(ev.exact_good_before, ev.exact_good_after, s=28, color=BLUE, edgecolor=SURF, lw=0.8,
                   label=f"even m (p > 0): {len(ev)} branches", zorder=3)
        ax.scatter(od.exact_good_before, od.exact_good_after, s=28, marker="s", color=ORANGE, edgecolor=SURF,
                   lw=0.8, label=f"odd m (p = 0): {len(od)} branches", zorder=3)
        ax.annotate("odd m: no balanced split\nexists; stays exactly 0", (0, 0), (0.22, 0.1), fontsize=8,
                    color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("P(balanced weighing) before: A_s|0>")
        ax.set_ylabel("P(balanced weighing) after: U|0>")
        ax.set_title(f"N={N}, k={k}, L={L} (B-queries in U: {L - 1})")
        ax.legend(loc="center right", fontsize=7.5)
    save(fig, "fig2_amplification")


# ------------------------------------------------------------------ figure 3
def fig_resources():
    df = pd.read_csv(RES / "resources.csv", dtype={"x": str, "s": str})
    df = df[df.w_variant == "paper"].sort_values(["k", "N"])
    x = np.arange(len(df))
    labs = [label(n, k) for n, k in zip(df.N, df.k)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    axes[0].bar(x, df.counted_queries, color=BLUE, width=0.6)
    axes[0].set_title("logical B-queries")
    axes[1].bar(x, df.qubits, color=BLUE, width=0.6)
    axes[1].set_title("qubits")
    axes[2].bar(x - 0.17, df.compiled_cx, color=BLUE, width=0.32, label="whole circuit")
    axes[2].bar(x + 0.17, df.compiled_cx_outside_oracle, color=AQUA, width=0.32, label="outside the oracle")
    axes[2].set_yscale("log")
    axes[2].set_title("compiled CX gates (log)")
    axes[2].legend(loc="upper left")
    axes[3].bar(x, df.compiled_depth, color=BLUE, width=0.6)
    axes[3].set_yscale("log")
    axes[3].set_title("compiled depth (log)")
    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels(labs, rotation=45)
        a.set_xlabel("(N, k)")
    fig.suptitle("Resources per circuit, basis {cx, u}, transpiler optimisation level 1, seed 1234",
                 color=INK, fontsize=10)
    fig.tight_layout()
    save(fig, "fig3_resources")


# ------------------------------------------------------------------ figure 4
def fig_delta():
    runs = pd.read_csv(RES / "delta_sweep_runs.csv", dtype={"x": str, "s": str})
    res = pd.read_csv(RES / "delta_sweep_resources.csv", dtype={"x": str, "s": str}).sort_values("delta")
    agg = runs.groupby("delta").agg(min_exact=("exact_success", "min"), bound=("bound_success", "min")).reset_index()
    N, k = int(runs.N.iloc[0]), int(runs.k.iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    axes[0].plot(res.delta, res.counted_queries, "o", color=BLUE, ms=7)
    axes[0].set_ylabel("logical B-queries (2L - 1)")
    axes[0].set_title("query budget (sampled δ only)")
    if "compiled_cx" in res:
        axes[1].plot(res.delta, res.compiled_cx, "o", color=BLUE, ms=7)
    axes[1].set_ylabel("compiled CX gates")
    axes[1].set_title("gate budget")
    axes[2].plot(agg.delta, 1 - agg.min_exact, "o-", color=BLUE, label="measured: 1 - min exact success over secrets")
    axes[2].plot(agg.delta, 1 - agg.bound, "s--", color=REF, lw=1.2, ms=4, label="rigorous bound on failure")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("failure probability (log)")
    axes[2].set_title("error")
    axes[2].legend(loc="lower left", fontsize=7.5)
    for a in axes:
        a.set_xscale("log")
        a.set_xlabel("per-branch tolerance δ (log)")
        a.invert_xaxis()
    fig.suptitle(f"Resource vs error frontier of the fixed-point schedule, N={N}, k={k}", color=INK, fontsize=10)
    fig.tight_layout()
    save(fig, "fig4_delta_tradeoff")


# ------------------------------------------------------------------ figure 5
def fig_classical():
    cl = pd.read_csv(RES / "classical.csv")
    sv = pd.read_csv(RES / "solver_runs.csv", dtype={"x": str})
    sv = sv[sv.label.isin(["one_coin", "find_star"])]
    q = sv.groupby(["N", "k"]).counted_queries.max().reset_index()
    m = cl.merge(q, on=["N", "k"], how="inner").sort_values(["k", "N"])
    ab = pd.read_csv(RES / "ablation_runs.csv", dtype={"x": str})
    tq = ab[ab.label == "tuned_overlap"].groupby(["N", "k"]).counted_queries.max().rename("tuned").reset_index()
    m = m.merge(tq, on=["N", "k"], how="left")
    m["tuned"] = m["tuned"].fillna(m["counted_queries"])
    eb = pd.read_csv(RES / "equal_budget_comparison.csv")
    asy = pd.read_csv(RES / "asymptotic_schedule.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), gridspec_kw={"width_ratios": [1, 0.9, 1.15]})

    ax = axes[0]
    x = np.arange(len(m))
    ax.bar(x - 0.27, m.minimax_optimal_worst_case, 0.25, color=ORANGE,
           label="classical optimum, worst case (exhaustive minimax)")
    ax.bar(x, m.counted_queries, 0.25, color=BLUE, label="quantum, δ = 0.1 schedule (counted)")
    ax.bar(x + 0.27, m.tuned, 0.25, color=AQUA, label="quantum, tuned schedule (counted)")
    ax.set_xticks(x)
    ax.set_xticklabels([label(n, k) for n, k in zip(m.N, m.k)])
    ax.set_xlabel("instance (N, k)")
    ax.set_ylabel("weighings / B-oracle queries")
    ax.set_title("(a) Oracle calls used (classical: exact; quantum: success ≥ 0.99)")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1]
    pick = []
    for (N, k), g in eb.groupby(["N", "k"], sort=False):
        t = g[g.schedule.str.startswith("tuned")]
        pick.append((t if len(t) else g).iloc[0])
    pk = pd.DataFrame(pick).sort_values(["k", "N"])
    pk = pk[pk.k > 1]
    x = np.arange(len(pk))
    ax.bar(x - 0.17, pk.quantum_bound_all_secrets, 0.32, color=BLUE,
           label="quantum: rigorous success bound (all secrets)")
    ax.bar(x + 0.17, pk.classical_success_cap, 0.32, color=ORANGE,
           label="classical: max possible success, same # calls (2^q / C(N,k))")
    for xx, qq in zip(x, pk.B_queries):
        ax.text(xx, 1.03, f"q={qq}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([label(n, k) for n, k in zip(pk.N, pk.k)])
    ax.set_ylim(0, 1.42)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("instance (N, k)")
    ax.set_ylabel("success probability (worst secret)")
    ax.set_title("(b) Equal query budget q: query separation at (8,3), (10,4), (12,5)")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[2]
    for N, col, mk in [(1024, BLUE, "o"), (4096, AQUA, "^")]:
        g = asy[asy.N == N].sort_values("k")
        ax.plot(g.k, g.schedule_queries, marker=mk, ms=4, color=col, lw=1.6,
                label=f"quantum δ=0.1 schedule, N={N} (derived, not simulated)")
    g = asy[asy.N == 1024].sort_values("k")
    ax.plot(g.k, g.classical_info_bound, color=ORANGE, lw=1.6, marker="s", ms=3.5,
            label="classical lower bound ⌈log2 C(N,k)⌉, N=1024")
    g = asy[asy.N == 4096].sort_values("k")
    ax.plot(g.k, g.classical_info_bound, color=ORANGE, lw=1.0, alpha=0.6, label="classical lower bound, N=4096")
    kk = np.logspace(0, np.log10(2047), 50)
    c = float(np.median(g.schedule_queries / g.k**0.25))
    ax.plot(kk, c * kk**0.25, color=REF, ls="--", lw=1.2, label=f"reference {c:.1f}·k^(1/4) (shape only)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of counterfeit coins k (log)")
    ax.set_ylabel("oracle calls (log)")
    ax.set_title("(c) Beyond simulation: implemented schedule vs classical bound")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig5_classical_comparison")


# ------------------------------------------------------------------ figure 6
def fig_ablation():
    rules = pd.read_csv(RES / "ablation_schedule_rules.csv", dtype={"x": str, "s": str})
    runs = pd.read_csv(RES / "ablation_runs.csv", dtype={"x": str, "s": str})
    names = {"generic_bound": ("generic N-free bound on p", ORANGE, "s"),
             "exact_p_min": ("exact finite-instance p_min", AQUA, "^"),
             "tuned_overlap": ("tuned to the rigorous average-over-branches bound", BLUE, "o")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    N = int(rules.N.iloc[0])
    for r, (nm, col, mk) in names.items():
        g = rules[rules.rule == r].sort_values("k")
        ax.plot(g.k, g.schedule_queries, marker=mk, ms=4, color=col, lw=1.6, label=nm)
    ax.set_xscale("log")
    ax.set_xlabel("k (log)")
    ax.set_ylabel("logical B-queries for success bound ≥ 0.99")
    ax.set_title(f"Schedule rules at fixed target, N={N} (derived from (N,k) only)")
    ax.legend(loc="upper left", fontsize=7.5)
    ax = axes[1]
    cases = list(dict.fromkeys(zip(runs.N, runs.k)))
    width = 0.24
    for j, (r, (nm, col, mk)) in enumerate(names.items()):
        vals, qs = [], []
        for N_, k_ in cases:
            g = runs[(runs.N == N_) & (runs.k == k_) & (runs.label == r)]
            vals.append(1 - g.exact_success.min())
            qs.append(int(g.counted_queries.iloc[0]))
        xs = np.arange(len(cases)) + (j - 1) * width
        ax.scatter(xs, vals, s=46, marker=mk, color=col, edgecolor=SURF, lw=0.8, label=nm, zorder=3)
        for xx, v, qq in zip(xs, vals, qs):
            ax.annotate(f"{qq}q", (xx, v), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK2)
    ax.axhline(0.01, color=REF, ls="--", lw=1.1, label="target failure 0.01")
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 2e-2)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([label(*c) for c in cases])
    ax.set_xlabel("instance (N, k)")
    ax.set_ylabel("measured worst failure over secrets (log)")
    ax.set_title("Simulated circuits: every rule meets the target (labels: B-queries)")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig6_schedule_ablation")


if __name__ == "__main__":
    style()
    for f in (fig_correctness, fig_amplification, fig_resources, fig_delta, fig_classical, fig_ablation):
        try:
            f()
        except FileNotFoundError as e:
            print("skipped", f.__name__, "-", e)
