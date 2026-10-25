"""Coherent amplitude amplification for branches with *different*, unknown
success probabilities: the fixed-point schedule of Yoder, Low & Chuang
(PRL 113, 210501, 2014), plus an analysis of why a fixed number of ordinary
Grover iterations cannot replace it here.

Why not the paper's QSearch as written: QSearch (BHMT Thm 3) picks random
iteration counts and *measures* to test for success. Inside the superposition
over selections s that measurement would collapse the s register and destroy
the phase oracle being simulated. A coherent replacement needs one unitary
that amplifies every branch with p(s, x) >= p_min while leaving p = 0
branches untouched. Ordinary Grover with a fixed iteration count oscillates in
p and cannot do this uniformly (``grover_worst_case``); the YLC sequence can.

YLC conventions (their eqs. 1, 7, 10, 11), implemented literally:
    S_s(alpha) = I - (1 - e^{-i alpha}) |s><s|,   S_t(beta) = I - (1 - e^{i beta}) |t><t|
    G(alpha, beta) = -S_s(alpha) S_t(beta),       S_L = G(alpha_l, beta_l) ... G(alpha_1, beta_1)
    alpha_j = -beta_{l-j+1} = 2 arccot( tan(2 pi j / L) sqrt(1 - gamma^2) ),  1/gamma = T_{1/L}(1/delta)
    P_L(lambda) = 1 - delta^2 T_L( T_{1/L}(1/delta) sqrt(1 - lambda) )^2 >= 1 - delta^2  for lambda >= w_L
with L = 2l + 1 and 2l = L - 1 calls of the bit-flip oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acosh, asin, atan, cos, cosh, pi, sin, sqrt, tan
from typing import Sequence

import numpy as np


def chebyshev_T(n: float, x: float) -> float:
    """T_n(x) for real n, real x (analytic continuation used by YLC)."""
    if abs(x) <= 1:
        return cos(n * np.arccos(x))
    if x > 1:
        return cosh(n * acosh(x))
    return (-1) ** int(n) * cosh(n * acosh(-x)) if float(n).is_integer() else float("nan")


def gamma_of(L: int, delta: float) -> float:
    return 1.0 / chebyshev_T(1.0 / L, 1.0 / delta)


def width(L: int, delta: float) -> float:
    """w_L: P_L(lambda) >= 1 - delta^2 for every lambda >= w_L."""
    g = gamma_of(L, delta)
    return 1.0 - g * g


def ylc_success(lam: float, L: int, delta: float) -> float:
    """Analytic P_L(lambda) (YLC eq. 1)."""
    x = chebyshev_T(1.0 / L, 1.0 / delta) * sqrt(max(0.0, 1.0 - lam))
    return 1.0 - delta**2 * chebyshev_T(L, x) ** 2


def _arccot(v: float) -> float:
    # any branch works: alpha enters as 2*arccot, and branches differ by pi.
    return pi / 2 if v == 0 else atan(1.0 / v)


@dataclass(frozen=True)
class FixedPointSchedule:
    L: int
    delta: float
    alphas: tuple[float, ...]
    betas: tuple[float, ...]

    @property
    def iterations(self) -> int:
        return len(self.alphas)

    @property
    def oracle_calls(self) -> int:
        """Bit-oracle calls in S_L: two per S_t (compute, phase, uncompute)."""
        return 2 * self.iterations

    @property
    def lambda_min(self) -> float:
        return width(self.L, self.delta)


def ylc_schedule(L: int, delta: float) -> FixedPointSchedule:
    if L < 1 or L % 2 == 0:
        raise ValueError("L must be an odd positive integer")
    if not (0 < delta <= 1):
        raise ValueError("delta must lie in (0, 1]")
    l = (L - 1) // 2
    g = gamma_of(L, delta)
    root = sqrt(max(0.0, 1.0 - g * g))
    alphas = tuple(2 * _arccot(tan(2 * pi * j / L) * root) for j in range(1, l + 1))
    betas = tuple(-alphas[l - j] for j in range(1, l + 1))  # beta_j = -alpha_{l-j+1}
    return FixedPointSchedule(L=L, delta=delta, alphas=alphas, betas=betas)


def choose_L(p_min: float, delta: float, L_max: int = 10_001) -> int:
    """Smallest odd L whose YLC width w_L is <= p_min (guarantee for all p >= p_min)."""
    if p_min >= 1:
        return 1
    L = 1
    while L <= L_max:
        if width(L, delta) <= p_min + 1e-15:
            return L
        L += 2
    raise ValueError("no admissible L below L_max")


def simulate_two_level(lam: float, sched: FixedPointSchedule) -> complex:
    """Exact 2x2 evolution of |s> = (sqrt(1-lam), sqrt(lam)) under S_L.

    Returns the final amplitude on |t> (good). Independent check of the
    analytic formula and of our sign conventions."""
    s = np.array([sqrt(1 - lam), sqrt(lam)], dtype=complex)
    state = s.copy()
    P = np.outer(s, s.conj())
    for a, b in zip(sched.alphas, sched.betas):
        St = np.diag([1.0, np.exp(1j * b)])
        Ss = np.eye(2) - (1 - np.exp(-1j * a)) * P
        state = -Ss @ (St @ state)
    return complex(state[1])


def two_level_phase_value(lam: float, sched: FixedPointSchedule) -> float:
    """<0| U^dag Z_bad U |0> for one branch = P_good - P_bad = 2 P_good - 1.

    This is the coefficient left on |s>|0_workspace> by the balance-to-parity
    step (U, phase on tilted, U^dag) — the exact per-branch phase value."""
    pg = abs(simulate_two_level(lam, sched)) ** 2
    return 2 * pg - 1


# ------------------------------------------------- the non-fixed-point baseline
def grover_success(p: float, j: int) -> float:
    """Ordinary amplitude amplification, j iterations: sin^2((2j+1) asin sqrt p)."""
    return sin((2 * j + 1) * asin(sqrt(p))) ** 2


def grover_worst_case(ps: Sequence[float], j_max: int = 8) -> list[tuple[int, float]]:
    """For each fixed iteration count j, the worst success over the given
    nonzero branch probabilities. Shows that no single j works uniformly."""
    return [(j, min(grover_success(p, j) for p in ps)) for j in range(j_max + 1)]
