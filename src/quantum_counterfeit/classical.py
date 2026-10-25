"""Classical adaptive baselines that interact ONLY through the binary,
equal-pan balance response (``BalanceOracle.classical_query``).

* ``GreedySolver`` — choose the legal query that splits the remaining
  candidate set most evenly (max-min split = max information for a binary
  answer); deterministic tie-break by enumeration order. A heuristic.
* ``minimax_queries`` — exact optimal worst-case number of weighings over
  all adaptive classical strategies (exhaustive game-tree search with
  memoisation). Proven optimal for the instances where it terminates.
* ``info_lower_bound`` — ceil(log2 C(N, k)), the information-theoretic lower
  bound for exact identification with binary answers. A bound, not a curve.
"""

from __future__ import annotations

import time
from functools import lru_cache
from math import ceil, comb, log2
from typing import Sequence

import numpy as np

from . import model


def info_lower_bound(N: int, k: int) -> int:
    return ceil(log2(comb(N, k)))


@lru_cache(maxsize=None)
def _tables(N: int, k: int):
    qs = np.array(list(model.legal_queries(N, include_empty=False)), dtype=np.int8)
    xs = np.array(model.all_secrets(N, k), dtype=np.int8)
    tilt = (qs.astype(np.int32) @ xs.T.astype(np.int32)) != 0  # [query, candidate]
    return qs, xs, tilt


class GreedySolver:
    def __init__(self, N: int, k: int):
        model.validate_params(N, k)
        self.N, self.k = N, k
        self.qs, self.xs, self.tilt = _tables(N, k)

    def solve(self, oracle) -> tuple[tuple[int, ...], int]:
        """Returns (identified secret, number of weighings used)."""
        cand = np.arange(len(self.xs))
        used = 0
        while cand.size > 1:
            t = self.tilt[:, cand].sum(axis=1)
            score = np.minimum(t, cand.size - t)
            j = int(np.argmax(score))  # first maximiser = deterministic tie-break
            if score[j] == 0:
                raise RuntimeError("remaining candidates are indistinguishable")
            ans = oracle.classical_query(tuple(int(v) for v in self.qs[j]))
            used += 1
            cand = cand[self.tilt[j, cand] == bool(ans)]
        return tuple(int(v) for v in self.xs[cand[0]]), used


def minimax_queries(N: int, k: int, time_limit: float = 120.0) -> int | None:
    """Optimal worst-case number of weighings (None if the search times out)."""
    _, xs, tilt = _tables(N, k)
    nx = len(xs)
    masks = []
    for row in tilt:
        m = 0
        for j in np.nonzero(~row)[0]:
            m |= 1 << int(j)
        masks.append(m)
    masks = sorted(set(masks))
    full = (1 << nx) - 1
    deadline = time.monotonic() + time_limit
    memo: dict[tuple[int, int], bool] = {}

    def solvable(S: int, d: int) -> bool:
        c = S.bit_count()
        if c <= 1:
            return True
        if d <= 0 or c > (1 << d):
            return False
        key = (S, d)
        if key in memo:
            return memo[key]
        if time.monotonic() > deadline:
            raise TimeoutError
        half = 1 << (d - 1)
        seen = set()
        splits = []
        for m in masks:
            a = S & m
            ca = a.bit_count()
            if ca == 0 or ca == c or a in seen:
                continue
            if ca > half or c - ca > half:
                continue
            seen.add(a)
            seen.add(S ^ a)
            splits.append((abs(2 * ca - c), a))
        splits.sort()
        ok = any(solvable(a, d - 1) and solvable(S ^ a, d - 1) for _, a in splits)
        memo[key] = ok
        return ok

    d = info_lower_bound(N, k)
    try:
        while not solvable(full, d):
            d += 1
        return d
    except TimeoutError:
        return None


def all_distinguishable(N: int, k: int) -> bool:
    """Every pair of admissible secrets is separated by some legal weighing."""
    _, _, tilt = _tables(N, k)
    cols = {tuple(tilt[:, j]) for j in range(tilt.shape[1])}
    return len(cols) == tilt.shape[1]


def parity_shortcut(x: Sequence[int], q: Sequence[int]) -> int:
    """WRONG for k > 1 (kept to demonstrate the failure): parity of selected counterfeits."""
    return sum(xi for xi, qi in zip(x, q) if qi) % 2


def or_shortcut(x: Sequence[int], q: Sequence[int]) -> int:
    """WRONG (kept to demonstrate the failure): 'some counterfeit is on a pan'."""
    return int(any(xi and qi for xi, qi in zip(x, q)))
