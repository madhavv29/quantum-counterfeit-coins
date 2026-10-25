"""Exact classical reference model of the balance (B-) oracle.

This module is the source of truth that every quantum component is tested
against. Nothing here is quantum and nothing here is approximate.

Conventions (match the paper, Iwama et al. 2012, §1, with 0-based indices):

* hidden secret ``x in {0,1}^N`` with Hamming weight ``k``, ``1 <= k < N/2``;
* a legal query ``q in {-1,0,+1}^N`` has as many ``+1`` (left pan) as ``-1``
  (right pan) entries, i.e. ``sum(q) == 0``;
* ``chi(x, q) = 0`` (balanced) iff ``sum_i x_i q_i == 0``, else ``1`` (tilted).

Circuit encoding used by the quantum oracle (see ``docs/oracle_contract.md``):
a query is stored in two N-bit registers ``sel`` and ``side`` with
``q_i = sel_i * (+1 if side_i else -1)``. ``side_i`` is a don't-care when
``sel_i == 0``. An encoding is *legal* iff the induced ``q`` is legal.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
from math import comb
from typing import Iterator, Sequence

Bits = tuple[int, ...]
Query = tuple[int, ...]


# ---------------------------------------------------------------- parameters
def validate_params(N: int, k: int) -> None:
    """Raise ``ValueError`` unless ``N >= 2`` and ``1 <= k < N/2``."""
    if not isinstance(N, int) or not isinstance(k, int):
        raise TypeError("N and k must be integers")
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    if not (1 <= k and 2 * k < N):
        raise ValueError(
            f"the model requires 1 <= k < N/2 (x and its complement are "
            f"indistinguishable otherwise); got N={N}, k={k}"
        )


def all_secrets(N: int, k: int) -> list[Bits]:
    """All admissible hidden vectors: weight-k strings of length N (lex order)."""
    validate_params(N, k)
    out = []
    for supp in combinations(range(N), k):
        x = [0] * N
        for i in supp:
            x[i] = 1
        out.append(tuple(x))
    return out


def complement(x: Sequence[int]) -> Bits:
    return tuple(1 - b for b in x)


def weight(x: Sequence[int]) -> int:
    return int(sum(x))


# ------------------------------------------------------------------- queries
def pan_sizes(q: Sequence[int]) -> tuple[int, int]:
    """(number of coins on left pan, number on right pan)."""
    if any(v not in (-1, 0, 1) for v in q):
        raise ValueError(f"query entries must be in {{-1,0,+1}}: {q}")
    return sum(1 for v in q if v == 1), sum(1 for v in q if v == -1)


def is_legal(q: Sequence[int]) -> bool:
    left, right = pan_sizes(q)
    return left == right


def weight_difference(x: Sequence[int], q: Sequence[int]) -> int:
    """sum_i x_i q_i = (#counterfeits on left) - (#counterfeits on right)."""
    if len(x) != len(q):
        raise ValueError("x and q must have equal length")
    return int(sum(xi * qi for xi, qi in zip(x, q)))


def chi(x: Sequence[int], q: Sequence[int]) -> int:
    """Binary balance response. Raises on an illegal query."""
    if not is_legal(q):
        raise ValueError(f"illegal query (unequal pan sizes): {q}")
    return 0 if weight_difference(x, q) == 0 else 1


def legal_queries(N: int, include_empty: bool = True) -> Iterator[Query]:
    """Enumerate every legal query of length N (deterministic order).

    Count = sum_l C(N, l) C(N-l, l)."""
    for q in product((0, 1, -1), repeat=N):
        if sum(q) == 0 and (include_empty or any(q)):
            yield q


def num_legal_queries(N: int, include_empty: bool = True) -> int:
    n = sum(comb(N, l) * comb(N - l, l) for l in range(N // 2 + 1))
    return n if include_empty else n - 1


# ------------------------------------------------- (sel, side) circuit encoding
def query_from_encoding(sel: Sequence[int], side: Sequence[int]) -> Query:
    """q_i = sel_i * (+1 if side_i else -1)."""
    return tuple((1 if sd else -1) if sl else 0 for sl, sd in zip(sel, side))


def encoding_is_legal(sel: Sequence[int], side: Sequence[int]) -> bool:
    return is_legal(query_from_encoding(sel, side))


def oracle_bit_reference(x: Sequence[int], sel: Sequence[int], side: Sequence[int]) -> int:
    """Value XORed into the output qubit by the circuit bit oracle.

    = chi(x, q) for a legal encoding, 0 for an illegal one (identity convention)."""
    q = query_from_encoding(sel, side)
    return chi(x, q) if is_legal(q) else 0


# --------------------------------------------- selections and their partitions
def support(s: Sequence[int]) -> tuple[int, ...]:
    """I(s) = {i : s_i = 1}."""
    return tuple(i for i, b in enumerate(s) if b)


def even_selections(N: int) -> list[Bits]:
    """Q_even: all s in {0,1}^N with even Hamming weight (2^(N-1) of them)."""
    return [s for s in product((0, 1), repeat=N) if sum(s) % 2 == 0]


def balanced_splits(s: Sequence[int]) -> list[Query]:
    """P_s as legal queries: ordered splits (L, R) of I(s) with |L| = |R| = w/2.

    Each split is returned as the query q with q_i = +1 on L, -1 on R."""
    I = support(s)
    w = len(I)
    if w % 2:
        raise ValueError("balanced splits only exist for even-weight selections")
    out = []
    for L in combinations(I, w // 2):
        Ls = set(L)
        q = [0] * len(s)
        for i in I:
            q[i] = 1 if i in Ls else -1
        out.append(tuple(q))
    return out


def m_of(s: Sequence[int], x: Sequence[int]) -> int:
    """m = |I(s) ∩ supp(x)| = number of counterfeit coins selected."""
    return int(sum(a & b for a, b in zip(s, x)))


def balanced_probability_enum(s: Sequence[int], x: Sequence[int]) -> Fraction:
    """Fraction of P_s on which the scale is balanced (exhaustive enumeration)."""
    splits = balanced_splits(s)
    good = sum(1 for q in splits if chi(x, q) == 0)
    return Fraction(good, len(splits))


def balanced_probability_formula(w: int, m: int) -> Fraction:
    """p(w, m) = C(m, m/2) C(w-m, (w-m)/2) / C(w, w/2) for even m, 0 for odd m.

    Requires w even and 0 <= m <= w (the paper's Lemma 2 expression)."""
    if w % 2 or not (0 <= m <= w):
        raise ValueError(f"need even w and 0 <= m <= w, got w={w}, m={m}")
    if m % 2:
        return Fraction(0)
    return Fraction(comb(m, m // 2) * comb(w - m, (w - m) // 2), comb(w, w // 2))


@lru_cache(maxsize=None)
def admissible_wm(N: int, k: int) -> tuple[tuple[int, int, int], ...]:
    """All (w, m, multiplicity) occurring over even s for ANY admissible secret.

    multiplicity = #{even s : |s| = w, |s ∩ supp x| = m} = C(k, m) C(N-k, w-m),
    which is the same for every weight-k x. Depends on (N, k) only."""
    validate_params(N, k)
    out = []
    for w in range(0, N + 1, 2):
        for m in range(0, min(k, w) + 1):
            if w - m > N - k:
                continue
            mult = comb(k, m) * comb(N - k, w - m)
            if mult:
                out.append((w, m, mult))
    return tuple(out)


def p_min(N: int, k: int) -> Fraction:
    """Smallest NONZERO balanced-partition probability over admissible (w, m).

    This is the finite-instance lower bound that fixes the amplification
    schedule. It uses only the public parameters (N, k), never the secret."""
    vals = [balanced_probability_formula(w, m) for w, m, _ in admissible_wm(N, k)]
    nz = [v for v in vals if v > 0]
    return min(nz)


def p_min_argmin(N: int, k: int) -> tuple[int, int]:
    best = None
    for w, m, _ in admissible_wm(N, k):
        p = balanced_probability_formula(w, m)
        if p > 0 and (best is None or p < best[0]):
            best = (p, w, m)
    return best[1], best[2]


# ------------------------------------------------------- ideal parity reference
def ideal_parity_state(x: Sequence[int]) -> dict[Bits, float]:
    """Amplitudes of  2^{-(N-1)/2} sum_{s even} (-1)^{s.x} |s>  (reference only)."""
    N = len(x)
    amp = 2 ** (-(N - 1) / 2)
    return {s: amp * (-1) ** m_of(s, x) for s in even_selections(N)}


def decode_with_weight(bits: Sequence[int], k: int) -> Bits | None:
    """Map a measured string to the weight-k representative of {z, complement(z)}.

    Returns None if neither has weight k (decoding failure)."""
    z = tuple(int(b) for b in bits)
    if weight(z) == k:
        return z
    if weight(complement(z)) == k:
        return complement(z)
    return None
