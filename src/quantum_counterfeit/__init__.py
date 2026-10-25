"""Quantum Counterfeit Coin Solver.

A tested, coherent circuit-level study of the multi-counterfeit algorithm of
Iwama, Nishimura, Raymond & Teruyama, "Quantum Counterfeit Coin Problems",
TCS 456 (2012). See README.md and docs/implementation_status.md for exactly
what is and is not implemented.
"""

from .b_oracle import BalanceOracle, NullOracle, count_oracle_calls
from .model import all_secrets, chi, is_legal, p_min, validate_params
from .solver import build_find_star, build_one_coin, run, solve

__all__ = [
    "BalanceOracle",
    "NullOracle",
    "count_oracle_calls",
    "all_secrets",
    "chi",
    "is_legal",
    "p_min",
    "validate_params",
    "build_find_star",
    "build_one_coin",
    "run",
    "solve",
]

__version__ = "0.1.0"
