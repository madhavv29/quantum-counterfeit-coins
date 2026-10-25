# Implementation status

| Component | Status | Evidence |
|---|---|---|
| Classical B-oracle | **implemented and tested** | `model.py`; exhaustive tests over all legal queries and secrets for `N ≤ 8` (`test_model.py`): equal pans, complement invariance, `k < N/2` enforced, required `N=6, k=2` counterexample, parity/OR shortcuts shown wrong |
| Reversible circuit B-oracle | **implemented and tested** | `b_oracle.py`; every `(sel, side, out)` encoding × every secret for `(4,1)…(7,3)`, ≈2.0 M inputs, 0 mismatches (`results/oracle_truth_table.csv`); clean scratch; full-basis bijectivity and self-inverse; dense `Operator` unitarity (`N=3`); phase-kickback version vs Qiskit `Statevector` and Aer |
| One-coin solver | **implemented and tested** | `solver.build_one_coin`; every secret, `N = 3…8, 10`, success 1 to 1e−12 with exactly **1** counted B-query; `(|x⟩+|x̄⟩)/√2` amplitudes checked after the Hadamards; Aer agreement |
| Uniform balanced partition state preparation | **implemented and tested** | `state_prep.partition_prep_circuit` (reversible, polynomial size, self-cleaning counters, no dense unitaries or `initialize`); overlap 1 with the uniform split state for every even `s`, `N = 4…8` |
| Controlled coherent partition preparation across `s` | **implemented and tested** | single simulation with `s` in superposition matches `Σ_s c|s⟩A_s|0⟩` exactly; `A†` round trip restores `|s⟩|0⟩` (`test_state_prep.py`) |
| Coherent amplitude amplification with nonzero-success guarantees | **implemented and tested** | `amplitude.py` + `parity_approx.amplification_circuit`: YLC fixed-point schedule, `L` from the secret-independent exact `p_min(N,k)`; formula verified vs 2×2 simulation; `P ≥ 1 − δ²` for all `p ≥ p_min`, `p = 0` preserved exactly; measured per-branch before/after (`results/amplification_branches.csv`). *Replaces the paper's QSearch; see `algorithm.md`.* |
| Balance-to-parity phase simulation | **implemented and tested** | `parity_approx.parity_block` (`U† O_B U`); per-branch coefficient and leakage measured from the coherent simulation agree with theory to 1e−9; odd-`m` branches exactly −1 with zero leakage |
| Full bounded-error `k>1` recovery | **implemented and tested** | `solver.build_find_star`; **every** secret of `(6,2)`, `(8,2)`, `(8,3)`, `(10,4)` and 24 sampled secrets of `(12,5)`; worst success 0.9871; call-graph B-query count = schedule count `2L − 1` (5 or 9); Aer state-vector and Aer shot agreement at `(6,2)` |
| Exact Appendix-B `Find(k)` and `Check` | **not implemented** | reasons and a possible route in `limitations.md` |
| Optimization over a reproducible baseline | **implemented and tested** | secret-independent schedule tuned to the exact average-over-branches bound: 9 → 5 B-queries at `(6,2)…(10,4)` for the same rigorous 0.99 target; all secrets re-simulated, every 5th for `(10,4)` (`results/ablation_*.csv`, figure 6). Improves constants only, not the exponent |

**Bottom line.** Everything up to and including the coherent bounded-error multi-counterfeit solver, plus the schedule optimisation, is implemented and tested. The exact algorithm `Find(k)` is not.

Test tags (`pytest -m …`): `reference` (classical/reference math), `k1` (single counterfeit only — says nothing about `k > 1`), `coherent_multi` (full coherent `k > 1` circuits), `aer`, `slow`. No test is tagged as exact, because nothing exact is claimed.
