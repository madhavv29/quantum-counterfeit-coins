# Literature and prior-implementation review

*Search performed 2026-10-24. Classification of every statement: **theorem** (proved in a cited paper), **verified** (checked by reading the source), **unverified** (could not be checked).*

## 1. The result being implemented

Iwama, Nishimura, Raymond, Teruyama, *Quantum Counterfeit Coin Problems*, TCS 456 (2012) 51–64, [arXiv:1009.0416](https://arxiv.org/abs/1009.0416) (read in full: §1, §2, §3, Appendices A–C).

* **Model (§1).** Hidden `x ∈ {0,1}^N` with `wt(x) = k` known. A query `q ∈ {−1,0,+1}^N` must contain equally many `+1`s and `−1`s; the answer is `χ = 0` iff `Σ q_i x_i = 0` (balanced), else `1` (tilted). The quantum oracle is the phase map `O_{B,x}|q⟩ = (−1)^{χ(x;q)}|q⟩`. The paper assumes `k < N/2` because `x` and `x̄` give identical answers.
* **Theorem 1 (theorem).** `Q(k, N) = O(k^{1/4})` quantum queries for exact identification, versus the classical `Ω(k log(N/k))`. No matching quantum lower bound; §3 proves `Ω(k^{1/4})` only for algorithms with the "big-pan" or "random-partition" property.
* **Lemma 1 (theorem).** One parity-restricted (even-weight) query to the inner-product (IP) oracle identifies any `x` with `wt(x) < N/2`, via a unitary `W: |x⟩ ↦ |ψ_x⟩ = 2^{-(N-1)/2} Σ_{s even} (−1)^{s·x}|s⟩`.
* **Find\*(k) (§2).** (1) `W|0⟩` gives the uniform even-weight superposition in register `R`. (2.1) Controlled by `s`, prepare the uniform superposition over equal-size splits of `I(s)` in `R'`. (2.2) Run `QSearch(A_s, χ̄)` — amplitude amplification with *unknown* success probability (BHMT Theorem 3) — for the balanced outcome. (2.3) Flip the phase if no balanced split was found. (2.4) Reverse 2.1–2.2. (3) Apply `W^{-1}` and measure.
* **Lemma 2 (theorem).** For even `m = |I(s) ∩ supp x|`, a uniform split is balanced with probability `p = C(m,m/2)C(w−m,(w−m)/2)/C(w,w/2) = Ω(1/√m)`; with `c₀ k^{1/4}` Grover-like steps the per-branch failure is ≤ 1/1600 and Find\* succeeds with probability ≥ 9/10.
* **Appendix A** gives `A_W` (GHZ-like cat state on `x, x̄`, a reversible "is the light representative" test, then Hadamards). Our `state_prep.w_paper_circuit` implements it literally; for even `N` we fix the paper's unspecified tie-break among weight-`N/2` strings by `x_0 = 0`.
* **Appendix B** builds the exact `Find(k)` from `Find*(k)` by rotating an ancilla so that the joint success probability is exactly 1/4 and applying one round of exact amplitude amplification with a `Check` procedure that uses `O(log k)` weighings.

**Gap we had to fill (verified by our own analysis, `amplitude.py`, tests).** Step 2.2 cites `QSearch`, which chooses random iteration counts and *measures*; inside the superposition over `s` a measurement would collapse `R`. The proof of Lemma 2 instead fixes the count at `c₀k^{1/4}` iterations. A fixed count of ordinary Grover iterations does **not** amplify every branch uniformly: branch probabilities range over `[p_min, 1]`, and for every fixed count `j ≥ 1` some admissible `p` is driven to low success (`grover_worst_case`; e.g. for `(N,k)=(8,2)` the worst success at `j = 1..5` is 0.07, 0.83, 0.006, 0.54, 0.0). The paper does not give gate-level detail for this coherent step, nor for the controlled partition preparation.

## 2. Tool used to fill the gap

Yoder, Low, Chuang, *Fixed-point quantum search with an optimal number of queries*, PRL 113, 210501 (2014), [arXiv:1409.3305](https://arxiv.org/abs/1409.3305) (read: construction, eqs. 1, 7, 10–11). For odd `L = 2l+1`, phases `α_j = −β_{l−j+1} = 2 cot⁻¹(tan(2πj/L)√(1−γ²))`, `γ⁻¹ = T_{1/L}(1/δ)`, give success `P_L(λ) = 1 − δ² T_L(T_{1/L}(1/δ)√(1−λ))² ≥ 1 − δ²` for all `λ ≥ w_L`, with `L − 1` bit-oracle calls (two per iterate). It needs only a lower bound on `λ`, uses no measurement, and leaves `λ = 0` untouched — exactly what the coherent branch-parallel setting requires. We verified the formula against an independent 2×2 simulation for `L ≤ 13` and many `δ` (`tests/test_phase_oracle.py`).

## 3. Prior implementations found

| Source | What it is | Implements B-oracle? | k > 1 algorithm of Iwama et al.? | Status of claim |
|---|---|---|---|---|
| Raymond & Imamichi, Qiskit community tutorial *Quantum Counterfeit Coin Problem* ([archived rendering](https://notebook.community/antoniomezzacapo/qiskit-tutorial/community/games/quantum_counterfeit_coin_problem)) | Single-query circuit for `k = 1` (Terhal–Smolin / Bernstein–Vazirani), legacy QISKit APIs | Yes for `k = 1` (where B = IP oracle) | No — the notebook states that `k > 1` is studied by Iwama et al. | verified (notebook text) |
| Qiskit Singapore hackathon 2019, [issue #15](https://github.com/qiskit-community/qiskit-hackathon-singapore-19/issues/15) | Team extended the `k = 1` demo with a redesigned "super-oracle" that reports *balanced* if any permutation of the selected coins balances; explicitly "instead of using the Iwama et al. algorithm" | No — that super-oracle is a different (stronger) oracle | No | verified (issue text); no repository link in the issue |
| [marcosfede/quantum-counterfeit-coin-algorithm](https://github.com/marcosfede/quantum-counterfeit-coin-algorithm) | LaTeX write-up + PDF on running the counterfeit-coin circuit on IBM QE backends (2017-era QISKit) | Not determined | Not determined | **unverified**: only the landing page was read; the PDF was not inspected |
| QASMBench ([arXiv:2005.13018](https://arxiv.org/pdf/2005.13018)), benchmark `cc` | 18 qubits, 34 gates, 17 CX; text cites Iwama et al. and the `k^{1/4}` bound | — | Gate counts are consistent only with the one-query `k = 1` circuit (one CX per coin); 34 gates cannot contain amplitude amplification | **unverified** inference from the published table; the QASM file was not inspected |
| Belovs, *Quantum algorithms for learning symmetric juntas via adversary bound* ([arXiv:1311.6777](https://arxiv.org/pdf/1311.6777)) | Theory; discusses the counterfeit-coin oracle and a related `Θ(k^{1/4})` result for EXACT-HALF | — | No implementation | verified (text) |
| [Quantum Algorithm Zoo](https://quantumalgorithmzoo.org/), entry "Counterfeit Coins" | Lists the `O(k^{1/4})` result; no implementation link for this entry | — | — | verified (page) |

**Search log (2026-10-24).** Web searches: `quantum counterfeit coin multiple false coins qiskit implementation github`; `"counterfeit coin" quantum "k^{1/4}" OR "quartic" implementation circuit`; `github "counterfeit coin" quantum "amplitude amplification" multiple counterfeit balance oracle`; `"quantum counterfeit coin" github k=2 OR "two counterfeit" OR "multiple counterfeit" circuit`; `Iwama Nishimura Raymond Teruyama counterfeit coin algorithm implementation simulation qiskit cirq pennylane`. GitHub's own code/repository search API was **not available** from our environment (HTTP 403), so repositories that are not indexed by web search were not examined.

**Conclusion.** We found no public, verifiable, coherent circuit implementation of the `k > 1` algorithm under the true binary B-oracle. Because our search could not cover GitHub code search, and two candidates above were not inspected, **we make no "first implementation" claim.**

## 4. What this project contributes (precisely)

1. An exhaustively tested reversible B-oracle circuit with explicit invalid-encoding semantics and clean scratch (paper: defines the oracle, gives no circuit).
2. A polynomial-size, coherent, `s`-controlled uniform balanced-partition preparation with self-cleaning counters (paper: Step 2.1 states the state, gives no circuit).
3. A coherent replacement for `QSearch` using YLC fixed-point amplification with a finite-instance, secret-independent `p_min(N, k)`; plus a demonstration of why a fixed count of ordinary Grover iterations fails here.
4. The assembled bounded-error circuit, run on every secret for `(6,2)`, `(8,2)`, `(8,3)`, `(10,4)` and sampled secrets for `(12,5)`, with call-graph-counted B-queries, exact and shot-based success, and compiled gate costs.
5. A secret-independent schedule optimisation (tuning to the exact average-over-branches bound) that reduces B-queries at a fixed success target; it improves constants, not the `k^{1/4}` exponent.

Not contributed: the exact algorithm `Find(k)` (Appendix B), any new theorem, any hardware run.
