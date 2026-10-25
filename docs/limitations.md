# Limitations and what is *not* established

## Not established by this project

* **No runtime, hardware, or asymptotic advantage is demonstrated.** What *is* measured is a finite-size query separation in the formal oracle model at `(8,3)`, `(10,4)` and `(12,5)`: with the same number of oracle calls (5, 5, 9) the circuit succeeds with probability ≥ 0.9987, ≥ 0.9963, ≥ 0.9865 on every secret, while no classical strategy can exceed 0.571, 0.152, 0.646 on its worst secret (`results.md` §8). At `(6,2)` and `(8,2)` the classical optimum needs no more calls than the circuit. The `O(k^{1/4})` vs `Ω(k log(N/k))` separation is an asymptotic **theorem** of Iwama et al.; our large-`k` curves are **derived from the implemented schedule**, not measured.
* **No scaling fit.** Simulated `k ≤ 5`. We do not fit an exponent to measured data; the `k^{1/4}` reference in figure 5 is a shape guide only.
* **No runtime or hardware claim.** Query complexity counts oracle calls. Our compiled circuits have 10⁴–10⁶ CX gates; the oracle itself is an instance-specific arithmetic circuit (a *formal* query primitive — it is not a way to weigh coins in superposition).
* **Not exact.** Find\* is bounded-error (success ≥ 0.987 at δ = 0.1 in every simulated instance, above the rigorous bound in every case). The exact algorithm is not implemented (see the last section).
* **Not a literal reconstruction of `QSearch`.** Step 2.2 is realised with YLC fixed-point amplification (justified in `algorithm.md`).
* **No "first implementation" claim** (`literature_review.md`: GitHub code search was unavailable; two candidate sources were not inspected).

## Engineering limits

* **Simulation memory.** Dense state vectors are used only up to 24 qubits (256 MiB). Full circuits (27–44 qubits) run on the exact sparse simulator, whose memory scales with the number of nonzero amplitudes. Measured final-state counts are 512, 8 192, 131 072 and ≈1.8 M for `N` = 6, 8, 10, 12 — about 16× per two extra coins (24 bytes each) — so `N ≈ 14` is the practical ceiling on a machine with a few GB of RAM, and `N = 16` would need tens of GB.
* **Gate counts are not optimised for hardware.** Multi-controlled gates are compiled by Qiskit's default ancilla-free synthesis; ancilla-assisted MCX would cut CX counts substantially. No T-count is reported (arbitrary-angle rotations need a synthesis tolerance we do not fix).
* **Encoding overhead.** `2N` query qubits (`sel`, `side`) instead of ≈1.58N; chosen for arithmetic simplicity (`oracle_contract.md`).
* **Invalid encodings** are handled by an identity convention; the solver never populates them, and tests check the convention rather than relying on that.
* **Success depends weakly on `x`.** Spread across secrets ≈ 6×10⁻⁵ at `(6,2)` and ≈ 10⁻³ at `(8,2)`; the clean-workspace component is exactly secret-independent, the leaked "junk" is not, because `A` is not permutation-equivariant away from `|0⟩`. This matters for the exact algorithm (below).

## Why the exact `Find(k)` (Appendix B) is not implemented

* **What it is.** `Find(k)`: run `Find*`, rotate an ancilla so the joint success probability is exactly 1/4, mark solutions with a reversible `Check`, reflect about the `Find*` output state, measure.
* **Mathematical interface needed.** (a) the success probability `a` of `Find*` must be known exactly and be the same for every secret; (b) a reversible `Check` acting on a register holding the candidate set `X` (paper: `O(log k)` weighings that compare unions of a partition of `X̄` into `k+1` groups, padded to big pans); (c) the reflection `I − 2U|0⟩⟨0|U†`.
* **What we found.** Condition (a) fails for our circuit as built: measured success differs between secrets (above). A fix exists on paper: count as "success" the event *S = x and the workspace is clean*. Its amplitude is exactly the overlap `F(N, k, schedule)` for every secret, so `a = F²` is known exactly from `(N, k)` alone and the clean-workspace test needs no queries.
* **What is missing.** A reversible `Check`. Building the paper's queries coherently from a superposed `X` requires reversible ranking of the coins outside `X` and assembling `k+1` equal groups — a second arithmetic construction comparable in size to `A`. Not attempted within this project.
* **Cheaper alternative.** Implement `Check` with more queries (e.g. per-coin comparisons, `O(k)` or `O(log N)` weighings): this would make `Find(k)` exact but give up the `O(k^{1/4})` bound for the checking step. Not implemented either.
* **What this means for the claims.** The repository implements and tests the bounded-error algorithm `Find*(k)` and one optimisation of it. It does not claim exactness or the full `Find(k)`.
