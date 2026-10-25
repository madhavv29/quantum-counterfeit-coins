# Oracle contract

## Mathematical object (fixed; never changed)

* Secret `x ∈ {0,1}^N`, `wt(x) = k`, `1 ≤ k < N/2` (`model.validate_params` raises otherwise).
* Legal query `q ∈ {−1,0,+1}^N` with `Σ q_i = 0` (equal pan sizes).
* `χ(x; q) = 0` if `Σ x_i q_i = 0` (balanced) else `1` (tilted). Binary: the direction of tilt is never reported.
* `χ(x; q) = χ(x̄; q)` for every legal `q` (tested exhaustively), which is why `k < N/2` is required.

## Query encoding in circuits

Two `N`-qubit registers `sel`, `side`:

```
q_i = 0              if sel_i = 0       (side_i is a don't-care)
q_i = +1 (left pan)  if sel_i = 1, side_i = 1
q_i = −1 (right pan) if sel_i = 1, side_i = 0
```

Trade-off: `2N` qubits instead of the information-theoretic `log₂ 3^N ≈ 1.58N`, in exchange for (a) a selection register that is *exactly* the paper's `R` register `q̃`, so `side` alone is the partition workspace `R'`, and (b) arithmetic that needs only controlled ±1 updates. An encoding is **legal** iff the induced `q` has equal pan sizes.

## Bit oracle `B_x` (one logical query)

Gate name `B_oracle`; qubit order `sel[N] | side[N] | out | E[bE] | D[bD]`, `bE = bitlen(N)`, `bD = bitlen(k)`.

| input | output |
|---|---|
| `|sel, side⟩|out⟩|0⟩_E|0⟩_D`, legal | `|sel, side⟩|out ⊕ χ(x;q)⟩|0⟩|0⟩` |
| `|sel, side⟩|out⟩|0⟩_E|0⟩_D`, illegal | unchanged (identity convention) |
| any input with dirty `E` or `D` | some fixed basis state (the gate is a permutation of the full basis) |

Properties verified in `tests/test_b_oracle.py` and `experiments/results/oracle_truth_table.csv`:

* exhaustive agreement with `model.oracle_bit_reference` on every `(sel, side, out)` for every secret of `(4,1)…(7,3)` (millions of inputs, zero mismatches);
* scratch `E, D` returned to `|0⟩` on every clean input;
* bijective and self-inverse on the **complete** basis including dirty scratch (`(4,1)`, `(5,2)`), and `U†U = I` as a dense `Operator` for `N = 3`;
* the required counterexample `q = (+1,+1,−1,−1,0,0)`: tilted for `x = 110000`, balanced for `x = 101000`.

Why `2^{bE} > N` and `2^{bD} > k`: the counters are computed modulo `2^b`; these widths make "counter ≡ 0" equivalent to "difference = 0" because `|E| ≤ N` and `|D| ≤ k`.

Why invalid encodings are harmless: the solver populates only states produced by `A` (every one a legal equal split of `I(s)`, `s` even) and its images under the amplification reflections, which stay in the span of `A_s|0⟩` and its balanced component. The tests check that illegal inputs are left unchanged rather than assuming they never occur.

## Phase oracle `O_B`

`B_phase` = `X, H` on `out` → `B_x` → `H, X` on `out`. With `out` entering `|0⟩` it acts as `(−1)^{legal·χ}` on `(sel, side)` and returns `out` to `|0⟩`: **one** logical query. Verified against Qiskit `Statevector`, Aer, and the classical reference.

## Generalised phase `S_t(β)` (inside amplification)

`B_x` → phase `e^{iβ}` on `out = 0` → `B_x`: **two** logical queries (a ±1 phase oracle cannot produce an arbitrary phase with one call and one qubit of ancilla). This matches the query accounting of Yoder–Low–Chuang (`L − 1` calls for a length-`L` sequence).

## Black-box discipline

* `BalanceOracle` stores `x` in a name-mangled attribute; `repr` hides it; no `x` / `secret` attribute exists.
* Solvers receive the oracle object and call only `N`, `k`, `layout`, `bit_gate()`. A test wraps the oracle in a guard that raises on any other attribute; another plants decoy `x`/`secret` attributes pointing to a different secret — the solver follows the gate. A source scan asserts the solver modules never reference the mangled secret, gate `.definition`s, or classical queries.
* The simulators *do* expand the gate's definition — that is executing the black box, the same thing hardware would do.
* Gate cost of the oracle is reported separately (`oracle_call_cx`, and compiled CX with the oracle replaced by an empty `NullOracle`), never treated as free.

## Counting

`b_oracle.count_oracle_calls` walks the circuit's call graph (through every package-built composite, including inverses `*_dg`) and counts `B_oracle`/`B_oracle_dg` instructions. Tests assert this equals the schedule's independent count `2L − 1` for several `L`.
