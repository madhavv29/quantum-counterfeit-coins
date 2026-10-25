# Algorithm ↔ code

Fidelity labels used below:

* **faithful** — implements the paper's construction as written;
* **faithful-to-the-idea** — same mathematical object or role, different realisation (explained);
* **new engineering** — the paper states the requirement but gives no circuit;
* **reference only** — used for testing, never inside the solver.

## Notation

`x` secret, `wt(x) = k`; `s` an even-weight selection, `I(s)` its support, `w = |I(s)|`, `m = |I(s) ∩ supp x|`; `c = 2^{-(N-1)/2}`; `p(w,m) = C(m,m/2)C(w−m,(w−m)/2)/C(w,w/2)` for even `m`, `0` for odd `m`.

## Pipeline of `Find*(k)` (`solver.build_find_star`)

```
|0>_S |0>_ws |0>_oracle |0>_W
  W                         S: c Σ_{s even} |s>                        (faithful, Appendix A)
  U = G_l…G_1 A             per branch s: |s> U_s|0>                    (new engineering + YLC)
  O_B                       phase −1 on tilted queries                 (1 B-query)
  U†                        per branch s: |s>(v_s|0> + |junk_s>)
  W†                        → |x>|0>  + small error                     (faithful)
  measure S, decode
```

with `v_s = (−1)^m` exactly for odd `m` and `v_s = 2P_L(p(w,m)) − 1 ≥ 1 − 2δ²` for even `m`.

## Components

| Component | Code | Paper | Label | Input / work / output qubits | Inverse | Valid input subspace | Phase convention | B-queries |
|---|---|---|---|---|---|---|---|---|
| Classical model `χ` | `model.chi` | §1 | faithful | — | — | legal `q` (raises otherwise) | — | — |
| Bit oracle `B_x` | `b_oracle._bit_oracle_circuit` | §1 (defined, no circuit) | new engineering | `sel[N] side[N] out` / `E[bitlen N] D[bitlen k]` / `out ⊕ χ` | itself | all basis states (illegal → identity, dirty scratch → some permutation) | — | 1 |
| Phase oracle `O_B` | `b_oracle.phase_oracle` | §1 | faithful | `sel side` / `out, E, D` returned `|0⟩` | itself | clean `out, E, D` | `(−1)^{legal·χ}` | 1 |
| `W` | `state_prep.w_paper_circuit` | App. A, `A_W` | faithful (tie-break for `wt = N/2` fixed by `x₀ = 0`) | `S[N]` / `R`, `cnt[bitlen N]` returned `|0⟩` | `.inverse()` | `x ∈ S_lh` | `W|x⟩ = |ψ_x⟩` exactly | 0 |
| `W'` (alternative) | `state_prep.w_ghz_circuit` | Lemma 1 | faithful-to-the-idea: labels `{x, x̄}` by the `x₀ = 0` member; decoding then complements classically | `S[N]` | `.inverse()` | `x₀ = 0` | `W'|0,d⟩ = |ψ_{(0,d)}⟩` | 0 |
| Partition prep `A` | `state_prep.partition_prep_circuit` | Step 2.1 (state given, no circuit) | new engineering | `S` (control) / `side[N] r[bitlen N] n[bitlen N/2]` | `.inverse()` | even `s` (option `support="all"` cleans odd `s` too) | real positive amplitudes | 0 |
| `S_0(α)` | `parity_approx._s0_circuit` | QSearch's `U_0` | faithful-to-the-idea (generalised phase) | `side r n` | `S_0(−α)` | — | `e^{−iα}` on `|0⟩_ws` | 0 |
| `S_t(β)` | `parity_approx._st_circuit` | `O_χ̄` | faithful-to-the-idea (generalised phase) | oracle qubits | `S_t(−β)` | clean `out, E, D` | `e^{iβ}` on balanced | 2 |
| Amplification `U` | `parity_approx.amplification_circuit` | Step 2.2 `QSearch` | **faithful-to-the-idea**: YLC fixed-point sequence instead of QSearch (see below) | all block qubits | `.inverse()` | `|s⟩|0⟩` | — | `L − 1` |
| Parity block `U† O_B U` | `parity_approx.parity_block` | Steps 2.1–2.4 | faithful-to-the-idea | all block qubits | — | `|s⟩|0⟩` | `≈ (−1)^{s·x}` | `2L − 1` |
| `k = 1` partition | `state_prep.alternating_partition_circuit` | §2 ("an arbitrary one half") | faithful | `S side` | reverse | even `s` | — | 0 |
| Ideal parity state | `model.ideal_parity_state` | Lemma 1 | reference only | — | — | — | — | — |
| `A_s|0⟩` amplitudes | `state_prep.partition_state_reference` | Step 2.1 | reference only | — | — | — | — | — |

### The partition preparation `A` (new engineering)

Coherent selection sampling. `r ← popcount(S)`, `n ← ⌊r/2⌋`; then for each coin `i` with `S_i = 1` rotate `side_i` by `RY(2 arcsin√(n/r))` (left pan with amplitude `√(n/r)`), then `n −= side_i`, `r −= 1`. The amplitude of every path is `Π √(choice probabilities) = 1/√C(w, w/2)` — the uniform state, with the counters ending at `r = n = 0` on every branch, so no uncomputation is needed. The rotation is a multiplexer over the `(r, n)` pairs reachable at coin `i` (computed exactly by `rotation_table`); the total count is polynomial (`≤ N³`, tested). `A` never touches the oracle and is block-diagonal in `s`, so `A S_0 A†` is the reflection about `A_s|0⟩` *in every branch at once*: coherent control over `s` costs nothing extra.

### Why QSearch was replaced, and by what

`QSearch` (BHMT Thm 3) measures, which would collapse the `s` register. A fixed number of ordinary Grover iterations cannot uniformly amplify branches whose `p` range over `[p_min, 1]` (`amplitude.grover_worst_case`, tested). The YLC fixed-point sequence needs only a lower bound: we use

* `p_min(N, k)` — the exact smallest nonzero `p(w, m)` over all `(w, m)` that can occur for **any** weight-`k` secret (`model.p_min`, verified against exhaustive enumeration). It depends on `(N, k)` only.
* `L` = the smallest odd length with YLC width `w_L(δ) ≤ p_min` (`amplitude.choose_L`). Then every branch with `p > 0` reaches `P_good ≥ 1 − δ²`, and `p = 0` branches stay at exactly 0.

The result is a **bounded-error, paper-inspired implementation** of the balance-to-parity step; it is not a step-by-step reconstruction of `QSearch`.

### Per-branch analysis (exact, and measured)

For branch `s` the amplification acts in the 2-D span of `A_s|0⟩` and its balanced component, so with `P = P_L(p)`:

```
U_s|0> = √P e^{iφ}|good> + √(1−P)|bad>
<0| U_s† O_B U_s |0> = P − (1 − P) = 2P − 1          (phases φ cancel: compute–phase–uncompute)
odd m:  P = 0  →  exactly −1, no leakage
even m: P ≥ 1 − δ²  →  v ≥ 1 − 2δ², leaked norm √(1 − v²)
```

`solver.branch_diagnostics` reads `v_s` and the leakage off the coherent simulation of all branches at once; `tests/test_phase_oracle.py` checks them against this formula to 1e−9.

### Success probability

The final state's overlap with the ideal `|x⟩|0⟩` is `F = E_s[(−1)^m v_s] = 2^{−(N−1)} Σ_{w,m} C(k,m)C(N−k,w−m)(−1)^m v(w,m)`, a function of `(N, k, schedule)` only (`solver.predicted_ideal_overlap`, vectorised as `experiments.overlap_fast`). Measuring `S` succeeds with probability `≥ F²` (junk can only add). Every simulated run satisfies this bound (tested).

## Query accounting

| Quantity | Value | Where counted |
|---|---|---|
| B-queries per `S_t` | 2 | `_st_circuit` |
| B-queries in `U` | `2l = L − 1` | `FixedPointSchedule.oracle_calls` |
| appearances of `U` / `U†` | 1 / 1 | `parity_block` |
| B-queries in `O_B` | 1 | `phase_oracle` |
| **total per Find\* run** | **`2L − 1`** | schedule (`expected_queries`) **and** independently by walking the call graph (`count_oracle_calls`); equality tested |
| `W`, `A`, `S_0` | 0 | — |
| `k = 1` circuit | 1 | — |

`L` grows like `log(2/δ)/√p_min` and `p_min ≥ 1/√(2k)` independently of `N` (tested), giving `O(k^{1/4} log(1/δ))` B-queries — the paper's scaling, reproduced as a property of the schedule, not as a measurement.

## Pseudocode

```python
def find_star(oracle, k, delta):
    N = oracle.N
    L = smallest_odd_L(width(L, delta) <= p_min(N, k))     # public data only
    alpha, beta = ylc_phases(L, delta)
    circuit:
        W(S)
        A(S; ws)                                            # uniform equal splits of I(s)
        for j in 1..l:
            B(S, ws; out); P_{beta_j}(out == 0); B(...)     # S_t(beta_j)
            A†; phase e^{-i alpha_j} on ws == 0; A           # S_s(alpha_j)
        X H (out); B(...); H X (out)                         # O_B: −1 on tilted
        ...the inverse of everything between W and O_B...
        W†(S)
    z = measure(S)
    return z if weight(z) == k else None
```
