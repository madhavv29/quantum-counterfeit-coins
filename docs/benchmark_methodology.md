# Benchmark methodology

## Provenance of every number

| Column prefix | Meaning | How produced |
|---|---|---|
| `exact_*` | exact probability / amplitude of the simulated circuit | full circuit executed gate by gate on the exact sparse simulator (cross-checked with Aer) |
| `shots_*` | finite-sample estimate | `shots` samples drawn with NumPy PCG64 (`seed` column) from the exact output distribution of the simulated circuit; 95 % Wilson intervals |
| `aer_shot_*` | finite-sample estimate | Qiskit Aer shot simulation of the measured circuit, fixed `seed_simulator` |
| `bound_*` | rigorous lower bound | `F²`, `F = E_s[(−1)^m v_s]`, computed from `(N, k, schedule)` only |
| `schedule_*` | derived resource count | from the implemented schedule rule at sizes we do not simulate |
| `theory_*` | published asymptotics | reference curves only (`k^{1/4}`, `k log₂(N/k)`) |
| `counted_queries` | logical B-queries | walk of the circuit call graph (forward and inverse calls) |
| `compiled_*` | physical gate cost | `qiskit.transpile`, basis `{cx, u}`, optimisation level 1, `seed_transpiler = 1234` |
| `*_outside_oracle` | gate cost excluding the oracle | same transpilation with every oracle replaced by an empty instruction of the same name (`NullOracle`) |
| `oracle_call_*` | cost of one oracle call | transpiling a single `B_oracle` gate |

Exact vs shot success differ only by sampling error: with 1000 shots and `p ≈ 0.99`, one standard error is ≈ 0.003, larger than the spread of exact values across secrets.

## Resource units are never mixed

Queries, qubits, CX gates, single-qubit gates, depth, simulator wall time and success probability each have their own column. Wall time is the classical CPU time of *simulation* (2-core cloud VM, see `results/run_metadata.json`) and says nothing about quantum runtime.

## Classical baselines (same oracle, same rules)

Classical algorithms see only `BalanceOracle.classical_query`, which accepts only legal equal-pan queries and returns the binary answer.

* **Exhaustive minimax** (`classical.minimax_queries`): game-tree search over candidate sets with memoisation, iterative deepening from the information bound, deduplication of queries by the split they induce. Returns the optimal worst-case number of weighings over *all* adaptive classical strategies — proven optimal for the instances where it terminates within the time limit (otherwise reported as missing, never guessed).
* **Greedy** (`classical.GreedySolver`): picks the legal query that splits the surviving candidates most evenly; first maximiser in enumeration order breaks ties. Run on every secret; we report mean and worst case.
* **Information bound** `⌈log₂ C(N,k)⌉`: worst-case lower bound for any classical strategy with binary answers. It is a bound, not an observed curve.

Fairness caveats: the classical numbers are for exact identification and the quantum numbers for bounded error (success ≥ 0.987); the quantum circuit's queries are superposed, the classical ones are not.

## Configurations

`experiments/configs/default.json` holds every case list, δ, seed and shot count. `experiments/run_all.py` regenerates all CSVs (`--quick` for a smoke run, `--only <section>`); `experiments/make_plots.py` draws all figures from those CSVs only.
