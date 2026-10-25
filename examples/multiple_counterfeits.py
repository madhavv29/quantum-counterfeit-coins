"""k > 1: the coherent bounded-error balance-to-parity circuit (Find*).

    python examples/multiple_counterfeits.py

The solver sees only the oracle's gate. It builds: W (paper App. A) ->
coherent uniform balanced-partition preparation A -> YLC fixed-point
amplification of balanced weighings -> phase on tilted -> uncompute -> W^dag,
then measures the selection register.
"""

from quantum_counterfeit import BalanceOracle, build_find_star, run
from quantum_counterfeit.solver import branch_diagnostics, predicted_ideal_overlap

N, k = 8, 3
secret = (0, 1, 0, 0, 1, 0, 0, 1)  # known only to the oracle

oracle = BalanceOracle(secret)
sc = build_find_star(oracle, delta=0.1)
print(f"N = {N}, k = {k}: p_min = {sc.p_min:.4f}, fixed-point length L = {sc.schedule.L}")
print(f"qubits: {sc.num_qubits}")
print(f"B-oracle queries: {sc.counted_queries} counted in the circuit, {sc.scheduled_queries} from the schedule")

result = run(sc)
bound = predicted_ideal_overlap(N, k, sc.schedule) ** 2
print(f"P(output = full counterfeit set) = {result.success_probability(secret):.6f}  (rigorous bound {bound:.6f})")

answer = result.sample(shots=1, seed=4)[0]
coins = "decoding failure" if answer is None else [i for i, b in enumerate(answer) if b]
print("measured answer:", answer, "-> counterfeit coins", coins)

# Per-branch view of the simulated parity oracle (analysis only; uses the
# secret to LABEL branches by m, never inside the circuit).
recs, _ = branch_diagnostics(oracle, secret, delta=0.1)
odd = [r for r in recs if r.m % 2]
even = [r for r in recs if r.m % 2 == 0]
print(f"odd-m branches:  {len(odd)}, coefficient range "
      f"[{min(r.value.real for r in odd):+.6f}, {max(r.value.real for r in odd):+.6f}] (ideal -1)")
print(f"even-m branches: {len(even)}, coefficient range "
      f"[{min(r.value.real for r in even):+.6f}, {max(r.value.real for r in even):+.6f}] (ideal +1)")
