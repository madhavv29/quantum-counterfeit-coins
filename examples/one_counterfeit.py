"""k = 1: find a single counterfeit coin with ONE quantum weighing.

    python examples/one_counterfeit.py

This is the Terhal-Smolin / Bernstein-Vazirani case. For k = 1 the balance
oracle coincides with the parity (inner-product) oracle on even selections,
so it says nothing about the k > 1 algorithm (see multiple_counterfeits.py).
"""

from quantum_counterfeit import BalanceOracle, build_one_coin, run

N = 8
secret = (0, 0, 0, 0, 0, 1, 0, 0)  # known only to the oracle

oracle = BalanceOracle(secret)
sc = build_one_coin(oracle)
result = run(sc)
answer = result.sample(shots=1, seed=1)[0]

print(f"N = {N}, k = 1")
print(f"qubits: {sc.num_qubits}, B-oracle queries (counted from the circuit): {sc.counted_queries}")
print(f"P(correct) = {result.success_probability(secret):.12f}")
print(f"measured answer: counterfeit coin index {answer.index(1)}")
assert answer == secret
