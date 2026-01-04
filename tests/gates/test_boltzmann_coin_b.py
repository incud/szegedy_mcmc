
from itertools import product
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

from szegedymcmc import IsingModel, MoveAcceptanceF, BoltzmannCoinRj, BoltzmannCoinB


def test_simple_example_rj():

    example_model = IsingModel.from_list_clauses(
        k=2, d=3, n=4, 
        Js=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.125],
        terms=[(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3), (1,3)])

    beta = 1.0
    n = example_model.n
    S = list(range(n))
    M = list(range(n, 2*n))
    C = 2*n

    def energy(x):
        spins = [1 - 2*((x >> i) & 1) for i in range(n)]  # 0->+1, 1->-1
        return sum(J * np.prod([spins[q] for q in om]) for J, om in zip(example_model.J, example_model.Omega))

    def approximate_dict(d, decimals=10):
        return {k: np.round(v, decimals) for k, v in d.items() if v > 1e-10}
    
    def prepare_state(qc, x, j, b):
        # prepare |x>_S
        for k in range(n):
            if (x >> k) & 1:
                qc.x(S[k])
        # prepare unary move |j>_M
        qc.x(M[j])
        # prepare coin |b>
        if b:
            qc.x(C)

    # --- 1) If the move in M is not j, then R_j must do nothing (coin stays |0>, S and M unchanged) ---
    print("Checking the action of R_j when the move register is m != j (R_j should be identity)")
    for j in range(n):
        Rj = BoltzmannCoinRj(example_model, beta, j)
        for x in range(2**n):
            for m in range(n):
                if m == j:
                    continue
                qc = QuantumCircuit(2*n + 1)
                prepare_state(qc, x, m, 0)
                probs_in = approximate_dict(Statevector.from_label("0" * (2*n + 1)).evolve(qc).probabilities_dict())
                qc.append(Rj, S + M + [C])
                probs_out = approximate_dict(Statevector.from_label("0" * (2*n + 1)).evolve(qc).probabilities_dict())
                print("CORRECT?", "Y" if probs_in == probs_out else "N", f" | x={x:0{n}b}, j={j}, m={m} | {probs_in} => {probs_out} ")
                assert probs_in == probs_out

    # --- 2) If the move in M is j and coin starts in |0>, then R_j should load A into the coin ---
    print("\nChecking the action of R_j when the move register is m = j (coin probability should match MH acceptance)")
    for j in range(n):
        Rj = BoltzmannCoinRj(example_model, beta, j)
        for x in range(2**n):
            qc = QuantumCircuit(2*n + 1)
            prepare_state(qc, x, j, 0)
            qc.append(Rj, S + M + [C])
            probs_out = Statevector.from_label("0" * (2*n + 1)).evolve(qc).probabilities_dict()

            y = x ^ (1 << j)
            Delta = energy(y) - energy(x)
            A = 1.0 if Delta <= 0 else min(1.0, float(np.exp(-beta * Delta)))

            p1 = sum(p for bs, p in probs_out.items() if ((int(bs, 2) >> C) & 1) == 1)
            ok_prob = np.isclose(p1, A, atol=1e-12)

            base = x | (1 << (n + j))  # system bits + unary move bit, coin masked out
            ok_regs = all(((int(bs, 2) & ~(1 << C)) == base) for bs in probs_out.keys())

            print(
                "CORRECT?", "Y" if (ok_prob and ok_regs) else "N",
                f" | x={x:0{n}b}, j={j}, Δ={Delta:+.6f}, A={A:.12f}, p(C=1)={p1:.12f}"
            )
            assert ok_prob and ok_regs
    