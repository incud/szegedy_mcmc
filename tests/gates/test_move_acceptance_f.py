from itertools import product
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

from szegedymcmc import IsingModel, MoveAcceptanceF


def test_simple_example():

    example_model = IsingModel.from_list_clauses(
        k=2, d=3, n=4, 
        Js=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.125],
        terms=[(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3), (1,3)])
    
    n = example_model.n
    F = MoveAcceptanceF(example_model, construction="linear")

    S = list(range(n))
    M = list(range(n, 2*n))
    C = 2*n

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

    # For |b>_C, b = 0 the unitary has no effect
    print("Checking the action of F with the coin b=0. The state should remain the same")
    for x in range(2**n):
        for j in range(n):
            qc = QuantumCircuit(2*n + 1)
            prepare_state(qc, x, j, 0)
            probs_in = Statevector.from_label("0" * (2*n + 1)).evolve(qc).probabilities_dict()
            qc.append(F, S+M+[C])
            probs_out = Statevector.from_label("0" * (2*n + 1)).evolve(qc).probabilities_dict()
            print("CORRECT?", "Y" if probs_in == probs_out else "N", " | ", probs_in, "=>", probs_out)
            assert probs_in == probs_out

    # For |b>_C, b = 1 should flip the j-th bit
    print("\nChecking the action of F with the coin b=1. The j-th bit should be flipped")
    for x in range(2**n):
        for j in range(n):
            qc = QuantumCircuit(2*n + 1)
            prepare_state(qc, x, j, 1)
            probs_in = Statevector.from_label("0" * (2*n + 1)).evolve(qc).probabilities_dict()
            bitstring_in = int(list(probs_in.keys())[0], base=2)
            qc.append(F, S+M+[C])
            probs_out = Statevector.from_label("0" * (2*n + 1)).evolve(qc).probabilities_dict()
            bitstring_out = int(list(probs_out.keys())[0], base=2)
            print("CORRECT?", "Y" if bitstring_in == bitstring_out ^ (1 << j) else "N", " | ", probs_in, "=>", probs_out)
            assert bitstring_in == bitstring_out ^ (1 << j)