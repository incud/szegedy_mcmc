import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from szegedymcmc.ising import IsingModel
from szegedymcmc.gates import SzegedyW, MovePreparationV, BoltzmannCoinB, MoveAcceptanceF, Reflection00
from szegedymcmc.ising.transition_matrix import build_transition_matrix_P


def build_U_W(model: IsingModel, beta: float) -> np.ndarray:

    n = model.n
    P = build_transition_matrix_P(model, beta)

    W_gate = SzegedyW(P)                 # your oracle implementation
    TW = Operator(W_gate).data           # matrix of W (2n qubits)

    qcLam = QuantumCircuit(2*n)
    for k in range(n):
        qcLam.swap(k, n+k)
    TLam = Operator(qcLam).data

    R0 = np.diag([1.0 if ((idx >> n) == 0) else -1.0 for idx in range(2**(2*n))])

    UW = R0 @ TW.conj().T @ TLam @ TW
    return UW