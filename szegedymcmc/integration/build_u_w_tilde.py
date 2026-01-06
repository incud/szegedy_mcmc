import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from szegedymcmc.ising import IsingModel
from szegedymcmc.gates import MovePreparationV, BoltzmannCoinB, MoveAcceptanceF, Reflection00


def build_U_W_tilde(model: IsingModel, beta: float) -> np.ndarray:

    n = model.n
    V = MovePreparationV(model)
    B = BoltzmannCoinB(model, beta)
    F = MoveAcceptanceF(model, construction="linear")
    R = Reflection00(model)

    S = list(range(n))
    M = list(range(n, 2*n))
    C = 2*n

    qc_tilde = QuantumCircuit(2*n + 1)
    qc_tilde.append(V, M)
    qc_tilde.append(B, S + M + [C])
    qc_tilde.append(F, S + M + [C])
    qc_tilde.append(B.inverse(), S + M + [C])
    qc_tilde.append(V.inverse(), M)
    qc_tilde.append(R, M + [C])

    U_W_tilde = Operator(qc_tilde).data
    return U_W_tilde
