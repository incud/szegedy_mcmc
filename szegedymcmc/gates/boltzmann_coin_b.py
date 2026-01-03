from itertools import product
import numpy as np
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.library import RYGate

from szegedymcmc.ising import IsingModel


class BoltzmannCoinRj(Gate):
    def __init__(self, ising_model: IsingModel, beta: float, j: int, label=None):
        self.ising_model = ising_model
        self.beta = float(beta)
        self.j = int(j)
        self.n = ising_model.n
        assert 0 <= j < self.n, f"The move index 0 <= j < n is invalid, {j=}"
        super().__init__(f"R{self.j}", 2 * self.n + 1, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        n = self.n
        qc = QuantumCircuit(2 * n + 1, name=f"R{self.j}")
        S = list(range(n))
        M = list(range(n, 2 * n))
        C = 2 * n

        # Step 1: determine the neighbors of spin j in the terms of the model
        Nj = self._find_neighbors_Nj()
        assert len(Nj) <= 1 + self.ising_model.d * (self.ising_model.k - 1)
        ctrls = [M[self.j]] + [S[q] for q in Nj] # control qubits

        # Step 2: enumerate all the possible assignments of the spins in N_j
        assignments = product([0, 1], repeat=len(Nj))
        
        for bits in assignments:
            # Step 3: precompute classically the corresponding angle
            angle = 2.0 * self._calculate_theta(Nj, bits)
            # Step 4: apply the controlled Ry
            ctrl_state = "".join("1" if b else "0" for b in bits[::-1]) + "1" # reversed as we accomodate qiskit's endianness
            qc.append(RYGate(angle).control(len(ctrls), ctrl_state=ctrl_state), ctrls + [C])

        return qc

    def _find_neighbors_Nj(self):
        Nj = {self.j}
        for om in self.ising_model.Omega:
            if self.j in om:
                Nj.update(om)
        return sorted(Nj)

    def _calculate_theta(self, Nj, bits):
        delta = self._calculate_delta(Nj, bits)
        A = 1.0 if delta <= 0 else min(1.0, float(np.exp(-self.beta * delta)))
        return float(np.arcsin(np.sqrt(A)))

    def _calculate_delta(self, Nj, bits):
        x = np.ones(self.n)
        x[list(Nj)] = 1 - 2*np.array(bits)
        return -2.0 * sum(J * np.prod(x[list(om)]) for J, om in zip(self.ising_model.J, self.ising_model.Omega) if self.j in om)


class BoltzmannCoinB(Gate):
    def __init__(self, model: IsingModel, beta: float, label=None):
        self.model = model
        self.beta = float(beta)
        self.n = int(model.n)
        super().__init__("B", 2 * self.n + 1, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(2 * self.n + 1, name="B")
        for j in range(self.n):
            qc.append(BoltzmannCoinRj(self.model, self.beta, j), list(range(2 * self.n + 1)))
        return qc
