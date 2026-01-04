import numpy as np
from qiskit.circuit import QuantumCircuit, Gate

from szegedymcmc.ising import IsingModel


class Reflection00(Gate):
    def __init__(self, ising_model: IsingModel, label=None):
        self.ising_model = ising_model
        self.N = int(self.ising_model.n)          # unary Move register size
        if self.N < 0:
            raise ValueError("n must be >= 0")
        super().__init__("R00", self.N + 1, [], label=label)  # (M of size N) + (coin C)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(self.N + 1, name="R00")
        if self.N == 0:
            qc.z(0)
            qc.global_phase += np.pi             # diag(-1,+1)
            return qc
        qs = list(range(self.N + 1))
        qc.x(qs)
        qc.mcp(np.pi, qs[:-1], qs[-1])           # -1 on |1...1>
        qc.x(qs)                                 # -> -1 on |0...0>
        qc.global_phase += np.pi
        return qc