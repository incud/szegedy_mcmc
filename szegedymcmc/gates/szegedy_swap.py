import numpy as np
import networkx as nx
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator


class SzegedySwap(Gate):
    
    def __init__(self, N: int, label=None):
        self.N = int(N)
        self.n = (self.N - 1).bit_length() if self.N > 1 else 1
        super().__init__("S", 2 * self.n, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(2 * self.n, name="SzegedyS")
        A = list(range(self.n))
        B = list(range(self.n, 2 * self.n))

        for j in range(self.n):
            qc.swap(A[j], B[j])

        return qc
