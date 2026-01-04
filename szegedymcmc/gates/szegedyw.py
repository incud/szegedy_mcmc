import numpy as np
import networkx as nx
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator


def check_row_stochasticity(P, tol=1e-10):
    if (P < 0).any() or not np.allclose(P.sum(1), 1, atol=tol):
        raise ValueError("Invalid MC.")
    

def check_irreduciblility(P, tol=1e-10):
    if not nx.is_strongly_connected(nx.from_numpy_array(P > tol, create_using=nx.DiGraph)):
        raise ValueError("Chain is not irreducible.")
    

def check_reversibility(P, tol=1e-10):
    G = nx.from_numpy_array(P > tol, create_using=nx.Graph)
    for cycle in nx.cycle_basis(G):
        fwd = bwd = 0.0
        for i in range(len(cycle)):
            a, b = cycle[i], cycle[(i + 1) % len(cycle)]
            if P[a, b] <= tol or P[b, a] <= tol:
                raise ValueError("Non-reversible: missing reverse edge.")
            fwd += np.log(P[a, b])
            bwd += np.log(P[b, a])
        if abs(fwd - bwd) > tol:
            raise ValueError("Detailed balance fails on a cycle.")


class SzegedyW(Gate):
    """
    Oracle W (Eq. 2 arXiv:1910.01659):
        W|x>|0> = |w_x>|x>,   |w_x> = sum_y sqrt(P[x,y]) |y|
    Acts on 2n qubits, n = ceil(log2(d)).
    """

    def __init__(self, P, tol=1e-10, label=None):
        self._validate(P, tol)
        self.P, self.tol = P, tol
        self.d = P.shape[0]
        self.n = int(np.ceil(np.log2(self.d))) if self.d > 1 else 1
        self.dim = 1 << self.n
        super().__init__("SzegedyW", 2 * self.n, [], label=label)
        self.definition = self._build_definition()

    def _validate(self, P, tol):
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("P must be square.")
        check_row_stochasticity(P, tol)
        check_irreduciblility(P, tol)
        check_reversibility(P, tol)
    
    def _build_definition(self):
        qc = QuantumCircuit(2 * self.n, name="SzegedyW")
        A, B = range(self.n), range(self.n, 2 * self.n)

        # |x>|0> → |0>|x>
        for k in range(self.n):
            qc.swap(A[k], B[k])

        # StatePreparation for each |w_x>
        for x in range(self.d):
            if not (self.P[x] > self.tol).any():
                continue
            amps = np.zeros(self.dim)
            amps[:self.d] = np.sqrt(self.P[x])
            amps /= np.linalg.norm(amps)
            qc.append(
                StatePreparation(amps).control(self.n, ctrl_state=x),
                list(B) + list(A),
            )

        return qc