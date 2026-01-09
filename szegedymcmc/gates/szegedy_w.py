import numpy as np
from qiskit.circuit import QuantumCircuit, Gate

from szegedymcmc.gates.szegedy_v import SzegedyV
from szegedymcmc.gates.szegedy_swap import SzegedySwap
from szegedymcmc.gates.szegedy_reflection_i0 import SzegedyReflectionI0


class SzegedyW(Gate):
    """
    Build the Szegedy walk operator W from:
      - V  : SzegedyV(P)
      - S  : SzegedySwap(N)
      - R  : SzegedyReflectionI0(N)  (I on first register A, reflection about |0..0> on second register B)

    Standard construction:
      R_A = V (I_A ⊗ R0_B) V†
      R_B = S R_A S
      W   = R_B R_A

    Circuit ordering note (Qiskit appends left-to-right but multiplies right-to-left):
      To implement U = V R V†, append: V†, then R, then V.
    """

    def __init__(self, P, tol=1e-10, label=None):
        self.P = np.asarray(P, dtype=float)
        self.N = int(self.P.shape[0])
        self.n = int(np.ceil(np.log2(self.N))) if self.N > 1 else 1

        # Instantiate primitives (these may validate P internally)
        self.V = SzegedyV(self.P, tol=tol)
        self.S = SzegedySwap(self.N)
        self.RI0 = SzegedyReflectionI0(self.N)

        # All gates must agree on register size
        if self.V.num_qubits != 2 * self.n:
            raise ValueError(
                f"SzegedyV qubit count mismatch: got {self.V.num_qubits}, expected {2*self.n}."
            )
        if self.S.num_qubits != 2 * self.n:
            raise ValueError(
                f"SzegedySwap qubit count mismatch: got {self.S.num_qubits}, expected {2*self.n}."
            )
        if self.RI0.num_qubits != 2 * self.n:
            raise ValueError(
                f"SzegedyReflectionI0 qubit count mismatch: got {self.RI0.num_qubits}, expected {2*self.n}."
            )

        super().__init__("SzegedyW", 2 * self.n, [], label=label)

        # Build reflections and W
        self.RA = self._build_RA()
        self.RB = self._build_RB(self.RA)
        self.definition = self._build_W(self.RA, self.RB)

    def _all_qubits(self):
        return list(range(2 * self.n))

    def _build_RA(self) -> Gate:
        """R_A = V (I ⊗ R0) V†"""
        qc = QuantumCircuit(2 * self.n, name="R_A")
        q = self._all_qubits()

        # Want: RA = V * RI0 * V†
        # Append: V† then RI0 then V
        qc.append(self.V.inverse(), q)
        qc.append(self.RI0, q)
        qc.append(self.V, q)

        return qc.to_gate(label="R_A")

    def _build_RB(self, RA: Gate) -> Gate:
        """R_B = S R_A S"""
        qc = QuantumCircuit(2 * self.n, name="R_B")
        q = self._all_qubits()

        # Append S, RA, S  -> overall unitary = S * RA * S
        qc.append(self.S, q)
        qc.append(RA, q)
        qc.append(self.S, q)

        return qc.to_gate(label="R_B")

    def _build_W(self, RA: Gate, RB: Gate) -> QuantumCircuit:
        """W = R_B R_A"""
        qc = QuantumCircuit(2 * self.n, name="W")
        q = self._all_qubits()

        # Append RA then RB -> overall unitary = RB * RA
        qc.append(RA, q)
        qc.append(RB, q)

        return qc
