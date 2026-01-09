import numpy as np
from qiskit.circuit import QuantumCircuit, Gate


class SzegedyReflectionI0(Gate):
    """
    R0 = I_A ⊗ (2|0..0><0..0| - I_B)

    Acts on 2n qubits:
      - A = first n qubits
      - B = last  n qubits

    Implementation:
      - For n=1: R0 on B is just Z.
      - For n>=2: implement phase flip on |0..0> via:
            X on all B
            MCZ on |1..1> (using H + MCX + H)
            X on all B
        This produces (-1) on |0..0> and (+1) elsewhere.
        Then add a global phase of π to get (+1) on |0..0> and (-1) elsewhere.
        (The global phase is physically irrelevant, but this makes the matrix exactly match 2|0><0|-I.)
    """

    def __init__(self, N: int, label=None):
        self.N = int(N)
        self.n = (self.N - 1).bit_length() if self.N > 1 else 1
        super().__init__("R_I0", 2 * self.n, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(2 * self.n, name="SzegedyR0")
        B = list(range(self.n, 2 * self.n))  # reflect about |0..0> on B only

        if self.n == 1:
            # 2|0><0| - I = Z for a single qubit
            qc.z(B[0])
            return qc

        # Step 1: X on all B maps |0..0> -> |1..1>
        for q in B:
            qc.x(q)

        # Step 2: apply multi-controlled-Z on state |1..1>
        # Implement MCZ using H on a target qubit + MCX + H.
        target = B[0]
        controls = B[1:]

        qc.h(target)
        qc.mcx(controls, target, mode="noancilla")
        qc.h(target)

        # Step 3: X back
        for q in B:
            qc.x(q)

        # This sequence flips phase on |0..0> only (i.e., gives -1 to |0..0>, +1 otherwise).
        # Multiply by a global -1 to obtain +1 on |0..0> and -1 on all others.
        qc.global_phase = np.pi

        return qc
