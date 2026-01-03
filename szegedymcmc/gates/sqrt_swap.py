from qiskit.circuit import QuantumCircuit, Gate


class SqrtSwap(Gate):
    def __init__(self, label=None):
        super().__init__("√SWAP", 2, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(2, name="√SWAP")
        qc.cx(0, 1)
        qc.csx(1, 0)
        qc.cx(0, 1)
        qc.s(1)  # phase-fix so the one-hot amplitudes align
        return qc
    