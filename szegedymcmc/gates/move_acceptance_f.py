from qiskit.circuit import QuantumCircuit, Gate

from szegedymcmc.ising import IsingModel


class MoveAcceptanceF(Gate):
    
    def __init__(self, ising_model: IsingModel, construction: str = "linear", label=None):
        self.ising_model = ising_model
        self.n = int(self.ising_model.n)
        self.construction = construction
        assert self.construction in ["linear", "log"], "The only allowed modes are 'linear' and 'log'"

        self.n_aux_qubits = 0 if self.construction == "linear" else self.n - 1
        super().__init__("F", 2 * self.n + 1 + self.n_aux_qubits, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        n = self.n

        if self.construction == "linear" or n <= 1:
            qc = QuantumCircuit(2 * n + 1, name="F")
            S = list(range(n))
            M = list(range(n, 2 * n))
            C = 2 * n
            for j in range(n):
                qc.ccx(M[j], C, S[j])
            return qc

        # log-depth: fanout coin C into (n-1) ancillas, use them as parallel coin-controls, then unfanout
        qc = QuantumCircuit(2 * n + 1 + (n - 1), name="F")
        S = list(range(n))
        M = list(range(n, 2 * n))
        C = 2 * n
        A = list(range(2 * n + 1, 2 * n + 1 + (n - 1)))  # coin copies

        raise ValueError("Not implemented yet")
