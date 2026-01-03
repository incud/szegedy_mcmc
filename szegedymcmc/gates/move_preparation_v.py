from qiskit.circuit import QuantumCircuit, Gate

from szegedymcmc.ising import IsingModel
from szegedymcmc.gates.sqrt_swap import SqrtSwap
from szegedymcmc.gates.utils import is_power_of_two


class MovePreparationV(Gate):

    def __init__(self, ising_model: IsingModel, label=None):
        self.ising_model = ising_model
        self.n = int(self.ising_model.n)
        if not is_power_of_two(self.n):
            raise ValueError("This simplified implementation works best with n being a power of two")
        super().__init__("V", self.n, [], label=label)
        self.definition = self._build_definition()

    def _build_definition(self):
        qc = QuantumCircuit(self.n, name="V")
        if self.n == 1:
            qc.x(0)
            return qc

        qc.x(0)  # |10..00>

        step = 1
        while step < self.n:
            block = 2 * step
            for start in range(0, self.n, block):
                for j in range(step):
                    a = start + j
                    b = start + j + step
                    qc.append(SqrtSwap(), [a, b])
                    
            step *= 2

        return qc