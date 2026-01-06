from itertools import product
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

from szegedymcmc import IsingModel, MoveAcceptanceF, Reflection00


def test_reflection_simple_example():

    example_model = IsingModel.from_list_clauses(
        k=2, d=3, n=4, 
        Js=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.125],
        terms=[(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3), (1,3)])
    
    n = example_model.n
    R00 = Reflection00(example_model)
    U = Operator(R00).data

    dim = 2 ** (n + 1)
    U_expected = -1 * np.eye(dim, dtype=complex)
    U_expected[0, 0] = 1.0                         # flip phase only on |0...0>_(M,C)

    assert np.allclose(U, U_expected)