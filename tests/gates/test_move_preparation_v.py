from itertools import product
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

from szegedymcmc import IsingModel, MovePreparationV


def test_simple_example():

    example_model = IsingModel.from_list_clauses(
        k=2, d=3, n=4, 
        Js=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.125],
        terms=[(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3), (1,3)])
    
    V = MovePreparationV(example_model)

    qc = QuantumCircuit(example_model.n)
    qc.append(V, range(example_model.n))

    sv = Statevector.from_label("0" * example_model.n).evolve(qc)
    probs_actual = sv.probabilities_dict()
    probs_expected = {"0001": 0.25, "0010": 0.25, "0100": 0.25, "1000": 0.25}

    for key_actual, key_expected in zip(probs_actual.keys(), probs_expected.keys()):
        assert str(key_actual) == str(key_expected)
        assert np.isclose(probs_actual[key_actual], probs_expected[key_expected])
