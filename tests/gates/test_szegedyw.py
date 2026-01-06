"""
These tests validate the Szegedy oracle gate W constructed by `SzegedyW(P)` for the
single–spin-flip Metropolis–Hastings transition matrix P of a small Ising model.

Background / conventions (matching the implementation):

- There are two n-qubit registers, A and B, so W acts on 2n qubits.
- Qiskit uses little-endian ordering for basis indices. With the common “flattened”
  indexing convention consistent with `np.kron(|B>, |A>)`, the computational basis
  state |a>_A |b>_B corresponds to the integer index `a + d*b`, where d = 2^n.

What `SzegedyW` implements:

1) It first swaps registers so that the logical action is
      W |x>_A |0>_B  =  |w_x>_A |x>_B,
   where |w_x> = sum_y sqrt(P[x,y]) |y> (padded to length d if needed).

2) It does so by, for each x, applying a multi-controlled `StatePreparation` on A
   controlled by the value x in B (after the swap). The net effect is the mapping above.

The tests below check:
(A) Action test: For every x, evolving |x>_A|0>_B by W produces a state whose B marginal
    is exactly |x> (probability 1), and whose A marginal matches |w_x| (probabilities P[x,*]).

(B) X-block test: The operator X := Π0 W† Λ W Π0 (restricted to the Π0-subspace) has matrix
    elements X[y,x] = <y,0| W† Λ W |x,0> = sqrt(P[x,y] P[y,x]). This is the key identity used
    in the spectral analysis: X is similar to P for reversible chains.

Both checks are “structural”: they confirm that the register swap and the controlled
state-preparation are wired consistently with the intended Szegedy construction.
"""

from itertools import product
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

from szegedymcmc.gates import SzegedyW
from szegedymcmc.ising.ising_model import IsingModel
from szegedymcmc.ising.transition_matrix import build_transition_matrix_P


def test_W_action_on_x0_prepares_wx_and_copies_x():

    model = IsingModel.from_list_clauses(
        k=2, d=3, n=4,
        Js=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.125],
        terms=[(0,), (1,), (2,), (3,), (0, 1), (0, 2), (0, 3), (1, 3)],
    )
    beta = 1.0
    P = build_transition_matrix_P(model, beta)

    n = model.n
    d = 2**n
    W = Operator(SzegedyW(P)).data

    # Basis index for |a>_A|b>_B is a + d*b (with vector ordering kron(|B>,|A|))
    idx = lambda a, b: a + d * b

    for x in range(d):
        # Input state |x>_A |0>_B
        ket_in = np.zeros(d * d, dtype=complex)
        ket_in[idx(x, 0)] = 1.0

        ket_out = W @ ket_in

        # 1) B must be exactly |x>: probability mass only on basis states (a, b=x) for varying a
        probs_by_b = np.zeros(d, dtype=float)
        for b in range(d):
            # sum over a of |amp(a,b)|^2
            s = 0.0
            for a in range(d):
                s += (abs(ket_out[idx(a, b)]) ** 2)
            probs_by_b[b] = s

        assert np.isclose(probs_by_b[x], 1.0, atol=1e-10), f"B not copied correctly for x={x}"
        assert np.isclose(probs_by_b.sum(), 1.0, atol=1e-10)
        assert np.allclose(np.delete(probs_by_b, x), 0.0, atol=1e-10)

        # 2) Conditional on B=x, the distribution on A must be P[x,*]
        probs_A = np.array([abs(ket_out[idx(a, x)]) ** 2 for a in range(d)], dtype=float)

        # W uses sqrt(P[x,y]) amplitudes on A, so probabilities should be exactly P[x,y].
        assert np.allclose(probs_A, P[x], atol=1e-10), f"A marginal mismatch for x={x}"


def test_W_induces_correct_X_block_via_Wdag_Lambda_W():
    
    model = IsingModel.from_list_clauses(
        k=2, d=3, n=4,
        Js=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.125],
        terms=[(0,), (1,), (2,), (3,), (0, 1), (0, 2), (0, 3), (1, 3)],
    )
    beta = 1.0
    P = build_transition_matrix_P(model, beta)

    n = model.n
    d = 2**n
    N = d * d

    W = Operator(SzegedyW(P)).data

    # Λ swaps A and B
    qcLam = QuantumCircuit(2 * n)
    for k in range(n):
        qcLam.swap(k, n + k)
    Lam = Operator(qcLam).data

    # T := W† Λ W
    T = W.conj().T @ Lam @ W

    # Π0 projects onto states with B=0. In the basis |a>_A|b>_B (index a + d*b),
    # Π0 keeps exactly the first d basis vectors (b=0).
    Pi0 = np.zeros((N, N), dtype=complex)
    Pi0[:d, :d] = np.eye(d, dtype=complex)

    # X = Π0 T Π0 restricted to the Π0-subspace is the top-left d×d block of Π0 T Π0,
    # i.e. X[y,x] = <y,0|T|x,0>.
    X = (Pi0 @ T @ Pi0)[:d, :d]

    # Check the defining identity: X[y,x] = sqrt(P[x,y] P[y,x])
    for x in range(d):
        for y in range(d):
            expected = np.sqrt(P[x, y] * P[y, x])
            got = X[y, x]
            assert np.isclose(got, expected, atol=1e-8), (
                f"X mismatch at (y,x)=({y},{x}): got {got}, expected {expected}"
            )
