"""
Integration tests for the Szegedy half-step walk operator U_W built from a Metropolis–Hastings
transition matrix P on d=2^n configurations.

Background and conventions
--------------------------
We work on the 2-register Hilbert space H = C^d ⊗ C^d, represented by 2n qubits in Qiskit.
We interpret the computational basis |a>_A |b>_B using Qiskit’s little-endian indexing, i.e.
the flattened basis index is idx(a,b) = a + d*b, so a is the “least significant” register.

Given a row-stochastic transition matrix P, define for each x the normalized “coin” state
|w_x> := sum_y sqrt(P[x,y]) |y>.  In the Szegedy construction, the oracle/isometry W satisfies
W |x>_B |0>_A = |w_x>_A |x>_B (up to register naming; the tests below follow the implementation’s
exact ordering).  The reflection about the clean-subspace E0 := span{|x>_B|0>_A} is
R0 = 2 Π0 - I where Π0 projects onto the subspace where the “second register” equals |0...0>.

From W and the swap Λ between the two d-dimensional registers, define the two canonical
Szegedy reflections:
    R_A := W R0 W†    (reflection about span{|w_x>_A |x>_B}_x, i.e. outgoing-transition subspace)
    R_B := Λ R_A Λ    (reflection about span{|x>_A |w_x>_B}_x, i.e. incoming-transition subspace)

The “half-step” unitary used in spectral analysis is
    U_W := R0 W† Λ W  = R0 (W† Λ W),
which is a product of reflections and therefore unitary.  In the decomposition
H = E0 ⊕ E0^⊥, the top-left d×d block of U_W restricted to E0 is the matrix
X with entries X_{y,x} = <y,0| W† Λ W |x,0> = sqrt(P[x,y] P[y,x]) (for reversible chains),
so on the clean-subspace U_W encodes the discriminant/symmetrized kernel rather than P itself.

What the tests verify
---------------------
1) test_build_U_W_pieces_simple_example:
   - Constructs W, Λ, and R0 explicitly, then forms R_A and R_B.
   - Verifies that R_A and R_B are genuine reflections (Hermitian and involutory).
   - Verifies the defining “fixed-subspace” property:
       R_A |w_x>_A |x>_B = |w_x>_A |x>_B
       R_B |x>_A |w_x>_B = |x>_A |w_x>_B
     for every configuration x, up to a global phase. This checks that the implemented W
     correctly prepares the outgoing-transition states |w_x> and that conjugation by Λ
     produces the incoming-transition reflection as expected.

2) test_build_U_W_simple_example:
   - Builds the full half-step unitary U_W via build_U_W (which computes U_W = R0 W† Λ W).
   - Checks, for single-spin-flip neighbors (x,y) (i.e. (x ^ y) has Hamming weight 1), that
     the clean-subspace matrix elements match the theoretical discriminant:
         <y,0|U_W|x,0> = sqrt(P[x,y] P[y,x]).
     This validates the key top-left block identity that underpins the spectral relation
     between U_W and the classical Markov chain.

Note on “full action”:
----------------------
Only the action of W on |x>|0> is fixed by construction; its action on |x>|y≠0> is an arbitrary
unitary completion (here produced by Qiskit's StatePreparation). For this reason, these tests
focus on properties that are uniquely determined by the Szegedy construction: invariance of
the outgoing/incoming subspaces under R_A/R_B and the clean-subspace matrix elements of U_W.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from szegedymcmc.ising import IsingModel
from szegedymcmc.gates import SzegedyW
from szegedymcmc.ising.transition_matrix import build_transition_matrix_P
from szegedymcmc.integration.build_u_w import build_U_W

from tests.test_utils import close_up_to_phase


def test_build_U_W_pieces_simple_example():

    # 0) give me the example model
    example_model = IsingModel.from_list_clauses(
        k=2, d=3, n=4,
        Js=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.125],
        terms=[(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3), (1,3)]
    )

    # (we also need P for the expected values)
    beta = 1.0
    P = build_transition_matrix_P(example_model, beta)
    n = example_model.n
    d = 2**n
    N = d * d

    # Build W and Λ explicitly (same conventions as build_U_W)
    TW = Operator(SzegedyW(P)).data

    qcLam = QuantumCircuit(2 * n)
    for k in range(n):
        qcLam.swap(k, n + k)
    TLam = Operator(qcLam).data

    # R0 = 2 Π0 - I, Π0 projects onto second register |0>
    R0 = np.diag([1.0 if ((idx >> n) == 0) else -1.0 for idx in range(N)])

    # Reflections about outgoing/incoming subspaces
    R_A = TW @ R0 @ TW.conj().T
    R_B = TLam @ R_A @ TLam

    # sanity: reflections
    assert np.allclose(R_A.conj().T, R_A, atol=1e-8) and np.allclose(R_A @ R_A, np.eye(N), atol=1e-8)
    assert np.allclose(R_B.conj().T, R_B, atol=1e-8) and np.allclose(R_B @ R_B, np.eye(N), atol=1e-8)

    # basis vectors: combined index = a + d*b corresponds to |a>_A |b>_B, so state is kron(|b>, |a|)
    for x in range(d):
        ex = np.zeros(d, dtype=complex); ex[x] = 1.0
        wx = np.sqrt(P[x]).astype(complex)   # ||wx||^2 = sum_y P[x,y] = 1

        psi_A = np.kron(ex, wx)              # |w_x>_A |x>_B   (B factor first)
        assert close_up_to_phase(R_A @ psi_A, psi_A, atol=1e-8)

        psi_B = TLam @ psi_A                 # |x>_A |w_x>_B
        assert close_up_to_phase(R_B @ psi_B, psi_B, atol=1e-8)


def test_build_U_W_simple_example():

    # 0) give me the example model
    example_model = IsingModel.from_list_clauses(
        k=2, d=3, n=4,
        Js=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.125],
        terms=[(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3), (1,3)]
    )

    # 1) instantiate U_W using build_U_W
    beta = 1.0
    U_W = build_U_W(example_model, beta)

    # (we also need P for the expected values)
    P = build_transition_matrix_P(example_model, beta)
    n = example_model.n
    d = 2**n

    # 2) iterate on each entry (x, y) in P
    # 2.1) make sure the move z = x ^ y has exactly one bit set
    # 2.2) check that <y,0|U_W|x,0> equals sqrt(P[x,y] P[y,x]) for single-spin flips
    def get_entry(x, y):
        # |x,0> and |y,0> live in the top-left d×d block (B=0), hence indices are just x and y
        return U_W[y, x]

    for x in range(d):
        for y in range(x + 1, d):
            if (x ^ y).bit_count() != 1:
                continue

            expected = np.sqrt(P[x, y] * P[y, x])
            got = get_entry(x, y)
            assert np.isclose(got, expected, atol=1e-8), (
                f"mismatch at x={x} y={y}: got {got}, expected {expected}"
            )


