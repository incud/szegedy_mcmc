"""
Integration tests for the “efficient” half-step walk implementation U_W_tilde.

Context and intent
------------------
The baseline Szegedy half-step operator U_W = R0 W† Λ W acts on H = C^d ⊗ C^d (2n qubits),
but a direct implementation of W generally requires expensive state preparation conditioned on x.
The paper’s Section 2 introduces an alternative construction that avoids preparing |w_x> explicitly
by working in an enlarged space that includes (i) an explicit Move register and (ii) a Coin qubit.
In our simplified single-spin-flip setting, the registers are:

- System register S: n qubits, computational basis |x> with x ∈ {0,1}^n (d = 2^n).
- Move register M: n qubits (unary / one-hot), intended basis states are |0> (no move)
  and |2^j> encoding “flip spin j”.
- Coin register C: 1 qubit, used to coherently encode accept/reject.

The implemented operator is the “tilde” half-step unitary (Eq. 25 in the paper’s notation),
constructed as a product of simple gates:
    U_W_tilde = R00 · V† · B† · F · B · V
(where the rightmost operation is applied first).

Pieces and their roles
----------------------
- V (MovePreparationV): prepares a uniform superposition over unary moves on M (for the single-spin
  flip proposal distribution). In our simplified construction n is assumed to be a power of two.
- B (BoltzmannCoinB): applies a move-controlled, local, x-dependent rotation on the coin qubit so that
  Pr(C=1) equals the Metropolis–Hastings acceptance A_{y x} for the proposed neighbor y = x ⊕ 2^j.
- F (MoveAcceptanceF): conditionally applies the selected move to S only when C=1 (accept branch),
  leaving S unchanged when C=0 (reject branch). This prevents unwanted superpositions over neighbors
  in the reject branch.
- R00 (Reflection00): a phase flip on the single state |0...0>_M |0>_C, acting as identity elsewhere;
  it is the analogue of the “clean-subspace reflection” in the enlarged space.

What these tests verify
-----------------------
1) test_build_U_W_tilde_pieces_simple_example:
   - Reconstructs U_W_tilde explicitly from its pieces (V, B, F, V†, B†, R00) and checks basic
     invariants that must hold independently of any “oracle completion” choices:
       * U_W_tilde is unitary on the full 2n+1 qubit space.
       * R00 is a reflection (Hermitian, involutory) and only flips |0...0>_(M,C).
     This test is aimed at catching register-ordering mistakes (S/M/C placement) and accidental
     non-unitarity from composing the pieces.

2) test_build_U_W_tilde_simple_example:
   - Checks the key clean-subspace matrix elements that are uniquely determined by the intended walk
     dynamics. With qubit order [S (n), M (n), C (1)], the state |x>_S |0>_M |0>_C corresponds to the
     basis index x. For single-spin-flip neighbors (x,y) (i.e. Hamming weight of x^y is 1), the test
     verifies:
         <y,0,0| U_W_tilde |x,0,0> = sqrt(P[x,y] P[y,x]),
     where P is the classical MH transition matrix under the same proposal/acceptance rules.

Note on scope
-------------
U_W_tilde acts on a larger Hilbert space than U_W. These tests therefore focus on (i) correctness of
the full composed unitary and (ii) the reduced action on the “clean” subspace |·>_S|0>_M|0>_C that
carries the intended Szegedy dynamics.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from szegedymcmc.ising import IsingModel
from szegedymcmc.gates import MovePreparationV, BoltzmannCoinB, MoveAcceptanceF, Reflection00
from szegedymcmc.ising.transition_matrix import build_transition_matrix_P
from szegedymcmc.integration.build_u_w_tilde import build_U_W_tilde

from tests.test_utils import close_up_to_phase


def test_build_U_W_tilde_pieces_simple_example():

    example_model = IsingModel.from_list_clauses(
        k=2, d=3, n=4,
        Js=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.125],
        terms=[(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3), (1,3)]
    )

    beta = 1.0
    n = example_model.n
    d = 2**n
    Ntilde = 2**(2*n + 1)

    V = MovePreparationV(example_model)
    B = BoltzmannCoinB(example_model, beta)
    F = MoveAcceptanceF(example_model, construction="linear")
    R00 = Reflection00(example_model)

    S = list(range(n))
    M = list(range(n, 2*n))
    C = 2*n

    qc = QuantumCircuit(2*n + 1)
    qc.append(V, M)
    qc.append(B, S + M + [C])
    qc.append(F, S + M + [C])
    qc.append(B.inverse(), S + M + [C])
    qc.append(V.inverse(), M)
    qc.append(R00, M + [C])

    U = Operator(qc).data

    # full unitarity of the composed circuit
    assert np.allclose(U.conj().T @ U, np.eye(Ntilde), atol=1e-8)

    # R00 is a reflection on (M,C) and flips only |0...0>_(M,C)
    R = Operator(R00).data
    assert np.allclose(R, R.conj().T, atol=1e-8) and np.allclose(R @ R, np.eye(2**(n+1)), atol=1e-8)

    e0 = np.zeros(2**(n+1), dtype=complex)
    e0[0] = 1.0
    assert close_up_to_phase(R @ e0, -e0, atol=1e-8)

    # a couple of nonzero basis states are unchanged (spot-check)
    for k in [1, 2, 2**n, 2**n + 3]:
        ek = np.zeros(2**(n+1), dtype=complex)
        ek[k] = 1.0
        assert close_up_to_phase(R @ ek, ek, atol=1e-8)


def test_build_U_W_tilde_simple_example():

    example_model = IsingModel.from_list_clauses(
        k=2, d=3, n=4,
        Js=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.125],
        terms=[(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3), (1,3)]
    )

    beta = 1.0
    P = build_transition_matrix_P(example_model, beta)
    n = example_model.n
    d = 2**n

    U_W_tilde = build_U_W_tilde(example_model, beta)

    def get_entry(x, y):
        # |x>_S|0>_M|0>_C and |y>_S|0>_M|0>_C correspond to indices x and y.
        return U_W_tilde[y, x]

    for x in range(d):
        for y in range(d):
            z = x ^ y
            if z.bit_count() != 1:
                continue

            expected = np.sqrt(P[x, y] * P[y, x])
            got = get_entry(x, y)

            assert np.isclose(got, expected, atol=1e-8), (
                f"mismatch at x={x} y={y}: got {got}, expected {expected}"
            )
