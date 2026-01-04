import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from szegedymcmc.ising import IsingModel
from szegedymcmc.gates import SzegedyW, MovePreparationV, BoltzmannCoinB, MoveAcceptanceF, Reflection00
from szegedymcmc.ising.transition_matrix import build_transition_matrix_P


def build_U_W(model: IsingModel, beta: float) -> np.ndarray:

    n = model.n
    P = build_transition_matrix_P(model, beta)

    W_gate = SzegedyW(P)                 # your oracle implementation
    TW = Operator(W_gate).data           # matrix of W (2n qubits)

    qcLam = QuantumCircuit(2*n)
    for k in range(n):
        qcLam.swap(k, n+k)
    TLam = Operator(qcLam).data

    R0 = np.diag([1.0 if ((idx >> n) == 0) else -1.0 for idx in range(2**(2*n))])

    UW = R0 @ TW.conj().T @ TLam @ TW
    return UW


def _test_build_U_W_simple_example():

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
    # PASSES correctly


def build_U_W_tilde(model: IsingModel, beta: float) -> np.ndarray:

    n = model.n
    V = MovePreparationV(model)
    B = BoltzmannCoinB(model, beta)
    F = MoveAcceptanceF(model, construction="linear")
    R = Reflection00(model)

    S = list(range(n))
    M = list(range(n, 2*n))
    C = 2*n

    qc_tilde = QuantumCircuit(2*n + 1)
    qc_tilde.append(V, M)
    qc_tilde.append(B, S + M + [C])
    qc_tilde.append(F, S + M + [C])
    qc_tilde.append(B.inverse(), S + M + [C])
    qc_tilde.append(V.inverse(), M)
    qc_tilde.append(R, M + [C])

    U_W_tilde = Operator(qc_tilde).data
    return U_W_tilde


def _test_build_U_W_tilde_simple_example():

    # 0) instantiate model + P
    example_model = IsingModel.from_list_clauses(
        k=2, d=3, n=4,
        Js=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.125],
        terms=[(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3), (1,3)]
    )

    beta = 1.0
    P = build_transition_matrix_P(example_model, beta)
    n = example_model.n
    d = 2**n

    # 1) instantiate U_W_tilde using build_U_W_tilde
    U_W_tilde = build_U_W_tilde(example_model, beta)

    # 2) iterate on each entry (x, y) in P
    # 2.1) make sure the move z = x ^ y has exactly one bit set
    # 2.2) check that <y,0,0| U_W_tilde |x,0,0> = sqrt(P[x,y] P[y,x])
    #
    # With qubit order S (n qubits) then M (n qubits) then C (1 qubit),
    # the basis index for |x>_S |0>_M |0>_C is just x.
    def get_entry(x, y):
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
    # PASSES correctly


def build_isometry_Y(n: int) -> np.ndarray:
    d = 2**n
    # maps |x> (meaning |x>_A|0>_B in the small UW setting) -> |x>_S|0>_M|0>_C
    Y = np.zeros((2**(2*n + 1), d), dtype=complex)
    for x in range(d):
        Y[x, x] = 1.0   # because M=0 and C=0 contribute no offset in little-endian indexing
    return Y

def close_up_to_phase(A, B, atol=1e-8) -> bool:
    ov = np.vdot(A.ravel(), B.ravel())
    phase = 1.0 if abs(ov) < 1e-14 else np.exp(1j * np.angle(ov))
    return np.allclose(A, B * phase, atol=atol)


def test_build_isometry_simple_example():

    example_model = IsingModel.from_list_clauses(
        k=2, d=3, n=4,
        Js=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.125],
        terms=[(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3), (1,3)]
    )
    beta = 1.0
    n = example_model.n
    d = 2**n

    U_W_original = build_U_W(example_model, beta)          # (2^(2n) x 2^(2n))
    U_W_tilde    = build_U_W_tilde(example_model, beta)    # (2^(2n+1) x 2^(2n+1))

    Y = build_isometry_Y(n)                                # (2^(2n+1) x 2^n)
    U_from_tilde = Y.conj().T @ U_W_tilde @ Y              # (2^n x 2^n)

    # original restricted to E0 (B=0) is the top-left d×d block
    U_supported = U_W_original[:d, :d]

    assert close_up_to_phase(U_from_tilde, U_supported, atol=1e-8)
