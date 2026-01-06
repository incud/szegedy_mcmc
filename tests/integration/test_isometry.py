import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from szegedymcmc.gates.szegedyw import SzegedyW
from szegedymcmc.ising import IsingModel
from szegedymcmc.gates import MovePreparationV, BoltzmannCoinB, MoveAcceptanceF, Reflection00
from szegedymcmc.ising.transition_matrix import build_transition_matrix_P
from szegedymcmc.integration.build_u_w import build_U_W
from szegedymcmc.integration.build_u_w_tilde import build_U_W_tilde

from tests.test_utils import close_up_to_phase

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# Assumed available in your project:
# - build_U_W(model, beta) -> np.ndarray          # 2^(2n) x 2^(2n)
# - build_U_W_tilde(model, beta) -> np.ndarray    # 2^(2n+1) x 2^(2n+1)
# - build_transition_matrix_P(model, beta) -> np.ndarray
# - SzegedyW(P) gate
# - MovePreparationV(model), BoltzmannCoinB(model,beta), MoveAcceptanceF(model,"linear")
# - close_up_to_phase(A,B,atol)

def close_up_to_phase(A, B, atol=1e-8):
    ov = np.vdot(A.ravel(), B.ravel())
    phase = 1.0 if abs(ov) < 1e-14 else np.exp(1j * np.angle(ov))
    return np.allclose(A, B / phase, atol=atol)


# -----------------------------------------------------------------------------
# 1) CLEAN-ANCILLA X-BLOCK TEST
# -----------------------------------------------------------------------------
# U_W lives on LR (2n qubits, dimension d^2). "Clean" means R=0:
#   |x>_L|0>_R   (note: this "0" is a *configuration basis state*, not a move)
#
# U_W_tilde lives on SMC (2n+1 qubits, dimension 2*d^2). "Clean" means M=0, C=0:
#   |x>_S|0>_M|0>_C
#
# Isometries from C^d into each space:
#   Y_small: |x> -> |x>_L|0>_R
#   Y_tilde: |x> -> |x>_S|0>_M|0>_C
#
# Then compare induced d×d operators:  X1 = Y_small† U_W Y_small,  X2 = Y_tilde† U_W_tilde Y_tilde.

def build_Y_clean_small(n: int) -> np.ndarray:
    d = 2**n
    Y = np.zeros((d*d, d), dtype=complex)
    # indexing convention used in your code: idx(|a>_L|b>_R) = a + d*b
    Y[np.arange(d), np.arange(d)] = 1.0  # b=0
    return Y

def build_Y_clean_tilde(n: int) -> np.ndarray:
    d = 2**n
    Y = np.zeros((2*d*d, d), dtype=complex)
    # qubit order S (n) then M (n) then C (1)
    # idx(|x>_S|0>_M|0>_C) = x
    Y[np.arange(d), np.arange(d)] = 1.0
    return Y

def _test_isometry_clean_X():

    model = IsingModel.from_list_clauses(
        k=2, d=3, n=4,
        Js=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.125],
        terms=[(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3), (1,3)],
    )
    beta = 1.0
    atol = 1e-8

    n = model.n
    d = 2**n

    U_W = build_U_W(model, beta)
    U_W_tilde = build_U_W_tilde(model, beta)

    Y1 = build_Y_clean_small(n)
    Y2 = build_Y_clean_tilde(n)

    X1 = Y1.conj().T @ U_W @ Y1
    X2 = Y2.conj().T @ U_W_tilde @ Y2

    assert close_up_to_phase(X1, X2, atol=atol), "clean-subspace X blocks mismatch (up to phase)"

    # Optional: compare to theory X_{yx} = sqrt(P[x,y] P[y,x]) for MH single-spin flips
    P = build_transition_matrix_P(model, beta)
    X_th = np.sqrt(P * P.T)
    assert np.allclose(X1, X_th, atol=atol)
    assert np.allclose(X2, X_th, atol=atol)


# -----------------------------------------------------------------------------
# 2) EDGE-SPACE TEST (OUTGOING AMPLITUDE MAP)
# -----------------------------------------------------------------------------
# This reconciles your "y=0 vs z=0" observation properly:
# - The clean block uses |0>_R (a fixed configuration) vs |0>_M|0>_C (clean ancillas).
# - If you want an "edge-labeled" comparison, you should compare objects that
#   actually *prepare edges* on both sides.
#
# In the Szegedy oracle W:
#   W |x>_L|0>_R = sum_y sqrt(P[x,y]) |y>_L |x>_R.
# So the amplitude on the *directed edge basis* (x -> y) is exactly sqrt(P[x,y]),
# visible as matrix element <y,x|W|x,0>.
#
# In the circuit construction, the prefix U_prop := F B V prepares the *accepted-edge*
# branch amplitudes:
#   starting from |x>_S|0>_M|0>_C, the amplitude on |y>_S|m(onehot j)>_M|1>_C
#   (with y = x xor 2^j) is sqrt(T_{xy} A_{xy}) = sqrt(P[x,y]) for off-diagonal edges,
#   since T_{xy}=1/n for single-spin flips.
#
# Therefore we compare two isometry-induced maps from |x> to an edge-labeled space:
#   S_W[x -> (x,j)]   := <y,x|W|x,0>
#   S_prop[x -> (x,j)] := <y,m,1| (F B V) |x,0,0>
# on the directed-edge index (x,j) with y = x xor 2^j.
#
# This is a clean, strong edge-space equivalence you can expect to hold.

def test_isometry_edge_outgoing_map():

    model = IsingModel.from_list_clauses(
        k=2, d=3, n=4,
        Js=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.125],
        terms=[(0,), (1,), (2,), (3,), (0,1), (0,2), (0,3), (1,3)],
    )
    beta = 1.0
    atol = 1e-8

    n = model.n
    d = 2**n
    P = build_transition_matrix_P(model, beta)

    # --- build W matrix (2n qubits) ---
    TW = Operator(SzegedyW(P)).data

    # --- build U_prop = F B V matrix (2n+1 qubits) ---
    V = MovePreparationV(model)
    B = BoltzmannCoinB(model, beta)
    F = MoveAcceptanceF(model, construction="linear")

    S = list(range(n))
    M = list(range(n, 2*n))
    C = 2*n

    qc_prop = QuantumCircuit(2*n + 1, name="Uprop=FBV")
    qc_prop.append(V, M)
    qc_prop.append(B, S + M + [C])
    qc_prop.append(F, S + M + [C])
    Tprop = Operator(qc_prop).data

    # --- build the two "edge amplitude" matrices of shape (d*n) x d ---
    # Row ordering: r = x*n + j corresponds to directed edge (x -> y=x xor 2^j).
    # Column is x (input configuration).
    SW = np.zeros((d*n, d), dtype=complex)
    SP = np.zeros((d*n, d), dtype=complex)

    # Indexing conventions:
    # - For W on LR: idx_in = x + d*0 = x, idx_out(|y>_L|x>_R) = y + d*x
    idx_in_W = lambda x: x
    idx_out_W = lambda y, x: y + d*x

    # - For FBV on SMC: idx_in = x (since M=0,C=0), idx_out(|y>_S|m>_M|1>_C) = y + d*m + d^2
    idx_in_prop = lambda x: x
    idx_out_prop = lambda y, m: y + d*m + d*d

    for x in range(d):
        for j in range(n):
            y = x ^ (1 << j)
            r = x*n + j
            m = 1 << j

            # W amplitude = <y,x|W|x,0> = sqrt(P[x,y]) (includes self-loop too, but y!=x here)
            SW[r, x] = TW[idx_out_W(y, x), idx_in_W(x)]
            SW_r_x = np.real_if_close(SW[r, x])

            # FBV accepted-edge amplitude = <y,m,1|FBV|x,0,0> = sqrt(P[x,y]) for y!=x
            SP[r, x] = Tprop[idx_out_prop(y, m), idx_in_prop(x)]
            SP_r_x = np.real_if_close(-1j * SP[r, x])

            # Optional: also compare each individually to sqrt(P[x,y])
            expected = np.sqrt(P[x, y])
            assert np.isclose(SW_r_x, expected, atol=atol), f"W edge amp mismatch at x={x}, j={j}"
            assert np.isclose(SP_r_x, expected, atol=atol), f"FBV edge amp mismatch at x={x}, j={j}"

    # Strong form: the two edge-amplitude maps agree (no phase ambiguity expected here)
    assert close_up_to_phase(SW, SP, atol=atol), "edge outgoing maps W vs (F B V) mismatch"

