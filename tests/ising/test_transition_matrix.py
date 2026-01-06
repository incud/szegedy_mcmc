"""
These tests validate `build_transition_matrix_P(model, beta)` for the simplified
Metropolis–Hastings chain on Ising configurations using *single spin-flip* proposals.

Model / state conventions:

- There are n spins, so the classical state space has size d = 2^n.
- A configuration is represented as an integer x in {0, ..., d-1}, interpreted in
  little-endian bit order: bit k of x is the 0/1 value of spin k.
- The Ising energy is computed from these bits via the project’s `energy_bits` logic
  (or equivalently via the model terms); `build_transition_matrix_P` uses energy
  differences Δ = E(y) - E(x) for y = x ^ (1<<j) (flip spin j).

Transition matrix construction being tested:

- Proposal kernel T: choose a spin j uniformly, propose y = x ^ (1<<j), so
    T[x,y] = 1/n  if y differs from x by exactly one bit, else 0.
- MH acceptance A(x->y): for inverse temperature beta,
    A(x->y) = 1                    if E(y) <= E(x)
            = exp(-beta (E(y)-E(x))) otherwise.
- Off-diagonal transitions:
    P[x,y] = T[x,y] * A(x->y) = (1/n) * A(x->y) for single-bit neighbors.
- Self-loop:
    P[x,x] = 1 - sum_{y != x} P[x,y]   so that each row sums to 1.

The tests check:
(A) Shape and row-stochasticity.
(B) Sparsity pattern: nonzero off-diagonals only for Hamming-distance-1 neighbors.
(C) Detailed balance via the MH ratio:
      P[x,y] / P[y,x] = exp(-beta (E(y)-E(x))) for single-spin-flip neighbors,
    which implies reversibility with respect to the Boltzmann distribution.
(D) Exact self-loop identity consistent with the constructed off-diagonals.

This test suite is intentionally small and direct: it does not rely on helper “run_check”
utilities and is suitable for pytest (no arguments in test functions).
"""

import numpy as np
from szegedymcmc.ising.ising_model import IsingModel
from szegedymcmc.ising.transition_matrix import build_transition_matrix_P


def _energy_bits(model: IsingModel, x_int: int) -> float:
    spins = 1 - 2 * np.array([(x_int >> k) & 1 for k in range(model.n)], dtype=float)  # 0->+1, 1->-1
    return sum(J * np.prod(spins[list(om)]) for J, om in zip(model.J, model.Omega))


def test_build_transition_matrix_P_single_spin_flip_simple_example():
    model = IsingModel.from_list_clauses(
        k=2, d=3, n=4,
        Js=[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.875, 0.125],
        terms=[(0,), (1,), (2,), (3,), (0, 1), (0, 2), (0, 3), (1, 3)],
    )
    beta = 1.0
    P = build_transition_matrix_P(model, beta)

    n = model.n
    d = 2**n

    # (A) shape + row-stochasticity + non-negativity
    assert P.shape == (d, d)
    assert np.all(P >= -1e-14)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)

    for x in range(d):
        Ex = _energy_bits(model, x)

        # (B) sparsity pattern + (C) MH ratio on neighbors + diagonal formula
        off_diag_sum = 0.0

        for y in range(d):
            if y == x:
                continue

            hd = (x ^ y).bit_count()
            if hd != 1:
                # must be exactly zero (up to numerical noise)
                assert abs(P[x, y]) < 1e-14, f"unexpected nonzero P[{x},{y}] for Hamming distance {hd}"
                continue

            # neighbor: y = x ^ (1<<j)
            j = (x ^ y).bit_length() - 1
            assert y == (x ^ (1 << j))

            Ey = _energy_bits(model, y)
            delta = Ey - Ex

            # Expected MH acceptance
            A_xy = 1.0 if delta <= 0 else float(np.exp(-beta * delta))
            expected = A_xy / n

            assert np.isclose(P[x, y], expected, atol=1e-12), (
                f"off-diagonal mismatch at x={x}, y={y} (j={j}): got {P[x,y]}, expected {expected}"
            )

            # Detailed balance ratio check:
            # P[x,y]/P[y,x] = exp(-beta (E(y)-E(x))) for neighbors
            # (handle both directions without special-casing acceptance)
            ratio = P[x, y] / P[y, x]
            rhs = float(np.exp(-beta * (Ey - Ex)))
            assert np.isclose(ratio, rhs, atol=1e-10), (
                f"DB ratio mismatch at (x,y)=({x},{y}): got {ratio}, expected {rhs}"
            )

            off_diag_sum += P[x, y]

        # (D) diagonal is 1 - sum_{y!=x} P[x,y]
        assert np.isclose(P[x, x], 1.0 - off_diag_sum, atol=1e-12), (
            f"diagonal mismatch at x={x}: got {P[x,x]}, expected {1.0 - off_diag_sum}"
        )
