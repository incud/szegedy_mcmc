import numpy as np

from szegedymcmc.ising.ising_model import IsingModel


def energy_bits(model: IsingModel, x_int: int) -> float:
    spins = 1 - 2*np.array([(x_int >> k) & 1 for k in range(model.n)], dtype=float)  # 0->+1, 1->-1
    return sum(J * np.prod(spins[list(om)]) for J, om in zip(model.J, model.Omega))


def build_transition_matrix_P(model: IsingModel, beta: float) -> np.ndarray:

    n = model.n
    d = 2**n
    P = np.zeros((d, d), dtype=float)

    for x in range(d):

        Ex = energy_bits(model, x)
        for j in range(n):
            y = x ^ (1 << j)
            Ey = energy_bits(model, y)
            A = 1.0 if (Ey - Ex) <= 0 else float(np.exp(-beta * (Ey - Ex)))
            P[x, y] = A / n

        P[x, x] = 1.0 - (P[x].sum() - P[x, x])

    return P
