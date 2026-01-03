from dataclasses import dataclass


@dataclass
class IsingModel:
    
    k: int  # upper bound on the number of spins interacting in a single term
    d: int  # upper bound on the number or terms containing a shared spin
    n: int  # number of spins
    m: int  # number of terms
    J: tuple      # sequence of floats containing the J value for each term
    Omega: tuple  # sequence of tuple[int, ...] containing the spins forming the term

    @staticmethod
    def from_list_clauses(k: int, d: int, n: int, Js: list[float], terms: list[tuple[int,...]]):
        assert 0 < n, "The number of spins n is not positive"
        assert 0 <= d < n, f"Invalid {d=}"
        assert 0 <= k <= n, f"Invalid {k=}"
        assert len(Js) == len(terms) > 0, "Invalid or mismatching number of terms"
        # add terms
        J, Omega = [], []
        for w, om in zip(Js, terms):
            assert len(om) > 0, f"Clause {om} invalid: there must be at least one spin in the interaction term"
            assert len(om) <= k, f"Clause {om} invalid: the number of spin in this term is {len(om)} while the maximum allowed is {k=}"
            assert len(set(om)) == len(om), f"Clause {om} invalid: there are duplicated spins"
            assert all(0 <= i < n for i in om), f"Clause {om} invalid: not all spins are between 0 and {n-1}=n-1"
            J.append(float(w))
            Omega.append(tuple(int(i) for i in sorted(om)))

        # constraint on the number or terms containing a shared spin 
        neigh = [set() for _ in range(n)]
        for om in Omega:
            for spin in om:
                neigh[spin].update(set(om) - {spin})
        if max(len(neigh[spin]) for spin in range(n)) > d:
            raise ValueError("Degree bound d violated.")

        return IsingModel(k=k, d=d, n=n, m=len(Omega), J=tuple(J), Omega=tuple(Omega))